"""
THEMIS C2I Batch Dispatcher - Batch API Mode for Large-Scale Evaluation

Workflow:
  Phase 1: Router batch  → all images get initial plans
  Phase 2: Judge batch   → all plans get reviewed
  Phase 3: Revision loop → rejected plans get revised (up to max_iterations)
  Phase 4: GPU execution → approved plans run through expert models

Uses OpenAI-compatible Batch API (Dashscope/Aliyun).
Cost: ~50% discount vs real-time API.
Latency: minutes to hours per batch (not real-time).

Usage:
  python c2i/run.py --mode batch --step 123 --limit 1000
  python c2i/run.py --mode batch --step 12 --limit 1000
  python c2i/run.py --mode batch --step 3
"""

import os
import sys
import json
import time
import base64
from pathlib import Path
from datetime import datetime
from openai import OpenAI

from common import (
    PROJECT_ROOT, C2I_DIR, IMAGE_DIR, CLASS_IDS_TXT,
    EXPERTS_REGISTRY_JSON, PLAN_DIR, APPROVED_DIR,
    JUDGE_FEEDBACK_DIR, EXPERT_RESULTS_DIR, BATCH_DIR,
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
    build_image_list, resolve_image_path, save_judge_feedback,
    preload_expert_managers,
)

from step1_router import (
    encode_image, load_experts_registry, get_taxonomy_info,
    build_router_prompt, build_router_revision_prompt,
    build_router_registry_summary, extract_expert_ids,
    validate_plan, parse_json_safely,
    _COMMON_ROUTER_INSTRUCTIONS, _build_context_block,
)
from step2_judge import (
    build_judge_prompt, build_judge_registry_summary,
)
from step3_execute import (
    execute_plan, save_testimony_bundle,
    load_approved_plans, resolve_image_path as resolve_image_path_global,
    collect_required_expert_ids, EXPERT_MODULE_MAP,
)

ROUTER_MODEL = "qwen3.6-plus"
JUDGE_MODEL = "qwen3.6-plus"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  JSONL Request Builders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_router_request(
    img_id: str,
    image_path: str,
    class_id: int,
    class_label: str,
    experts_registry_str: str,
) -> dict:
    """Build a single batch request line for Router."""
    taxonomy_info = get_taxonomy_info(class_id)
    base64_image = encode_image(image_path)
    prompt = build_router_prompt(class_label, taxonomy_info, experts_registry_str)
    _, expert_ids_str, registry_summary = _build_context_block(
        class_label, taxonomy_info, experts_registry_str
    )
    formatted_instructions = _COMMON_ROUTER_INSTRUCTIONS.format(expert_ids_str=expert_ids_str)

    system_msg = (
        "You are a highly logical Router Agent for image auditing. "
        "You must prioritize the provided Taxonomy Knowledge as the source of truth. "
        "Output JSON only."
    )
    system_parts = [system_msg, formatted_instructions,
                    f"**[Expert Registry (Available Tools)]**\n{registry_summary}"]
    system_text = "\n\n".join(system_parts)

    return {
        "custom_id": f"router_{img_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": ROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system_text},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        },
    }


def _build_judge_request(
    img_id: str,
    image_path: str,
    class_id: int,
    class_label: str,
    plan: dict,
    experts_registry_str: str,
) -> dict:
    """Build a single batch request line for Judge."""
    taxonomy_info = get_taxonomy_info(class_id)
    base64_image = encode_image(image_path)
    prompt, registry_summary = build_judge_prompt(
        plan, class_label, taxonomy_info, experts_registry_str
    )

    system_msg = (
        "You are a meta-cognitive Judge Agent. You rigorously analyze evaluation strategies "
        "for logical rigor, scientific validity, and completeness. You must output JSON only."
    )
    system_text = f"{system_msg}\n\n**[Expert Registry]**\n{registry_summary}"

    return {
        "custom_id": f"judge_{img_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": JUDGE_MODEL,
            "messages": [
                {"role": "system", "content": system_text},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        },
    }


def _build_revision_request(
    img_id: str,
    image_path: str,
    class_id: int,
    class_label: str,
    previous_plan: dict,
    feedback_history: list[dict],
    experts_registry_str: str,
) -> dict:
    """Build a single batch request line for Router revision."""
    taxonomy_info = get_taxonomy_info(class_id)
    base64_image = encode_image(image_path)
    prompt = build_router_revision_prompt(
        class_label, taxonomy_info, experts_registry_str,
        previous_plan, feedback_history,
    )
    _, expert_ids_str, registry_summary = _build_context_block(
        class_label, taxonomy_info, experts_registry_str
    )
    formatted_instructions = _COMMON_ROUTER_INSTRUCTIONS.format(expert_ids_str=expert_ids_str)

    system_msg = (
        "You are a highly logical Router Agent for image auditing. "
        "You must prioritize the provided Taxonomy Knowledge as the source of truth. "
        "Output JSON only."
    )
    system_parts = [system_msg, formatted_instructions,
                    f"**[Expert Registry (Available Tools)]**\n{registry_summary}"]
    system_text = "\n\n".join(system_parts)

    return {
        "custom_id": f"revision_{img_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": ROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system_text},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        },
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Batch API Operations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def write_batch_jsonl(requests: list[dict], output_path: Path) -> Path:
    """Write batch requests to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    print(f"  [BATCH] Wrote {len(requests)} requests -> {output_path.name}")
    return output_path


def submit_batch(client: OpenAI, jsonl_path: Path, description: str = "") -> str:
    """Upload JSONL and submit a batch job. Returns batch_id."""
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )
    print(f"  [BATCH] Submitted: {batch.id} (file: {file_obj.id})")
    return batch.id


def poll_batch(client: OpenAI, batch_id: str, poll_interval: int = 30) -> dict:
    """Poll batch status until completion. Returns batch object."""
    print(f"  [BATCH] Polling {batch_id}...")
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        completed = batch.request_counts.completed if batch.request_counts else 0
        total = batch.request_counts.total if batch.request_counts else 0

        print(f"    Status: {status} | Progress: {completed}/{total}", end="\r")

        if status in ("completed", "failed", "cancelled", "expired"):
            print(f"\n  [BATCH] Final status: {status}")
            return batch

        time.sleep(poll_interval)


def download_batch_results(client: OpenAI, batch: object) -> list[dict]:
    """Download and parse batch output file."""
    if batch.status != "completed":
        print(f"  [BATCH] Cannot download: batch status is {batch.status}")
        if batch.errors:
            for err in batch.errors.data[:5]:
                print(f"    Error: {err.message}")
        return []

    output_file_id = batch.output_file_id
    if not output_file_id:
        print("  [BATCH] No output file available")
        return []

    content = client.files.content(output_file_id)
    lines = content.text.strip().split("\n")

    results = []
    for line in lines:
        if line.strip():
            results.append(json.loads(line))

    print(f"  [BATCH] Downloaded {len(results)} results")
    return results


def parse_batch_response(result: dict) -> tuple[str, str | None]:
    """Extract custom_id and response content from a batch result line."""
    custom_id = result.get("custom_id", "")
    response = result.get("response", {})

    if response.get("status_code") != 200:
        error = response.get("body", {}).get("error", {}).get("message", "Unknown error")
        print(f"  [BATCH] {custom_id} failed: {error}")
        return custom_id, None

    body = response.get("body", {})
    choices = body.get("choices", [])
    if not choices:
        return custom_id, None

    content = choices[0].get("message", {}).get("content", "")
    return custom_id, content


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Batch Pipeline Orchestration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_batch_step12(
    valid_images: list[tuple],
    image_dir: Path,
    experts_registry_str: str,
    max_iterations: int,
    plan_dir: Path,
    approved_dir: Path,
    judge_feedback_dir: Path | None,
    batch_dir: Path,
    poll_interval: int,
) -> dict:
    """Run Step 1+2 via Batch API with iteration support."""
    client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
    stats = {"router_ok": 0, "router_fail": 0, "judge_approved": 0, "judge_rejected": 0}
    batch_dir.mkdir(parents=True, exist_ok=True)
    plan_dir.mkdir(parents=True, exist_ok=True)
    approved_dir.mkdir(parents=True, exist_ok=True)

    # Build image info map
    image_info: dict[str, dict] = {}
    for img_name, img_id, class_id, class_label in valid_images:
        img_path = resolve_image_path(image_dir, img_id)
        if img_path is None:
            print(f"  [WARN] Image not found: {img_id}")
            stats["router_fail"] += 1
            continue
        image_info[img_id] = {
            "image_path": str(img_path),
            "class_id": class_id,
            "class_label": class_label,
        }

    # ── Phase 1: Router Batch ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 1: Router Batch ({len(image_info)} images)")
    print(f"{'='*60}")

    router_requests = []
    for img_id, info in image_info.items():
        req = _build_router_request(
            img_id, info["image_path"], info["class_id"],
            info["class_label"], experts_registry_str,
        )
        router_requests.append(req)

    jsonl_path = write_batch_jsonl(
        router_requests, batch_dir / "router_batch.jsonl"
    )
    batch_id = submit_batch(client, jsonl_path, "THEMIS Router Batch")
    batch_obj = poll_batch(client, batch_id, poll_interval)
    router_results = download_batch_results(client, batch_obj)

    # Parse router results into plans
    plans: dict[str, dict] = {}
    for result in router_results:
        custom_id, content = parse_batch_response(result)
        img_id = custom_id.replace("router_", "")

        if content is None:
            stats["router_fail"] += 1
            continue

        plan = parse_json_safely(content)
        if plan is None:
            print(f"  [{img_id}] Router returned unparseable JSON")
            stats["router_fail"] += 1
            continue

        info = image_info[img_id]
        plan["metadata"] = {
            "original_image": info["image_path"],
            "class_id": info["class_id"],
            "class_label": info["class_label"],
            "router_cost_seconds": 0,
            "plan_valid": validate_plan(plan, experts_registry_str),
        }

        plans[img_id] = plan
        stats["router_ok"] += 1

        save_path = plan_dir / f"plan_{img_id}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=4, ensure_ascii=False)

    print(f"  Router results: {stats['router_ok']} OK, {stats['router_fail']} failed")

    # ── Phase 2+3: Judge Batch (with iterations) ───────────────
    pending_plans = dict(plans)
    feedback_history: dict[str, list[dict]] = {img_id: [] for img_id in pending_plans}

    for iteration in range(1, max_iterations + 1):
        if not pending_plans:
            break

        print(f"\n{'='*60}")
        print(f"  Phase 2: Judge Batch - Iteration {iteration}/{max_iterations} "
              f"({len(pending_plans)} plans)")
        print(f"{'='*60}")

        judge_requests = []
        for img_id, plan in pending_plans.items():
            info = image_info[img_id]
            req = _build_judge_request(
                img_id, info["image_path"], info["class_id"],
                info["class_label"], plan, experts_registry_str,
            )
            judge_requests.append(req)

        jsonl_path = write_batch_jsonl(
            judge_requests, batch_dir / f"judge_batch_iter{iteration}.jsonl"
        )
        batch_id = submit_batch(client, jsonl_path, f"THEMIS Judge Iter{iteration}")
        batch_obj = poll_batch(client, batch_id, poll_interval)
        judge_results = download_batch_results(client, batch_obj)

        # Process judge verdicts
        approved_this_round: list[str] = []
        rejected_this_round: list[str] = []

        for result in judge_results:
            custom_id, content = parse_batch_response(result)
            img_id = custom_id.replace("judge_", "")

            if content is None or img_id not in pending_plans:
                continue

            verdict = parse_json_safely(content)
            if verdict is None:
                continue

            is_approved = verdict.get("is_approved", False)

            if judge_feedback_dir:
                judge_feedback_dir.mkdir(parents=True, exist_ok=True)
                save_judge_feedback(
                    judge_feedback_dir, img_id, iteration,
                    verdict, pending_plans[img_id],
                    image_info[img_id]["class_label"],
                )

            if is_approved:
                approved_this_round.append(img_id)
                plans[img_id]["metadata"]["judge_approved"] = True
                plans[img_id]["metadata"]["judge_iterations"] = iteration
                stats["judge_approved"] += 1
            else:
                rejected_this_round.append(img_id)
                feedback_history[img_id].append({
                    "reasons_for_rejection": verdict.get("reasons_for_rejection", ""),
                    "suggestions": verdict.get("suggestions", []),
                })

        # Remove approved from pending
        for img_id in approved_this_round:
            pending_plans.pop(img_id, None)

        print(f"  Approved: {len(approved_this_round)}, "
              f"Rejected: {len(rejected_this_round)}")

        # If last iteration, force-approve remaining
        if iteration == max_iterations and pending_plans:
            print(f"  [WARN] Force-approving {len(pending_plans)} remaining plans")
            for img_id in list(pending_plans.keys()):
                plans[img_id]["metadata"]["judge_approved"] = False
                plans[img_id]["metadata"]["judge_forced"] = True
                plans[img_id]["metadata"]["judge_iterations"] = iteration
                stats["judge_rejected"] += 1
            pending_plans.clear()
            break

        # Phase 3: Revision batch for rejected plans
        if rejected_this_round and iteration < max_iterations:
            print(f"\n  Phase 3: Revision Batch ({len(rejected_this_round)} plans)")

            revision_requests = []
            for img_id in rejected_this_round:
                info = image_info[img_id]
                req = _build_revision_request(
                    img_id, info["image_path"], info["class_id"],
                    info["class_label"], pending_plans[img_id],
                    feedback_history[img_id], experts_registry_str,
                )
                revision_requests.append(req)

            jsonl_path = write_batch_jsonl(
                revision_requests, batch_dir / f"revision_batch_iter{iteration}.jsonl"
            )
            batch_id = submit_batch(client, jsonl_path, f"THEMIS Revision Iter{iteration}")
            batch_obj = poll_batch(client, batch_id, poll_interval)
            revision_results = download_batch_results(client, batch_obj)

            for result in revision_results:
                custom_id, content = parse_batch_response(result)
                img_id = custom_id.replace("revision_", "")

                if content is None or img_id not in pending_plans:
                    continue

                revised_plan = parse_json_safely(content)
                if revised_plan is None:
                    continue

                info = image_info[img_id]
                revised_plan["metadata"] = {
                    "original_image": info["image_path"],
                    "class_id": info["class_id"],
                    "class_label": info["class_label"],
                    "router_cost_seconds": 0,
                    "plan_valid": validate_plan(revised_plan, experts_registry_str),
                }

                pending_plans[img_id] = revised_plan
                plans[img_id] = revised_plan

                save_path = plan_dir / f"plan_{img_id}_rev{iteration}.json"
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(revised_plan, f, indent=4, ensure_ascii=False)

    # ── Save all approved plans ────────────────────────────────
    for img_id, plan in plans.items():
        approved_path = approved_dir / f"approved_plan_{img_id}.json"
        with open(approved_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=4, ensure_ascii=False)

    print(f"\n  [BATCH] Saved {len(plans)} approved plans to {approved_dir}")
    return stats


def run_batch_step3(
    approved_dir: Path,
    expert_results_dir: Path,
    expert_managers: list,
    image_id_filter: str = "",
    limit: int = 0,
) -> dict:
    """Run Step 3 (GPU execution) on approved plans from batch."""
    from dispatch_async import run_step3_async
    import asyncio

    stats = asyncio.run(run_step3_async(
        approved_dir=approved_dir,
        expert_results_dir=expert_results_dir,
        expert_managers=expert_managers,
        image_id_filter=image_id_filter,
        limit=limit,
    ))
    return stats


def run_batch_pipeline(
    valid_images: list[tuple],
    image_dir: Path,
    experts_registry_str: str,
    max_iterations: int,
    plan_dir: Path,
    approved_dir: Path,
    judge_feedback_dir: Path | None,
    expert_results_dir: Path,
    batch_dir: Path,
    expert_managers: list | None,
    poll_interval: int,
    step: str,
) -> dict:
    """Full batch pipeline orchestration."""
    stats = {}

    run_step12 = step in ("12", "123")
    run_step3 = step in ("3", "123")

    if run_step12:
        step12_stats = run_batch_step12(
            valid_images=valid_images,
            image_dir=image_dir,
            experts_registry_str=experts_registry_str,
            max_iterations=max_iterations,
            plan_dir=plan_dir,
            approved_dir=approved_dir,
            judge_feedback_dir=judge_feedback_dir,
            batch_dir=batch_dir,
            poll_interval=poll_interval,
        )
        stats.update(step12_stats)

    if run_step3:
        if expert_managers is None or not expert_managers:
            print("[ERROR] No expert managers loaded for Step 3.")
            return stats

        step3_stats = run_batch_step3(
            approved_dir=approved_dir,
            expert_results_dir=expert_results_dir,
            expert_managers=expert_managers,
            limit=0,
        )
        stats.update(step3_stats)

    return stats

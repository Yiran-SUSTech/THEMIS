"""
THEMIS T2I Sync Dispatcher - Sequential single-image processing.

Called from run.py with --mode sync.
Pipeline: Step 0 (Atomize) → Step 1 (Router) → Step 2 (Judge) → Step 3 (Expert) → Step 4 (Reflector)
"""

import os
import sys
import json
import time
from pathlib import Path
from openai import OpenAI

from common import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
    resolve_image_path, save_judge_feedback,
    ATOMIZED_DIR, GENEVAL2_DATA_JSONL,
    load_geneval2_data,
)

from step0_atomize import atomize_prompt, save_atomized_prompt
from step1_router import generate_plan, revise_plan, load_experts_registry
from step2_judge import review_plan
from step4_reflector import run_reflector, save_final_report, print_final_summary

# Import expert execution from c2i_harness (reused)
from c2i_harness.step3_execute import (
    ExpertManager, execute_plan, save_testimony_bundle,
    load_approved_plans, resolve_image_path as resolve_image_path_global,
)


def _run_single_image(
    client: OpenAI,
    image_path: str,
    img_id: str,
    prompt_text: str,
    atomized_data: dict,
    experts_registry_str: str,
    max_iterations: int,
    plan_dir: Path,
    approved_dir: Path,
    judge_feedback_dir: Path | None,
    api_retry: int = 0,
    temp_router: float = 0.0,
    temp_judge: float = 0.0,
) -> dict | None:
    """Run Router+Judge loop for a single image (synchronous)."""
    print(f"\n{'#'*60}")
    print(f"  Image: {img_id} | Prompt: {prompt_text[:80]}")
    print(f"{'#'*60}")

    print(f"\n  [Step 1] Router generating initial plan...")
    current_plan = generate_plan(
        client, image_path, img_id, prompt_text, atomized_data,
        experts_registry_str,
        api_retry=api_retry, temperature=temp_router,
    )
    if current_plan is None:
        print(f"  [Step 1] FAILED - Router could not generate plan\n")
        return None

    is_valid = current_plan.get("metadata", {}).get("plan_valid", False)
    print(f"  [Step 1] Plan generated | Valid: {is_valid} | "
          f"Cost: {current_plan['metadata']['router_cost_seconds']:.2f}s")

    plan_save_path = plan_dir / f"plan_{img_id}.json"
    with open(plan_save_path, "w", encoding="utf-8") as f:
        json.dump(current_plan, f, indent=4, ensure_ascii=False)

    feedback_history: list[dict] = []
    iteration_log: list[str] = []

    for iteration in range(1, max_iterations + 1):
        print(f"\n  [Step 2] Judge reviewing plan (iteration {iteration}/{max_iterations})...")
        judge_result = review_plan(
            client, image_path, img_id, prompt_text, atomized_data,
            current_plan, experts_registry_str,
            api_retry=api_retry, temperature=temp_judge,
        )

        if judge_result is None:
            print(f"  [Step 2] FAILED - Judge returned no valid response")
            iteration_log.append(f"Iteration {iteration}: Judge Error")
            break

        is_approved = judge_result.get("is_approved", False)
        reasons = judge_result.get("reasons_for_rejection", "")
        suggestions = judge_result.get("suggestions", [])

        print(f"  [Step 2] Verdict: {'Approved' if is_approved else 'Rejected'}")
        if not is_approved:
            print(f"  [Step 2] Reasons: {reasons}")

        if judge_feedback_dir is not None:
            save_judge_feedback(
                judge_feedback_dir, img_id, iteration,
                judge_result, current_plan, prompt_text,
            )

        if is_approved:
            iteration_log.append(f"Iteration {iteration}: Approved!")
            current_plan["metadata"]["judge_approved"] = True
            current_plan["metadata"]["judge_iterations"] = iteration
            break
        else:
            iteration_log.append(f"Iteration {iteration}: Rejected")

            if iteration == max_iterations:
                print(f"\n  [WARN] Max iterations reached. Forcing approval.")
                iteration_log.append(f"Iteration {iteration}: Max iterations - forced")
                current_plan["metadata"]["judge_approved"] = False
                current_plan["metadata"]["judge_forced"] = True
                current_plan["metadata"]["judge_iterations"] = iteration
                break

            feedback_history.append({
                "reasons_for_rejection": reasons,
                "suggestions": suggestions,
            })

            print(f"  [Step 1] Router revising plan...")
            revised_plan = revise_plan(
                client, image_path, img_id, prompt_text, atomized_data,
                experts_registry_str, current_plan, feedback_history,
                api_retry=api_retry, temperature=temp_router,
            )

            if revised_plan is not None:
                current_plan = revised_plan

    current_plan["metadata"]["iteration_log"] = iteration_log

    approved_save_path = approved_dir / f"approved_plan_{img_id}.json"
    with open(approved_save_path, "w", encoding="utf-8") as f:
        json.dump(current_plan, f, indent=4, ensure_ascii=False)
    print(f"\n  [SAVED] Approved plan -> {approved_save_path.name}")

    return current_plan


def run_sync_pipeline(
    valid_images: list[tuple],
    image_dir: Path,
    experts_registry_str: str,
    max_iterations: int,
    plan_dir: Path,
    approved_dir: Path,
    judge_feedback_dir: Path | None,
    expert_results_dir: Path,
    expert_managers: list,
    step: str,
    temp_router: float = 0.0,
    temp_judge: float = 0.0,
    temp_reflector: float = 0.5,
    final_reports_dir: Path | None = None,
    api_retry: int = 0,
) -> dict:
    """Run the full pipeline in synchronous serial mode.

    valid_images: list of (img_name, img_id, prompt_text) tuples.
    """
    stats = {
        "atomize_ok": 0, "atomize_fail": 0,
        "api_ok": 0, "api_fail": 0,
        "gpu_ok": 0, "gpu_fail": 0,
        "step4_ok": 0, "step4_fail": 0,
    }

    run_step12 = step in ("1", "2", "12", "123", "1234")
    run_step3 = step in ("3", "123", "1234")
    run_step4 = step in ("4", "1234")

    client = None
    if run_step12 or run_step4:
        client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

    expert_manager = expert_managers[0] if expert_managers else None

    # ── Step 0 + Step 1 + Step 2: Atomize + Router + Judge ───────
    if run_step12 and client:
        geneval2_data = load_geneval2_data(GENEVAL2_DATA_JSONL)

        for img_name, img_id, prompt_text in valid_images:
            img_path = resolve_image_path(image_dir, img_id)
            if img_path is None:
                print(f"[WARN] Image not found: {img_id}")
                stats["api_fail"] += 1
                continue

            # Step 0: Atomize prompt
            print(f"\n  [Step 0] Atomizing prompt for {img_id}...")
            prompt_data = geneval2_data.get(img_id, {})
            if not prompt_data:
                print(f"  [WARN] No GenEval2 data for prompt_id={img_id}")
                stats["atomize_fail"] += 1
                stats["api_fail"] += 1
                continue

            atomized_data = atomize_prompt(prompt_data)
            save_atomized_prompt(atomized_data, ATOMIZED_DIR, img_id)
            stats["atomize_ok"] += 1
            print(f"  [Step 0] Atoms: {atomized_data['atom_count']}, "
                  f"Objects: {len(atomized_data['objects'])}")

            # Steps 1+2: Router + Judge
            plan = _run_single_image(
                client, str(img_path), img_id, prompt_text, atomized_data,
                experts_registry_str, max_iterations,
                plan_dir, approved_dir, judge_feedback_dir,
                api_retry=api_retry, temp_router=temp_router, temp_judge=temp_judge,
            )
            if plan is not None:
                stats["api_ok"] += 1
            else:
                stats["api_fail"] += 1

    # ── Step 3 + Step 4: Expert Execution + Reflector ───────────
    if run_step3 or run_step4:
        if run_step3 and not expert_manager:
            print("[ERROR] No expert managers loaded for Step 3.")
            return stats

        plans = load_approved_plans(approved_dir)
        if not plans:
            print("[ERROR] No approved plans found. Run Step 1+2 first.")
            return stats

        # Load geneval2 data for atomization (needed for Reflector)
        geneval2_data = load_geneval2_data(GENEVAL2_DATA_JSONL)

        for idx, plan in enumerate(plans, 1):
            metadata = plan.get("metadata", {})
            image_path_raw = metadata.get("original_image", "")
            prompt_text = metadata.get("prompt_text", "")
            prompt_id = metadata.get("prompt_id", "")

            image_path = resolve_image_path_global(image_path_raw)
            if image_path is None:
                stats["gpu_fail"] += 1
                if run_step4:
                    stats["step4_fail"] += 1
                continue

            image_id = os.path.splitext(os.path.basename(image_path))[0]
            print(f"\n[{idx}/{len(plans)}] {os.path.basename(image_path)}")

            # Load or re-atomize prompt data for Reflector
            atomized_path = ATOMIZED_DIR / f"atomized_{image_id}.json"
            atomized_data = None
            if atomized_path.exists():
                with open(atomized_path, "r", encoding="utf-8") as f:
                    atomized_data = json.load(f)
            elif prompt_id and prompt_id in geneval2_data:
                atomized_data = atomize_prompt(geneval2_data[prompt_id])
                save_atomized_prompt(atomized_data, ATOMIZED_DIR, image_id)

            bundle = None
            if run_step3:
                try:
                    bundle = execute_plan(
                        plan, expert_manager, image_path, prompt_text,
                    )
                    save_testimony_bundle(bundle, expert_results_dir)
                    stats["gpu_ok"] += 1
                except Exception as e:
                    print(f"  [FATAL] {type(e).__name__}: {e}")
                    stats["gpu_fail"] += 1

            if run_step4 and bundle is not None and client:
                print(f"\n  [Step 4] Reflector evaluating {image_id}...")
                try:
                    report = run_reflector(
                        client=client,
                        image_path=image_path,
                        prompt_id=image_id,
                        prompt_text=prompt_text,
                        atomized_data=atomized_data or {},
                        expert_results=bundle,
                        experts_registry_str=experts_registry_str,
                        router_plan=plan,
                        api_retry=api_retry,
                        temperature=temp_reflector,
                    )
                    if report is None:
                        print(f"  [Step 4] FAILED - Reflector returned no valid response")
                        stats["step4_fail"] += 1
                    else:
                        if final_reports_dir:
                            final_reports_dir.mkdir(parents=True, exist_ok=True)
                            save_final_report(report, final_reports_dir)
                        print_final_summary(report)
                        stats["step4_ok"] += 1
                except Exception as e:
                    print(f"  [Step 4] FATAL: {type(e).__name__}: {e}")
                    stats["step4_fail"] += 1
            elif run_step4 and bundle is None:
                stats["step4_fail"] += 1

    return stats

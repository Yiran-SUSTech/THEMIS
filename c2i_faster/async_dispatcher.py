"""
THEMIS C2I Async Dispatcher - Pipeline-Parallel Batch Evaluation

Architecture:
  - API Track: asyncio event loop with N concurrent Router+Judge tasks
  - GPU Track: M parallel ExpertManager instances (each on a GPU group)
  - Coordination: asyncio.Queue bridges API output to GPU input

Performance vs. sync dispatcher.py:
  - API calls (40s/image) overlap with GPU execution (16s/image)
  - 2 GPU groups process 2 images simultaneously
  - Expected throughput: ~8s/image (GPU-bound) vs. 60s/image (serial)

Usage:
  python c2i/async_dispatcher.py --step 123 --limit 100 --gpu-groups 2
  python c2i/async_dispatcher.py --step 12 --api-concurrency 5
  python c2i/async_dispatcher.py --step 3 --gpu-groups 2
"""

import os
import sys
import re
import json
import time
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
C2I_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(C2I_DIR) not in sys.path:
    sys.path.insert(0, str(C2I_DIR))

IMAGE_DIR = PROJECT_ROOT / "test_images"
CLASS_IDS_TXT = IMAGE_DIR / "class_ids.txt"
IMAGENET_CLASSES_JSON = PROJECT_ROOT / "imagenet_classes.json"
EXPERTS_REGISTRY_JSON = PROJECT_ROOT / "expert_registry.json"

OUTPUT_DIR = C2I_DIR / os.environ.get("C2I_OUTPUT_DIR_NAME", "output")
PLAN_DIR = OUTPUT_DIR / "plans"
APPROVED_DIR = OUTPUT_DIR / "approved_plans"
JUDGE_FEEDBACK_DIR = OUTPUT_DIR / "judge_feedback"
EXPERT_RESULTS_DIR = OUTPUT_DIR / "expert_results"

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from step1_router import generate_plan, revise_plan, load_experts_registry
from step2_judge import review_plan
from step3_execute import (
    ExpertManager,
    execute_plan,
    save_testimony_bundle,
    load_approved_plans,
    resolve_image_path as resolve_image_path_global,
    collect_required_expert_ids,
    DEFAULT_GPU_CONFIG,
    EXPERT_MODULE_MAP,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GPU Group Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_gpu_group_configs(num_groups: int, base_config: dict) -> list[dict]:
    """Split 8 GPUs into N groups, each with a full set of experts.

    Group 0: cuda:0-3 (or maca:0-1 for MACA devices)
    Group 1: cuda:4-7 (or maca:2-3 for MACA devices)

    Each group gets its own device assignments so experts don't conflict.
    """
    if num_groups == 1:
        return [base_config]

    group_configs = []
    for g in range(num_groups):
        offset = g * 4
        maca_offset = g * 2
        config = {
            "perceptual_quality_auditor": {"device": f"cuda:{offset}", "num_gpus": 1},
            "animal_pose_auditor": {"device": f"cuda:{offset + 2}", "num_gpus": 1},
            "geometric_depth_auditor": {"device": f"maca:{maca_offset}", "num_gpus": 1},
            "fine_grained_classifier": {"device": f"maca:{maca_offset + 1}", "num_gpus": 1},
            "open_vocabulary_detector": {"device": "cpu", "num_gpus": 0},
            "topology_boundary_auditor": {"device": "cpu", "num_gpus": 0},
            "image_text_auditor": {"device": "cpu", "num_gpus": 0},
        }
        group_configs.append(config)

    return group_configs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Utility Functions (same as dispatcher.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_class_ids(txt_path: str) -> dict[str, int]:
    img_to_class = {}
    if not os.path.exists(txt_path):
        print(f"[WARN] class_ids.txt not found: {txt_path}")
        return img_to_class
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if len(parts) >= 2:
                try:
                    img_to_class[parts[0]] = int(parts[1])
                except ValueError:
                    continue
    return img_to_class


def load_imagenet_classes(json_path: str) -> dict[int, str]:
    class_map = {}
    if not os.path.exists(json_path):
        print(f"[WARN] imagenet_classes.json not found: {json_path}")
        return class_map
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for k, v in data.items():
        try:
            class_map[int(k)] = v
        except (ValueError, TypeError):
            continue
    return class_map


def resolve_image_path(image_dir: Path, img_id: str) -> Path | None:
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = image_dir / f"{img_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def save_judge_feedback(
    judge_feedback_dir: Path,
    img_id: str,
    iteration: int,
    judge_result: dict,
    plan: dict,
    class_label: str,
) -> None:
    feedback_record = {
        "image_id": img_id,
        "class_label": class_label,
        "iteration": iteration,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "judge_verdict": {
            "is_approved": judge_result.get("is_approved", False),
            "reasons_for_rejection": judge_result.get("reasons_for_rejection", ""),
            "suggestions": judge_result.get("suggestions", []),
        },
        "audited_plan": plan,
    }
    feedback_path = judge_feedback_dir / f"judge_feedback_{img_id}_iter{iteration}.json"
    with open(feedback_path, "w", encoding="utf-8") as f:
        json.dump(feedback_record, f, indent=4, ensure_ascii=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Async API Track: Router + Judge (runs in thread pool)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _sync_router_judge(
    client: OpenAI,
    image_path: str,
    img_id: str,
    class_id: int,
    class_label: str,
    experts_registry_str: str,
    max_iterations: int,
    plan_dir: Path,
    approved_dir: Path,
    judge_feedback_dir: Path | None,
) -> dict | None:
    """Synchronous Router+Judge pipeline for a single image.

    This runs in a thread so it can be called from asyncio without blocking.
    """
    current_plan = generate_plan(
        client, image_path, class_id, class_label, experts_registry_str,
    )
    if current_plan is None:
        print(f"  [{img_id}] Router FAILED")
        return None

    plan_save_path = plan_dir / f"plan_{img_id}.json"
    with open(plan_save_path, "w", encoding="utf-8") as f:
        json.dump(current_plan, f, indent=4, ensure_ascii=False)

    feedback_history: list[dict] = []
    iteration_log: list[str] = []

    for iteration in range(1, max_iterations + 1):
        judge_result = review_plan(
            client, image_path, class_id, class_label,
            current_plan, experts_registry_str,
        )

        if judge_result is None:
            iteration_log.append(f"Iteration {iteration}: Judge Error")
            break

        is_approved = judge_result.get("is_approved", False)

        if judge_feedback_dir is not None:
            save_judge_feedback(
                judge_feedback_dir, img_id, iteration,
                judge_result, current_plan, class_label,
            )

        if is_approved:
            iteration_log.append(f"Iteration {iteration}: Approved!")
            current_plan["metadata"]["judge_approved"] = True
            current_plan["metadata"]["judge_iterations"] = iteration
            break
        else:
            iteration_log.append(f"Iteration {iteration}: Rejected")

            if iteration == max_iterations:
                iteration_log.append(f"Iteration {iteration}: Max iterations - forced")
                current_plan["metadata"]["judge_approved"] = False
                current_plan["metadata"]["judge_forced"] = True
                current_plan["metadata"]["judge_iterations"] = iteration
                break

            feedback_history.append({
                "reasons_for_rejection": judge_result.get("reasons_for_rejection", ""),
                "suggestions": judge_result.get("suggestions", []),
            })

            revised_plan = revise_plan(
                client, image_path, class_id, class_label,
                experts_registry_str, current_plan, feedback_history,
            )

            if revised_plan is not None:
                current_plan = revised_plan
                revision_path = plan_dir / f"plan_{img_id}_rev{iteration}.json"
                with open(revision_path, "w", encoding="utf-8") as f:
                    json.dump(revised_plan, f, indent=4, ensure_ascii=False)

    current_plan["metadata"]["iteration_log"] = iteration_log

    approved_save_path = approved_dir / f"approved_plan_{img_id}.json"
    with open(approved_save_path, "w", encoding="utf-8") as f:
        json.dump(current_plan, f, indent=4, ensure_ascii=False)

    print(f"  [{img_id}] Plan approved ({iteration_log[-1] if iteration_log else 'OK'})")
    return current_plan


async def api_worker(
    task_queue: asyncio.Queue,
    plan_queue: asyncio.Queue,
    client: OpenAI,
    experts_registry_str: str,
    max_iterations: int,
    plan_dir: Path,
    approved_dir: Path,
    judge_feedback_dir: Path | None,
    api_semaphore: asyncio.Semaphore,
    stats: dict,
) -> None:
    """Async worker: pulls image tasks, runs Router+Judge in thread pool."""
    loop = asyncio.get_event_loop()

    while True:
        item = await task_queue.get()
        if item is None:
            task_queue.task_done()
            break

        img_id, image_path, class_id, class_label = item

        async with api_semaphore:
            try:
                plan = await loop.run_in_executor(
                    None,
                    _sync_router_judge,
                    client, image_path, img_id, class_id, class_label,
                    experts_registry_str, max_iterations,
                    plan_dir, approved_dir, judge_feedback_dir,
                )

                if plan is not None:
                    await plan_queue.put((img_id, image_path, class_label, plan))
                    stats["api_ok"] += 1
                else:
                    stats["api_fail"] += 1
            except Exception as e:
                print(f"  [{img_id}] API worker error: {type(e).__name__}: {e}")
                stats["api_fail"] += 1

        task_queue.task_done()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Async GPU Track: Expert Execution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def gpu_worker(
    plan_queue: asyncio.Queue,
    expert_manager: ExpertManager,
    expert_results_dir: Path,
    gpu_semaphore: asyncio.Semaphore,
    worker_id: int,
    stats: dict,
    done_event: asyncio.Event,
) -> None:
    """Async worker: pulls approved plans, runs experts on GPU group."""
    loop = asyncio.get_event_loop()

    while True:
        try:
            item = await asyncio.wait_for(plan_queue.get(), timeout=2.0)
        except asyncio.TimeoutError:
            if done_event.is_set() and plan_queue.empty():
                break
            continue

        if item is None:
            plan_queue.task_done()
            break

        img_id, image_path, class_label, plan = item

        async with gpu_semaphore:
            try:
                resolved_path = resolve_image_path_global(image_path)
                if resolved_path is None:
                    resolved_path = image_path

                bundle = await loop.run_in_executor(
                    None,
                    execute_plan,
                    plan, expert_manager, resolved_path, class_label,
                )
                await loop.run_in_executor(
                    None,
                    save_testimony_bundle,
                    bundle, expert_results_dir,
                )
                stats["gpu_ok"] += 1
                print(f"  [GPU-{worker_id}][{img_id}] Done "
                      f"({bundle['execution_summary']['total_execution_time_ms']:.0f}ms)")
            except Exception as e:
                print(f"  [GPU-{worker_id}][{img_id}] FATAL: {type(e).__name__}: {e}")
                stats["gpu_fail"] += 1

        plan_queue.task_done()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main Async Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def run_full_pipeline_async(
    valid_images: list[tuple],
    client: OpenAI,
    experts_registry_str: str,
    max_iterations: int,
    plan_dir: Path,
    approved_dir: Path,
    judge_feedback_dir: Path | None,
    expert_results_dir: Path,
    expert_managers: list[ExpertManager],
    api_concurrency: int,
) -> dict:
    """Run the full Step 1→2→3 pipeline with async overlap.

    API calls run concurrently (up to api_concurrency).
    GPU execution runs on M parallel expert groups.
    API and GPU tracks run simultaneously via asyncio.
    """
    stats = {"api_ok": 0, "api_fail": 0, "gpu_ok": 0, "gpu_fail": 0}

    task_queue: asyncio.Queue = asyncio.Queue()
    plan_queue: asyncio.Queue = asyncio.Queue()

    api_semaphore = asyncio.Semaphore(api_concurrency)
    gpu_semaphore = asyncio.Semaphore(len(expert_managers))
    done_event = asyncio.Event()

    for img_name, img_id, class_id, class_label in valid_images:
        image_path = resolve_image_path(IMAGE_DIR, img_id)
        if image_path is None:
            print(f"[WARN] Image not found: {img_id}")
            stats["api_fail"] += 1
            continue
        await task_queue.put((img_id, str(image_path), class_id, class_label))

    num_api_workers = min(api_concurrency, len(valid_images))
    for _ in range(num_api_workers):
        await task_queue.put(None)

    api_tasks = [
        asyncio.create_task(api_worker(
            task_queue, plan_queue, client, experts_registry_str,
            max_iterations, plan_dir, approved_dir, judge_feedback_dir,
            api_semaphore, stats,
        ))
        for _ in range(num_api_workers)
    ]

    gpu_tasks = [
        asyncio.create_task(gpu_worker(
            plan_queue, em, expert_results_dir,
            gpu_semaphore, i, stats, done_event,
        ))
        for i, em in enumerate(expert_managers)
    ]

    await asyncio.gather(*api_tasks)
    done_event.set()

    await plan_queue.join()

    for _ in gpu_tasks:
        await plan_queue.put(None)
    await asyncio.gather(*gpu_tasks)

    return stats


async def run_step12_async(
    valid_images: list[tuple],
    client: OpenAI,
    experts_registry_str: str,
    max_iterations: int,
    plan_dir: Path,
    approved_dir: Path,
    judge_feedback_dir: Path | None,
    image_dir: Path,
    api_concurrency: int,
) -> dict:
    """Run only Step 1+2 with async API concurrency (no GPU)."""
    stats = {"api_ok": 0, "api_fail": 0}

    api_semaphore = asyncio.Semaphore(api_concurrency)
    loop = asyncio.get_event_loop()

    async def process_one(img_name, img_id, class_id, class_label):
        image_path = resolve_image_path(image_dir, img_id)
        if image_path is None:
            stats["api_fail"] += 1
            return

        async with api_semaphore:
            plan = await loop.run_in_executor(
                None,
                _sync_router_judge,
                client, str(image_path), img_id, class_id, class_label,
                experts_registry_str, max_iterations,
                plan_dir, approved_dir, judge_feedback_dir,
            )
            if plan is not None:
                stats["api_ok"] += 1
            else:
                stats["api_fail"] += 1

    tasks = [
        asyncio.create_task(process_one(n, iid, cid, cl))
        for n, iid, cid, cl in valid_images
    ]
    await asyncio.gather(*tasks)
    return stats


async def run_step3_async(
    approved_dir: Path,
    expert_results_dir: Path,
    expert_managers: list[ExpertManager],
    image_id_filter: str = "",
    limit: int = 0,
) -> dict:
    """Run only Step 3 with M parallel GPU groups."""
    stats = {"gpu_ok": 0, "gpu_fail": 0}

    plans = load_approved_plans(approved_dir)
    if image_id_filter:
        plans = [p for p in plans if image_id_filter in p.get("_source_file", "")]
    if limit > 0:
        plans = plans[:limit]

    if not plans:
        print("[ERROR] No approved plans found.")
        return stats

    plan_queue: asyncio.Queue = asyncio.Queue()
    gpu_semaphore = asyncio.Semaphore(len(expert_managers))
    done_event = asyncio.Event()

    for plan in plans:
        metadata = plan.get("metadata", {})
        image_path = metadata.get("original_image", "")
        class_label = metadata.get("class_label", "")
        image_id = Path(image_path).stem if image_path else "unknown"
        await plan_queue.put((image_id, image_path, class_label, plan))

    done_event.set()

    gpu_tasks = [
        asyncio.create_task(gpu_worker(
            plan_queue, em, expert_results_dir,
            gpu_semaphore, i, stats, done_event,
        ))
        for i, em in enumerate(expert_managers)
    ]

    await plan_queue.join()

    for _ in gpu_tasks:
        await plan_queue.put(None)
    await asyncio.gather(*gpu_tasks)

    return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(
        description="THEMIS C2I Async Dispatcher - Pipeline-parallel evaluation"
    )
    parser.add_argument(
        "--step", type=str, default="123",
        choices=["12", "3", "123"],
        help="Which steps to run: '12'=Router+Judge, '3'=Expert, '123'=Full (default)",
    )
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--image-dir", type=str, default=str(IMAGE_DIR))
    parser.add_argument("--class-ids", type=str, default=str(CLASS_IDS_TXT))
    parser.add_argument("--plan-dir", type=str, default=str(PLAN_DIR))
    parser.add_argument("--approved-dir", type=str, default=str(APPROVED_DIR))
    parser.add_argument("--expert-results-dir", type=str, default=str(EXPERT_RESULTS_DIR))
    parser.add_argument("--save-feedback", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--image-id", type=str, default="")
    parser.add_argument("--api-concurrency", type=int, default=5,
                        help="Max concurrent API calls (default: 5)")
    parser.add_argument("--gpu-groups", type=int, default=2,
                        help="Number of parallel GPU groups (default: 2, max experts on 8 GPUs)")
    parser.add_argument("--gpu-config", type=str, default=None,
                        help="Path to custom GPU config JSON (overrides --gpu-groups)")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    plan_dir = Path(args.plan_dir)
    approved_dir = Path(args.approved_dir)
    expert_results_dir = Path(args.expert_results_dir)

    plan_dir.mkdir(parents=True, exist_ok=True)
    approved_dir.mkdir(parents=True, exist_ok=True)
    expert_results_dir.mkdir(parents=True, exist_ok=True)

    judge_feedback_dir = None
    if args.save_feedback:
        judge_feedback_dir = JUDGE_FEEDBACK_DIR
        judge_feedback_dir.mkdir(parents=True, exist_ok=True)

    step = args.step
    run_step12 = step in ("12", "123")
    run_step3 = step in ("3", "123")

    # ── Load image list ─────────────────────────────────────────
    img_to_class = parse_class_ids(str(Path(args.class_ids)))
    imagenet_classes = load_imagenet_classes(str(IMAGENET_CLASSES_JSON))
    experts_registry_str = load_experts_registry(str(EXPERTS_REGISTRY_JSON))

    image_files = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    valid_images = []
    for img_name in image_files:
        img_id = os.path.splitext(img_name)[0]
        if img_id not in img_to_class:
            continue
        class_id = img_to_class[img_id]
        class_label = imagenet_classes.get(class_id, f"class_{class_id}")
        valid_images.append((img_name, img_id, class_id, class_label))

    if args.image_id:
        valid_images = [
            (n, iid, cid, cl) for n, iid, cid, cl in valid_images
            if iid == args.image_id
        ]
        if not valid_images:
            print(f"[ERROR] Image ID '{args.image_id}' not found")
            return

    if args.limit > 0:
        valid_images = valid_images[:args.limit]

    # ── Pre-load Expert Managers ────────────────────────────────
    expert_managers = []
    if run_step3:
        if args.gpu_config:
            with open(args.gpu_config, "r") as f:
                custom_config = json.load(f)
            group_configs = [custom_config]
        else:
            group_configs = build_gpu_group_configs(
                args.gpu_groups, DEFAULT_GPU_CONFIG,
            )

        required_ids = list(EXPERT_MODULE_MAP.keys())

        print(f"\n{'='*60}")
        print(f"  Pre-loading {args.gpu_groups} GPU group(s)")
        print(f"{'='*60}")

        for g, cfg in enumerate(group_configs):
            print(f"\n  --- GPU Group {g} ---")
            em = ExpertManager(gpu_config=cfg)
            em.load_all(required_ids)
            expert_managers.append(em)

        total_loaded = sum(len(em.loaded_experts) for em in expert_managers)
        print(f"\n  Total expert instances loaded: {total_loaded}")
        print(f"{'='*60}")

    # ── Run Pipeline ────────────────────────────────────────────
    total_start = time.time()

    print(f"\n{'='*60}")
    print(f"  THEMIS Async Dispatcher")
    print(f"  Step:             {step}")
    print(f"  Images:           {len(valid_images)}")
    print(f"  API concurrency:  {args.api_concurrency}")
    print(f"  GPU groups:       {args.gpu_groups if run_step3 else 'N/A'}")
    print(f"{'='*60}\n")

    if step == "123":
        if not DASHSCOPE_API_KEY:
            print("[ERROR] DASHSCOPE_API_KEY not set.")
            return

        client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

        stats = asyncio.run(run_full_pipeline_async(
            valid_images=valid_images,
            client=client,
            experts_registry_str=experts_registry_str,
            max_iterations=args.max_iterations,
            plan_dir=plan_dir,
            approved_dir=approved_dir,
            judge_feedback_dir=judge_feedback_dir,
            expert_results_dir=expert_results_dir,
            expert_managers=expert_managers,
            api_concurrency=args.api_concurrency,
        ))

    elif step == "12":
        if not DASHSCOPE_API_KEY:
            print("[ERROR] DASHSCOPE_API_KEY not set.")
            return

        client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

        stats = asyncio.run(run_step12_async(
            valid_images=valid_images,
            client=client,
            experts_registry_str=experts_registry_str,
            max_iterations=args.max_iterations,
            plan_dir=plan_dir,
            approved_dir=approved_dir,
            judge_feedback_dir=judge_feedback_dir,
            image_dir=image_dir,
            api_concurrency=args.api_concurrency,
        ))

    elif step == "3":
        stats = asyncio.run(run_step3_async(
            approved_dir=approved_dir,
            expert_results_dir=expert_results_dir,
            expert_managers=expert_managers,
            image_id_filter=args.image_id,
            limit=args.limit,
        ))

    total_elapsed = time.time() - total_start

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Async Pipeline Summary")
    print(f"{'='*60}")
    print(f"  Total images:     {len(valid_images)}")
    for k, v in stats.items():
        print(f"  {k:16s}  {v}")
    print(f"  Total elapsed:    {total_elapsed:.2f}s")
    if len(valid_images) > 0 and total_elapsed > 0:
        throughput = len(valid_images) / total_elapsed
        print(f"  Throughput:       {throughput:.2f} img/s ({1/throughput:.1f} s/img)")
    print(f"{'='*60}")

    for em in expert_managers:
        em.cleanup()


if __name__ == "__main__":
    main()

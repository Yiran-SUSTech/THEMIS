"""
THEMIS C2I Async Dispatcher - Pipeline-parallel processing.

Called from run.py with --mode async.
Reuses core async logic from async_dispatcher.py.
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from openai import OpenAI

from common import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
    resolve_image_path, save_judge_feedback,
)

from step1_router import generate_plan, revise_plan, load_experts_registry
from step2_judge import review_plan
from step3_execute import (
    ExpertManager, execute_plan, save_testimony_bundle,
    load_approved_plans, resolve_image_path as resolve_image_path_global,
    collect_required_expert_ids, EXPERT_MODULE_MAP,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Core Async Workers
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
    """Synchronous Router+Judge for one image. Runs in thread pool."""
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


async def _api_worker(
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


async def _gpu_worker(
    plan_queue: asyncio.Queue,
    expert_manager: ExpertManager,
    expert_results_dir: Path,
    gpu_semaphore: asyncio.Semaphore,
    worker_id: int,
    stats: dict,
    done_event: asyncio.Event,
) -> None:
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
                    None, execute_plan, plan, expert_manager, resolved_path, class_label,
                )
                await loop.run_in_executor(
                    None, save_testimony_bundle, bundle, expert_results_dir,
                )
                stats["gpu_ok"] += 1
                print(f"  [GPU-{worker_id}][{img_id}] Done "
                      f"({bundle['execution_summary']['total_execution_time_ms']:.0f}ms)")
            except Exception as e:
                print(f"  [GPU-{worker_id}][{img_id}] FATAL: {type(e).__name__}: {e}")
                stats["gpu_fail"] += 1

        plan_queue.task_done()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Public Async Pipelines
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _run_full_pipeline(
    valid_images: list[tuple],
    image_dir: Path,
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
    stats = {"api_ok": 0, "api_fail": 0, "gpu_ok": 0, "gpu_fail": 0}

    task_queue: asyncio.Queue = asyncio.Queue()
    plan_queue: asyncio.Queue = asyncio.Queue()

    api_semaphore = asyncio.Semaphore(api_concurrency)
    gpu_semaphore = asyncio.Semaphore(len(expert_managers))
    done_event = asyncio.Event()

    for img_name, img_id, class_id, class_label in valid_images:
        image_path = resolve_image_path(image_dir, img_id)
        if image_path is None:
            stats["api_fail"] += 1
            continue
        await task_queue.put((img_id, str(image_path), class_id, class_label))

    num_api_workers = min(api_concurrency, len(valid_images))
    for _ in range(num_api_workers):
        await task_queue.put(None)

    api_tasks = [
        asyncio.create_task(_api_worker(
            task_queue, plan_queue, client, experts_registry_str,
            max_iterations, plan_dir, approved_dir, judge_feedback_dir,
            api_semaphore, stats,
        ))
        for _ in range(num_api_workers)
    ]

    gpu_tasks = [
        asyncio.create_task(_gpu_worker(
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


async def _run_step12_only(
    valid_images: list[tuple],
    image_dir: Path,
    client: OpenAI,
    experts_registry_str: str,
    max_iterations: int,
    plan_dir: Path,
    approved_dir: Path,
    judge_feedback_dir: Path | None,
    api_concurrency: int,
) -> dict:
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
    expert_managers: list,
    image_id_filter: str = "",
    limit: int = 0,
) -> dict:
    """Run Step 3 with parallel GPU groups. Public API for batch mode too."""
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
        asyncio.create_task(_gpu_worker(
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Entry Point (called from run.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_async_pipeline(
    valid_images: list[tuple],
    image_dir: Path,
    experts_registry_str: str,
    max_iterations: int,
    plan_dir: Path,
    approved_dir: Path,
    judge_feedback_dir: Path | None,
    expert_results_dir: Path,
    expert_managers: list,
    api_concurrency: int,
    step: str,
) -> dict:
    """Public entry: run async pipeline. Called from run.py."""
    run_step12 = step in ("1", "2", "12", "123")
    run_step3 = step in ("3", "123")

    if run_step12 and run_step3 and expert_managers:
        client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
        return asyncio.run(_run_full_pipeline(
            valid_images=valid_images,
            image_dir=image_dir,
            client=client,
            experts_registry_str=experts_registry_str,
            max_iterations=max_iterations,
            plan_dir=plan_dir,
            approved_dir=approved_dir,
            judge_feedback_dir=judge_feedback_dir,
            expert_results_dir=expert_results_dir,
            expert_managers=expert_managers,
            api_concurrency=api_concurrency,
        ))

    elif run_step12 and not run_step3:
        client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
        return asyncio.run(_run_step12_only(
            valid_images=valid_images,
            image_dir=image_dir,
            client=client,
            experts_registry_str=experts_registry_str,
            max_iterations=max_iterations,
            plan_dir=plan_dir,
            approved_dir=approved_dir,
            judge_feedback_dir=judge_feedback_dir,
            api_concurrency=api_concurrency,
        ))

    elif run_step3 and not run_step12:
        if not expert_managers:
            print("[ERROR] No expert managers for Step 3.")
            return {"gpu_ok": 0, "gpu_fail": 0}
        return asyncio.run(run_step3_async(
            approved_dir=approved_dir,
            expert_results_dir=expert_results_dir,
            expert_managers=expert_managers,
        ))

    return {}

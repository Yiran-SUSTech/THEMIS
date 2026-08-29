"""
THEMIS T2I Async Dispatcher - Pipeline-parallel processing.

Called from run.py with --mode async.
Pipeline: Step 0 (Atomize) → Step 1 (Router) → Step 2 (Judge) → Step 3 (Expert) → Step 4 (Reflector)
"""

import os
import sys
import json
import time
import asyncio
import traceback
from pathlib import Path
from openai import OpenAI

from common import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
    resolve_image_path, save_judge_feedback,
    ATOMIZED_DIR, GENEVAL2_DATA_JSONL,
    load_geneval2_data, record_failure, bump_progress,
)

from step0_atomize import atomize_prompt, save_atomized_prompt, enrich_with_generic_taxonomy
from step1_router import generate_plan, revise_plan, load_experts_registry
from step2_judge import review_plan
from step4_reflector import (
    run_reflector, save_final_report, print_final_summary,
    build_checklist_annotation, save_checklist_annotation,
)

from c2i_harness.step3_execute import (
    ExpertManager, execute_plan, save_testimony_bundle,
    load_approved_plans, resolve_image_path as resolve_image_path_global,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Core Async Workers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _sync_router_judge(
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
    stats: dict | None = None,
) -> dict | None:
    """Synchronous Router+Judge for one image. Runs in thread pool."""
    start_time = time.time()
    current_plan = generate_plan(
        client, image_path, img_id, prompt_text, atomized_data,
        experts_registry_str,
        api_retry=api_retry, temperature=temp_router,
    )
    if current_plan is None:
        print(f"  [{img_id}] Router FAILED "
              f"(see [Router][{img_id}] errors above / output debug_raw/)")
        record_failure(stats, img_id, "step1_router",
                       "Router returned no valid plan (empty content, unparseable JSON, or API error)")
        return None

    plan_save_path = plan_dir / f"plan_{img_id}.json"
    with open(plan_save_path, "w", encoding="utf-8") as f:
        json.dump(current_plan, f, indent=4, ensure_ascii=False)

    feedback_history: list[dict] = []
    iteration_log: list[str] = []

    for iteration in range(1, max_iterations + 1):
        judge_result = review_plan(
            client, image_path, img_id, prompt_text, atomized_data,
            current_plan, experts_registry_str,
            api_retry=api_retry, temperature=temp_judge,
        )

        if judge_result is None:
            iteration_log.append(f"Iteration {iteration}: Judge Error")
            print(f"  [{img_id}] WARN: Judge error at iteration {iteration} "
                  f"— saving plan WITHOUT approval and continuing pipeline")
            record_failure(stats, img_id, "step2_judge",
                           "Judge returned no valid response (empty content, unparseable JSON, or API error)")
            break

        is_approved = judge_result.get("is_approved", False)

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
                client, image_path, img_id, prompt_text, atomized_data,
                experts_registry_str, current_plan, feedback_history,
                api_retry=api_retry, temperature=temp_router,
            )

            if revised_plan is not None:
                current_plan = revised_plan
                revision_path = plan_dir / f"plan_{img_id}_rev{iteration}.json"
                with open(revision_path, "w", encoding="utf-8") as f:
                    json.dump(revised_plan, f, indent=4, ensure_ascii=False)
            else:
                print(f"  [{img_id}] WARN: Router revision failed at iteration "
                      f"{iteration}, keeping previous plan")
                record_failure(stats, img_id, "step1_router_revision",
                               "revise_plan returned None; keeping pre-revision plan")

    current_plan["metadata"]["iteration_log"] = iteration_log

    approved_save_path = approved_dir / f"approved_plan_{img_id}.json"
    with open(approved_save_path, "w", encoding="utf-8") as f:
        json.dump(current_plan, f, indent=4, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"  [{img_id}] Plan approved "
          f"({iteration_log[-1] if iteration_log else 'OK'}) in {elapsed:.1f}s")
    return current_plan


async def _api_worker(
    task_queue: asyncio.Queue,
    plan_queue: asyncio.Queue,
    client: OpenAI,
    experts_registry_str: str,
    geneval2_data: dict,
    max_iterations: int,
    plan_dir: Path,
    approved_dir: Path,
    judge_feedback_dir: Path | None,
    api_semaphore: asyncio.Semaphore,
    stats: dict,
    api_retry: int = 0,
    temp_router: float = 0.0,
    temp_judge: float = 0.0,
    total: int = 0,
) -> None:
    """Async worker: atomize prompt → Router → Judge loop."""
    loop = asyncio.get_event_loop()

    while True:
        item = await task_queue.get()
        if item is None:
            task_queue.task_done()
            break

        img_id, image_path, prompt_text = item

        async with api_semaphore:
            try:
                # Step 0: Atomize prompt (in thread to avoid blocking)
                prompt_data = geneval2_data.get(img_id, {})
                if not prompt_data:
                    print(f"  [{img_id}] No GenEval2 data found, skipping")
                    record_failure(stats, img_id, "step0_atomize",
                                   f"No GenEval2 record for prompt_id={img_id} in JSONL")
                    stats["api_fail"] += 1
                    task_queue.task_done()
                    continue

                atomize_start = time.time()
                atomized_data = await loop.run_in_executor(
                    None, atomize_prompt, prompt_data,
                )
                await loop.run_in_executor(
                    None, save_atomized_prompt, atomized_data, ATOMIZED_DIR, img_id,
                )

                # Step 0d: Enrich with generic taxonomy
                atomized_data = await loop.run_in_executor(
                    None,
                    lambda: enrich_with_generic_taxonomy(
                        atomized_data, client,
                        api_retry=api_retry, temperature=0.0,
                        ctx_id=img_id,
                    ),
                )
                await loop.run_in_executor(
                    None, save_atomized_prompt, atomized_data, ATOMIZED_DIR, img_id,
                )
                stats["atomize_ok"] += 1
                print(f"  [{img_id}] Atomize ok: {atomized_data['atom_count']} atoms, "
                      f"{len(atomized_data['objects'])} objects "
                      f"({time.time() - atomize_start:.1f}s)")

                # Steps 1+2: Router + Judge
                plan = await loop.run_in_executor(
                    None,
                    _sync_router_judge,
                    client, image_path, img_id, prompt_text, atomized_data,
                    experts_registry_str, max_iterations,
                    plan_dir, approved_dir, judge_feedback_dir,
                    api_retry, temp_router, temp_judge, stats,
                )

                if plan is not None:
                    await plan_queue.put((img_id, image_path, prompt_text, atomized_data, plan))
                    stats["api_ok"] += 1
                else:
                    stats["api_fail"] += 1
                bump_progress("Step1+2", total)
            except Exception as e:
                print(f"  [{img_id}] API worker error: {type(e).__name__}: {e}")
                print(traceback.format_exc())
                record_failure(stats, img_id, "step0_12_api_worker",
                               f"{type(e).__name__}: {e}")
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
    reflector_queue: asyncio.Queue | None = None,
    cpu_semaphore: object | None = None,
    total: int = 0,
) -> None:
    """Async worker: execute expert plan on GPU."""
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

        img_id, image_path, prompt_text, atomized_data, plan = item

        async with gpu_semaphore:
            try:
                resolved_path = resolve_image_path_global(image_path)
                if resolved_path is None:
                    print(f"  [GPU-{worker_id}][{img_id}] WARN: image not found at "
                          f"'{image_path}', passing raw path to experts")
                    resolved_path = image_path

                bundle = await loop.run_in_executor(
                    None, execute_plan, plan, expert_manager, resolved_path, prompt_text,
                    False, cpu_semaphore,
                )
                await loop.run_in_executor(
                    None, save_testimony_bundle, bundle, expert_results_dir,
                )
                stats["gpu_ok"] += 1

                # Per-expert failure visibility: the bundle may succeed overall
                # while individual expert calls failed — surface those here.
                testimonies = bundle.get("expert_testimonies", [])
                failed_experts = [
                    t.get("expert_id", "?") for t in testimonies
                    if t.get("status") != "success"
                ]
                if failed_experts:
                    print(f"  [GPU-{worker_id}][{img_id}] WARN: "
                          f"{len(failed_experts)}/{len(testimonies)} expert call(s) "
                          f"failed: {failed_experts} (details in "
                          f"{expert_results_dir}/expert_results_{img_id}.json)")

                print(f"  [GPU-{worker_id}][{img_id}] Done "
                      f"({bundle['execution_summary']['total_execution_time_ms']:.0f}ms)")

                if reflector_queue is not None:
                    await reflector_queue.put((
                        img_id, str(resolved_path), prompt_text, atomized_data,
                        bundle, plan,
                    ))
            except Exception as e:
                print(f"  [GPU-{worker_id}][{img_id}] FATAL: {type(e).__name__}: {e}")
                print(traceback.format_exc())
                record_failure(stats, img_id, "step3_expert",
                               f"{type(e).__name__}: {e}")
                stats["gpu_fail"] += 1
            bump_progress("Step3", total)

        plan_queue.task_done()


def _sync_reflector(
    client: OpenAI,
    image_path: str,
    img_id: str,
    prompt_text: str,
    atomized_data: dict,
    expert_results: dict,
    experts_registry_str: str,
    router_plan: dict,
    final_reports_dir: Path,
    ref_image_dir: Path | None = None,
    enable_self_reflection: bool = True,
    enable_checklist: bool = False,
    checklist_dir: Path | None = None,
    api_retry: int = 0,
    temperature: float = 0.5,
) -> dict | None:
    """Synchronous Reflector for one image. Runs in thread pool."""
    report = run_reflector(
        client=client,
        image_path=image_path,
        prompt_id=img_id,
        prompt_text=prompt_text,
        atomized_data=atomized_data,
        expert_results=expert_results,
        experts_registry_str=experts_registry_str,
        router_plan=router_plan,
        ref_image_dir=ref_image_dir,
        enable_self_reflection=enable_self_reflection,
        enable_checklist=enable_checklist,
        temperature=temperature,
        api_retry=api_retry,
    )
    if report is None:
        print(f"  [{img_id}] Reflector FAILED")
        return None

    save_final_report(report, final_reports_dir)
    if enable_checklist and checklist_dir is not None:
        annotation = build_checklist_annotation(
            report, img_id, os.path.basename(image_path), prompt_text,
        )
        save_checklist_annotation(annotation, checklist_dir)
    print_final_summary(report)
    return report


async def _reflector_worker(
    reflector_queue: asyncio.Queue,
    client: OpenAI,
    experts_registry_str: str,
    final_reports_dir: Path,
    api_semaphore: asyncio.Semaphore,
    stats: dict,
    done_event: asyncio.Event,
    ref_image_dir: Path | None = None,
    enable_self_reflection: bool = True,
    enable_checklist: bool = False,
    checklist_dir: Path | None = None,
    api_retry: int = 0,
    temp_reflector: float = 0.5,
    total: int = 0,
) -> None:
    """Async worker: pull GPU results from reflector_queue, call Reflector API."""
    loop = asyncio.get_event_loop()

    while True:
        try:
            item = await asyncio.wait_for(reflector_queue.get(), timeout=2.0)
        except asyncio.TimeoutError:
            if done_event.is_set() and reflector_queue.empty():
                break
            continue

        if item is None:
            reflector_queue.task_done()
            break

        img_id, image_path, prompt_text, atomized_data, expert_results, router_plan = item

        async with api_semaphore:
            try:
                report = await loop.run_in_executor(
                    None,
                    _sync_reflector,
                    client, image_path, img_id, prompt_text, atomized_data,
                    expert_results, experts_registry_str, router_plan,
                    final_reports_dir, ref_image_dir, enable_self_reflection,
                    enable_checklist, checklist_dir,
                    api_retry, temp_reflector,
                )
                if report is not None:
                    stats["reflector_ok"] += 1
                else:
                    stats["reflector_fail"] += 1
                    record_failure(stats, img_id, "step4_reflector",
                                   "Reflector returned no valid report (see [Reflector] errors above / debug_raw/)")
            except Exception as e:
                print(f"  [{img_id}] Reflector worker error: {type(e).__name__}: {e}")
                print(traceback.format_exc())
                record_failure(stats, img_id, "step4_reflector",
                               f"{type(e).__name__}: {e}")
                stats["reflector_fail"] += 1
            bump_progress("Step4", total)

        reflector_queue.task_done()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Public Async Pipelines
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _run_full_pipeline(
    valid_images: list[tuple],
    image_dir: Path,
    client: OpenAI,
    experts_registry_str: str,
    geneval2_data: dict,
    max_iterations: int,
    plan_dir: Path,
    approved_dir: Path,
    judge_feedback_dir: Path | None,
    expert_results_dir: Path,
    expert_managers: list[ExpertManager],
    api_concurrency: int,
    final_reports_dir: Path | None = None,
    temp_router: float = 0.0,
    temp_judge: float = 0.0,
    temp_reflector: float = 0.5,
    cpu_semaphore: object | None = None,
    ref_image_dir: Path | None = None,
    enable_self_reflection: bool = True,
    enable_checklist: bool = False,
    checklist_dir: Path | None = None,
    api_retry: int = 0,
) -> dict:
    """Run full pipeline: Atomize → Router → Judge → Expert → Reflector."""
    run_step4 = final_reports_dir is not None
    stats = {
        "atomize_ok": 0,
        "api_ok": 0, "api_fail": 0,
        "gpu_ok": 0, "gpu_fail": 0,
        "reflector_ok": 0, "reflector_fail": 0,
    }

    task_queue: asyncio.Queue = asyncio.Queue()
    plan_queue: asyncio.Queue = asyncio.Queue()
    reflector_queue: asyncio.Queue | None = asyncio.Queue() if run_step4 else None

    api_semaphore = asyncio.Semaphore(api_concurrency)
    gpu_semaphore = asyncio.Semaphore(len(expert_managers))
    gpu_done_event = asyncio.Event()
    reflector_done_event = asyncio.Event()

    stats["resumed"] = 0
    queued = 0
    for img_name, img_id, prompt_text in valid_images:
        image_path = resolve_image_path(image_dir, img_id)
        if image_path is None:
            print(f"  [WARN] Image file not found for prompt_id={img_id} "
                  f"(expected {img_id}.png/.jpg/.jpeg in {image_dir})")
            record_failure(stats, img_id, "image_lookup",
                           f"No image file found in {image_dir} for prompt_id {img_id}")
            stats["api_fail"] += 1
            continue

        # Resume: skip images that already have a final report
        # (and a checklist annotation when checklist mode is on, so that
        # re-running with --enable-checklist backfills missing checklists
        # instead of silently skipping completed images)
        if run_step4 and final_reports_dir is not None:
            report_path = final_reports_dir / f"final_evaluation_report_{img_id}.json"
            checklist_path = (
                checklist_dir / f"checklist_{img_id}.json"
                if checklist_dir is not None else None
            )
            done = report_path.exists() and (
                checklist_path is None or checklist_path.exists()
            )
            if done:
                stats["resumed"] += 1
                continue

        await task_queue.put((img_id, str(image_path), prompt_text))
        queued += 1

    if stats["resumed"] > 0:
        print(f"  [RESUME] {stats['resumed']} image(s) skipped — final report already exists")
        print(f"  [RESUME] {queued} image(s) to process")

    num_api_workers = min(api_concurrency, max(queued, 1))
    for _ in range(num_api_workers):
        await task_queue.put(None)

    api_tasks = [
        asyncio.create_task(_api_worker(
            task_queue, plan_queue, client, experts_registry_str,
            geneval2_data, max_iterations, plan_dir, approved_dir,
            judge_feedback_dir, api_semaphore, stats,
            api_retry, temp_router, temp_judge, queued,
        ))
        for _ in range(num_api_workers)
    ]

    gpu_tasks = [
        asyncio.create_task(_gpu_worker(
            plan_queue, em, expert_results_dir,
            gpu_semaphore, i, stats, gpu_done_event,
            reflector_queue=reflector_queue,
            cpu_semaphore=cpu_semaphore,
            total=queued,
        ))
        for i, em in enumerate(expert_managers)
    ]

    reflector_tasks = []
    if run_step4:
        reflector_api_semaphore = asyncio.Semaphore(api_concurrency)
        num_reflector_workers = min(api_concurrency, max(queued, 1))
        for _ in range(num_reflector_workers):
            reflector_tasks.append(
                asyncio.create_task(_reflector_worker(
                    reflector_queue, client, experts_registry_str,
                    final_reports_dir, reflector_api_semaphore,
                    stats, reflector_done_event,
                    ref_image_dir, enable_self_reflection,
                    enable_checklist, checklist_dir,
                    api_retry, temp_reflector, queued,
                ))
            )

    # Wait for API workers to finish
    await asyncio.gather(*api_tasks)
    # Signal GPU workers that no more plans will arrive
    gpu_done_event.set()
    await plan_queue.join()

    # Send sentinel values to GPU workers
    for _ in gpu_tasks:
        await plan_queue.put(None)
    await asyncio.gather(*gpu_tasks)

    if run_step4:
        reflector_done_event.set()
        await reflector_queue.join()
        for _ in reflector_tasks:
            await reflector_queue.put(None)
        await asyncio.gather(*reflector_tasks)

    if not run_step4:
        stats.pop("reflector_ok", None)
        stats.pop("reflector_fail", None)

    return stats


async def _run_step12_only(
    valid_images: list[tuple],
    image_dir: Path,
    client: OpenAI,
    experts_registry_str: str,
    geneval2_data: dict,
    max_iterations: int,
    plan_dir: Path,
    approved_dir: Path,
    judge_feedback_dir: Path | None,
    api_concurrency: int,
    temp_router: float = 0.0,
    temp_judge: float = 0.0,
    api_retry: int = 0,
) -> dict:
    """Run Step 0+1+2 only (Atomize + Router + Judge)."""
    stats = {"atomize_ok": 0, "api_ok": 0, "api_fail": 0, "resumed": 0}
    api_semaphore = asyncio.Semaphore(api_concurrency)
    loop = asyncio.get_event_loop()

    todo: list[tuple] = []
    for img_name, img_id, prompt_text in valid_images:
        image_path = resolve_image_path(image_dir, img_id)
        if image_path is None:
            print(f"  [WARN] Image file not found for prompt_id={img_id} "
                  f"(expected {img_id}.png/.jpg/.jpeg in {image_dir})")
            record_failure(stats, img_id, "image_lookup",
                           f"No image file found in {image_dir} for prompt_id {img_id}")
            stats["api_fail"] += 1
            continue

        # Resume: skip images with an existing approved plan
        if (approved_dir / f"approved_plan_{img_id}.json").exists():
            stats["resumed"] += 1
            continue
        todo.append((img_name, img_id, prompt_text))

    if stats["resumed"] > 0:
        print(f"  [RESUME] {stats['resumed']} image(s) skipped — approved plan already exists")
        print(f"  [RESUME] {len(todo)} image(s) to process")

    total = len(todo)

    async def process_one(img_name, img_id, prompt_text):
        image_path = resolve_image_path(image_dir, img_id)
        if image_path is None:
            stats["api_fail"] += 1
            return

        async with api_semaphore:
            try:
                # Step 0: Atomize
                prompt_data = geneval2_data.get(img_id, {})
                if not prompt_data:
                    print(f"  [{img_id}] No GenEval2 data found, skipping")
                    record_failure(stats, img_id, "step0_atomize",
                                   f"No GenEval2 record for prompt_id={img_id} in JSONL")
                    stats["api_fail"] += 1
                    return

                atomized_data = await loop.run_in_executor(
                    None, atomize_prompt, prompt_data,
                )
                await loop.run_in_executor(
                    None, save_atomized_prompt, atomized_data, ATOMIZED_DIR, img_id,
                )

                # Step 0d: Enrich with generic taxonomy
                atomized_data = await loop.run_in_executor(
                    None,
                    lambda: enrich_with_generic_taxonomy(
                        atomized_data, client,
                        api_retry=api_retry, temperature=0.0,
                        ctx_id=img_id,
                    ),
                )
                await loop.run_in_executor(
                    None, save_atomized_prompt, atomized_data, ATOMIZED_DIR, img_id,
                )
                stats["atomize_ok"] += 1

                # Steps 1+2: Router + Judge
                plan = await loop.run_in_executor(
                    None,
                    _sync_router_judge,
                    client, str(image_path), img_id, prompt_text, atomized_data,
                    experts_registry_str, max_iterations,
                    plan_dir, approved_dir, judge_feedback_dir,
                    api_retry, temp_router, temp_judge, stats,
                )
                if plan is not None:
                    stats["api_ok"] += 1
                else:
                    stats["api_fail"] += 1
                bump_progress("Step1+2", total)
            except Exception as e:
                print(f"  [{img_id}] Step12 worker error: {type(e).__name__}: {e}")
                print(traceback.format_exc())
                record_failure(stats, img_id, "step0_12_api_worker",
                               f"{type(e).__name__}: {e}")
                stats["api_fail"] += 1

    tasks = [
        asyncio.create_task(process_one(n, iid, pt))
        for n, iid, pt in todo
    ]
    await asyncio.gather(*tasks)
    return stats


async def run_step3_async(
    approved_dir: Path,
    expert_results_dir: Path,
    expert_managers: list,
    cpu_semaphore: object | None = None,
) -> dict:
    """Run Step 3 with parallel GPU groups."""
    stats = {"gpu_ok": 0, "gpu_fail": 0, "resumed": 0}

    plans = load_approved_plans(approved_dir)
    if not plans:
        print("[ERROR] No approved plans found.")
        return stats

    plan_queue: asyncio.Queue = asyncio.Queue()
    gpu_semaphore = asyncio.Semaphore(len(expert_managers))
    done_event = asyncio.Event()

    geneval2_data = load_geneval2_data(GENEVAL2_DATA_JSONL)

    queued = 0
    for plan in plans:
        try:
            metadata = plan.get("metadata", {})
            image_path = metadata.get("original_image", "")
            prompt_text = metadata.get("prompt_text", "")
            prompt_id = metadata.get("prompt_id", "")
            image_id = Path(image_path).stem if image_path else "unknown"

            # Resume: skip images with existing expert results
            if (expert_results_dir / f"expert_results_{image_id}.json").exists():
                stats["resumed"] += 1
                continue

            # Load atomized data for reflector queue
            atomized_path = ATOMIZED_DIR / f"atomized_{image_id}.json"
            atomized_data = {}
            if atomized_path.exists():
                with open(atomized_path, "r", encoding="utf-8") as f:
                    atomized_data = json.load(f)
            elif prompt_id and prompt_id in geneval2_data:
                atomized_data = atomize_prompt(geneval2_data[prompt_id])
                save_atomized_prompt(atomized_data, ATOMIZED_DIR, image_id)

            await plan_queue.put((image_id, image_path, prompt_text, atomized_data, plan))
            queued += 1
        except Exception as e:
            prompt_id = plan.get("metadata", {}).get("prompt_id", "?")
            print(f"  [{prompt_id}] Step3 plan loading error: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            record_failure(stats, prompt_id, "step3_plan_load",
                           f"{type(e).__name__}: {e}")
            stats["gpu_fail"] += 1

    if stats["resumed"] > 0:
        print(f"  [RESUME] {stats['resumed']} image(s) skipped — expert results already exist")
        print(f"  [RESUME] {queued} image(s) to process")

    done_event.set()

    gpu_tasks = [
        asyncio.create_task(_gpu_worker(
            plan_queue, em, expert_results_dir,
            gpu_semaphore, i, stats, done_event,
            cpu_semaphore=cpu_semaphore,
            total=queued,
        ))
        for i, em in enumerate(expert_managers)
    ]

    await plan_queue.join()

    for _ in gpu_tasks:
        await plan_queue.put(None)
    await asyncio.gather(*gpu_tasks)

    return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step 4 Only (load from disk)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _run_step4_only(
    valid_images: list[tuple],
    image_dir: Path,
    experts_registry_str: str,
    expert_results_dir: Path,
    approved_dir: Path,
    final_reports_dir: Path,
    api_concurrency: int,
    ref_image_dir: Path | None = None,
    enable_self_reflection: bool = True,
    enable_checklist: bool = False,
    checklist_dir: Path | None = None,
    temp_reflector: float = 0.5,
    api_retry: int = 0,
) -> dict:
    """Run Step 4 (Reflector) only, loading expert results and plans from disk."""
    stats = {"reflector_ok": 0, "reflector_fail": 0, "resumed": 0}
    client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
    loop = asyncio.get_event_loop()
    api_semaphore = asyncio.Semaphore(api_concurrency)

    geneval2_data = load_geneval2_data(GENEVAL2_DATA_JSONL)

    todo: list[tuple] = []
    for img_name, img_id, prompt_text in valid_images:
        # Resume: skip images with an existing final report (and checklist
        # annotation when checklist mode is on — see _run_full_pipeline)
        report_exists = (final_reports_dir / f"final_evaluation_report_{img_id}.json").exists()
        checklist_ok = (
            checklist_dir is None
            or (checklist_dir / f"checklist_{img_id}.json").exists()
        )
        if report_exists and checklist_ok:
            stats["resumed"] += 1
            continue
        todo.append((img_name, img_id, prompt_text))

    if stats["resumed"] > 0:
        print(f"  [RESUME] {stats['resumed']} image(s) skipped — final report"
              f"{' + checklist' if checklist_dir is not None else ''} already exists")
        print(f"  [RESUME] {len(todo)} image(s) to process")

    total = len(todo)

    async def process_one(img_name, img_id, prompt_text):
        async with api_semaphore:
            try:
                image_path = resolve_image_path(image_dir, img_id)
                if image_path is None:
                    print(f"  [{img_id}] No image file found, skipping")
                    record_failure(stats, img_id, "image_lookup",
                                   f"No image file found in {image_dir} for prompt_id {img_id}")
                    stats["reflector_fail"] += 1
                    return

                # Load approved plan
                plan_path = approved_dir / f"approved_plan_{img_id}.json"
                if not plan_path.exists():
                    print(f"  [{img_id}] No approved plan found, skipping")
                    record_failure(stats, img_id, "step4_missing_plan",
                                   f"No approved plan at {plan_path} — run Step 1+2 first")
                    stats["reflector_fail"] += 1
                    return
                with open(plan_path, "r", encoding="utf-8") as f:
                    plan = json.load(f)

                # Load expert results bundle
                bundle_path = expert_results_dir / f"expert_results_{img_id}.json"
                if not bundle_path.exists():
                    print(f"  [{img_id}] No expert results found, skipping")
                    record_failure(stats, img_id, "step4_missing_expert_results",
                                   f"No expert results at {bundle_path} — run Step 3 first")
                    stats["reflector_fail"] += 1
                    return
                with open(bundle_path, "r", encoding="utf-8") as f:
                    bundle = json.load(f)

                # Load atomized data
                atomized_path = ATOMIZED_DIR / f"atomized_{img_id}.json"
                atomized_data = {}
                if atomized_path.exists():
                    with open(atomized_path, "r", encoding="utf-8") as f:
                        atomized_data = json.load(f)
                elif img_id in geneval2_data:
                    atomized_data = atomize_prompt(geneval2_data[img_id])
                    save_atomized_prompt(atomized_data, ATOMIZED_DIR, img_id)

                report = await loop.run_in_executor(
                    None,
                    _sync_reflector,
                    client, str(image_path), img_id, prompt_text, atomized_data,
                    bundle, experts_registry_str, plan, final_reports_dir,
                    ref_image_dir, enable_self_reflection,
                    enable_checklist, checklist_dir,
                    api_retry, temp_reflector,
                )
                if report is not None:
                    stats["reflector_ok"] += 1
                else:
                    stats["reflector_fail"] += 1
                    record_failure(stats, img_id, "step4_reflector",
                                   "Reflector returned no valid report (see [Reflector] errors above / debug_raw/)")
                bump_progress("Step4", total)
            except Exception as e:
                print(f"  [{img_id}] Step4 worker error: {type(e).__name__}: {e}")
                print(traceback.format_exc())
                record_failure(stats, img_id, "step4_reflector",
                               f"{type(e).__name__}: {e}")
                stats["reflector_fail"] += 1

    tasks = [
        asyncio.create_task(process_one(n, iid, pt))
        for n, iid, pt in todo
    ]
    await asyncio.gather(*tasks)
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
    final_reports_dir: Path | None = None,
    temp_router: float = 0.0,
    temp_judge: float = 0.0,
    temp_reflector: float = 0.5,
    cpu_semaphore: object | None = None,
    ref_image_dir: Path | None = None,
    enable_self_reflection: bool = True,
    enable_checklist: bool = False,
    checklist_dir: Path | None = None,
    api_retry: int = 0,
) -> dict:
    """Public entry: run async pipeline. Called from run.py."""
    geneval2_data = load_geneval2_data(GENEVAL2_DATA_JSONL)

    run_step12 = step in ("1", "2", "12", "123", "1234")
    run_step3 = step in ("3", "123", "1234")
    run_step4 = step in ("4", "1234")

    # Step 4 alone: load expert results from disk and run Reflector
    if run_step4 and not run_step12 and not run_step3:
        return asyncio.run(_run_step4_only(
            valid_images=valid_images,
            image_dir=image_dir,
            experts_registry_str=experts_registry_str,
            expert_results_dir=expert_results_dir,
            approved_dir=approved_dir,
            final_reports_dir=final_reports_dir,
            api_concurrency=api_concurrency,
            ref_image_dir=ref_image_dir,
            enable_self_reflection=enable_self_reflection,
            enable_checklist=enable_checklist,
            checklist_dir=checklist_dir,
            temp_reflector=temp_reflector,
            api_retry=api_retry,
        ))

    if run_step12 and run_step3 and expert_managers:
        client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
        reports_dir = final_reports_dir if run_step4 else None
        return asyncio.run(_run_full_pipeline(
            valid_images=valid_images,
            image_dir=image_dir,
            client=client,
            experts_registry_str=experts_registry_str,
            geneval2_data=geneval2_data,
            max_iterations=max_iterations,
            plan_dir=plan_dir,
            approved_dir=approved_dir,
            judge_feedback_dir=judge_feedback_dir,
            expert_results_dir=expert_results_dir,
            expert_managers=expert_managers,
            api_concurrency=api_concurrency,
            final_reports_dir=reports_dir,
            temp_router=temp_router,
            temp_judge=temp_judge,
            temp_reflector=temp_reflector,
            cpu_semaphore=cpu_semaphore,
            ref_image_dir=ref_image_dir,
            enable_self_reflection=enable_self_reflection,
            enable_checklist=enable_checklist,
            checklist_dir=checklist_dir,
            api_retry=api_retry,
        ))

    elif run_step12 and not run_step3:
        client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
        return asyncio.run(_run_step12_only(
            valid_images=valid_images,
            image_dir=image_dir,
            client=client,
            experts_registry_str=experts_registry_str,
            geneval2_data=geneval2_data,
            max_iterations=max_iterations,
            plan_dir=plan_dir,
            approved_dir=approved_dir,
            judge_feedback_dir=judge_feedback_dir,
            api_concurrency=api_concurrency,
            temp_router=temp_router,
            temp_judge=temp_judge,
            api_retry=api_retry,
        ))

    elif run_step3 and not run_step12:
        if not expert_managers:
            print("[ERROR] No expert managers for Step 3.")
            return {"gpu_ok": 0, "gpu_fail": 0}
        return asyncio.run(run_step3_async(
            approved_dir=approved_dir,
            expert_results_dir=expert_results_dir,
            expert_managers=expert_managers,
            cpu_semaphore=cpu_semaphore,
        ))

    return {}

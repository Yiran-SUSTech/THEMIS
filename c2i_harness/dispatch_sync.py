"""
THEMIS C2I Sync Dispatcher - Sequential single-image processing.

Called from run.py with --mode sync.
"""

import os
import sys
import json
import time
from pathlib import Path
from openai import OpenAI

from common import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
    resolve_image_path, save_judge_feedback, compute_router_scores,
)

from step1_router import generate_plan, revise_plan, load_experts_registry, generate_direct_score, save_direct_score_report
from step2_judge import review_plan
from step3_execute import (
    ExpertManager, execute_plan, save_testimony_bundle,
    load_approved_plans, resolve_image_path as resolve_image_path_global,
    collect_required_expert_ids, EXPERT_MODULE_MAP,
)


def _run_single_image(
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
    api_retry: int = 0,
    temp_router: float = 0.0,
    temp_judge: float = 0.0,
) -> dict | None:
    """Run Router+Judge loop for a single image (synchronous)."""
    print(f"\n{'#'*60}")
    print(f"  Image: {img_id} | Class: {class_label}")
    print(f"{'#'*60}")

    print(f"\n  [Step 1] Router generating initial plan...")
    current_plan = generate_plan(
        client, image_path, class_id, class_label, experts_registry_str,
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
            client, image_path, class_id, class_label,
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
                client, image_path, class_id, class_label,
                experts_registry_str, current_plan, feedback_history,
                api_retry=api_retry, temperature=temp_router,
            )

            if revised_plan is not None:
                current_plan = revised_plan

    current_plan["metadata"]["iteration_log"] = iteration_log

    router_scores = compute_router_scores(current_plan)
    current_plan["router_scores"] = router_scores

    cs = router_scores["checkpoint_summary"]
    a_s = router_scores["artifact_summary"]
    print(f"\n  --- Router Assessment ---")
    print(f"  Alignment: {router_scores['router_alignment_score']:.2f}/5.0 "
          f"({cs['present']}/{cs['testable']} passed, {cs['untestable']} untestable)")
    if a_s["count"] > 0:
        print(f"  Artifact:  {router_scores['router_artifact_score']:.2f}/5.0 "
              f"({a_s['count']} artifacts, max severity={a_s['max_severity']:.1f}, "
              f"{a_s['severe_count']} severe, {a_s['minor_count']} minor)")
    else:
        print(f"  Artifact:  5.00/5.0 (no artifacts observed)")

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
    save_pose_viz: bool = False,
    ref_enable: bool = False,
    enable_checklist: bool = False,
    checklist_dir: Path | None = None,
    api_retry: int = 0,
    without_expert: bool = False,
    without_expert_dir: Path | None = None,
    pose_hard_cap: bool = False,
) -> dict:
    """Run the full pipeline in synchronous serial mode."""
    stats = {
        "api_ok": 0, "api_fail": 0,
        "gpu_ok": 0, "gpu_fail": 0,
        "step4_ok": 0, "step4_fail": 0,
    }

    # Without-expert ablation mode: router-only direct scoring
    if without_expert:
        client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
        output_dir = without_expert_dir if without_expert_dir is not None else Path("output/without_expert_reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = {"router_ok": 0, "router_fail": 0}

        for img_name, img_id, class_id, class_label in valid_images:
            img_path = resolve_image_path(image_dir, img_id)
            if img_path is None:
                print(f"[WARN] Image not found: {img_id}")
                stats["router_fail"] += 1
                continue

            print(f"\n{'#'*60}")
            print(f"  [Without-Expert] Image: {img_id} | Class: {class_label}")
            print(f"{'#'*60}")

            result = generate_direct_score(
                client, str(img_path), class_id, class_label,
                experts_registry_str, api_retry=api_retry, temperature=temp_router,
            )
            if result is not None:
                save_direct_score_report(result, output_dir)
                stats["router_ok"] += 1
                al = result.get("alignment_score", 0.0)
                ar = result.get("artifact_score", 0.0)
                print(f"  [{img_id}] DirectScore: alignment={al:.2f} artifact={ar:.2f}")
            else:
                stats["router_fail"] += 1

        return stats

    run_step12 = step in ("1", "2", "12", "123", "1234")
    run_step3 = step in ("3", "123", "1234")
    run_step4 = step in ("4", "1234")

    client = None
    if run_step12 or run_step4:
        client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

    expert_manager = expert_managers[0] if expert_managers else None

    if run_step12 and client:
        for img_name, img_id, class_id, class_label in valid_images:
            img_path = resolve_image_path(image_dir, img_id)
            if img_path is None:
                print(f"[WARN] Image not found: {img_id}")
                stats["api_fail"] += 1
                continue

            plan = _run_single_image(
                client, str(img_path), img_id, class_id, class_label,
                experts_registry_str, max_iterations,
                plan_dir, approved_dir, judge_feedback_dir,
                api_retry=api_retry, temp_router=temp_router, temp_judge=temp_judge,
            )
            if plan is not None:
                stats["api_ok"] += 1
            else:
                stats["api_fail"] += 1

    if run_step3 or run_step4:
        if run_step3 and not expert_manager:
            print("[ERROR] No expert managers loaded for Step 3.")
            return stats

        plans = load_approved_plans(approved_dir)
        if not plans:
            print("[ERROR] No approved plans found. Run Step 1+2 first.")
            return stats

        for idx, plan in enumerate(plans, 1):
            metadata = plan.get("metadata", {})
            image_path_raw = metadata.get("original_image", "")
            class_label = metadata.get("class_label", "")
            class_id = metadata.get("class_id", 0)

            image_path = resolve_image_path_global(image_path_raw)
            if image_path is None:
                stats["gpu_fail"] += 1
                if run_step4:
                    stats["step4_fail"] += 1
                continue

            image_id = os.path.splitext(os.path.basename(image_path))[0]
            print(f"\n[{idx}/{len(plans)}] {os.path.basename(image_path)}")

            bundle = None
            if run_step3:
                try:
                    bundle = execute_plan(
                        plan, expert_manager, image_path, class_label,
                        save_pose_viz=save_pose_viz,
                    )
                    save_testimony_bundle(bundle, expert_results_dir)
                    stats["gpu_ok"] += 1
                except Exception as e:
                    print(f"  [FATAL] {type(e).__name__}: {e}")
                    stats["gpu_fail"] += 1

            if run_step4 and bundle is not None and client:
                print(f"\n  [Step 4] Reflector evaluating {image_id}...")
                try:
                    from step4_reflector import run_reflector, save_final_report, print_final_summary, select_reference_images, build_checklist_annotation, save_checklist_annotation
                    from step1_router import get_taxonomy_info, get_structured_taxonomy_info

                    ref_images = None
                    if ref_enable:
                        exclude_name = os.path.basename(str(image_path))
                        ref_images = select_reference_images(
                            class_id, image_dir, exclude_image_name=exclude_name,
                        )

                    report = run_reflector(
                        client=client,
                        image_path=image_path,
                        class_id=class_id,
                        class_label=class_label,
                        expert_results=bundle,
                        experts_registry_str=experts_registry_str,
                        router_plan=plan,
                        ref_images=ref_images,
                        enable_checklist=enable_checklist,
                        api_retry=api_retry,
                        temperature=temp_reflector,
                        pose_hard_cap=pose_hard_cap,
                    )
                    if report is None:
                        print(f"  [Step 4] FAILED - Reflector returned no valid response")
                        stats["step4_fail"] += 1
                    else:
                        if final_reports_dir:
                            final_reports_dir.mkdir(parents=True, exist_ok=True)
                            save_final_report(report, final_reports_dir)
                        if enable_checklist and checklist_dir is not None:
                            image_name = os.path.basename(str(image_path))
                            annotation = build_checklist_annotation(report, class_id, class_label, image_name)
                            save_checklist_annotation(annotation, checklist_dir)
                        print_final_summary(report)
                        stats["step4_ok"] += 1
                except Exception as e:
                    print(f"  [Step 4] FATAL: {type(e).__name__}: {e}")
                    stats["step4_fail"] += 1
            elif run_step4 and bundle is None:
                stats["step4_fail"] += 1

    return stats

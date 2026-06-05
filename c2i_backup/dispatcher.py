import os
import sys
import re
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
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

PLAN_DIR = Path(__file__).resolve().parent / "output" / "plans"
APPROVED_DIR = Path(__file__).resolve().parent / "output" / "approved_plans"
JUDGE_FEEDBACK_DIR = Path(__file__).resolve().parent / "output" / "judge_feedback"
EXPERT_RESULTS_DIR = Path(__file__).resolve().parent / "output" / "expert_results"
FINAL_REPORTS_DIR = Path(__file__).resolve().parent / "output" / "final_reports"

EXPERT_OUTPUT_DIRS = {
    "topology_boundary_auditor": Path(__file__).resolve().parent / "output" / "sam_masks",
    "geometric_depth_auditor": Path(__file__).resolve().parent / "output" / "depth_maps",
}

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from step1_router import generate_plan, revise_plan, validate_plan, load_experts_registry
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
from step4_reflector import run_reflector, save_final_report, print_final_summary


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
        "judge_cost_seconds": judge_result.get("judge_cost_seconds", 0),
        "audited_plan": plan,
    }
    feedback_path = judge_feedback_dir / f"judge_feedback_{img_id}_iter{iteration}.json"
    with open(feedback_path, "w", encoding="utf-8") as f:
        json.dump(feedback_record, f, indent=4, ensure_ascii=False)
    print(f"  [SAVED] Judge feedback -> {feedback_path.name}")


def run_pipeline_for_image(
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
    print(f"\n{'#'*60}")
    print(f"  Image: {img_id} | Class: {class_label}")
    print(f"{'#'*60}")

    print(f"\n  [Step 1] Router generating initial plan...")
    current_plan = generate_plan(
        client, image_path, class_id, class_label, experts_registry_str,
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
    print(f"  [SAVED] Initial plan -> {plan_save_path.name}")

    feedback_history: list[dict] = []
    iteration_log: list[str] = []

    for iteration in range(1, max_iterations + 1):
        print(f"\n  [Step 2] Judge reviewing plan (iteration {iteration}/{max_iterations})...")
        judge_result = review_plan(
            client, image_path, class_id, class_label,
            current_plan, experts_registry_str,
        )

        if judge_result is None:
            print(f"  [Step 2] FAILED - Judge returned no valid response")
            iteration_log.append(f"Iteration {iteration}: Judge Error")
            break

        is_approved = judge_result.get("is_approved", False)
        reasons = judge_result.get("reasons_for_rejection", "")
        suggestions = judge_result.get("suggestions", [])
        judge_cost = judge_result.get("judge_cost_seconds", 0)

        print(f"  [Step 2] Verdict: {'Approved' if is_approved else 'Rejected'} | Cost: {judge_cost:.2f}s")
        if not is_approved:
            print(f"  [Step 2] Reasons: {reasons}")
            print(f"  [Step 2] Suggestions: {json.dumps(suggestions, ensure_ascii=False)}")

        if judge_feedback_dir is not None:
            save_judge_feedback(
                judge_feedback_dir, img_id, iteration,
                judge_result, current_plan, class_label,
            )

        if is_approved:
            print(f"\n  Iteration {iteration}: Approved!")
            iteration_log.append(f"Iteration {iteration}: Approved!")
            current_plan["metadata"]["judge_approved"] = True
            current_plan["metadata"]["judge_iterations"] = iteration
            break
        else:
            print(f"  Iteration {iteration}: Rejected -> Dispatching to Router for revision")
            iteration_log.append(f"Iteration {iteration}: Rejected")

            if iteration == max_iterations:
                print(f"\n  [WARN] Max iterations ({max_iterations}) reached. Forcing approval of latest plan.")
                iteration_log.append(
                    f"Iteration {iteration}: Max iterations reached - forced approval"
                )
                current_plan["metadata"]["judge_approved"] = False
                current_plan["metadata"]["judge_forced"] = True
                current_plan["metadata"]["judge_iterations"] = iteration
                break

            feedback_history.append({
                "reasons_for_rejection": reasons,
                "suggestions": suggestions,
            })

            print(f"  [Step 1] Router revising plan based on Judge feedback...")
            revised_plan = revise_plan(
                client, image_path, class_id, class_label,
                experts_registry_str, current_plan, feedback_history,
            )

            if revised_plan is None:
                print(f"  [Step 1] Revision FAILED - keeping previous plan")
                iteration_log.append(f"Iteration {iteration}: Router revision failed")
                continue

            is_valid = revised_plan.get("metadata", {}).get("plan_valid", False)
            print(f"  [Step 1] Plan revised | Valid: {is_valid} | "
                  f"Cost: {revised_plan['metadata']['router_cost_seconds']:.2f}s")

            revision_save_path = plan_dir / f"plan_{img_id}_rev{iteration}.json"
            with open(revision_save_path, "w", encoding="utf-8") as f:
                json.dump(revised_plan, f, indent=4, ensure_ascii=False)
            print(f"  [SAVED] Revised plan -> {revision_save_path.name}")

            current_plan = revised_plan

    current_plan["metadata"]["iteration_log"] = iteration_log

    approved_save_path = approved_dir / f"approved_plan_{img_id}.json"
    with open(approved_save_path, "w", encoding="utf-8") as f:
        json.dump(current_plan, f, indent=4, ensure_ascii=False)
    print(f"\n  [SAVED] Approved plan -> {approved_save_path.name}")

    print(f"\n  --- Iteration Log ---")
    for entry in iteration_log:
        print(f"  {entry}")
    print(f"  ---------------------")

    return current_plan


def run_step3_pipeline(
    approved_dir: Path,
    expert_results_dir: Path,
    gpu_config: dict,
    image_id_filter: str = "",
    limit: int = 0,
    expert_manager: ExpertManager | None = None,
    save_pose_viz: bool = False,
) -> None:
    """Step 3: Load expert models, read approved plans, execute local expert evaluation.

    This function:
    1. Loads all required expert models onto designated GPUs at startup
    2. Reads approved_plan_*.json from the approved_plans directory
    3. For each plan, orchestrates expert execution (with dependency resolution)
    4. Saves standardized Expert Testimony Bundles for Step 4 (Reflector)

    If expert_manager is provided (pre-loaded at program startup), it will be
    reused directly instead of creating a new one. Cleanup is only performed
    when the manager was created internally (not passed from outside).
    """
    plans = load_approved_plans(approved_dir)

    if image_id_filter:
        plans = [p for p in plans if image_id_filter in p.get("_source_file", "")]
        if not plans:
            print(f"[ERROR] No approved plan found for image ID: {image_id_filter}")
            return

    if limit > 0:
        plans = plans[:limit]

    if not plans:
        print("[ERROR] No approved plans found. Run Step 1+2 first.")
        return

    required_ids = collect_required_expert_ids(plans)
    print(f"\n[Step 3] Plans to execute: {len(plans)}")
    print(f"[Step 3] Required experts: {required_ids}")

    owns_manager = expert_manager is None
    if owns_manager:
        expert_manager = ExpertManager(gpu_config=gpu_config, expert_output_dirs=EXPERT_OUTPUT_DIRS)
        expert_manager.load_all(required_ids)

    if not expert_manager.loaded_experts:
        print("[ERROR] No experts loaded successfully. Cannot proceed with Step 3.")
        return

    print(f"\n{'=' * 60}")
    print(f"  Step 3: Executing {len(plans)} Approved Plans")
    print(f"  Loaded experts: {expert_manager.get_loaded_ids()}")
    print(f"{'=' * 60}")

    success_count = 0
    failed_count = 0
    total_start = time.time()

    for idx, plan in enumerate(plans, 1):
        metadata = plan.get("metadata", {})
        image_path_raw = metadata.get("original_image", "")
        class_label = metadata.get("class_label", "")
        source_file = plan.get("_source_file", "unknown")

        image_path = resolve_image_path_global(image_path_raw)
        if image_path is None:
            print(f"\n[{idx}/{len(plans)}] Image not found: {image_path_raw} "
                  f"(Plan: {source_file})")
            failed_count += 1
            continue

        print(f"\n[{idx}/{len(plans)}] {os.path.basename(image_path)} "
              f"(Plan: {source_file})")

        try:
            bundle = execute_plan(plan, expert_manager, image_path, class_label, save_pose_viz=save_pose_viz)
            save_testimony_bundle(bundle, expert_results_dir)
            success_count += 1
        except Exception as e:
            print(f"  [FATAL] Pipeline crashed: {type(e).__name__}: {e}")
            failed_count += 1

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 60}")
    print(f"  Step 3 Execution Summary")
    print(f"{'=' * 60}")
    print(f"  Total plans:     {len(plans)}")
    print(f"  Successful:      {success_count}")
    print(f"  Failed:          {failed_count}")
    print(f"  Total elapsed:   {total_elapsed:.2f}s")
    print(f"{'=' * 60}")

    if owns_manager:
        expert_manager.cleanup()


def _load_expert_results(expert_results_dir: Path) -> list[dict]:
    results = []
    if not expert_results_dir.exists():
        print(f"[WARN] Expert results directory not found: {expert_results_dir}")
        return results
    for f in sorted(expert_results_dir.glob("expert_results_*.json")):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                result = json.load(fp)
            result["_source_file"] = f.name
            results.append(result)
        except Exception as e:
            print(f"[WARN] Failed to load expert result {f.name}: {e}")
    return results


def run_step4_pipeline(
    client: OpenAI,
    expert_results_dir: Path,
    final_reports_dir: Path,
    experts_registry_str: str,
    image_id_filter: str = "",
    limit: int = 0,
) -> None:
    expert_results_list = _load_expert_results(expert_results_dir)

    if image_id_filter:
        expert_results_list = [
            r for r in expert_results_list
            if image_id_filter in r.get("image_id", "")
        ]
        if not expert_results_list:
            print(f"[ERROR] No expert results found for image ID: {image_id_filter}")
            return

    if limit > 0:
        expert_results_list = expert_results_list[:limit]

    if not expert_results_list:
        print("[ERROR] No expert results found. Run Step 3 first.")
        return

    print(f"\n{'=' * 60}")
    print(f"  Step 4: Reflector Final Evaluation")
    print(f"  Expert results to evaluate: {len(expert_results_list)}")
    print(f"  Final reports dir: {final_reports_dir}")
    print(f"{'=' * 60}")

    final_reports_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failed_count = 0
    total_start = time.time()

    for idx, expert_result in enumerate(expert_results_list, 1):
        image_id = expert_result.get("image_id", "unknown")
        class_id = expert_result.get("class_id")
        class_label = expert_result.get("class_label", "N/A")
        source_file = expert_result.get("_source_file", "unknown")

        if class_id is None:
            print(f"\n[{idx}/{len(expert_results_list)}] Skipping {source_file}: missing class_id")
            failed_count += 1
            continue

        image_path_raw = expert_result.get("image_path", "")
        image_path = resolve_image_path_global(image_path_raw)
        if image_path is None:
            img_stem = image_id
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = IMAGE_DIR / f"{img_stem}{ext}"
                if candidate.exists():
                    image_path = str(candidate)
                    break

        if image_path is None:
            print(f"\n[{idx}/{len(expert_results_list)}] Image not found for ID: {image_id}")
            failed_count += 1
            continue

        print(f"\n[{idx}/{len(expert_results_list)}] {os.path.basename(image_path)} "
              f"| Class: {class_label} (Result: {source_file})")

        try:
            report = run_reflector(
                client=client,
                image_path=image_path,
                class_id=class_id,
                class_label=class_label,
                expert_results=expert_result,
                experts_registry_str=experts_registry_str,
            )

            if report is None:
                print(f"  [Step 4] FAILED - Reflector returned no valid response")
                failed_count += 1
                continue

            save_final_report(report, final_reports_dir)
            print_final_summary(report)

            success_count += 1
        except Exception as e:
            print(f"  [FATAL] Reflector pipeline crashed: {type(e).__name__}: {e}")
            failed_count += 1

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 60}")
    print(f"  Step 4 Reflector Summary")
    print(f"{'=' * 60}")
    print(f"  Total evaluations: {len(expert_results_list)}")
    print(f"  Successful:        {success_count}")
    print(f"  Failed:            {failed_count}")
    print(f"  Total elapsed:     {total_elapsed:.2f}s")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="THEMIS C2I Dispatcher - Orchestrates the full evaluation pipeline"
    )
    parser.add_argument(
        "--step",
        type=str,
        default="12",
        choices=["1", "2", "12", "3", "123", "4", "1234"],
        help=(
            "Which step(s) to run: "
            "'1'=Router only, '2'=Judge only, '12'=Router+Judge (default), "
            "'3'=Expert execution only, '4'=Reflector only, "
            "'123'=Full pipeline (Steps 1-3), '1234'=Full pipeline (Steps 1-4)"
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Maximum Judge-Router iteration rounds (default: 2)",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=str(IMAGE_DIR),
        help="Directory containing test images",
    )
    parser.add_argument(
        "--class-ids",
        type=str,
        default=str(CLASS_IDS_TXT),
        help="Path to class_ids.txt",
    )
    parser.add_argument(
        "--plan-dir",
        type=str,
        default=str(PLAN_DIR),
        help="Directory to save intermediate plan JSON files",
    )
    parser.add_argument(
        "--approved-dir",
        type=str,
        default=str(APPROVED_DIR),
        help="Directory to save/read approved plan JSON files",
    )
    parser.add_argument(
        "--expert-results-dir",
        type=str,
        default=str(EXPERT_RESULTS_DIR),
        help="Directory to save expert_results_*.json files (Step 3)",
    )
    parser.add_argument(
        "--final-reports-dir",
        type=str,
        default=str(FINAL_REPORTS_DIR),
        help="Directory to save final_evaluation_report_*.json files (Step 4)",
    )
    parser.add_argument(
        "--save-feedback",
        action="store_true",
        default=False,
        help="Save Judge feedback and verdict for each iteration to disk",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of images to process (0 = all)",
    )
    parser.add_argument(
        "--image-id",
        type=str,
        default="",
        help="Process a single image by its ID (e.g., 000000)",
    )
    parser.add_argument(
        "--gpu-config",
        type=str,
        default=None,
        help="Path to a custom GPU allocation JSON file (Step 3)",
    )
    parser.add_argument(
        "--save-pose-viz",
        action="store_true",
        default=False,
        help="Save animal_pose_auditor keypoint visualization with ID and confidence labels",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    class_ids_txt = Path(args.class_ids)
    plan_dir = Path(args.plan_dir)
    approved_dir = Path(args.approved_dir)
    expert_results_dir = Path(args.expert_results_dir)
    final_reports_dir = Path(args.final_reports_dir)

    plan_dir.mkdir(parents=True, exist_ok=True)
    approved_dir.mkdir(parents=True, exist_ok=True)
    expert_results_dir.mkdir(parents=True, exist_ok=True)
    final_reports_dir.mkdir(parents=True, exist_ok=True)

    judge_feedback_dir = None
    if args.save_feedback:
        judge_feedback_dir = JUDGE_FEEDBACK_DIR
        judge_feedback_dir.mkdir(parents=True, exist_ok=True)

    gpu_config = DEFAULT_GPU_CONFIG
    if args.gpu_config:
        try:
            with open(args.gpu_config, "r", encoding="utf-8") as f:
                gpu_config = json.load(f)
            print(f"[INFO] Loaded custom GPU config from: {args.gpu_config}")
        except Exception as e:
            print(f"[WARN] Failed to load GPU config, using defaults: {e}")

    step = args.step
    run_step12 = step in ("1", "2", "12", "123", "1234")
    run_step3 = step in ("3", "123", "1234")
    run_step4 = step in ("4", "1234")
    is_full_pipeline = step in ("123", "1234")

    # ── Pre-load Expert Models at Startup ───────────────────────────────
    # When running Step 3 (alone or as part of 123), load all expert models
    # onto GPUs immediately at program startup, before any other work begins.
    expert_manager = None
    if run_step3:
        print(f"\n{'='*60}")
        print(f"  Pre-loading Expert Models at Startup")
        print(f"{'='*60}")

        if step == "3":
            plans = load_approved_plans(approved_dir)
            if args.image_id:
                plans = [p for p in plans if args.image_id in p.get("_source_file", "")]
            if args.limit > 0:
                plans = plans[:args.limit]
            if plans:
                required_ids = collect_required_expert_ids(plans)
            else:
                required_ids = list(EXPERT_MODULE_MAP.keys())
        else:
            required_ids = list(EXPERT_MODULE_MAP.keys())

        expert_manager = ExpertManager(gpu_config=gpu_config, expert_output_dirs=EXPERT_OUTPUT_DIRS)
        expert_manager.load_all(required_ids)

        if not expert_manager.loaded_experts:
            print("[ERROR] No experts loaded successfully. Step 3 will fail.")
            if step == "3":
                return

        print(f"\n  Pre-loaded experts: {expert_manager.get_loaded_ids()}")
        print(f"{'='*60}")

    # ── Full Pipeline (Step 1→2→3 per image) ────────────────────────────
    # For --step 123, process each image through all three steps sequentially
    # before moving to the next image. Expert models stay loaded throughout.
    if is_full_pipeline:
        if not DASHSCOPE_API_KEY:
            print("[ERROR] DASHSCOPE_API_KEY not set. Required for Step 1+2.")
            return

        client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

        img_to_class = parse_class_ids(str(class_ids_txt))
        imagenet_classes = load_imagenet_classes(str(IMAGENET_CLASSES_JSON))
        experts_registry_str = load_experts_registry(str(EXPERTS_REGISTRY_JSON))

        if not img_to_class:
            print("[ERROR] No image-class mappings loaded. Check class_ids.txt path.")
            return

        image_files = sorted(
            f
            for f in os.listdir(image_dir)
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
                (name, iid, cid, cl) for name, iid, cid, cl in valid_images
                if iid == args.image_id
            ]
            if not valid_images:
                print(f"[ERROR] Image ID '{args.image_id}' not found in class_ids.txt")
                return

        if args.limit > 0:
            valid_images = valid_images[:args.limit]

        print(f"\n{'='*60}")
        if run_step4:
            print(f"THEMIS C2I Full Pipeline (Step 1→2→3→4 per image)")
        else:
            print(f"THEMIS C2I Full Pipeline (Step 1→2→3 per image)")
        print(f"Images to process: {len(valid_images)}")
        print(f"Max iterations:    {args.max_iterations}")
        print(f"Expert models:     LOADED (resident in GPU memory)")
        print(f"Plan directory:    {plan_dir}")
        print(f"Approved dir:      {approved_dir}")
        print(f"Expert results:    {expert_results_dir}")
        if run_step4:
            print(f"Final reports:     {final_reports_dir}")
        print(f"{'='*60}")

        step12_ok = 0
        step12_fail = 0
        step3_ok = 0
        step3_fail = 0
        step4_ok = 0
        step4_fail = 0
        total_start = time.time()

        for idx, (img_name, img_id, class_id, class_label) in enumerate(valid_images, 1):
            image_path = resolve_image_path(image_dir, img_id)
            if image_path is None:
                print(f"\n[{idx}/{len(valid_images)}] Image not found for ID: {img_id}")
                step12_fail += 1
                step3_fail += 1
                if run_step4:
                    step4_fail += 1
                continue

            print(f"\n{'#'*60}")
            print(f"  [{idx}/{len(valid_images)}] {img_name} | {class_label}")
            print(f"{'#'*60}")

            # ── Step 1+2: Router + Judge ─────────────────────────────
            approved_plan = run_pipeline_for_image(
                client=client,
                image_path=str(image_path),
                img_id=img_id,
                class_id=class_id,
                class_label=class_label,
                experts_registry_str=experts_registry_str,
                max_iterations=args.max_iterations,
                plan_dir=plan_dir,
                approved_dir=approved_dir,
                judge_feedback_dir=judge_feedback_dir,
            )

            if approved_plan is None:
                step12_fail += 1
                step3_fail += 1
                if run_step4:
                    step4_fail += 1
                continue

            step12_ok += 1

            # ── Step 3: Execute Approved Plan ────────────────────────
            resolved_path = resolve_image_path_global(
                approved_plan.get("metadata", {}).get("original_image", "")
            )
            if resolved_path is None:
                print(f"  [Step 3] Image path resolution failed, skipping execution.")
                step3_fail += 1
                if run_step4:
                    step4_fail += 1
                continue

            bundle = None
            try:
                bundle = execute_plan(
                    approved_plan, expert_manager, resolved_path, class_label,
                    save_pose_viz=args.save_pose_viz,
                )
                save_testimony_bundle(bundle, expert_results_dir)
                step3_ok += 1
            except Exception as e:
                print(f"  [Step 3] FATAL: {type(e).__name__}: {e}")
                step3_fail += 1

            # ── Step 4: Reflector (inline per image) ─────────────────
            if run_step4 and bundle is not None:
                print(f"\n  [Step 4] Reflector evaluating {img_id}...")
                try:
                    report = run_reflector(
                        client=client,
                        image_path=resolved_path,
                        class_id=class_id,
                        class_label=class_label,
                        expert_results=bundle,
                        experts_registry_str=experts_registry_str,
                    )
                    if report is None:
                        print(f"  [Step 4] FAILED - Reflector returned no valid response")
                        step4_fail += 1
                    else:
                        save_final_report(report, final_reports_dir)
                        print_final_summary(report)
                        step4_ok += 1
                except Exception as e:
                    print(f"  [Step 4] FATAL: {type(e).__name__}: {e}")
                    step4_fail += 1
            elif run_step4 and bundle is None:
                step4_fail += 1

        total_elapsed = time.time() - total_start

        print(f"\n{'='*60}")
        print(f"  Full Pipeline Summary")
        print(f"{'='*60}")
        print(f"  Total images:       {len(valid_images)}")
        print(f"  Step 1+2 OK:        {step12_ok}")
        print(f"  Step 1+2 Failed:    {step12_fail}")
        print(f"  Step 3 OK:          {step3_ok}")
        print(f"  Step 3 Failed:      {step3_fail}")
        if run_step4:
            print(f"  Step 4 OK:          {step4_ok}")
            print(f"  Step 4 Failed:      {step4_fail}")
        print(f"  Total elapsed:      {total_elapsed:.2f}s")
        print(f"{'='*60}")

        if expert_manager is not None:
            expert_manager.cleanup()

    # ── Step 4: Reflector (standalone only, not when already inlined in 1234) ───
    if run_step4 and not is_full_pipeline:
        if not DASHSCOPE_API_KEY:
            print("[ERROR] DASHSCOPE_API_KEY not set. Required for Step 4 (Reflector).")
            return

        client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
        experts_registry_str = load_experts_registry(str(EXPERTS_REGISTRY_JSON))

        run_step4_pipeline(
            client=client,
            expert_results_dir=expert_results_dir,
            final_reports_dir=final_reports_dir,
            experts_registry_str=experts_registry_str,
            image_id_filter=args.image_id,
            limit=args.limit,
        )

    if is_full_pipeline:
        return

    # ── Step 1+2 only (batch mode) ──────────────────────────────────────
    if run_step12 and not is_full_pipeline:
        if not DASHSCOPE_API_KEY:
            print("[ERROR] DASHSCOPE_API_KEY not set. Required for Step 1+2.")
            print("  export DASHSCOPE_API_KEY=your_api_key_here  (Linux/Mac)")
            print("  set DASHSCOPE_API_KEY=your_api_key_here     (Windows CMD)")
            return

        client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

        img_to_class = parse_class_ids(str(class_ids_txt))
        imagenet_classes = load_imagenet_classes(str(IMAGENET_CLASSES_JSON))
        experts_registry_str = load_experts_registry(str(EXPERTS_REGISTRY_JSON))

        if not img_to_class:
            print("[ERROR] No image-class mappings loaded. Check class_ids.txt path.")
            return

        image_files = sorted(
            f
            for f in os.listdir(image_dir)
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
                (name, iid, cid, cl) for name, iid, cid, cl in valid_images
                if iid == args.image_id
            ]
            if not valid_images:
                print(f"[ERROR] Image ID '{args.image_id}' not found in class_ids.txt")
                return

        if args.limit > 0:
            valid_images = valid_images[:args.limit]

        print(f"\n{'='*60}")
        print(f"THEMIS C2I Dispatcher - Step 1+2 (Router + Judge)")
        print(f"Images to process: {len(valid_images)}")
        print(f"Max iterations:    {args.max_iterations}")
        print(f"Save feedback:     {args.save_feedback}")
        print(f"Plan directory:    {plan_dir}")
        print(f"Approved dir:      {approved_dir}")
        if judge_feedback_dir:
            print(f"Feedback dir:      {judge_feedback_dir}")
        print(f"{'='*60}")

        success_count = 0
        failed_count = 0
        total_start = time.time()

        for idx, (img_name, img_id, class_id, class_label) in enumerate(valid_images, 1):
            image_path = resolve_image_path(image_dir, img_id)
            if image_path is None:
                print(f"[ERROR] Image not found for ID: {img_id}")
                failed_count += 1
                continue

            print(f"\n[{idx}/{len(valid_images)}] Processing: {img_name}")

            result = run_pipeline_for_image(
                client=client,
                image_path=str(image_path),
                img_id=img_id,
                class_id=class_id,
                class_label=class_label,
                experts_registry_str=experts_registry_str,
                max_iterations=args.max_iterations,
                plan_dir=plan_dir,
                approved_dir=approved_dir,
                judge_feedback_dir=judge_feedback_dir,
            )

            if result is not None:
                success_count += 1
            else:
                failed_count += 1

        total_cost = time.time() - total_start

        print(f"\n{'='*60}")
        print(f"Step 1+2 Summary")
        print(f"{'='*60}")
        print(f"  Total images:    {len(valid_images)}")
        print(f"  Successful:      {success_count}")
        print(f"  Failed:          {failed_count}")
        print(f"  Total elapsed:   {total_cost:.2f}s")
        print(f"{'='*60}")

    # ── Step 3 only (batch mode) ────────────────────────────────────────
    if run_step3 and not is_full_pipeline:
        print(f"\n{'='*60}")
        print(f"THEMIS C2I Dispatcher - Step 3 (Expert Execution)")
        print(f"Approved dir:       {approved_dir}")
        print(f"Expert results dir: {expert_results_dir}")
        print(f"{'='*60}")

        run_step3_pipeline(
            approved_dir=approved_dir,
            expert_results_dir=expert_results_dir,
            gpu_config=gpu_config,
            image_id_filter=args.image_id,
            limit=args.limit,
            expert_manager=expert_manager,
            save_pose_viz=args.save_pose_viz,
        )


if __name__ == "__main__":
    main()

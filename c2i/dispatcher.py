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

IMAGE_DIR = PROJECT_ROOT / "test_images"
CLASS_IDS_TXT = IMAGE_DIR / "class_ids.txt"
IMAGENET_CLASSES_JSON = PROJECT_ROOT / "imagenet_classes.json"
EXPERTS_REGISTRY_JSON = PROJECT_ROOT / "expert_registry.json"

PLAN_DIR = Path(__file__).resolve().parent / "output" / "plans"
APPROVED_DIR = Path(__file__).resolve().parent / "output" / "approved_plans"
JUDGE_FEEDBACK_DIR = Path(__file__).resolve().parent / "output" / "judge_feedback"

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from step1_router import generate_plan, revise_plan, validate_plan, load_experts_registry
from step2_judge import review_plan


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

        print(f"  [Step 2] Verdict: {'Approved' if is_approved else 'Rejected'}")
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


def main():
    parser = argparse.ArgumentParser(
        description="THEMIS C2I Dispatcher - Orchestrates Router and Judge iteration loop"
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
        help="Directory to save approved plan JSON files",
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
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    class_ids_txt = Path(args.class_ids)
    plan_dir = Path(args.plan_dir)
    approved_dir = Path(args.approved_dir)

    plan_dir.mkdir(parents=True, exist_ok=True)
    approved_dir.mkdir(parents=True, exist_ok=True)

    judge_feedback_dir = None
    if args.save_feedback:
        judge_feedback_dir = JUDGE_FEEDBACK_DIR
        judge_feedback_dir.mkdir(parents=True, exist_ok=True)

    if not DASHSCOPE_API_KEY:
        print("[ERROR] DASHSCOPE_API_KEY not set. Please run:")
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
        valid_images = valid_images[: args.limit]

    print(f"\n{'='*60}")
    print(f"THEMIS C2I Dispatcher")
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
    print(f"Dispatcher Summary")
    print(f"{'='*60}")
    print(f"  Total images:    {len(valid_images)}")
    print(f"  Successful:      {success_count}")
    print(f"  Failed:          {failed_count}")
    print(f"  Total elapsed:   {total_cost:.2f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

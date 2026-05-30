import os
import sys
import re
import json
import base64
import time
import argparse
from pathlib import Path
from openai import OpenAI

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "test_images"
CLASS_IDS_TXT = IMAGE_DIR / "class_ids.txt"
TAXONOMY_DIR = PROJECT_ROOT / "taxonomy_info"
EXPERTS_REGISTRY_JSON = PROJECT_ROOT / "expert_registry.json"
IMAGENET_CLASSES_JSON = PROJECT_ROOT / "imagenet_classes.json"
OUTPUT_PLAN_DIR = Path(__file__).resolve().parent / "output" / "plans"

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ROUTER_MODEL = "qwen3.6-plus"


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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


def load_experts_registry(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.dumps(json.load(f), indent=2)


def get_taxonomy_info(class_id: int) -> dict | None:
    batch_num = class_id // 10
    batch_file = TAXONOMY_DIR / f"taxonomy_enriched_Batch_{batch_num}.json"
    if not batch_file.exists():
        return None
    with open(batch_file, "r", encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        if item.get("class_id") == class_id:
            return item
    return None


def build_router_prompt(
    class_label: str,
    taxonomy_info: dict | None,
    experts_registry_str: str,
) -> str:
    taxonomy_desc = "No specific taxonomy prior knowledge found for this class."
    taxonomy_class_name = class_label
    if taxonomy_info:
        taxonomy_class_name = taxonomy_info.get("class_name", class_label)
        taxonomy_desc = taxonomy_info.get(
            "enriched_description", taxonomy_desc
        )

    return f"""You are the Lead Strategic Planner (Router) for an advanced AI image evaluation system.
Your task is to analyze the provided image and its specific class category, then formulate a rigorous evaluation plan using the available Expert Registry.

**[Input Data]**
- **Class Label:** {class_label}
- **Taxonomy Class Name:** {taxonomy_class_name}
- **Taxonomy Prior Knowledge (Ground Truth):** {taxonomy_desc}
- **Expert Registry (Available Tools):** {experts_registry_str}

**[Strategic Instruction]**
1. **Identify Category Archetype:** Determine if the class "{class_label}" is an **Organism** (animal/plant), a **Rigid Object** (architecture/tool/vehicle), or a **Natural Scene** (landscape/texture).
2. **Feature Mapping:** Based on the Taxonomy Prior Knowledge, extract 2-10 "Non-negotiable" diagnostic features that must be verified in the image (e.g., specific symmetry for buildings, anatomical counts for animals, textural coherence for landscapes).
3. **Visual Risk Assessment:** Scrutinize the image for category-specific flaws:
   - *Organisms:* Look for "Melting" limbs, missing parts, or anatomical hallucinations.
   - *Rigid Objects:* Look for warped lines, perspective distortion, or "fusing" into the background.
   - *Scenes:* Look for repetitive patterns (mode collapse) or illogical spatial bleeding.
4. **Expert Selection:** Map the identified risks to specific `expert_name` values from the Registry. You MUST use the exact `expert_id` values from the Expert Registry as the `expert_name` field. Available expert_ids: animal_pose_auditor, geometric_depth_auditor, topology_boundary_auditor, open_vocabulary_detector, fine_grained_classifier, perceptual_quality_auditor, image_text_auditor.
5. **Weight Allocation:** Assign weights based on "Structural Criticality" — anatomy/structure experts should receive higher weights for organisms, geometric experts for rigid objects, etc. All weights for selected experts must sum to 1.0.

**[Output Requirements]**
Return a pure JSON object (no Markdown wrapping) with this exact schema:
{{
  "image_class": "The ImageNet class label string",
  "selected_experts": [
    {{
      "expert_name": "string (must exactly match an expert_id from the Expert Registry)",
      "reason": "Why this expert is selected for this specific class and image",
      "weight": 0.0
    }}
  ],
  "focus_areas": ["string (e.g., feet, facial_details, background, limb_integrity, textural_coherence)"],
  "custom_prompts_for_reflector": "string (special audit hints for the Reflector in later steps, e.g., 'Pay extra attention to whether the tail fuses with the background')"
}}

**[Constraints]**
- You MUST include "fine_grained_classifier" as one of the selected experts for identity verification.
- You MUST include "open_vocabulary_detector" if the class requires locating specific body parts or components.
- All weights must be positive and sum to 1.0.
- Select 3-5 experts appropriate for the category.
- Output ONLY the JSON object, no additional text."""


def call_router(
    client: OpenAI,
    base64_image: str,
    prompt: str,
) -> dict | None:
    try:
        completion = client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a highly logical Router Agent for image auditing. "
                        "You must prioritize the provided Taxonomy Knowledge as the source of truth. "
                        "Output JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"  [ERROR] API call failed: {e}")
        return None


def validate_plan(plan: dict) -> bool:
    if "image_class" not in plan:
        print("  [WARN] Plan missing 'image_class'")
        return False
    if "selected_experts" not in plan or not isinstance(plan["selected_experts"], list):
        print("  [WARN] Plan missing or invalid 'selected_experts'")
        return False
    if "focus_areas" not in plan or not isinstance(plan["focus_areas"], list):
        print("  [WARN] Plan missing or invalid 'focus_areas'")
        return False
    if "custom_prompts_for_reflector" not in plan:
        print("  [WARN] Plan missing 'custom_prompts_for_reflector'")
        return False

    valid_expert_ids = {
        "animal_pose_auditor",
        "geometric_depth_auditor",
        "topology_boundary_auditor",
        "open_vocabulary_detector",
        "fine_grained_classifier",
        "perceptual_quality_auditor",
        "image_text_auditor",
    }
    total_weight = 0.0
    for expert in plan["selected_experts"]:
        if "expert_name" not in expert:
            print(f"  [WARN] Expert entry missing 'expert_name': {expert}")
            return False
        if expert["expert_name"] not in valid_expert_ids:
            print(
                f"  [WARN] Invalid expert_name '{expert['expert_name']}', "
                f"must be one of: {valid_expert_ids}"
            )
            return False
        if "weight" not in expert:
            print(f"  [WARN] Expert '{expert['expert_name']}' missing 'weight'")
            return False
        total_weight += expert["weight"]

    if abs(total_weight - 1.0) > 0.05:
        print(
            f"  [WARN] Expert weights sum to {total_weight:.2f}, expected ~1.0"
        )
    return True


def run_router_for_image(
    client: OpenAI,
    img_name: str,
    class_id: int,
    class_label: str,
    experts_registry_str: str,
    image_dir: Path,
) -> dict | None:
    img_path = image_dir / img_name
    if not img_path.exists():
        print(f"  [ERROR] Image not found: {img_path}")
        return None

    taxonomy_info = get_taxonomy_info(class_id)
    if taxonomy_info is None:
        print(f"  [WARN] No taxonomy info for class_id={class_id}, proceeding without prior knowledge.")

    base64_image = encode_image(str(img_path))
    prompt = build_router_prompt(class_label, taxonomy_info, experts_registry_str)

    start_time = time.time()
    plan = call_router(client, base64_image, prompt)
    cost_time = time.time() - start_time

    if plan is None:
        return None

    plan["metadata"] = {
        "original_image": str(img_path),
        "image_filename": img_name,
        "class_id": class_id,
        "class_label": class_label,
        "router_cost_seconds": round(cost_time, 2),
    }

    is_valid = validate_plan(plan)
    plan["metadata"]["plan_valid"] = is_valid

    return plan


def process_images(
    client: OpenAI,
    images: list[tuple[str, str, int, str]],
    experts_registry_str: str,
    image_dir: Path,
    output_plan_dir: Path,
    start_idx: int = 1,
) -> tuple[int, list[dict], list[tuple[str, str, int, str]]]:
    success_count = 0
    time_records = []
    failed_images = []

    for idx, (img_name, img_id, class_id, class_label) in enumerate(
        images, start=start_idx
    ):
        print(
            f"[{idx}/{start_idx - 1 + len(images)}] Processing: {img_name} "
            f"(class_id={class_id}, label={class_label})"
        )

        plan = run_router_for_image(
            client, img_name, class_id, class_label, experts_registry_str, image_dir
        )

        if plan is None:
            print(f"  -> FAILED\n")
            failed_images.append((img_name, img_id, class_id, class_label))
            continue

        save_path = output_plan_dir / f"plan_{img_id}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=4, ensure_ascii=False)

        cost = plan["metadata"]["router_cost_seconds"]
        time_records.append({"img_name": img_name, "cost": cost})
        success_count += 1

        valid_tag = "VALID" if plan["metadata"]["plan_valid"] else "INVALID"
        print(
            f"  -> Saved to {save_path.name} | "
            f"Cost: {cost:.2f}s | Plan: {valid_tag}\n"
        )

    return success_count, time_records, failed_images


def print_summary(
    success_count: int,
    total_count: int,
    time_records: list[dict],
    failed_images: list[tuple[str, str, int, str]],
) -> None:
    print(f"\n{'='*60}")
    print(f"Router Performance Summary")
    print(f"{'='*60}")
    if time_records:
        total = sum(r["cost"] for r in time_records)
        avg = total / len(time_records)
        print(f"  Successfully planned : {success_count}/{total_count}")
        print(f"  Total elapsed time   : {total:.2f}s")
        print(f"  Average plan time    : {avg:.2f}s per image")
    else:
        print(f"  Successfully planned : 0/{total_count}")
    if failed_images:
        print(f"\n  Failed images ({len(failed_images)}):")
        for img_name, _, class_id, class_label in failed_images:
            print(f"    - {img_name} (class_id={class_id}, label={class_label})")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="THEMIS C2I Step 1: Router - Generate evaluation plans for test images"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of images to process (0 = all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_PLAN_DIR),
        help="Output directory for plan JSON files",
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
        "--max-retries",
        type=int,
        default=3,
        help="Max retry rounds for failed images (default: 3)",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    class_ids_txt = Path(args.class_ids)
    output_plan_dir = Path(args.output)
    output_plan_dir.mkdir(parents=True, exist_ok=True)

    if not DASHSCOPE_API_KEY:
        print("[ERROR] DASHSCOPE_API_KEY not set. Please run:")
        print("  export DASHSCOPE_API_KEY=your_api_key_here  (Linux/Mac)")
        print("  set DASHSCOPE_API_KEY=your_api_key_here      (Windows CMD)")
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

    if args.limit > 0:
        valid_images = valid_images[: args.limit]

    print(f"\n{'='*60}")
    print(f"THEMIS C2I Router - Step 1")
    print(f"Images to process: {len(valid_images)}")
    print(f"Output directory:  {output_plan_dir}")
    print(f"{'='*60}\n")

    all_time_records = []
    total_success = 0
    total_count = len(valid_images)

    success_count, time_records, failed_images = process_images(
        client, valid_images, experts_registry_str, image_dir, output_plan_dir
    )
    all_time_records.extend(time_records)
    total_success += success_count

    retry_round = 0
    while failed_images and retry_round < args.max_retries:
        retry_round += 1
        print(f"\n{'='*60}")
        print(f"Retry Round {retry_round}/{args.max_retries}")
        print(f"Failed images: {len(failed_images)}")
        for img_name, _, class_id, class_label in failed_images:
            print(f"  - {img_name} (class_id={class_id}, label={class_label})")
        print(f"{'='*60}")

        answer = input(f"\nRetry these {len(failed_images)} failed images? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("Retry skipped by user.")
            break

        print()
        success_count, time_records, failed_images = process_images(
            client, failed_images, experts_registry_str, image_dir, output_plan_dir
        )
        all_time_records.extend(time_records)
        total_success += success_count

    print_summary(total_success, total_count, all_time_records, failed_images)


if __name__ == "__main__":
    main()

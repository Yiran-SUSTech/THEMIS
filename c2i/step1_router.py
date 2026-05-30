import os
import sys
import re
import json
import base64
import time
from pathlib import Path
from openai import OpenAI

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TAXONOMY_DIR = PROJECT_ROOT / "taxonomy_info"
EXPERTS_REGISTRY_JSON = PROJECT_ROOT / "expert_registry.json"

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ROUTER_MODEL = "qwen3.6-plus"


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_experts_registry(file_path: str = str(EXPERTS_REGISTRY_JSON)) -> str:
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


def extract_expert_ids(experts_registry_str: str) -> list[str]:
    try:
        registry = json.loads(experts_registry_str)
        return [item["expert_id"] for item in registry if "expert_id" in item]
    except (json.JSONDecodeError, TypeError):
        return []


def clean_json_response(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    elif text.startswith("```"):
        text = text[len("```"):]
    if text.endswith("```"):
        text = text[:-len("```")]
    return text.strip()


def parse_json_safely(raw_text: str) -> dict | None:
    cleaned = clean_json_response(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
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
        taxonomy_desc = taxonomy_info.get("enriched_description", taxonomy_desc)

    expert_ids = extract_expert_ids(experts_registry_str)
    expert_ids_str = ", ".join(expert_ids) if expert_ids else "See Expert Registry above"

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
4. **Expert Selection:** Map the identified risks to specific `expert_name` values from the Registry. You MUST use the exact `expert_id` values from the Expert Registry as the `expert_name` field. Available expert_ids: {expert_ids_str}.
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


def build_router_revision_prompt(
    class_label: str,
    taxonomy_info: dict | None,
    experts_registry_str: str,
    previous_plan: dict,
    feedback_history: list[dict],
) -> str:
    taxonomy_desc = "No specific taxonomy prior knowledge found for this class."
    taxonomy_class_name = class_label
    if taxonomy_info:
        taxonomy_class_name = taxonomy_info.get("class_name", class_label)
        taxonomy_desc = taxonomy_info.get("enriched_description", taxonomy_desc)

    expert_ids = extract_expert_ids(experts_registry_str)
    expert_ids_str = ", ".join(expert_ids) if expert_ids else "See Expert Registry above"

    feedback_text = ""
    for i, fb in enumerate(feedback_history, 1):
        feedback_text += f"\n--- Feedback Round {i} ---\n"
        feedback_text += f"Reasons for Rejection: {fb.get('reasons_for_rejection', 'N/A')}\n"
        feedback_text += f"Suggestions: {json.dumps(fb.get('suggestions', []), ensure_ascii=False)}\n"

    return f"""You are the Lead Strategic Planner (Router) for an advanced AI image evaluation system.
Your previous plan was REJECTED by the Judge. You must revise it based on the feedback below.

**[Input Data]**
- **Class Label:** {class_label}
- **Taxonomy Class Name:** {taxonomy_class_name}
- **Taxonomy Prior Knowledge (Ground Truth):** {taxonomy_desc}
- **Expert Registry (Available Tools):** {experts_registry_str}

**[Your Previous Plan (REJECTED)]**
{json.dumps(previous_plan, indent=2, ensure_ascii=False)}

**[Judge Feedback History]**
{feedback_text}

**[Strategic Instruction]**
1. Carefully read the Judge's feedback and understand what was wrong with your previous plan.
2. Revise the plan to address ALL issues raised by the Judge.
3. **Identify Category Archetype:** Determine if the class "{class_label}" is an **Organism** (animal/plant), a **Rigid Object** (architecture/tool/vehicle), or a **Natural Scene** (landscape/texture).
4. **Feature Mapping:** Based on the Taxonomy Prior Knowledge, extract 2-10 "Non-negotiable" diagnostic features that must be verified in the image.
5. **Visual Risk Assessment:** Scrutinize the image for category-specific flaws:
   - *Organisms:* Look for "Melting" limbs, missing parts, or anatomical hallucinations.
   - *Rigid Objects:* Look for warped lines, perspective distortion, or "fusing" into the background.
   - *Scenes:* Look for repetitive patterns (mode collapse) or illogical spatial bleeding.
6. **Expert Selection:** Map the identified risks to specific `expert_name` values from the Registry. You MUST use the exact `expert_id` values from the Expert Registry as the `expert_name` field. Available expert_ids: {expert_ids_str}.
7. **Weight Allocation:** Assign weights based on "Structural Criticality" — anatomy/structure experts should receive higher weights for organisms, geometric experts for rigid objects, etc. All weights for selected experts must sum to 1.0.

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
  "custom_prompts_for_reflector": "string (special audit hints for the Reflector in later steps)"
}}

**[Constraints]**
- You MUST include "fine_grained_classifier" as one of the selected experts for identity verification.
- You MUST include "open_vocabulary_detector" if the class requires locating specific body parts or components.
- All weights must be positive and sum to 1.0.
- Select 3-5 experts appropriate for the category.
- Output ONLY the JSON object, no additional text."""


def validate_plan(plan: dict, experts_registry_str: str = "") -> bool:
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

    valid_expert_ids = set(extract_expert_ids(experts_registry_str)) if experts_registry_str else {
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


def _call_router_api(
    client: OpenAI,
    base64_image: str,
    prompt: str,
    system_msg: str,
) -> dict | None:
    try:
        completion = client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
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
        raw_content = completion.choices[0].message.content
        result = parse_json_safely(raw_content)
        if result is None:
            print(f"  [ERROR] Router returned unparseable JSON: {raw_content[:200]}")
        return result
    except Exception as e:
        print(f"  [ERROR] Router API call failed: {e}")
        return None


def generate_plan(
    client: OpenAI,
    image_path: str,
    class_id: int,
    class_label: str,
    experts_registry_str: str,
) -> dict | None:
    taxonomy_info = get_taxonomy_info(class_id)
    if taxonomy_info is None:
        print(f"  [WARN] No taxonomy info for class_id={class_id}, proceeding without prior knowledge.")

    base64_image = encode_image(image_path)
    prompt = build_router_prompt(class_label, taxonomy_info, experts_registry_str)
    system_msg = (
        "You are a highly logical Router Agent for image auditing. "
        "You must prioritize the provided Taxonomy Knowledge as the source of truth. "
        "Output JSON only."
    )

    start_time = time.time()
    plan = _call_router_api(client, base64_image, prompt, system_msg)
    cost_time = time.time() - start_time

    if plan is None:
        return None

    plan["metadata"] = {
        "original_image": image_path,
        "class_id": class_id,
        "class_label": class_label,
        "router_cost_seconds": round(cost_time, 2),
        "plan_valid": validate_plan(plan, experts_registry_str),
    }

    return plan


def revise_plan(
    client: OpenAI,
    image_path: str,
    class_id: int,
    class_label: str,
    experts_registry_str: str,
    previous_plan: dict,
    feedback_history: list[dict],
) -> dict | None:
    taxonomy_info = get_taxonomy_info(class_id)
    if taxonomy_info is None:
        print(f"  [WARN] No taxonomy info for class_id={class_id}, proceeding without prior knowledge.")

    base64_image = encode_image(image_path)
    prompt = build_router_revision_prompt(
        class_label, taxonomy_info, experts_registry_str,
        previous_plan, feedback_history,
    )
    system_msg = (
        "You are a highly logical Router Agent for image auditing. "
        "Your previous plan was rejected and you must revise it based on Judge feedback. "
        "You must prioritize the provided Taxonomy Knowledge as the source of truth. "
        "Output JSON only, and strictly follow the format as in the previous plan."
    )

    start_time = time.time()
    plan = _call_router_api(client, base64_image, prompt, system_msg)
    cost_time = time.time() - start_time

    if plan is None:
        return None

    plan["metadata"] = {
        "original_image": image_path,
        "class_id": class_id,
        "class_label": class_label,
        "router_cost_seconds": round(cost_time, 2),
        "plan_valid": validate_plan(plan, experts_registry_str),
        "is_revision": True,
        "revision_feedback_count": len(feedback_history),
    }

    return plan

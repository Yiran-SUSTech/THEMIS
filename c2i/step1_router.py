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


def build_router_registry_summary(experts_registry_str: str) -> str:
    try:
        registry = json.loads(experts_registry_str)
    except (json.JSONDecodeError, TypeError):
        return experts_registry_str

    summary = []
    for e in registry:
        entry = {
            "expert_id": e.get("expert_id"),
            "best_for": e.get("best_for"),
            "applicable_scenes": e.get("applicable_scenes"),
        }
        if e.get("topology_map"):
            entry["topology_map"] = e["topology_map"]
            entry["morphology_note"] = "ONLY for limbed subjects (humans, dogs, cats). NOT for limbless (fish, snakes)."
        summary.append(entry)

    return json.dumps(summary, indent=2, ensure_ascii=False)


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


_COMMON_ROUTER_INSTRUCTIONS = """**[Strategic Instruction]**
You MUST follow these steps IN ORDER. Your output JSON MUST include the `image_description` field as the result of Step 0.

**Step 0 — Image Content Inventory (MANDATORY)**
Before selecting any expert, you MUST first describe what you see in the image. This description drives all subsequent decisions.
- List ALL visible entities: the class subject, any people, animals, objects, text, and notable background elements.
- For each entity, note its role (main subject, secondary subject, background element) and whether it could exhibit AI-generation artifacts.
- This step ensures you do NOT miss any detectable entity (e.g., a person in the background of a fish image).

**Step 1 — Subject Inventory**
Based on Step 0, identify ALL visible subjects (class subject + auxiliary subjects like people, background objects) that could exhibit AI-generation artifacts.

**Step 2 — Feature Mapping**
Based on Taxonomy AND image content from Step 0, extract 2-10 "Non-negotiable" diagnostic features.

**Step 3 — Visual Risk Assessment**
Scrutinize for category-specific flaws:
- *Organisms:* Melting limbs, missing parts, anatomical hallucinations.
- *Rigid Objects:* Warped lines, perspective distortion, fusing into background.
- *Scenes:* Repetitive patterns (mode collapse), illogical spatial bleeding.

**Step 4 — Expert Selection Rules**
Map risks to expert_ids ({expert_ids_str}). Key rules:
- Each expert's `target_subject` MUST be morphologically compatible with that expert's capabilities (check applicable_scenes/best_for/topology_map in Registry).
- MUST include "fine_grained_classifier" for class subject identity verification.
- MUST include "open_vocabulary_detector" if the class requires locating specific body parts or components.
- **CRITICAL — Detect ALL limbed entities:** If Step 0 identified ANY person, animal, or limbed creature in the image (whether main subject, secondary subject, or background element), you MUST include "animal_pose_auditor" with that entity as `target_subject`. For example, if the image shows a fish being held by a person, include animal_pose_auditor targeting "person" in addition to fish-targeted experts. Do NOT ignore background people — they are common sources of AI artifacts.
- **CRITICAL — Detect ALL text:** If Step 0 identified ANY text in the image (signs, labels, watermarks), you MUST include "image_text_auditor" with that text region as `target_subject`.
- **CRITICAL — Same expert, different targets:** The same expert_id MAY appear multiple times in `selected_experts` with DIFFERENT `target_subject` values. For example, if the image contains both a person and a monkey, you should include TWO entries of "animal_pose_auditor" — one targeting "person" and one targeting "monkey". Similarly, if the image contains multiple distinct objects that each need boundary checking, include multiple "topology_boundary_auditor" entries with different targets.
- MUST specify a `target_subject` for every selected expert.
- Select 3-8 expert entries appropriate for the category and image content. You may select up to 8 expert entries (counting duplicates) to cover all detectable entities.

**Step 5 — Weight Allocation**
Assign weights by Structural Criticality — class subject's experts get higher weights; auxiliary subjects' experts get lower weights. All weights must be positive and sum to 1.0.

**[Output Requirements]**
Return a pure JSON object (no Markdown wrapping) with this exact schema:
{{
  "image_description": "A brief description of ALL visible entities in the image, their roles, and potential artifact risks. This is the output of Step 0.",
  "image_class": "The ImageNet class label string",
  "selected_experts": [
    {{
      "expert_name": "string (must exactly match an expert_id from the Expert Registry)",
      "target_subject": "string (which subject in the image this expert should be applied to)",
      "reason": "Why this expert is selected for this specific target_subject",
      "weight": 0.0
    }}
  ],
  "focus_areas": ["string (e.g., feet, facial_details, background, limb_integrity)"],
  "custom_prompts_for_reflector": "string (special audit hints for the Reflector)"
}}

Output ONLY the JSON object, no additional text."""


def _build_context_block(
    class_label: str,
    taxonomy_info: dict | None,
    experts_registry_str: str,
) -> tuple[str, str, str]:
    taxonomy_desc = "No specific taxonomy prior knowledge found for this class."
    taxonomy_class_name = class_label
    if taxonomy_info:
        taxonomy_class_name = taxonomy_info.get("class_name", class_label)
        taxonomy_desc = taxonomy_info.get("enriched_description", taxonomy_desc)

    expert_ids = extract_expert_ids(experts_registry_str)
    expert_ids_str = ", ".join(expert_ids) if expert_ids else "See Expert Registry above"
    registry_summary = build_router_registry_summary(experts_registry_str)

    variable_context = (
        f"- **Class Label:** {class_label}\n"
        f"- **Taxonomy Class Name:** {taxonomy_class_name}\n"
        f"- **Taxonomy Prior Knowledge (Ground Truth):** {taxonomy_desc}"
    )
    return variable_context, expert_ids_str, registry_summary


def build_router_prompt(
    class_label: str,
    taxonomy_info: dict | None,
    experts_registry_str: str,
) -> str:
    variable_context, _, _ = _build_context_block(class_label, taxonomy_info, experts_registry_str)

    return f"""Analyze the provided image AND its class category, then formulate a rigorous evaluation plan using the Expert Registry and Strategic Instructions provided in the system context.

**[Input Data]**
{variable_context}"""


def _build_revision_user_prompt_session(
    class_label: str,
    taxonomy_info: dict | None,
    experts_registry_str: str,
    previous_plan: dict,
    feedback_history: list[dict],
) -> str:
    variable_context, _, _ = _build_context_block(class_label, taxonomy_info, experts_registry_str)

    latest_feedback = feedback_history[-1] if feedback_history else {}
    reasons = latest_feedback.get('reasons_for_rejection', 'N/A')
    suggestions = latest_feedback.get('suggestions', [])

    return f"""[Router Role] Your previous plan was REJECTED by the Judge. Revise it based on the feedback.

**[Input Data]**
{variable_context}

**[Judge's Latest Rejection Reasons]**
{reasons}

**[Judge's Suggestions]**
{json.dumps(suggestions, ensure_ascii=False)}

**[Revision Directive]**
Address ALL issues raised by the Judge. Output the revised plan as JSON."""


def build_router_revision_prompt(
    class_label: str,
    taxonomy_info: dict | None,
    experts_registry_str: str,
    previous_plan: dict,
    feedback_history: list[dict],
) -> str:
    variable_context, _, _ = _build_context_block(class_label, taxonomy_info, experts_registry_str)

    feedback_text = ""
    for i, fb in enumerate(feedback_history, 1):
        feedback_text += f"\n--- Feedback Round {i} ---\n"
        feedback_text += f"Reasons for Rejection: {fb.get('reasons_for_rejection', 'N/A')}\n"
        feedback_text += f"Suggestions: {json.dumps(fb.get('suggestions', []), ensure_ascii=False)}\n"

    return f"""Your previous plan was REJECTED by the Judge. Revise it based on the feedback below, following the Expert Registry and Strategic Instructions provided in the system context.

**[Input Data]**
{variable_context}

**[Your Previous Plan (REJECTED)]**
{json.dumps(previous_plan, indent=2, ensure_ascii=False)}

**[Judge Feedback History]**
{feedback_text}

**[Revision Directive]**
Address ALL issues raised by the Judge."""


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
    if "image_description" not in plan:
        print("  [WARN] Plan missing 'image_description' (required since Step 0)")
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
        if "target_subject" not in expert or not expert["target_subject"].strip():
            print(f"  [WARN] Expert '{expert['expert_name']}' missing 'target_subject'")
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


def _build_cached_system_message(
    system_msg: str,
    registry_summary: str = "",
    formatted_instructions: str = "",
) -> dict:
    parts = [system_msg]
    if formatted_instructions:
        parts.append(formatted_instructions)
    if registry_summary:
        parts.append(f"**[Expert Registry (Available Tools)]**\n{registry_summary}")
    combined_text = "\n\n".join(parts)
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": combined_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }


def _call_router_api(
    client: OpenAI,
    base64_image: str,
    prompt: str,
    system_msg: str,
    registry_summary: str = "",
    formatted_instructions: str = "",
) -> dict | None:
    if registry_summary:
        system_message = _build_cached_system_message(
            system_msg, registry_summary, formatted_instructions
        )
    else:
        system_message = {"role": "system", "content": system_msg}

    try:
        completion = client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=[
                system_message,
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
            temperature=0,
        )
        raw_content = completion.choices[0].message.content
        result = parse_json_safely(raw_content)
        if result is None:
            print(f"  [ERROR] Router returned unparseable JSON: {raw_content[:200]}")

        usage = getattr(completion, "usage", None)
        if usage:
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details else 0
            created = getattr(details, "cache_creation_input_tokens", 0) if details else 0
            if cached or created:
                print(f"  [CACHE] Router: hit={cached} tokens, created={created} tokens")

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
    session=None,
) -> dict | None:
    taxonomy_info = get_taxonomy_info(class_id)
    if taxonomy_info is None:
        print(f"  [WARN] No taxonomy info for class_id={class_id}, proceeding without prior knowledge.")

    base64_image = encode_image(image_path)
    prompt = build_router_prompt(class_label, taxonomy_info, experts_registry_str)

    start_time = time.time()

    if session is not None:
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
        ]
        session.add_user(user_content)
        raw_content, completion = session.call_api(
            client, ROUTER_MODEL, response_format={"type": "json_object"},
        )
        plan = parse_json_safely(raw_content)
        if plan is None:
            print(f"  [ERROR] Router returned unparseable JSON: {raw_content[:200]}")
    else:
        _, expert_ids_str, registry_summary = _build_context_block(class_label, taxonomy_info, experts_registry_str)
        formatted_instructions = _COMMON_ROUTER_INSTRUCTIONS.format(expert_ids_str=expert_ids_str)
        system_msg = (
            "You are a highly logical Router Agent for image auditing. "
            "You must prioritize the provided Taxonomy Knowledge as the source of truth. "
            "Output JSON only."
        )
        plan = _call_router_api(
            client, base64_image, prompt, system_msg,
            registry_summary=registry_summary,
            formatted_instructions=formatted_instructions,
        )

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
    session=None,
) -> dict | None:
    taxonomy_info = get_taxonomy_info(class_id)
    if taxonomy_info is None:
        print(f"  [WARN] No taxonomy info for class_id={class_id}, proceeding without prior knowledge.")

    base64_image = encode_image(image_path)

    start_time = time.time()

    if session is not None:
        prompt = _build_revision_user_prompt_session(
            class_label, taxonomy_info, experts_registry_str,
            previous_plan, feedback_history,
        )
        user_content = [
            {"type": "text", "text": prompt},
        ]
        session.add_user(user_content)
        raw_content, completion = session.call_api(
            client, ROUTER_MODEL, response_format={"type": "json_object"},
        )
        plan = parse_json_safely(raw_content)
        if plan is None:
            print(f"  [ERROR] Router returned unparseable JSON: {raw_content[:200]}")
    else:
        prompt = build_router_revision_prompt(
            class_label, taxonomy_info, experts_registry_str,
            previous_plan, feedback_history,
        )
        _, expert_ids_str, registry_summary = _build_context_block(class_label, taxonomy_info, experts_registry_str)
        formatted_instructions = _COMMON_ROUTER_INSTRUCTIONS.format(expert_ids_str=expert_ids_str)
        system_msg = (
            "You are a highly logical Router Agent for image auditing. "
            "You must prioritize the provided Taxonomy Knowledge as the source of truth. "
            "Output JSON only."
        )
        plan = _call_router_api(
            client, base64_image, prompt, system_msg,
            registry_summary=registry_summary,
            formatted_instructions=formatted_instructions,
        )

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

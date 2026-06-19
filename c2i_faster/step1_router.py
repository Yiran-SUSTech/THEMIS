import os
import sys
import re
import json
import base64
import time
from pathlib import Path
from openai import OpenAI
from common import api_call_with_retry

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TAXONOMY_DIR = PROJECT_ROOT / "taxonomy_info"
TAXONOMY_STRUCTURAL_DIR = PROJECT_ROOT / "taxonomy_info_structural"
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
            "description": e.get("description"),
            "best_for": e.get("best_for"),
        }
        if e.get("expert_id") == "animal_pose_auditor":
            entry["note"] = "ONLY for limbed subjects (humans, dogs, cats). NOT for limbless (fish, snakes)."
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


def get_structured_taxonomy_info(class_id: int) -> dict | None:
    """Read structured taxonomy info (diagnostic_checkpoints) from taxonomy_info_structural/."""
    batch_num = class_id // 10
    batch_file = TAXONOMY_STRUCTURAL_DIR / f"taxonomy_enriched_Batch_{batch_num}_structured.json"
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
    if raw_text is None or raw_text.strip() == "":
        return None
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


_COMMON_ROUTER_INSTRUCTIONS = """You are a Router Agent for AI-generated image evaluation. Follow these steps in order and output a single JSON object.

**Step 1 — Checkpoint Verification (STRICT)**
You are given `diagnostic_checkpoints` organized by body-part categories. For EACH checkpoint:
- Is it testable? Only mark `is_testable: false` if the feature is genuinely impossible to see (completely occluded or outside frame). When in doubt, mark as testable and give your best judgment.
- If testable, does the image match the checkpoint description? Be critical — even subtle deviations (wrong color shade, slightly wrong proportion, partial but incomplete match) should be marked `is_present: false`. A checkpoint is present only if the feature clearly and fully matches.
- Brief reasoning for both decisions.

**Step 2 — Artifact Detection (THOROUGH)**
Scan the ENTIRE image carefully for AI-generation artifacts, including subtle ones. For each artifact found:
- Type: melting, fusion, extra_limbs, missing_parts, structural_collapse, blur, texture_anomaly, perspective_distortion, text_gibberish, other.
- Location and severity (0-5 scale: 0=none, 1=barely noticeable on close inspection, 2=noticeable but minor, 3=moderately severe, 4=severe structural failure, 5=catastrophic nonsensical region).
- Brief reasoning.
- Pay special attention to: subtle edge bleeding between subject and background, slight texture inconsistencies in skin/fur/feathers, minor perspective warping, faint ghost limbs, small areas of melting or fusion that are easy to overlook.
- If no artifacts found after thorough inspection, output empty list.

**Step 3 — Expert Selection**
Select experts from ({expert_ids_str}) based on visible entities and artifact risks. Rules:
- MUST include "fine_grained_classifier" for the class subject.
- Include "animal_pose_auditor" ONLY for limbed subjects (people, dogs, cats, etc.), NOT for limbless ones (fish, snakes).
- Include "image_text_auditor" ONLY if text is visible.
- Same expert_id may appear multiple times with different `target_subject`.
- Select 3-8 experts. Specify `target_subject` for each.

**Step 4 — Weights**
Class subject's experts get higher weights; auxiliary subjects' get lower. All weights positive, sum to 1.0.

**Output JSON schema:**
{{
  "image_description": "Brief description of all visible entities and their roles",
  "image_class": "ImageNet class label",
  "checkpoint_verdicts": [{{"checkpoint": "str", "category": "str", "is_testable": bool, "is_present": bool, "reasoning": "str"}}],
  "artifact_observations": [{{"artifact_type": "str", "location": "str", "severity": float, "reasoning": "str"}}],
  "selected_experts": [{{"expert_name": "str", "target_subject": "str", "reason": "str", "weight": float}}],
  "focus_areas": ["str"],
  "custom_prompts_for_reflector": "str"
}}"""


def _build_context_block(
    class_label: str,
    taxonomy_info: dict | None,
    experts_registry_str: str,
    structured_taxonomy_info: dict | None = None,
) -> tuple[str, str, str]:
    taxonomy_desc = "No specific taxonomy prior knowledge found for this class."
    taxonomy_class_name = class_label
    if taxonomy_info:
        taxonomy_class_name = taxonomy_info.get("class_name", class_label)
        taxonomy_desc = taxonomy_info.get("enriched_description", taxonomy_desc)

    expert_ids = extract_expert_ids(experts_registry_str)
    expert_ids_str = ", ".join(expert_ids) if expert_ids else "See Expert Registry above"
    registry_summary = build_router_registry_summary(experts_registry_str)

    # Build diagnostic checkpoints context
    checkpoints_text = "No structured diagnostic checkpoints available for this class."
    if structured_taxonomy_info:
        checkpoints = structured_taxonomy_info.get("diagnostic_checkpoints", {})
        if checkpoints:
            checkpoints_text = json.dumps(checkpoints, indent=2, ensure_ascii=False)

    variable_context = (
        f"- **Class Label:** {class_label}\n"
        f"- **Taxonomy Class Name:** {taxonomy_class_name}\n"
        f"- **Taxonomy Prior Knowledge (Ground Truth):** {taxonomy_desc}\n"
        f"- **Diagnostic Checkpoints (for Step 1 verification):**\n{checkpoints_text}"
    )
    return variable_context, expert_ids_str, registry_summary


def build_router_prompt(
    class_label: str,
    taxonomy_info: dict | None,
    experts_registry_str: str,
    structured_taxonomy_info: dict | None = None,
) -> str:
    variable_context, _, _ = _build_context_block(
        class_label, taxonomy_info, experts_registry_str, structured_taxonomy_info,
    )

    return f"""Analyze the provided image AND its class category, then formulate a rigorous evaluation plan using the Expert Registry and Strategic Instructions provided in the system context.

**[Input Data]**
{variable_context}"""


def _build_revision_user_prompt_session(
    class_label: str,
    taxonomy_info: dict | None,
    experts_registry_str: str,
    previous_plan: dict,
    feedback_history: list[dict],
    structured_taxonomy_info: dict | None = None,
) -> str:
    variable_context, _, _ = _build_context_block(
        class_label, taxonomy_info, experts_registry_str, structured_taxonomy_info,
    )

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
    structured_taxonomy_info: dict | None = None,
) -> str:
    variable_context, _, _ = _build_context_block(
        class_label, taxonomy_info, experts_registry_str, structured_taxonomy_info,
    )

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

    # Validate checkpoint_verdicts
    if "checkpoint_verdicts" not in plan or not isinstance(plan["checkpoint_verdicts"], list):
        print("  [WARN] Plan missing or invalid 'checkpoint_verdicts'")
        return False
    for cv in plan["checkpoint_verdicts"]:
        if "checkpoint" not in cv:
            print(f"  [WARN] checkpoint_verdict missing 'checkpoint': {cv}")
            return False
        if "is_testable" not in cv:
            print(f"  [WARN] checkpoint_verdict missing 'is_testable': {cv}")
            return False
        if "is_present" not in cv:
            print(f"  [WARN] checkpoint_verdict missing 'is_present': {cv}")
            return False

    # Validate artifact_observations
    if "artifact_observations" not in plan or not isinstance(plan["artifact_observations"], list):
        print("  [WARN] Plan missing or invalid 'artifact_observations'")
        return False
    for ao in plan["artifact_observations"]:
        if "artifact_type" not in ao:
            print(f"  [WARN] artifact_observation missing 'artifact_type': {ao}")
            return False
        if "severity" not in ao:
            print(f"  [WARN] artifact_observation missing 'severity': {ao}")
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
    api_retry: int = 0,
) -> dict | None:
    if registry_summary:
        system_message = _build_cached_system_message(
            system_msg, registry_summary, formatted_instructions
        )
    else:
        system_message = {"role": "system", "content": system_msg}

    try:
        completion = api_call_with_retry(
            client.chat.completions.create,
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
            max_retries=api_retry,
            label="Router",
        )
        raw_content = completion.choices[0].message.content
        if raw_content is None or raw_content.strip() == "":
            print(f"  [ERROR] Router returned empty content (content is {'None' if raw_content is None else 'empty string'})")
            return None
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
        print(f"  [ERROR] Router API call failed: {type(e).__name__}: {e}")
        return None


def generate_plan(
    client: OpenAI,
    image_path: str,
    class_id: int,
    class_label: str,
    experts_registry_str: str,
    session=None,
    api_retry: int = 0,
) -> dict | None:
    taxonomy_info = get_taxonomy_info(class_id)
    if taxonomy_info is None:
        print(f"  [WARN] No taxonomy info for class_id={class_id}, proceeding without prior knowledge.")

    structured_taxonomy_info = get_structured_taxonomy_info(class_id)
    if structured_taxonomy_info is None:
        print(f"  [WARN] No structured taxonomy info for class_id={class_id}, proceeding without diagnostic checkpoints.")

    base64_image = encode_image(image_path)
    prompt = build_router_prompt(class_label, taxonomy_info, experts_registry_str, structured_taxonomy_info)

    start_time = time.time()

    if session is not None:
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
        ]
        session.add_user(user_content)
        try:
            raw_content, completion = session.call_api(
                client, ROUTER_MODEL, response_format={"type": "json_object"},
                label="Router",
            )
        except Exception as e:
            print(f"  [ERROR] Router API call failed: {type(e).__name__}: {e}")
            return None
        if raw_content is None or raw_content.strip() == "":
            print(f"  [ERROR] Router returned empty content")
            return None
        plan = parse_json_safely(raw_content)
        if plan is None:
            print(f"  [ERROR] Router returned unparseable JSON: {raw_content[:200]}")
    else:
        _, expert_ids_str, registry_summary = _build_context_block(
            class_label, taxonomy_info, experts_registry_str, structured_taxonomy_info,
        )
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
            api_retry=api_retry,
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
    api_retry: int = 0,
) -> dict | None:
    taxonomy_info = get_taxonomy_info(class_id)
    if taxonomy_info is None:
        print(f"  [WARN] No taxonomy info for class_id={class_id}, proceeding without prior knowledge.")

    structured_taxonomy_info = get_structured_taxonomy_info(class_id)

    base64_image = encode_image(image_path)

    start_time = time.time()

    if session is not None:
        prompt = _build_revision_user_prompt_session(
            class_label, taxonomy_info, experts_registry_str,
            previous_plan, feedback_history, structured_taxonomy_info,
        )
        user_content = [
            {"type": "text", "text": prompt},
        ]
        session.add_user(user_content)
        try:
            raw_content, completion = session.call_api(
                client, ROUTER_MODEL, response_format={"type": "json_object"},
                label="Router",
            )
        except Exception as e:
            print(f"  [ERROR] Router API call failed: {type(e).__name__}: {e}")
            return None
        if raw_content is None or raw_content.strip() == "":
            print(f"  [ERROR] Router returned empty content")
            return None
        plan = parse_json_safely(raw_content)
        if plan is None:
            print(f"  [ERROR] Router returned unparseable JSON: {raw_content[:200]}")
    else:
        prompt = build_router_revision_prompt(
            class_label, taxonomy_info, experts_registry_str,
            previous_plan, feedback_history, structured_taxonomy_info,
        )
        _, expert_ids_str, registry_summary = _build_context_block(
            class_label, taxonomy_info, experts_registry_str, structured_taxonomy_info,
        )
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
            api_retry=api_retry,
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

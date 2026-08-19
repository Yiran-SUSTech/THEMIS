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

**Step 3 — Expert Verification Plan**
Based on your preliminary judgments in Steps 1-2, specify which points can be
verified by expert models. For each verification need:

a) For each Taxonomy checkpoint that you marked is_present=true or is_present=false:
   - Can an expert model provide hard evidence to confirm/deny your judgment?
   - If yes, assign the appropriate expert with specific verification_goals.

b) For checkpoints you marked is_present=false (potential mismatch):
   - Strongly consider assigning "fine_grained_classifier" to verify species identity.
   - Consider "open_vocabulary_detector" if object presence is in question.

c) For artifact observations you found:
   - Assign "perceptual_quality_auditor" to verify distortion/artifact severity.
   - Consider "topology_boundary_auditor" for structural/shape issues.

d) For checkpoints with no expert available:
   - Leave unassigned. Your visual observation is the primary evidence.
   - List them in "unverifiable_points".

Expert Capability Mapping:
  - open_vocabulary_detector: Object detection + bounding boxes
  - fine_grained_classifier: ImageNet species classification (top-3 labels)
  - animal_pose_auditor: Limb/keypoint verification (limbed subjects only)
  - topology_boundary_auditor: Shape/contour verification (segmentation)
  - geometric_depth_auditor: Spatial relationship verification (depth map)
  - perceptual_quality_auditor: Artifact/distortion verification
  - image_text_auditor: Text verification (OCR)

Rules:
- Select 3-8 experts. Same expert_id may appear with different target_subjects.
- Each expert entry MUST specify verification_goals (list of checkpoint descriptions).
- "fine_grained_classifier" is recommended for the class subject.
- "animal_pose_auditor" ONLY for limbed subjects (people, dogs, cats, etc.).
- "image_text_auditor" ONLY if text is visible.

**Step 4 — Weights**
Class subject's experts get higher weights; auxiliary subjects' get lower. All weights positive, sum to 1.0.

**Output JSON schema:**
{{
  "image_description": "Brief description of all visible entities and their roles",
  "image_class": "ImageNet class label",
  "checkpoint_verdicts": [{{"checkpoint": "str", "category": "str", "is_testable": bool, "is_present": bool, "reasoning": "str"}}],
  "artifact_observations": [{{"artifact_type": "str", "location": "str", "severity": float, "reasoning": "str"}}],
  "expert_verification_plan": [{{"expert_name": "str", "target_subject": "str", "verification_goals": ["str"], "reason": "str", "weight": float}}],
  "unverifiable_points": ["str"],
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
    if "expert_verification_plan" not in plan or not isinstance(plan["expert_verification_plan"], list):
        print("  [WARN] Plan missing or invalid 'expert_verification_plan'")
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
    # Validate expert_verification_plan entries (new schema)
    total_weight = 0.0
    for ev in plan["expert_verification_plan"]:
        if "expert_name" not in ev:
            print(f"  [WARN] expert_verification_plan entry missing 'expert_name': {ev}")
            return False
        if ev["expert_name"] not in valid_expert_ids:
            print(
                f"  [WARN] Invalid expert_name '{ev['expert_name']}', "
                f"must be one of: {valid_expert_ids}"
            )
            return False
        if "target_subject" not in ev or not str(ev.get("target_subject", "")).strip():
            print(f"  [WARN] Expert '{ev['expert_name']}' missing 'target_subject'")
            return False
        if "verification_goals" not in ev or not ev["verification_goals"]:
            print(f"  [WARN] Expert '{ev['expert_name']}' has no verification_goals")
            return False
        if "weight" in ev:
            total_weight += ev["weight"]

    # Backward-compat: validate selected_experts if present
    if "selected_experts" in plan and isinstance(plan["selected_experts"], list):
        for expert in plan["selected_experts"]:
            if "expert_name" not in expert:
                print(f"  [WARN] selected_experts entry missing 'expert_name': {expert}")
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
    temperature: float = 0.0,
    model_name: str = "",
) -> dict | None:
    if not model_name:
        model_name = ROUTER_MODEL
    if registry_summary:
        system_message = _build_cached_system_message(
            system_msg, registry_summary, formatted_instructions
        )
    else:
        system_message = {"role": "system", "content": system_msg}

    try:
        completion = api_call_with_retry(
            client.chat.completions.create,
            model=model_name,
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
            temperature=temperature,
            max_retries=api_retry,
            label="Router",
            extra_body={"enable_thinking": False},
        )
        raw_content = completion.choices[0].message.content
        finish_reason = getattr(completion.choices[0], "finish_reason", "unknown")
        usage = getattr(completion, "usage", None)
        if raw_content is None or raw_content.strip() == "":
            reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
            usage_info = f"prompt_tokens={usage.prompt_tokens}, completion_tokens={usage.completion_tokens}" if usage else "no usage info"
            print(f"  [ERROR] Router returned empty content (content is {'None' if raw_content is None else 'empty string'}, finish_reason={finish_reason}, {usage_info})")
            if reasoning:
                print(f"  [WARN] Router reasoning_content found ({len(reasoning)} chars), attempting to extract JSON")
                raw_content = reasoning
            else:
                msg = completion.choices[0].message
                print(f"  [DEBUG] Router full message: content={repr(msg.content)}, role={getattr(msg, 'role', 'N/A')}, function_call={getattr(msg, 'function_call', None)}, tool_calls={getattr(msg, 'tool_calls', None)}, refusal={getattr(msg, 'refusal', None)}")
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
    api_retry: int = 0,
    temperature: float = 0.0,
    model_name: str = "",
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
        temperature=temperature,
        model_name=model_name,
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

    # Backward-compat: generate selected_experts from expert_verification_plan
    if "expert_verification_plan" in plan and "selected_experts" not in plan:
        plan["selected_experts"] = [
            {
                "expert_name": ev.get("expert_name", ""),
                "target_subject": ev.get("target_subject", ""),
                "reason": ev.get("reason", ""),
                "weight": ev.get("weight", 0.0),
            }
            for ev in plan.get("expert_verification_plan", [])
        ]

    return plan


def revise_plan(
    client: OpenAI,
    image_path: str,
    class_id: int,
    class_label: str,
    experts_registry_str: str,
    previous_plan: dict,
    feedback_history: list[dict],
    api_retry: int = 0,
    temperature: float = 0.0,
    model_name: str = "",
) -> dict | None:
    taxonomy_info = get_taxonomy_info(class_id)
    if taxonomy_info is None:
        print(f"  [WARN] No taxonomy info for class_id={class_id}, proceeding without prior knowledge.")

    structured_taxonomy_info = get_structured_taxonomy_info(class_id)

    base64_image = encode_image(image_path)

    start_time = time.time()

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
        temperature=temperature,
        model_name=model_name,
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

    if "expert_verification_plan" in plan and "selected_experts" not in plan:
        plan["selected_experts"] = [
            {
                "expert_name": ev.get("expert_name", ""),
                "target_subject": ev.get("target_subject", ""),
                "reason": ev.get("reason", ""),
                "weight": ev.get("weight", 0.0),
            }
            for ev in plan.get("expert_verification_plan", [])
        ]

    return plan


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Direct Scoring Mode (--without-expert ablation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_ROUTER_DIRECT_SCORE_INSTRUCTIONS = """You are a Router Agent for AI-generated image evaluation. Follow these steps in order and output a single JSON object.

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

**Step 3 — Direct Scoring**
Based on your checkpoint verdicts and artifact observations above, produce final scores directly:
- `alignment_score` (0.0-5.0 continuous): How well does the image match the target class? 5.0 = perfect class conformance (all testable checkpoints present), 0.0 = completely wrong class. Be critical — partial matches should score in the middle range. Multiple checkpoint failures compound.
- `artifact_score` (0.0-5.0 continuous): How artifact-free is the image? 5.0 = no artifacts at all, 0.0 = catastrophic structural failure. Consider both the severity and count of artifacts. Multiple minor artifacts compound.
- Scores should be precise continuous values (e.g., 3.82, 1.47, 4.63), NOT rounded to 0.5 increments.
- A truly excellent image (full class conformance + zero artifacts) should score near 5.0.
- Any notable issue should produce a meaningfully lower score. Multiple minor issues compound.

**Output JSON schema:**
{{
  "image_description": "Brief description of all visible entities and their roles",
  "image_class": "ImageNet class label",
  "checkpoint_verdicts": [{{"checkpoint": "str", "category": "str", "is_testable": bool, "is_present": bool, "reasoning": "str"}}],
  "artifact_observations": [{{"artifact_type": "str", "location": "str", "severity": float, "reasoning": "str"}}],
  "alignment_score": 0.0,
  "artifact_score": 0.0,
  "alignment_reasoning": "Concise: how many checkpoints passed/testable, key mismatches, overall class conformance",
  "artifact_reasoning": "Concise: artifacts found + severities, overall quality assessment"
}}"""


def build_direct_score_prompt(
    class_label: str,
    taxonomy_info: dict | None,
    experts_registry_str: str,
    structured_taxonomy_info: dict | None = None,
) -> str:
    """Build user prompt for direct-scoring mode (without experts)."""
    variable_context, _, _ = _build_context_block(
        class_label, taxonomy_info, experts_registry_str, structured_taxonomy_info,
    )

    return f"""Analyze the provided image AND its class category, then evaluate the image following the Strategic Instructions provided in the system context.

**[Input Data]**
{variable_context}"""


def generate_direct_score(
    client: OpenAI,
    image_path: str,
    class_id: int,
    class_label: str,
    experts_registry_str: str,
    api_retry: int = 0,
    temperature: float = 0.0,
    model_name: str = "",
) -> dict | None:
    """Generate direct alignment and artifact scores without expert models.

    Used in --without-expert ablation mode. The router directly scores the image
    based on checkpoint verdicts and artifact observations, without invoking experts,
    judge, or reflector. The prompt is kept as similar as possible to the normal
    router prompt (Steps 1 and 2 are identical), with Step 3 replaced by direct
    scoring instead of expert selection.
    """
    taxonomy_info = get_taxonomy_info(class_id)
    if taxonomy_info is None:
        print(f"  [WARN] No taxonomy info for class_id={class_id}, proceeding without prior knowledge.")

    structured_taxonomy_info = get_structured_taxonomy_info(class_id)
    if structured_taxonomy_info is None:
        print(f"  [WARN] No structured taxonomy info for class_id={class_id}, proceeding without diagnostic checkpoints.")

    base64_image = encode_image(image_path)
    prompt = build_direct_score_prompt(
        class_label, taxonomy_info, experts_registry_str, structured_taxonomy_info,
    )

    start_time = time.time()

    _, expert_ids_str, registry_summary = _build_context_block(
        class_label, taxonomy_info, experts_registry_str, structured_taxonomy_info,
    )
    formatted_instructions = _ROUTER_DIRECT_SCORE_INSTRUCTIONS
    system_msg = (
        "You are a highly logical Router Agent for image auditing. "
        "You must prioritize the provided Taxonomy Knowledge as the source of truth. "
        "Output JSON only."
    )
    result = _call_router_api(
        client, base64_image, prompt, system_msg,
        registry_summary=registry_summary,
        formatted_instructions=formatted_instructions,
        api_retry=api_retry,
        temperature=temperature,
        model_name=model_name,
    )

    cost_time = time.time() - start_time

    if result is None:
        return None

    # Clamp scores to [0, 5]
    alignment_score = result.get("alignment_score", 0.0)
    artifact_score = result.get("artifact_score", 0.0)
    try:
        alignment_score = max(0.0, min(5.0, float(alignment_score)))
        artifact_score = max(0.0, min(5.0, float(artifact_score)))
    except (TypeError, ValueError):
        alignment_score = 0.0
        artifact_score = 0.0
    result["alignment_score"] = round(alignment_score, 2)
    result["artifact_score"] = round(artifact_score, 2)

    result["metadata"] = {
        "original_image": image_path,
        "class_id": class_id,
        "class_label": class_label,
        "router_cost_seconds": round(cost_time, 2),
        "mode": "without_expert",
    }

    return result


def save_direct_score_report(
    report: dict,
    output_dir,
) -> str:
    """Save a direct-score report (without-expert mode) as a JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = report.get("metadata", {})
    image_id = "unknown"
    original_image = metadata.get("original_image", "")
    if original_image:
        image_id = Path(original_image).stem

    filename = f"direct_score_{image_id}.json"
    filepath = output_dir / filename

    safe_report = _sanitize_report(report)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(safe_report, f, indent=4, ensure_ascii=False)

    return str(filepath)


def _sanitize_report(obj):
    """Recursively sanitize numpy types in a report for JSON serialization."""
    try:
        import numpy as np
    except ImportError:
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_report(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_report(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

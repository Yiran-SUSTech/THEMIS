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
T2I_DIR = Path(__file__).resolve().parent
C2I_DIR = PROJECT_ROOT / "c2i_harness"

# Add c2i_harness at the END of sys.path so it doesn't shadow t2i_harness modules
# (c2i_harness contains common.py and same-named step1/step2 modules).
if str(C2I_DIR) not in sys.path:
    sys.path.append(str(C2I_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(T2I_DIR) not in sys.path:
    sys.path.insert(0, str(T2I_DIR))

from common import api_call_with_retry, dump_debug_raw

# common.py may have inserted c2i_harness at position 0 (e.g. due to drive-letter
# casing differences on Windows), which would shadow t2i_harness modules of the
# same name. Re-assert t2i_harness at the front of sys.path.
_t2i_dir_str = str(T2I_DIR)
if _t2i_dir_str in sys.path:
    sys.path.remove(_t2i_dir_str)
sys.path.insert(0, _t2i_dir_str)

TAXONOMY_DIR = PROJECT_ROOT / "taxonomy_info"
TAXONOMY_STRUCTURAL_DIR = PROJECT_ROOT / "taxonomy_info_structural"
EXPERTS_REGISTRY_JSON = PROJECT_ROOT / "expert_registry.json"

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ROUTER_MODEL = "qwen3.6-plus"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Utility Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Router Instructions (T2I-specific)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_COMMON_ROUTER_INSTRUCTIONS = """You are a Router Agent for AI-generated image evaluation (Text-to-Image mode). The image was generated from a text prompt. Follow these steps in order and output a single JSON object.

**Step 1 — Atom QA Verdicts (STRICT)**
You are given a list of `atoms` — each atom is a question derived from the prompt with an expected answer. For EACH atom:
- Predict the answer (`predicted`) based on what you actually see in the image.
- Mark `is_correct: true` if your predicted answer matches the expected answer, `false` otherwise.
- Provide a `confidence` score (0.0-1.0) reflecting how certain you are about your prediction.
- Brief `reasoning` for your prediction.

**Step 2 — Taxonomy Checkpoint Verification (STRICT)**
For EACH object that has `diagnostic_checkpoints` organized by body-part categories, verify EACH checkpoint:
- Is it testable? Only mark `is_testable: false` if the feature is genuinely impossible to see (completely occluded or outside frame). If a feature is partially visible but blurry, unclear, or distorted — mark it as `is_testable: true, is_present: false`. Do NOT use `is_testable: false` as a way to skip difficult judgments; "blurry" or "hard to tell" means testable-but-absent, NOT untestable.
- If testable, does the image match the checkpoint description? Be critical — even subtle deviations (wrong color shade, slightly wrong proportion, partial but incomplete match) should be marked `is_present: false`. A checkpoint is present only if the feature clearly and fully matches.
- Brief reasoning for both decisions.
- Include the `object` name for each checkpoint verdict.

**Step 3 — Artifact Detection (THOROUGH)**
Scan the ENTIRE image carefully for AI-generation artifacts, including subtle ones. For each artifact found:
- Type: melting, fusion, extra_limbs, missing_parts, structural_collapse, blur, texture_anomaly, perspective_distortion, text_gibberish, other.
- Location and severity (0-5 scale: 0=none, 1=barely noticeable on close inspection, 2=noticeable but minor, 3=moderately severe, 4=severe structural failure, 5=catastrophic nonsensical region).
- Brief reasoning.
- Pay special attention to: subtle edge bleeding between subject and background, slight texture inconsistencies in skin/fur/feathers, minor perspective warping, faint ghost limbs, small areas of melting or fusion that are easy to overlook.
- If no artifacts found after thorough inspection, output empty list.

**Step 4 — Expert Verification Plan**
Based on your preliminary judgments in Steps 1-3, specify which points can be
verified by expert models. For each verification need:

a) For each Taxonomy checkpoint that you marked is_present=true or is_present=false:
   - Can an expert model provide hard evidence to confirm/deny your judgment?
   - If yes, assign the appropriate expert (see capability mapping below).

b) For each count-type atom (e.g., "How many monkeys?"):
   - Assign "open_vocabulary_detector" with the target_subject to verify count.

c) For each object-presence atom (e.g., "Are there monkeys?"):
   - Assign "open_vocabulary_detector" for detection evidence.
   - Assign "fine_grained_classifier" for classification evidence.

d) For each attribute atom (e.g., "Are the monkeys brown?"):
   - If the attribute is a color/material: "fine_grained_classifier" may help
     but your visual observation is primary. Only assign if classification
     labels contain color/material cues.
   - If no expert can verify: leave unassigned (VLM-only judgment).

e) For taxonomy checkpoints with no expert available:
   - Leave unassigned. Your visual observation is the primary evidence.

Expert Capability Mapping:
  - open_vocabulary_detector: Object detection + counting (bounding boxes)
  - fine_grained_classifier: ImageNet species classification (top-3 labels)
  - animal_pose_auditor: Limb/keypoint verification (limbed subjects only)
  - topology_boundary_auditor: Shape/contour verification (segmentation)
  - geometric_depth_auditor: Spatial relationship verification (depth map)
  - perceptual_quality_auditor: Artifact/distortion verification
  - image_text_auditor: Text verification (OCR)

Rules:
- Select 3-8 experts. Same expert_id may appear with different target_subjects.
- Each expert entry MUST specify verification_goals (list of checkpoint/atom IDs).
- "fine_grained_classifier" is recommended for each main object.
- "animal_pose_auditor" ONLY for limbed subjects (people, dogs, cats, etc.).
- "image_text_auditor" ONLY if text is visible.

**Step 5 — Weights**
Main objects' experts get higher weights; auxiliary objects' get lower. All weights positive, sum to 1.0.

**Output JSON schema:**
{{
  "image_description": "Brief description of all visible entities and their roles",
  "atom_verdicts": [{{"atom_index": int, "question": "str", "expected": "str", "predicted": "str", "is_correct": bool, "confidence": float, "reasoning": "str"}}],
  "checkpoint_verdicts": [{{"object": "str", "checkpoint": "str", "category": "str", "is_testable": bool, "is_present": bool, "reasoning": "str"}}],
  "artifact_observations": [{{"artifact_type": "str", "location": "str", "severity": float, "reasoning": "str"}}],
  "expert_verification_plan": [{{"expert_name": "str", "target_subject": "str", "verification_goals": ["str"], "reason": "str", "weight": float}}],
  "unverifiable_points": ["str"],
  "focus_areas": ["str"]
}}"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Context Building (T2I-specific)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_context_block(
    prompt_text: str,
    atoms: list,
    objects_taxonomy: list,
    experts_registry_str: str,
) -> tuple[str, str, str]:
    expert_ids = extract_expert_ids(experts_registry_str)
    expert_ids_str = ", ".join(expert_ids) if expert_ids else "See Expert Registry above"
    registry_summary = build_router_registry_summary(experts_registry_str)

    # Build atoms text
    atoms_text = "No atoms available."
    if atoms:
        atoms_lines = []
        for atom in atoms:
            idx = atom.get("atom_index", "?")
            question = atom.get("question", "?")
            expected = atom.get("expected", "?")
            atoms_lines.append(
                f"  - Atom {idx}: Q: {question} | Expected: {expected}"
            )
        atoms_text = "\n".join(atoms_lines)

    # Build objects taxonomy text
    objects_text = "No objects with taxonomy info available."
    if objects_taxonomy:
        obj_lines = []
        for obj in objects_taxonomy:
            obj_name = obj.get("object", "unknown")
            tax_info = obj.get("taxonomy_info")
            struct_info = obj.get("structured_taxonomy_info")

            tax_desc = "No taxonomy prior knowledge found for this object."
            class_name = obj_name
            if tax_info:
                class_name = tax_info.get("class_name", obj_name)
                tax_desc = tax_info.get("enriched_description", tax_desc)

            checkpoints_text = "No structured diagnostic checkpoints available for this object."
            if struct_info:
                checkpoints = struct_info.get("diagnostic_checkpoints", {})
                if checkpoints:
                    checkpoints_text = json.dumps(checkpoints, indent=2, ensure_ascii=False)

            obj_lines.append(
                f"  ### Object: {obj_name}\n"
                f"  - Taxonomy Class Name: {class_name}\n"
                f"  - Taxonomy Prior Knowledge (Ground Truth): {tax_desc}\n"
                f"  - Diagnostic Checkpoints (for Step 2 verification):\n{checkpoints_text}"
            )
        objects_text = "\n".join(obj_lines)

    variable_context = (
        f"- **Prompt:** {prompt_text}\n"
        f"- **Atoms (QA questions derived from the prompt):**\n{atoms_text}\n"
        f"- **Objects with Taxonomy Info:**\n{objects_text}"
    )
    return variable_context, expert_ids_str, registry_summary


def build_router_prompt(
    prompt_text: str,
    atoms: list,
    objects_taxonomy: list,
    experts_registry_str: str,
) -> str:
    variable_context, _, _ = _build_context_block(
        prompt_text, atoms, objects_taxonomy, experts_registry_str,
    )

    return f"""Analyze the provided image AND the text prompt it was generated from, then formulate a rigorous evaluation plan using the Expert Registry and Strategic Instructions provided in the system context.

**[Input Data]**
{variable_context}"""


def build_router_revision_prompt(
    prompt_text: str,
    atoms: list,
    objects_taxonomy: list,
    experts_registry_str: str,
    previous_plan: dict,
    feedback_history: list[dict],
) -> str:
    variable_context, _, _ = _build_context_block(
        prompt_text, atoms, objects_taxonomy, experts_registry_str,
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Plan Validation (T2I-specific)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_plan(plan: dict, experts_registry_str: str = "", ctx_id: str = "") -> bool:
    tag = f"[Router][{ctx_id}]" if ctx_id else "[Router]"
    if "image_description" not in plan:
        print(f"  {tag} [WARN] Plan missing 'image_description'")
        return False
    if "selected_experts" not in plan or not isinstance(plan["selected_experts"], list):
        # selected_experts may be absent if the model only emitted
        # expert_verification_plan; the backward-compat mapping in
        # generate_plan/revise_plan normally synthesizes it.
        if "expert_verification_plan" not in plan:
            print(f"  {tag} [WARN] Plan missing or invalid 'selected_experts' and 'expert_verification_plan'")
            return False
        print(f"  {tag} [WARN] Plan missing 'selected_experts' (will be synthesized from expert_verification_plan)")
    # NEW (Change C): validate expert_verification_plan
    if "expert_verification_plan" not in plan or not isinstance(plan["expert_verification_plan"], list):
        print(f"  {tag} [WARN] Plan missing or invalid 'expert_verification_plan'")
        return False
    for ev in plan["expert_verification_plan"]:
        if "expert_name" not in ev:
            print(f"  {tag} [WARN] expert_verification_plan entry missing 'expert_name': {ev}")
            return False
        if "verification_goals" not in ev or not ev["verification_goals"]:
            print(f"  {tag} [WARN] Expert '{ev.get('expert_name')}' has no verification_goals")
            return False
        if "target_subject" not in ev:
            print(f"  {tag} [WARN] expert_verification_plan entry missing 'target_subject': {ev}")
            return False
    if "focus_areas" not in plan or not isinstance(plan["focus_areas"], list):
        print(f"  {tag} [WARN] Plan missing or invalid 'focus_areas'")
        return False

    # Validate atom_verdicts
    if "atom_verdicts" not in plan or not isinstance(plan["atom_verdicts"], list):
        print(f"  {tag} [WARN] Plan missing or invalid 'atom_verdicts'")
        return False
    for av in plan["atom_verdicts"]:
        if "atom_index" not in av:
            print(f"  {tag} [WARN] atom_verdict missing 'atom_index': {av}")
            return False
        if "is_correct" not in av:
            print(f"  {tag} [WARN] atom_verdict missing 'is_correct': {av}")
            return False

    # Validate checkpoint_verdicts
    if "checkpoint_verdicts" not in plan or not isinstance(plan["checkpoint_verdicts"], list):
        print(f"  {tag} [WARN] Plan missing or invalid 'checkpoint_verdicts'")
        return False
    for cv in plan["checkpoint_verdicts"]:
        if "checkpoint" not in cv:
            print(f"  {tag} [WARN] checkpoint_verdict missing 'checkpoint': {cv}")
            return False
        if "is_testable" not in cv:
            print(f"  {tag} [WARN] checkpoint_verdict missing 'is_testable': {cv}")
            return False
        if "is_present" not in cv:
            print(f"  {tag} [WARN] checkpoint_verdict missing 'is_present': {cv}")
            return False

    # Validate artifact_observations
    if "artifact_observations" not in plan or not isinstance(plan["artifact_observations"], list):
        print(f"  {tag} [WARN] Plan missing or invalid 'artifact_observations'")
        return False
    for ao in plan["artifact_observations"]:
        if "artifact_type" not in ao:
            print(f"  {tag} [WARN] artifact_observation missing 'artifact_type': {ao}")
            return False
        if "severity" not in ao:
            print(f"  {tag} [WARN] artifact_observation missing 'severity': {ao}")
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
    for expert in plan.get("selected_experts", []):
        if "expert_name" not in expert:
            print(f"  {tag} [WARN] Expert entry missing 'expert_name': {expert}")
            return False
        if expert["expert_name"] not in valid_expert_ids:
            print(
                f"  {tag} [WARN] Invalid expert_name '{expert['expert_name']}', "
                f"must be one of: {valid_expert_ids}"
            )
            return False
        if "target_subject" not in expert or not expert["target_subject"].strip():
            print(f"  {tag} [WARN] Expert '{expert['expert_name']}' missing 'target_subject'")
            return False
        if "weight" not in expert:
            print(f"  {tag} [WARN] Expert '{expert['expert_name']}' missing 'weight'")
            return False
        total_weight += expert["weight"]

    if abs(total_weight - 1.0) > 0.05:
        print(
            f"  {tag} [WARN] Expert weights sum to {total_weight:.2f}, expected ~1.0"
        )
    return True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Cached System Message & API Call
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
    ctx_id: str = "",
) -> dict | None:
    if not model_name:
        model_name = ROUTER_MODEL
    tag = f"[Router][{ctx_id}]" if ctx_id else "[Router]"
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
            print(f"  {tag} [ERROR] Returned empty content (content is {'None' if raw_content is None else 'empty string'}, finish_reason={finish_reason}, {usage_info})")
            if reasoning:
                print(f"  {tag} [WARN] reasoning_content found ({len(reasoning)} chars), attempting to extract JSON")
                raw_content = reasoning
            else:
                msg = completion.choices[0].message
                print(f"  {tag} [DEBUG] Full message: content={repr(msg.content)}, role={getattr(msg, 'role', 'N/A')}, function_call={getattr(msg, 'function_call', None)}, tool_calls={getattr(msg, 'tool_calls', None)}, refusal={getattr(msg, 'refusal', None)}")
                dump_debug_raw("router_empty_content", ctx_id or "unknown",
                               repr(msg.content),
                               note=f"{tag} empty content, finish_reason={finish_reason}")
                return None
        result = parse_json_safely(raw_content)
        if result is None:
            dump_path = dump_debug_raw("router_unparseable", ctx_id or "unknown",
                                       raw_content, note=f"{tag} unparseable JSON response")
            hint = f" (full response saved to: {dump_path})" if dump_path else ""
            print(f"  {tag} [ERROR] Returned unparseable JSON{hint}")
            print(f"  {tag} [ERROR] Head: {raw_content[:200]}")

        usage = getattr(completion, "usage", None)
        if usage:
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details else 0
            created = getattr(details, "cache_creation_input_tokens", 0) if details else 0
            if cached or created:
                print(f"  [CACHE] {tag}: hit={cached} tokens, created={created} tokens")

        return result
    except Exception as e:
        print(f"  {tag} [ERROR] API call failed: {type(e).__name__}: {e}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Plan Generation & Revision (T2I-specific)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _resolve_objects_taxonomy(atomized_data: dict) -> list:
    """Build objects_taxonomy list from atomized_data, looking up taxonomy info as needed.

    atomized_data is the output of step0_atomize and is expected to have:
        {
            "atoms": [{"atom_index": 0, "question": "...", "expected": "..."}, ...],
            "objects": [{"object": "monkey", "class_id": 376}, ...]
        }

    Objects may already carry pre-resolved taxonomy_info / structured_taxonomy_info;
    if not, they are looked up from class_id.
    """
    objects_raw = atomized_data.get("objects", [])
    objects_taxonomy = []
    for obj in objects_raw:
        # step0_atomize uses "object_name"; fall back to "object" for compatibility
        obj_name = obj.get("object_name") or obj.get("object", "unknown")
        class_id = obj.get("class_id")
        tax_info = obj.get("taxonomy_info")
        struct_info = obj.get("structured_taxonomy_info")
        if tax_info is None and class_id is not None:
            tax_info = get_taxonomy_info(class_id)
        if struct_info is None and class_id is not None:
            struct_info = get_structured_taxonomy_info(class_id)
        objects_taxonomy.append({
            "object": obj_name,
            "class_id": class_id,
            "taxonomy_info": tax_info,
            "structured_taxonomy_info": struct_info,
        })
    return objects_taxonomy


def generate_plan(
    client: OpenAI,
    image_path: str,
    prompt_id: str,
    prompt_text: str,
    atomized_data: dict,
    experts_registry_str: str,
    api_retry: int = 0,
    temperature: float = 0.0,
    model_name: str = "",
) -> dict | None:
    atoms = atomized_data.get("atoms", [])
    objects_taxonomy = _resolve_objects_taxonomy(atomized_data)

    base64_image = encode_image(image_path)
    prompt = build_router_prompt(prompt_text, atoms, objects_taxonomy, experts_registry_str)

    start_time = time.time()

    _, expert_ids_str, registry_summary = _build_context_block(
        prompt_text, atoms, objects_taxonomy, experts_registry_str,
    )
    formatted_instructions = _COMMON_ROUTER_INSTRUCTIONS.format(expert_ids_str=expert_ids_str)
    system_msg = (
        "You are a highly logical Router Agent for image auditing (Text-to-Image mode). "
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
        ctx_id=prompt_id,
    )

    cost_time = time.time() - start_time

    if plan is None:
        return None

    # Backward-compat (Change D): map expert_verification_plan to selected_experts
    # so that Step 3 (execute_plan) can parse the plan without modification.
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

    plan["metadata"] = {
        "original_image": image_path,
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "router_cost_seconds": round(cost_time, 2),
        "plan_valid": validate_plan(plan, experts_registry_str, ctx_id=prompt_id),
    }

    return plan


def revise_plan(
    client: OpenAI,
    image_path: str,
    prompt_id: str,
    prompt_text: str,
    atomized_data: dict,
    experts_registry_str: str,
    previous_plan: dict,
    feedback_history: list[dict],
    api_retry: int = 0,
    temperature: float = 0.0,
    model_name: str = "",
) -> dict | None:
    atoms = atomized_data.get("atoms", [])
    objects_taxonomy = _resolve_objects_taxonomy(atomized_data)

    base64_image = encode_image(image_path)

    start_time = time.time()

    prompt = build_router_revision_prompt(
        prompt_text, atoms, objects_taxonomy, experts_registry_str,
        previous_plan, feedback_history,
    )
    _, expert_ids_str, registry_summary = _build_context_block(
        prompt_text, atoms, objects_taxonomy, experts_registry_str,
    )
    formatted_instructions = _COMMON_ROUTER_INSTRUCTIONS.format(expert_ids_str=expert_ids_str)
    system_msg = (
        "You are a highly logical Router Agent for image auditing (Text-to-Image mode). "
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
        ctx_id=f"{prompt_id}/rev",
    )

    cost_time = time.time() - start_time

    if plan is None:
        return None

    # Backward-compat: map expert_verification_plan to selected_experts so that
    # Step 3 (execute_plan) can parse the revised plan without modification.
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

    plan["metadata"] = {
        "original_image": image_path,
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "router_cost_seconds": round(cost_time, 2),
        "plan_valid": validate_plan(plan, experts_registry_str, ctx_id=f"{prompt_id}/rev"),
        "is_revision": True,
        "revision_feedback_count": len(feedback_history),
    }

    return plan

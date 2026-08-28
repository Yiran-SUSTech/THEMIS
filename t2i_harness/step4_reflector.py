"""T2I Reflector (Step 4) - Final evaluation combining atomic QA, taxonomy, and expert evidence.

Produces:
  - per_atom_scores: 0-1 per atomic QA (qa_score x tax_score)
  - alignment_score: mean(per_atom_scores) x 5.0
  - authenticity_score: 0-5 image quality / authenticity assessment
"""

import os
import sys
import re
import json
import base64
import time
from pathlib import Path
from statistics import mean
from functools import lru_cache
from openai import OpenAI

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
T2I_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(T2I_DIR) not in sys.path:
    sys.path.insert(0, str(T2I_DIR))

from common import api_call_with_retry, dump_debug_raw

TAXONOMY_DIR = PROJECT_ROOT / "taxonomy_info"
TAXONOMY_STRUCTURAL_DIR = PROJECT_ROOT / "taxonomy_info_structural"
EXPERTS_REGISTRY_JSON = PROJECT_ROOT / "expert_registry.json"
T2I_REF_ANNOTATIONS_JSON = T2I_DIR / "t2i_ref_annotations.json"

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
REFLECTOR_MODEL = "qwen3.7-plus"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Utility Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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


def load_experts_registry(file_path: str = str(EXPERTS_REGISTRY_JSON)) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.dumps(json.load(f), indent=2)


@lru_cache(maxsize=1)
def _load_t2i_ref_annotations() -> dict:
    """加载 T2I 人类打分参考数据。"""
    if not T2I_REF_ANNOTATIONS_JSON.exists():
        return {}
    with open(T2I_REF_ANNOTATIONS_JSON, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def select_t2i_reference_images(
    image_dir: Path,
    prompt_text: str,
    num_refs: int = 3,
) -> list[dict]:
    """选择与当前 prompt 最相似的参考图片。"""
    annotations = _load_t2i_ref_annotations()
    if not annotations:
        return []

    prompt_words = set(prompt_text.lower().split())
    scored = []
    for img_name, info in annotations.items():
        ref_prompt = info.get("prompt", "")
        ref_words = set(ref_prompt.lower().split())
        overlap = len(prompt_words & ref_words)
        if overlap > 0:
            scored.append((overlap, img_name, info))

    if not scored:
        scored = [(0, name, info) for name, info in annotations.items()]

    scored.sort(key=lambda x: x[0], reverse=True)

    high = [(s, n, i) for s, n, i in scored if i.get("alignment_score", 0) >= 3.5]
    mid = [(s, n, i) for s, n, i in scored if 1.5 <= i.get("alignment_score", 0) < 3.5]
    low = [(s, n, i) for s, n, i in scored if i.get("alignment_score", 0) < 1.5]

    def _get_authenticity(info: dict) -> float:
        score = info.get("authenticity_score")
        if score is None:
            # Legacy key in human-annotated t2i_ref_annotations.json
            score = info.get("artifact_score", 0)
        return score

    selected = []
    for bucket in [high, mid, low]:
        if bucket and len(selected) < num_refs:
            _, img_name, info = bucket[0]
            img_path = Path(image_dir) / img_name
            if img_path.exists():
                selected.append({
                    "image_name": img_name,
                    "image_path": str(img_path),
                    "alignment_score": info.get("alignment_score", 0),
                    "authenticity_score": _get_authenticity(info),
                    "prompt": info.get("prompt", ""),
                })

    return selected[:num_refs]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Expert Blind Spots
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXPERT_BLIND_SPOTS = {
    "fine_grained_classifier": [
        "Cannot detect structural artifacts (melting, extra limbs, fusion).",
        "Cannot verify if specific anatomical features (barbels, whiskers, etc.) are visually present — only predicts class identity.",
        "A high-confidence Top-1 prediction does NOT mean the image is artifact-free.",
    ],
    "open_vocabulary_detector": [
        "Cannot assess structural integrity of detected objects.",
        "Failure to detect an object may indicate the object is malformed or fused with surroundings, not just a model limitation.",
    ],
    "animal_pose_auditor": [
        "Only detects keypoints for limbed subjects (humans, dogs, cats). Cannot audit fish, snakes, or limbless creatures.",
        "Low confidence keypoints indicate structural uncertainty — do NOT dismiss as model noise.",
        "Does not directly detect texture artifacts, color anomalies, or background defects.",
    ],
    "topology_boundary_auditor": [
        "High mask confidence only means the segmentation model found a coherent region — it does NOT mean the boundary is artifact-free.",
        "Cannot detect subtle fusion between subject and background if the color/texture is similar.",
        "Does not check anatomical correctness inside the mask.",
    ],
    "geometric_depth_auditor": [
        "Depth statistics alone do not indicate artifact presence or absence.",
        "Cannot detect texture-level artifacts, anatomical errors, or color anomalies.",
    ],
    "perceptual_quality_auditor": [
        "Only detects graphics-level distortions (noise, blur, compression, darkening).",
        "Cannot detect structural/anatomical artifacts (extra limbs, melting, fusion, semantic inconsistencies).",
        "A 'null' distortion verdict does NOT mean the image is artifact-free — it means no graphics-level distortion was found.",
    ],
    "image_text_auditor": [
        "Only audits text regions. Cannot detect any non-text artifacts.",
    ],
}


def _build_expert_context_str(
    expert_results: dict,
    experts_registry_str: str,
) -> str:
    registry_list = json.loads(experts_registry_str)
    registry_map = {e["expert_id"]: e for e in registry_list}

    testimonies = expert_results.get("expert_testimonies", [])
    custom_prompts = expert_results.get("custom_prompts_for_reflector", "")
    focus_areas = expert_results.get("focus_areas", [])
    image_description = expert_results.get("image_description", "")

    parts = []

    if image_description:
        parts.append(f"Router Image Description: {image_description}")

    if focus_areas:
        parts.append(f"Focus Areas: {json.dumps(focus_areas, ensure_ascii=False)}")

    if custom_prompts:
        parts.append(f"Custom Audit Hints: {custom_prompts}")

    for t in testimonies:
        eid = t.get("expert_id", "unknown")
        target_subject = t.get("target_subject", "N/A")
        weight = t.get("weight", 0.0)
        status = t.get("status", "unknown")
        evidence = t.get("evidence", {})
        error = t.get("error")

        reg_entry = registry_map.get(eid, {})
        expert_specialty = reg_entry.get("specialty", "N/A")
        diagnostic_criteria = reg_entry.get("diagnostic_criteria", {})

        block = f'--- Expert: {eid} ---\nTarget: "{target_subject}" | Weight: {weight} | Specialty: {expert_specialty}\nDiagnostic Criteria: {json.dumps(diagnostic_criteria, ensure_ascii=False)}\nStatus: {status}'

        blind_spots = EXPERT_BLIND_SPOTS.get(eid, [])
        if blind_spots:
            block += f"\nNOT_Capable_Of: {json.dumps(blind_spots, ensure_ascii=False)}"

        if status == "success":
            evidence_clean = _sanitize_evidence(evidence)
            block += f"\nEvidence:\n{json.dumps(evidence_clean, indent=2, ensure_ascii=False)}"
        else:
            block += f"\nError: {error or 'Unknown error'}"

        parts.append(block)

    blind_spot_warning = (
        "\n--- CRITICAL: Expert Blind Spot Warning ---\n"
        "Each expert has limitations (see NOT_Capable_Of above). "
        "'No defect reported' does NOT equal 'no defect exists'. "
        "Experts can only detect defects within their specialty. "
        "Structural artifacts (melting, fusion, extra/missing parts) may go undetected "
        "if no expert covers that specific check. "
        "You MUST verify areas that experts cannot cover based on Router's observations."
    )
    parts.append(blind_spot_warning)

    return "\n\n".join(parts)


def _sanitize_evidence(evidence):
    import numpy as np
    if isinstance(evidence, dict):
        return {k: _sanitize_evidence(v) for k, v in evidence.items()}
    elif isinstance(evidence, list):
        return [_sanitize_evidence(v) for v in evidence]
    elif isinstance(evidence, (np.integer,)):
        return int(evidence)
    elif isinstance(evidence, (np.floating,)):
        return float(evidence)
    elif isinstance(evidence, np.ndarray):
        return evidence.tolist()
    elif isinstance(evidence, (str, int, float, bool)) or evidence is None:
        return evidence
    else:
        return str(evidence)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Reflector System Template (T2I)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_REFLECTOR_SYSTEM_TEMPLATE = r"""You are the Reflector of a T2I image evaluation system. Review the Router's assessment, expert evidence, and atomic QA verdicts, then produce the final evaluation. Output JSON only, no markdown.

**Core Principles:**
1. For each atom, evaluate TWO layers:
   a) QA correctness: Does the image answer the atom's question correctly?
   b) Taxonomy conformance (if applicable): Does the detected object match the taxonomy diagnostic_checkpoints for its class?
   The final atom_score = qa_score × tax_score. If taxonomy is not applicable, atom_score = qa_score.
2. Expert detector/classifier hard data is more reliable than Router's visual impression for object presence and counting.
3. For attribute verification: Router's visual observation is primary; experts are supplementary.
4. For authenticity: Router's direct visual observation is primary; experts are supplementary. Expert silence does NOT override Router's findings.
5. Be critical — do NOT rubber-stamp the Router's assessment.

**Independent Visual Verification:**
- Do NOT assume a feature is present just because the text prompt or class label suggests it should be. You MUST look at the image and verify the feature is actually visible and correctly rendered.
- The Router's checkpoint verdicts are provided at the END of the prompt for reference only. Form your own assessment FIRST by examining the image, then compare with the Router.
- AI-generated images frequently fail to render the MOST distinctive features of a class correctly. For example, a "hammerhead shark" image may show a shark WITHOUT the hammer-shaped head; a "flamingo" image may show a bird WITHOUT the long curved neck. Always scrutinize the defining feature.

**Scoring:**
- per_atom_scores: For each atomic QA, assign 0.0-1.0 based on:
  - qa_score (0-1): Is the answer correct?
  - tax_score (0-1): Does the object conform to taxonomy checkpoints? (1.0 if no taxonomy applicable)
  - atom_score = qa_score × tax_score
- alignment_score: mean(per_atom_scores) × 5.0
- authenticity_score (0-5): How authentic is the image? Higher = fewer artifacts, better quality. A flawless image scores 5.0. Structural defects severely reduce this score.
- Scores should be precise continuous values (e.g., 3.82, 1.47, 4.63).
- If more than 25% of checkpoints are marked untestable, you must deduct an appropriate amount from alignment_score, because fewer testable checkpoints usually means the image does not contain enough taxonomy features for a high alignment score.

**Output JSON:**
{
  "atom_reviews": [
    {"atom_index": 0, "question": "...", "expected": "...",
     "qa_score": 1.0, "tax_score": 0.9, "atom_score": 0.9,
     "expert_evidence": "...", "reasoning": "..."}
  ],
  "authenticity_review": "...",
  "alignment_score": 0.0,
  "authenticity_score": 0.0,
  "per_atom_scores": [0.9],
  "key_defects": ["..."]
}

6. HUMAN REFERENCE CALIBRATION:
   - You will be provided with human-scored reference images.
   - Use these as calibration anchors: if the target image has similar quality
     to a reference with alignment_score=4.0, assign a similar alignment_score.
   - If the target image has similar artifact severity to a reference with
     authenticity_score=2.0, assign a similar authenticity_score.
   - References are anchors, not templates. Adjust based on the target's
     specific features and expert evidence.
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Prompt Building
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_reflector_prompt(
    prompt_text: str,
    atomized_data: dict,
    expert_results_str: str,
    router_plan: dict | None = None,
    ref_images: list[dict] | None = None,
) -> str:
    """Build the user prompt for the T2I Reflector.

    Context includes: text prompt, atomic QA pairs, taxonomy prior knowledge
    (objects with diagnostic_checkpoints), and the Router's assessment
    (atom_verdicts, checkpoint_verdicts, artifact_observations).
    """
    atoms = atomized_data.get("atoms", [])
    objects = atomized_data.get("objects", [])

    # Build atomic QA pairs text
    atoms_text = ""
    if atoms:
        atoms_lines = ["**[Atomic QA Pairs]**"]
        for i, atom in enumerate(atoms):
            question = atom.get("question", "")
            answer = atom.get("answer", "")
            skill = atom.get("skill", "")
            target_object = atom.get("target_object", "")
            atoms_lines.append(
                f"{i}. Q: {question} -> Expected: {answer} "
                f"(skill: {skill}, target_object: {target_object})"
            )
        atoms_text = "\n".join(atoms_lines) + "\n"

    # Build taxonomy prior knowledge text
    taxonomy_text = ""
    if objects:
        tax_lines = ["**[Taxonomy Prior Knowledge]**"]
        for i, obj in enumerate(objects, 1):
            obj_name = obj.get("object_name", "")
            class_name = obj.get("class_name", "")
            class_id = obj.get("class_id")
            tax_desc = obj.get("taxonomy_description", "")
            checkpoints = obj.get("diagnostic_checkpoints", {})
            tax_lines.append(
                f"Object {i}: {obj_name} (class: {class_name}, class_id: {class_id})"
            )
            if tax_desc:
                tax_lines.append(f"  Taxonomy Description: {tax_desc}")
            if checkpoints:
                tax_lines.append(
                    f"  Diagnostic Checkpoints: "
                    f"{json.dumps(checkpoints, indent=2, ensure_ascii=False)}"
                )
            else:
                tax_lines.append(
                    "  Diagnostic Checkpoints: (none - no taxonomy prior applicable)"
                )
        taxonomy_text = "\n".join(tax_lines) + "\n"

    # Build Router's assessment context
    router_assessment = ""
    if router_plan:
        atom_verdicts = router_plan.get("atom_verdicts", [])
        checkpoint_verdicts = router_plan.get("checkpoint_verdicts", [])
        artifact_observations = router_plan.get("artifact_observations", [])
        image_description = router_plan.get("image_description", "")

        router_assessment = f"""
**[Router's Assessment]**
- Image: {image_description}
- Atom Verdicts: {json.dumps(atom_verdicts, indent=2, ensure_ascii=False)}
- Checkpoint Verdicts: {json.dumps(checkpoint_verdicts, indent=2, ensure_ascii=False)}
- Artifact Observations: {json.dumps(artifact_observations, indent=2, ensure_ascii=False)}
"""

    prompt = f"""Review the image, expert evidence, and Router's preliminary assessment to produce the final evaluation.

**IMPORTANT: Independent Assessment First**
Before reading the Router's verdicts (provided at the END for reference), examine the image yourself and form your own opinion about each atom and checkpoint. The Router's verdicts are preliminary and may be wrong — you MUST verify each one by looking at the actual image.

**[Context]**
- Text Prompt: {prompt_text}
{atoms_text}
{taxonomy_text}
**[Expert Testimonies]**
{expert_results_str}

**[Router's Preliminary Assessment — FOR REFERENCE ONLY, DO NOT ANCHOR]**
{router_assessment}"""

    if ref_images:
        ref_lines = []
        for ref in ref_images:
            ref_lines.append(
                f"  - Image: {ref['image_name']} | "
                f"Prompt: {ref.get('prompt', 'N/A')} | "
                f"Human Alignment: {ref['alignment_score']} | "
                f"Human Authenticity: {ref['authenticity_score']}"
            )
        prompt += (
            "\n\n**[Human-Annotated Reference Images]**\n"
            "Compare the target image against these references. "
            "If the target has similar quality, assign a similar score.\n"
            + "\n".join(ref_lines)
        )

    return prompt


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Score Calibration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def compute_alignment_score(per_atom_scores: list[float]) -> float:
    """Compute alignment_score from per_atom_scores.

    alignment_score = mean(per_atom_scores) x 5.0
    Returns 0.0 if per_atom_scores is empty.
    """
    if not per_atom_scores:
        return 0.0
    return round(mean(per_atom_scores) * 5.0, 2)


def _calibrate_scores(result: dict) -> dict:
    """Post-process Reflector output with hard rules that cannot be violated.

    T2I version:
    - Compute alignment_score = mean(per_atom_scores) x 5.0 from per_atom_scores
    - Keep authenticity_score from Reflector's judgment (clamped to [0, 5])
    - No classifier-based alignment capping (T2I handles this through atom scores)
    - No pose hard cap
    """
    adjustments = []

    # Compute alignment_score from per_atom_scores
    # (Reflector prompt explicitly requires per_atom_scores in output JSON)
    per_atom_scores = result.get("per_atom_scores", [])
    computed_alignment = compute_alignment_score(per_atom_scores)
    reflector_alignment = result.get("alignment_score", 0.0)
    try:
        reflector_alignment = float(reflector_alignment)
    except (TypeError, ValueError):
        reflector_alignment = 0.0
    if abs(computed_alignment - reflector_alignment) > 0.01:
        adjustments.append(
            f"Alignment recalculated from per_atom_scores: "
            f"{reflector_alignment:.2f} -> {computed_alignment}"
        )
    result["alignment_score"] = computed_alignment

    # Clamp authenticity_score to [0, 5]
    authenticity_score = result.get("authenticity_score", 0.0)
    try:
        authenticity_score = float(authenticity_score)
    except (TypeError, ValueError):
        authenticity_score = 0.0
    authenticity_score = max(0.0, min(5.0, authenticity_score))
    result["authenticity_score"] = round(authenticity_score, 2)

    if adjustments:
        result["code_adjustments"] = adjustments

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Auxiliary Images
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _collect_auxiliary_images(expert_results: dict) -> list[str]:
    paths = []
    for t in expert_results.get("expert_testimonies", []):
        if t.get("status") != "success":
            continue
        evidence = t.get("evidence", {})
        for key in ("saved_mask_path", "saved_depth_path", "mask_visualization_path"):
            p = evidence.get(key)
            if p and os.path.exists(p):
                paths.append(p)
    return paths


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Self-Reflection (Round 2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


_REFLECTOR_SELF_REFLECTION_TEMPLATE = """You are the Reflector performing self-reflection on your initial assessment. You have just completed a preliminary evaluation of an AI-generated image. Now critically review your own assessment and produce the final, revised evaluation.

**Self-Reflection Checklist:**
1. Score-Reasoning Consistency: Do your scores align with your reasoning?
   - If your reasoning describes serious issues but the score is high → lower it.
   - If your reasoning is positive but the score is low → raise it.
   - Look for contradictions between per_atom_scores and the reasoning text.

2. Expert Evidence Utilization: Did you properly consider ALL expert testimony?
   - Were there classifier results (top-3 labels) you ignored or underweighted?
   - Were there detector counts that contradict your atom verdicts?
   - Were there auxiliary images (depth maps, segmentation masks) you didn't reference?

3. Reference Calibration: If human-scored reference images were provided:
   - Are your scores consistent with the reference anchors?
   - If a reference with similar quality has alignment=4.0, is your score in a similar range?

4. Leniency Bias: Are you rubber-stamping the Router's assessment too readily?
   - The Router's checkpoint verdicts are preliminary — did you independently verify them?
   - Did you accept is_present=true without checking expert evidence?

5. Harshness Bias: Are you over-penalizing minor issues?
   - A minor texture anomaly (severity 1) should not drop authenticity_score by more than 0.5.
   - Multiple minor issues compound, but one minor issue should not dominate.

6. Atom Score Review: For each atom:
   - Is qa_score justified by the expert evidence (or VLM observation if no expert)?
   - Is tax_score appropriate? If no taxonomy info, tax_score should be 1.0.
   - Did you form your OWN independent verdict by looking at the image, or did you just copy the Router's?

7. Upward Override Justification: For every checkpoint where your final assessment is MORE lenient than the Router's (e.g., you agreed with is_present=true where the Router was uncertain, or you raised a score above the base score):
   - You MUST provide independent visual evidence from the image itself (not just "the Router said so").
   - If you cannot point to a specific visual feature that confirms the checkpoint, you should not mark it as present.
   - This prevents rubber-stamping by forcing you to independently verify each positive verdict.

**Output the SAME JSON schema as your initial assessment, with revised scores.**
Add a "self_reflection_notes" field documenting:
  - What you changed and why
  - Which checklist items triggered adjustments
  - Whether your final scores are higher, lower, or same as initial (with reasoning)
"""


def _run_self_reflection_round(
    client: OpenAI,
    system_message: dict,
    user_content: list[dict],
    round1_result: dict,
    api_retry: int = 0,
    temperature: float = 0.5,
    model_name: str = "",
) -> dict | None:
    """执行 Round 2 Self-Reflection API 调用。"""
    if not model_name:
        model_name = REFLECTOR_MODEL
    self_reflection_system = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": _REFLECTOR_SELF_REFLECTION_TEMPLATE,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }

    round2_prompt = (
        "Review your assessment above using the Self-Reflection Checklist. "
        "Output revised JSON with the same schema, plus a 'self_reflection_notes' field."
    )

    messages = [
        self_reflection_system,
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": json.dumps(round1_result, indent=2, ensure_ascii=False)},
        {"role": "user", "content": round2_prompt},
    ]

    try:
        completion = api_call_with_retry(
            client.chat.completions.create,
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_retries=api_retry,
            label="Reflector-SelfReflection",
            extra_body={"enable_thinking": False},
        )
        raw_content = completion.choices[0].message.content
        if raw_content is None or raw_content.strip() == "":
            reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
            if reasoning:
                raw_content = reasoning
            else:
                return None
        result = parse_json_safely(raw_content)
        return result
    except Exception as e:
        print(f"  [WARN] Self-reflection round failed: {type(e).__name__}: {e}")
        return None


def _merge_self_reflection(round1: dict, round2: dict) -> dict:
    """合并两轮结果。Round 2 的分数优先，但保留 Round 1 的数据用于审计。"""
    round1_scores = {
        "alignment_score": round1.get("alignment_score"),
        "authenticity_score": round1.get("authenticity_score"),
        "per_atom_scores": round1.get("per_atom_scores", []),
    }

    merged = round2.copy()
    merged["preliminary_scores"] = round1_scores
    merged["self_reflection_notes"] = round2.get("self_reflection_notes", "")

    r1_align = round1.get("alignment_score", 0)
    r2_align = round2.get("alignment_score", 0)
    if abs(r1_align - r2_align) > 0.01:
        merged["score_changes"] = {
            "alignment_score": f"{r1_align:.2f} → {r2_align:.2f}",
            "authenticity_score": f"{round1.get('authenticity_score', 0):.2f} → {round2.get('authenticity_score', 0):.2f}",
        }

    return merged


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main Reflector Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_reflector(
    client: OpenAI,
    image_path: str,
    prompt_id: str,
    prompt_text: str,
    atomized_data: dict,
    expert_results: dict,
    experts_registry_str: str,
    router_plan: dict | None = None,
    ref_image_dir: Path | None = None,
    enable_self_reflection: bool = True,
    api_retry: int = 0,
    temperature: float = 0.5,
    model_name: str = "",
) -> dict | None:
    if not model_name:
        model_name = REFLECTOR_MODEL
    tag = f"[Reflector][{prompt_id}]" if prompt_id else "[Reflector]"
    expert_results_str = _build_expert_context_str(expert_results, experts_registry_str)
    base64_image = encode_image(image_path)

    start_time = time.time()

    ref_images = []
    if ref_image_dir:
        ref_images = select_t2i_reference_images(Path(ref_image_dir), prompt_text)

    prompt = build_reflector_prompt(
        prompt_text, atomized_data, expert_results_str, router_plan,
        ref_images=ref_images,
    )

    user_content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            },
        },
    ]

    aux_images = _collect_auxiliary_images(expert_results)
    for aux_path in aux_images:
        try:
            aux_b64 = encode_image(aux_path)
            aux_label = Path(aux_path).stem
            user_content.append({
                "type": "text",
                "text": f"[Auxiliary Expert Output Image: {aux_label}]",
            })
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{aux_b64}"
                },
            })
        except Exception as e:
            print(f"  {tag} [WARN] Failed to encode auxiliary image {aux_path}: {e}")

    for ref in ref_images:
        try:
            ref_b64 = encode_image(ref["image_path"])
            user_content.append({
                "type": "text",
                "text": f"[Reference: {ref['image_name']} | "
                        f"Alignment={ref['alignment_score']} | "
                        f"Authenticity={ref['authenticity_score']}]",
            })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{ref_b64}"},
            })
        except Exception as e:
            print(f"  {tag} [WARN] Failed to load reference image {ref['image_path']}: {e}")

    system_message = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": _REFLECTOR_SYSTEM_TEMPLATE,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }

    try:
        completion = api_call_with_retry(
            client.chat.completions.create,
            model=model_name,
            messages=[
                system_message,
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_retries=api_retry,
            label="Reflector",
            extra_body={"enable_thinking": False},
        )
        raw_content = completion.choices[0].message.content
        cost_time = time.time() - start_time
        finish_reason = getattr(completion.choices[0], "finish_reason", "unknown")
        usage = getattr(completion, "usage", None)

        if raw_content is None or raw_content.strip() == "":
            reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
            usage_info = (
                f"prompt_tokens={usage.prompt_tokens}, "
                f"completion_tokens={usage.completion_tokens}"
            ) if usage else "no usage info"
            print(
                f"  {tag} [ERROR] Returned empty content "
                f"(content is {'None' if raw_content is None else 'empty string'}, "
                f"finish_reason={finish_reason}, {usage_info})"
            )
            if reasoning:
                print(
                    f"  {tag} [WARN] reasoning_content found "
                    f"({len(reasoning)} chars), attempting to extract JSON"
                )
                raw_content = reasoning
            else:
                msg = completion.choices[0].message
                print(
                    f"  {tag} [DEBUG] Full message: content={repr(msg.content)}, "
                    f"role={getattr(msg, 'role', 'N/A')}, "
                    f"function_call={getattr(msg, 'function_call', None)}, "
                    f"tool_calls={getattr(msg, 'tool_calls', None)}, "
                    f"refusal={getattr(msg, 'refusal', None)}"
                )
                dump_debug_raw("reflector_empty_content", prompt_id or "unknown",
                               repr(msg.content),
                               note=f"{tag} empty content, finish_reason={finish_reason}")
                return None
        result = parse_json_safely(raw_content)
        if result is None:
            dump_path = dump_debug_raw("reflector_unparseable", prompt_id or "unknown",
                                       raw_content, note=f"{tag} unparseable JSON response")
            hint = f" (full response saved to: {dump_path})" if dump_path else ""
            print(f"  {tag} [ERROR] Returned unparseable JSON{hint}")
            print(f"  {tag} [ERROR] Head: {raw_content[:300]}")
            return None

        round2_result = None
        if enable_self_reflection:
            round2_result = _run_self_reflection_round(
                client, system_message, user_content, result,
                api_retry=api_retry, temperature=temperature,
                model_name=model_name,
            )
            if round2_result is not None:
                result = _merge_self_reflection(result, round2_result)
                print(f"  {tag} [INFO] Self-reflection completed. "
                      f"Alignment: {result.get('alignment_score', 'N/A')}, "
                      f"Authenticity: {result.get('authenticity_score', 'N/A')}")
            else:
                print(f"  {tag} [WARN] Self-reflection round failed, using Round 1 scores")

        result["metadata"] = {
            "original_image": image_path,
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "auxiliary_images_included": [Path(p).name for p in aux_images],
            "reflector_cost_seconds": round(cost_time, 2),
            "self_reflection_enabled": enable_self_reflection,
            "self_reflection_succeeded": round2_result is not None if enable_self_reflection else None,
        }
        result = _calibrate_scores(result)

        return result

    except Exception as e:
        cost_time = time.time() - start_time
        print(f"  {tag} [ERROR] API call failed after {cost_time:.1f}s: {type(e).__name__}: {e}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Report Saving & Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def save_final_report(
    report: dict,
    output_dir: str | Path | None = None,
) -> str:
    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent
            / os.environ.get("T2I_OUTPUT_DIR_NAME", "output")
            / "final_reports"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = report.get("metadata", {})
    prompt_id = metadata.get("prompt_id", "unknown")

    filename = f"final_evaluation_report_{prompt_id}.json"
    filepath = output_dir / filename

    safe_report = _sanitize_evidence(report)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(safe_report, f, indent=4, ensure_ascii=False)

    print(f"  [SAVED] {filename}")
    return str(filepath)


def print_final_summary(report: dict) -> None:
    metadata = report.get("metadata", {})
    prompt_text = metadata.get("prompt_text", "N/A")
    alignment_score = report.get("alignment_score", 0.0)
    authenticity_score = report.get("authenticity_score", 0.0)
    key_defects = report.get("key_defects", [])
    code_adjustments = report.get("code_adjustments", [])

    print(f"\n--- Final Evaluation Complete ---")
    print(f"Prompt: {prompt_text}")
    print(
        f"Alignment: {alignment_score:.2f}/5.0 | "
        f"Authenticity: {authenticity_score:.2f}/5.0"
    )
    if code_adjustments:
        for adj in code_adjustments:
            print(f"  [Code Adjustment] {adj}")
    if key_defects:
        print(f"Key Defects: {', '.join(key_defects)}")

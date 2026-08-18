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
C2I_DIR = Path(__file__).resolve().parent

TAXONOMY_DIR = PROJECT_ROOT / "taxonomy_info"
TAXONOMY_STRUCTURAL_DIR = PROJECT_ROOT / "taxonomy_info_structural"
EXPERTS_REGISTRY_JSON = PROJECT_ROOT / "expert_registry.json"
REF_ANNOTATIONS_JSON = C2I_DIR / "ref_annotations.json"

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
REFLECTOR_MODEL = "qwen3.7-plus"


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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Reference Image Selection (for Reflector anchoring)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_REF_ANNOTATIONS_CACHE: dict | None = None
_SUPER_CATEGORY_CACHE: dict[int, str | None] = {}

# Artifact score segment boundaries (0-5 scale split into thirds)
_REF_LOW_MID_BOUNDARY = 5.0 / 3.0    # ~1.667
_REF_MID_HIGH_BOUNDARY = 10.0 / 3.0  # ~3.333


def _get_super_category_for_class(class_id: int) -> str | None:
    """Look up the super_category for a given class_id from taxonomy_info_structural."""
    if class_id in _SUPER_CATEGORY_CACHE:
        return _SUPER_CATEGORY_CACHE[class_id]

    batch_num = class_id // 10
    batch_file = TAXONOMY_STRUCTURAL_DIR / f"taxonomy_enriched_Batch_{batch_num}_structured.json"
    super_cat: str | None = None
    if batch_file.exists():
        try:
            with open(batch_file, "r", encoding="utf-8") as f:
                items = json.load(f)
            for item in items:
                if item.get("class_id") == class_id:
                    super_cat = item.get("super_category")
                    break
        except (json.JSONDecodeError, OSError):
            super_cat = None

    _SUPER_CATEGORY_CACHE[class_id] = super_cat
    return super_cat


def _load_ref_annotations() -> dict:
    """Load ref_annotations.json (cached at module level)."""
    global _REF_ANNOTATIONS_CACHE
    if _REF_ANNOTATIONS_CACHE is not None:
        return _REF_ANNOTATIONS_CACHE
    if not REF_ANNOTATIONS_JSON.exists():
        print(f"  [WARN] ref_annotations.json not found: {REF_ANNOTATIONS_JSON}")
        _REF_ANNOTATIONS_CACHE = {}
        return _REF_ANNOTATIONS_CACHE
    try:
        with open(REF_ANNOTATIONS_JSON, "r", encoding="utf-8") as f:
            _REF_ANNOTATIONS_CACHE = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [WARN] Failed to load ref_annotations.json: {e}")
        _REF_ANNOTATIONS_CACHE = {}
    return _REF_ANNOTATIONS_CACHE


def select_reference_images(
    class_id: int,
    image_dir: Path,
    exclude_image_name: str | None = None,
    num_refs: int = 3,
) -> list[dict]:
    """Select reference images for the Reflector.

    Selection rules:
      1. Same super_category as the evaluated image's class.
      2. Cover low/mid/high artifact_score segments for calibration.
      3. Exclude the image being evaluated.
      4. Image file must exist in image_dir.

    Returns a list of dicts with keys:
      image_name, image_path, class_id, class_name,
      alignment_score, artifact_score, super_category, segment
    """
    target_super_cat = _get_super_category_for_class(class_id)
    if target_super_cat is None:
        print(f"  [WARN] No super_category found for class_id={class_id}, skipping reference images.")
        return []

    annotations = _load_ref_annotations()
    if not annotations:
        return []

    image_dir = Path(image_dir)
    candidates: list[dict] = []

    for img_name, ann in annotations.items():
        if exclude_image_name and img_name == exclude_image_name:
            continue

        ann_class_id = ann.get("class_id")
        if ann_class_id is None:
            continue
        ann_super_cat = _get_super_category_for_class(ann_class_id)
        if ann_super_cat != target_super_cat:
            continue

        img_path = image_dir / img_name
        if not img_path.exists():
            continue

        scores = ann.get("scores", {})
        artifact_score = scores.get("artifact_score")
        alignment_score = scores.get("alignment_score")
        if artifact_score is None:
            continue

        try:
            artifact_score = float(artifact_score)
            alignment_score = float(alignment_score) if alignment_score is not None else None
        except (TypeError, ValueError):
            continue

        if artifact_score < _REF_LOW_MID_BOUNDARY:
            segment = "low"
        elif artifact_score < _REF_MID_HIGH_BOUNDARY:
            segment = "mid"
        else:
            segment = "high"

        candidates.append({
            "image_name": img_name,
            "image_path": str(img_path),
            "class_id": ann_class_id,
            "class_name": ann.get("class_name", ""),
            "alignment_score": alignment_score,
            "artifact_score": artifact_score,
            "super_category": ann_super_cat,
            "segment": segment,
        })

    if not candidates:
        print(f"  [WARN] No reference candidates found for super_category={target_super_cat}")
        return []

    low_group = [c for c in candidates if c["segment"] == "low"]
    mid_group = [c for c in candidates if c["segment"] == "mid"]
    high_group = [c for c in candidates if c["segment"] == "high"]

    def _pick_representative(group: list[dict], low: float, high: float) -> dict | None:
        if not group:
            return None
        midpoint = (low + high) / 2.0
        return min(group, key=lambda c: abs(c["artifact_score"] - midpoint))

    selected: list[dict] = []
    for group, lo, hi in (
        (low_group, 0.0, _REF_LOW_MID_BOUNDARY),
        (mid_group, _REF_LOW_MID_BOUNDARY, _REF_MID_HIGH_BOUNDARY),
        (high_group, _REF_MID_HIGH_BOUNDARY, 5.0),
    ):
        rep = _pick_representative(group, lo, hi)
        if rep is not None:
            selected.append(rep)

    # If fewer than num_refs, fill from remaining candidates
    if len(selected) < num_refs:
        selected_names = {s["image_name"] for s in selected}
        remaining = [c for c in candidates if c["image_name"] not in selected_names]
        remaining.sort(key=lambda c: abs(c["artifact_score"] - 2.5))
        for c in remaining:
            if len(selected) >= num_refs:
                break
            selected.append(c)

    return selected[:num_refs]


def _build_ref_images_text(ref_images: list[dict]) -> str:
    """Build a text description of the reference images for the Reflector prompt."""
    if not ref_images:
        return ""

    lines = [
        "**[Human-Annotated Reference Images]**",
        "Below are reference images with human annotations. Use them as anchors to calibrate your scoring.",
        "Higher artifact_score = fewer artifacts (better image quality).",
        "Higher alignment_score = better class conformance.",
        "",
    ]
    for i, ref in enumerate(ref_images, 1):
        al = ref["alignment_score"] if ref["alignment_score"] is not None else "N/A"
        al_str = f"{al:.2f}" if isinstance(al, float) else str(al)
        lines.append(
            f"Reference {i} [{ref['segment']}]: {ref['image_name']} | "
            f"class={ref['class_name']} | alignment_score={al_str} | "
            f"artifact_score={ref['artifact_score']:.2f}"
        )
    lines.append("")
    lines.append(
        "Compare the target image against these references. If the target has similar quality "
        "to a reference, assign a similar score. This ensures your scores are calibrated to "
        "human judgment and remain stable across images."
    )
    return "\n".join(lines)


def _append_ref_images_to_content(
    user_content: list[dict],
    ref_images: list[dict],
) -> None:
    """Append base64-encoded reference images to the API request content."""
    for ref in ref_images:
        try:
            ref_b64 = encode_image(ref["image_path"])
            al = ref["alignment_score"]
            al_str = f"{al:.2f}" if isinstance(al, float) else str(al)
            user_content.append({
                "type": "text",
                "text": (
                    f"[Reference Image: {ref['image_name']} | "
                    f"alignment={al_str} | artifact={ref['artifact_score']:.2f}]"
                ),
            })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{ref_b64}"},
            })
        except Exception as e:
            print(f"  [WARN] Failed to encode reference image {ref['image_path']}: {e}")


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


_REFLECTOR_SYSTEM_TEMPLATE = r"""You are the Reflector of an AI image evaluation system. Review the Router's assessment and expert evidence, then produce the final evaluation. Output JSON only, no markdown.

**Core Principles:**
1. For alignment: expert classifier hard data is more reliable than Router's visual impression.
2. For artifacts: Router's direct visual observation is primary; experts are supplementary. Expert silence does NOT override Router's findings.
3. Structural defects outweigh aesthetic quality.
4. Be critical — do NOT rubber-stamp the Router's assessment. Actively look for issues the Router may have missed or underestimated.
5. When in doubt about an artifact, lean toward flagging it rather than ignoring it.

**Scoring Guidelines:**
- The base scores shown below are computed by a formula and are ONLY a starting reference. You MUST give your own independent scores based on your holistic judgment.
- Your scores should be precise continuous values (e.g., 3.82, 1.47, 4.63), NOT rounded to 0.5 increments.
- A truly excellent image (full class conformance + zero artifacts) should score near 5.0.
- Any notable issue should produce a meaningfully lower score. Multiple minor issues compound.
- You may score higher or lower than the base scores if your judgment warrants it.

**Pose Evidence Interpretation:**
- The animal_pose_auditor (ViTPose) is trained on real animal photographs. When evaluating AI-generated images, its keypoint confidence scores may be artificially low due to domain shift (different texture, style, and edge statistics).
- Treat low-confidence keypoints as a WEAK signal, NOT definitive evidence of artifacts. Only use this signal when:
  1. You VISUALLY confirm structural anomalies in the same region as low-confidence keypoints, OR
  2. Keypoint count is physically impossible (e.g., more than 4 legs, asymmetric distribution), OR
  3. Low confidence coincides with clear visual artifacts (melting, fusion, structural collapse).
- Ignore low-confidence keypoints if they could be explained by: subject too small, rare pose, occlusion, or normal variance in AI-generated imagery.
- High-confidence keypoints are a STRONG signal: if most keypoints have high confidence AND you don't see visual artifacts, the image likely has good structural integrity.

**Your Task:**
- Review the Router's checkpoint verdicts: for each, consider whether the Router was too lenient. Did it mark a checkpoint as present when the match is only partial? Did it skip a checkpoint by marking it untestable when it could have been judged?
- Review the Router's artifact observations: for each, consider whether the severity was underestimated. Look for additional artifacts the Router missed, especially subtle ones revealed by expert evidence.
- Note any new artifacts found by experts that the Router missed.
- Produce final alignment_score and artifact_score (0-5 continuous).

**Output JSON:**
{
  "checkpoint_review": "For each checkpoint, agree/disagree with Router's is_testable and is_present, with reasoning. Flag any checkpoints the Router was too lenient on.",
  "artifact_review": "For each artifact, agree/disagree with Router's severity, with reasoning. Note new artifacts from experts or your own observation. Flag any severities the Router underestimated.",
  "alignment_score": 0.0,
  "artifact_score": 0.0,
  "alignment_reasoning": "Concise: how many checkpoints passed/testable, expert classifier confirmation, adjustments made",
  "artifact_reasoning": "Concise: Router's artifacts + severities, expert support/contradiction, new findings",
  "key_defects": ["string"]
}"""


_REFLECTOR_CHECKLIST_SYSTEM_TEMPLATE = r"""You are the Reflector of an AI image evaluation system. Review the Router's assessment and expert evidence, then produce the final evaluation. Output JSON only, no markdown.

**Core Principles:**
1. For alignment: expert classifier hard data is more reliable than Router's visual impression.
2. For artifacts: Router's direct visual observation is primary; experts are supplementary. Expert silence does NOT override Router's findings.
3. Structural defects outweigh aesthetic quality.
4. Be critical — do NOT rubber-stamp the Router's assessment. Actively look for issues the Router may have missed or underestimated.
5. When in doubt about an artifact, lean toward flagging it rather than ignoring it.

**Scoring Guidelines:**
- The base scores shown below are computed by a formula and are ONLY a starting reference. You MUST give your own independent scores based on your holistic judgment.
- Your scores should be precise continuous values (e.g., 3.82, 1.47, 4.63), NOT rounded to 0.5 increments.
- A truly excellent image (full class conformance + zero artifacts) should score near 5.0.
- Any notable issue should produce a meaningfully lower score. Multiple minor issues compound.
- You may score higher or lower than the base scores if your judgment warrants it.

**Pose Evidence Interpretation:**
- The animal_pose_auditor (ViTPose) is trained on real animal photographs. When evaluating AI-generated images, its keypoint confidence scores may be artificially low due to domain shift (different texture, style, and edge statistics).
- Treat low-confidence keypoints as a WEAK signal, NOT definitive evidence of artifacts. Only use this signal when:
  1. You VISUALLY confirm structural anomalies in the same region as low-confidence keypoints, OR
  2. Keypoint count is physically impossible (e.g., more than 4 legs, asymmetric distribution), OR
  3. Low confidence coincides with clear visual artifacts (melting, fusion, structural collapse).
- Ignore low-confidence keypoints if they could be explained by: subject too small, rare pose, occlusion, or normal variance in AI-generated imagery.
- High-confidence keypoints are a STRONG signal: if most keypoints have high confidence AND you don't see visual artifacts, the image likely has good structural integrity.

**Your Task:**
- Review the Router's checkpoint verdicts: for each, consider whether the Router was too lenient. Did it mark a checkpoint as present when the match is only partial? Did it skip a checkpoint by marking it untestable when it could have been judged?
- Review the Router's artifact observations: for each, consider whether the severity was underestimated. Look for additional artifacts the Router missed, especially subtle ones revealed by expert evidence.
- Note any new artifacts found by experts that the Router missed.
- Produce final alignment_score and artifact_score (0-5 continuous).

**Checklist Annotation (fine_grained_details):**
- You MUST produce a `fine_grained_details` object that mirrors the Diagnostic Checkpoints structure.
- For EACH checkpoint description listed under each category, assign one of three status values:
  - "🟢 Checked" — the feature is clearly present and correctly rendered in the image.
  - "🔴 Missing" — the feature is absent, malformed, or incorrect.
  - "⚪ N/A" — the feature cannot be evaluated (e.g., not visible, occluded, or genuinely inapplicable to this view).
- Use the EXACT checkpoint description strings from the Diagnostic Checkpoints as keys. Do NOT rephrase or invent new keys.
- Your checklist must cover EVERY checkpoint from EVERY category in the Diagnostic Checkpoints — no omissions.
- Base your status on your own holistic judgment, informed by the Router's verdicts and expert evidence, but do not blindly copy the Router.

**Veto Mechanism (veto_activated):**
- Set `veto_activated` to true if the image has catastrophic structural failure (e.g., complete structural collapse, severe melting making the subject unrecognizable, or multiple major anatomical errors) that makes meaningful evaluation impossible.
- Otherwise set it to false.

**Output JSON:**
{
  "checkpoint_review": "For each checkpoint, agree/disagree with Router's is_testable and is_present, with reasoning. Flag any checkpoints the Router was too lenient on.",
  "artifact_review": "For each artifact, agree/disagree with Router's severity, with reasoning. Note new artifacts from experts or your own observation. Flag any severities the Router underestimated.",
  "alignment_score": 0.0,
  "artifact_score": 0.0,
  "alignment_reasoning": "Concise: how many checkpoints passed/testable, expert classifier confirmation, adjustments made",
  "artifact_reasoning": "Concise: Router's artifacts + severities, expert support/contradiction, new findings",
  "key_defects": ["string"],
  "veto_activated": false,
  "fine_grained_details": {
    "Category_Name": {
      "Exact checkpoint description string from diagnostic_checkpoints": "🟢 Checked"
    }
  }
}"""


_REFLECTOR_SELF_REFLECTION_TEMPLATE = """You are the Reflector performing self-reflection on your initial assessment. You have just completed a preliminary evaluation of an AI-generated image. Now critically review your own assessment and produce the final, revised evaluation.

**Self-Reflection Checklist:**
1. Score-Reasoning Consistency: Do your scores align with your reasoning?
   - If your alignment_reasoning describes checkpoint mismatches but alignment_score is high → lower it.
   - If your artifact_reasoning describes severe issues but artifact_score is high → lower it.
   - Look for contradictions between the reasoning text and the numerical scores.

2. Expert Evidence Utilization: Did you properly consider ALL expert testimony?
   - Were there classifier results (top-3 labels) you ignored or underweighted?
   - Did the classifier Top-1 match the target class? If not, did you adequately cap alignment?
   - Were there auxiliary images (depth maps, segmentation masks) you didn't reference?
   - Did the pose auditor's keypoint analysis reveal structural issues you overlooked?

3. Reference Calibration: If human-scored reference images were provided:
   - Are your scores consistent with the reference anchors?
   - If a reference with similar quality has alignment=4.0, is your score in a similar range?
   - Reference anchors should prevent both inflated and deflated scores.

4. Leniency Bias: Are you rubber-stamping the Router's assessment too readily?
   - The Router's checkpoint verdicts are preliminary — did you independently verify them?
   - Did you accept is_present=true without checking expert evidence?
   - The Router may miss subtle artifacts — did you look for additional issues?

5. Harshness Bias: Are you over-penalizing minor issues?
   - A minor texture anomaly (severity 1) should not drop artifact_score by more than 0.5.
   - Multiple minor issues compound, but one minor issue should not dominate.
   - Pose low-confidence keypoints alone (without visual confirmation) are a weak signal.

6. Checkpoint Review: For each checkpoint:
   - Did you agree/disagree with the Router's is_present verdict?
   - If you disagreed, did you explain why?
   - If the Router was too lenient, did you flag it?

**Output the SAME JSON schema as your initial assessment, with revised scores.**
Add a "self_reflection_notes" field documenting:
  - What you changed and why
  - Which checklist items triggered adjustments
  - Whether your final scores are higher, lower, or same as initial (with reasoning)
"""


def build_reflector_prompt(
    class_label: str,
    taxonomy_info: dict | None,
    expert_results_str: str,
    router_plan: dict | None = None,
    structured_taxonomy_info: dict | None = None,
) -> str:
    taxonomy_desc = taxonomy_info.get("enriched_description", "No specific taxonomy found.") if taxonomy_info else "No specific taxonomy found."

    # Build Router's assessment context
    router_assessment = ""
    base_scores_text = ""
    if router_plan:
        checkpoint_verdicts = router_plan.get("checkpoint_verdicts", [])
        artifact_observations = router_plan.get("artifact_observations", [])
        image_description = router_plan.get("image_description", "")

        # Compute base scores (for reference — Reflector may adjust)
        testable = [cv for cv in checkpoint_verdicts if cv.get("is_testable", False)]
        present = [cv for cv in testable if cv.get("is_present", False)]
        untestable = [cv for cv in checkpoint_verdicts if not cv.get("is_testable", False)]
        base_alignment = 5.0 * len(present) / len(testable) if testable else 0.0

        if untestable:
            untestable_penalty = 0.5 * len(untestable) / len(checkpoint_verdicts) * base_alignment
            base_alignment -= untestable_penalty

        if artifact_observations:
            max_severity = max(ao.get("severity", 0.0) for ao in artifact_observations)
            severe_count = sum(1 for ao in artifact_observations if ao.get("severity", 0.0) >= 2.0)
            minor_count = len(artifact_observations) - severe_count
            base_artifact = 5.0 - max_severity - 0.3 * severe_count - 0.15 * minor_count
            base_artifact = max(0.0, base_artifact)
        else:
            base_artifact = 5.0

        base_scores_text = f"Formula Reference (starting point only, NOT your final score): Alignment={base_alignment:.2f} ({len(present)}/{len(testable)} passed, {len(untestable)} untestable) | Artifact={base_artifact:.2f}"

        router_assessment = f"""
**[Router's Assessment]**
- Image: {image_description}
- Checkpoint Verdicts: {json.dumps(checkpoint_verdicts, indent=2, ensure_ascii=False)}
- Artifact Observations: {json.dumps(artifact_observations, indent=2, ensure_ascii=False)}
- {base_scores_text}
"""

    # Build diagnostic checkpoints context
    checkpoints_text = ""
    if structured_taxonomy_info:
        checkpoints = structured_taxonomy_info.get("diagnostic_checkpoints", {})
        if checkpoints:
            checkpoints_text = f"\n**[Diagnostic Checkpoints]**\n{json.dumps(checkpoints, indent=2, ensure_ascii=False)}\n"

    return f"""Review the Router's assessment and expert evidence to produce the final evaluation.

**[Context]**
- Target Class: {class_label}
- Taxonomy: {taxonomy_desc}
{checkpoints_text}
{router_assessment}
**[Expert Testimonies]**
{expert_results_str}"""


def _calibrate_scores(
    result: dict,
    expert_results: dict,
    router_plan: dict | None = None,
    pose_hard_cap: bool = False,
    enable_classifier_cap: bool = True,
) -> dict:
    """Post-process Reflector output with hard rules that cannot be violated.
    These rules were previously in the prompt but are now enforced in code for reliability.

    Args:
        pose_hard_cap: If True, apply hard caps to artifact_score based on pose low-confidence analysis.
                       If False (default), skip pose-based artifact caps to avoid domain-shift bias.
        enable_classifier_cap: If True (default), cap alignment_score based on fine_grained_classifier
                               Top-1/Top-3 mismatch. If False, trust the Reflector's judgment entirely.
    """
    alignment_score = result.get("alignment_score", 0.0)
    artifact_score = result.get("artifact_score", 0.0)
    adjustments = []

    # --- Alignment calibration based on expert classifier ---
    if enable_classifier_cap and expert_results:
        for t in expert_results.get("expert_testimonies", []):
            if t.get("expert_id") == "fine_grained_classifier" and t.get("status") == "success":
                evidence = t.get("evidence", {})
                top1 = evidence.get("top1_label", "")
                top3 = evidence.get("top3_labels", [])
                class_label = result.get("metadata", {}).get("class_label", "")

                # Top-1 mismatch → alignment ≤ 2.0
                if top1 and top1 != class_label:
                    if alignment_score > 2.0:
                        adjustments.append(f"Classifier Top-1 '{top1}' != target '{class_label}', capping alignment {alignment_score:.2f} → 2.0")
                        alignment_score = 2.0

                # Top-3 mismatch → alignment ≤ 1.0
                if top3 and class_label not in top3:
                    if alignment_score > 1.0:
                        adjustments.append(f"Target '{class_label}' not in Top-3 {top3}, capping alignment {alignment_score:.2f} → 1.0")
                        alignment_score = 1.0
                break

    # --- Artifact calibration based on pose low-confidence ---
    # Only apply when pose_hard_cap is True (disabled by default due to domain-shift concerns)
    if pose_hard_cap and expert_results:
        for t in expert_results.get("expert_testimonies", []):
            if t.get("expert_id") == "animal_pose_auditor" and t.get("status") == "success":
                evidence = t.get("evidence", {})
                lca = evidence.get("low_confidence_analysis", {})
                if lca:
                    risk_level = lca.get("artifact_risk_level", "")
                    low_ratio = lca.get("low_confidence_ratio", 0.0)

                    if risk_level == "HIGH" or low_ratio >= 0.40:
                        if artifact_score > 1.5:
                            adjustments.append(f"Pose HIGH risk (ratio={low_ratio:.2f}), capping artifact {artifact_score:.2f} → 1.5")
                            artifact_score = 1.5
                    elif risk_level == "MEDIUM" or low_ratio >= 0.25:
                        if artifact_score > 2.0:
                            adjustments.append(f"Pose MEDIUM risk (ratio={low_ratio:.2f}), capping artifact {artifact_score:.2f} → 2.0")
                            artifact_score = 2.0
                    elif risk_level == "LOW" or low_ratio >= 0.15:
                        if artifact_score > 3.0:
                            adjustments.append(f"Pose LOW risk (ratio={low_ratio:.2f}), capping artifact {artifact_score:.2f} → 3.0")
                            artifact_score = 3.0
                break

    # Clamp scores to [0, 5]
    alignment_score = max(0.0, min(5.0, alignment_score))
    artifact_score = max(0.0, min(5.0, artifact_score))

    result["alignment_score"] = round(alignment_score, 2)
    result["artifact_score"] = round(artifact_score, 2)
    if adjustments:
        result["code_adjustments"] = adjustments

    return result


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


def _run_self_reflection_round(
    client: OpenAI,
    system_message: dict,  # Round 1 system message (unused; kept for API compatibility)
    user_content: list[dict],
    round1_result: dict,
    api_retry: int = 0,
    temperature: float = 0.5,
) -> dict | None:
    """执行 Round 2 Self-Reflection API 调用。

    利用对话历史：[self_reflection_system, round1_user, round1_assistant, round2_user]
    模型可以看到自己的初步评分并据此修订。
    """
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
            model=REFLECTOR_MODEL,
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
        "artifact_score": round1.get("artifact_score"),
    }

    merged = round2.copy()
    merged["preliminary_scores"] = round1_scores
    merged["self_reflection_notes"] = round2.get("self_reflection_notes", "")

    r1_align = round1.get("alignment_score", 0)
    r2_align = round2.get("alignment_score", 0)
    r1_artifact = round1.get("artifact_score", 0)
    r2_artifact = round2.get("artifact_score", 0)

    if abs(r1_align - r2_align) > 0.01 or abs(r1_artifact - r2_artifact) > 0.01:
        merged["score_changes"] = {
            "alignment_score": f"{r1_align:.2f} → {r2_align:.2f}",
            "artifact_score": f"{r1_artifact:.2f} → {r2_artifact:.2f}",
        }

    return merged


def run_reflector(
    client: OpenAI,
    image_path: str,
    class_id: int,
    class_label: str,
    expert_results: dict,
    experts_registry_str: str,
    router_plan: dict | None = None,
    ref_images: list[dict] | None = None,
    enable_checklist: bool = False,
    enable_self_reflection: bool = True,
    api_retry: int = 0,
    temperature: float = 0.5,
    pose_hard_cap: bool = False,
    enable_classifier_cap: bool = True,
) -> dict | None:
    taxonomy_info = get_taxonomy_info(class_id)
    if taxonomy_info is None:
        print(f"  [WARN] No taxonomy info for class_id={class_id}, proceeding without prior knowledge.")

    structured_taxonomy_info = get_structured_taxonomy_info(class_id)

    expert_results_str = _build_expert_context_str(expert_results, experts_registry_str)

    base64_image = encode_image(image_path)

    ref_images_text = _build_ref_images_text(ref_images) if ref_images else ""

    start_time = time.time()

    prompt = build_reflector_prompt(
        class_label, taxonomy_info, expert_results_str,
        router_plan, structured_taxonomy_info,
    )
    if ref_images_text:
        prompt = prompt + "\n\n" + ref_images_text

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
            print(f"  [WARN] Failed to encode auxiliary image {aux_path}: {e}")

    if ref_images:
        _append_ref_images_to_content(user_content, ref_images)

    system_msg = _REFLECTOR_CHECKLIST_SYSTEM_TEMPLATE if enable_checklist else _REFLECTOR_SYSTEM_TEMPLATE

    system_message = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": system_msg,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }

    try:
        completion = api_call_with_retry(
            client.chat.completions.create,
            model=REFLECTOR_MODEL,
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
            usage_info = f"prompt_tokens={usage.prompt_tokens}, completion_tokens={usage.completion_tokens}" if usage else "no usage info"
            print(f"  [ERROR] Reflector returned empty content (content is {'None' if raw_content is None else 'empty string'}, finish_reason={finish_reason}, {usage_info})")
            if reasoning:
                print(f"  [WARN] Reflector reasoning_content found ({len(reasoning)} chars), attempting to extract JSON")
                raw_content = reasoning
            else:
                msg = completion.choices[0].message
                print(f"  [DEBUG] Reflector full message: content={repr(msg.content)}, role={getattr(msg, 'role', 'N/A')}, function_call={getattr(msg, 'function_call', None)}, tool_calls={getattr(msg, 'tool_calls', None)}, refusal={getattr(msg, 'refusal', None)}")
                return None
        result = parse_json_safely(raw_content)
        if result is None:
            print(f"  [ERROR] Reflector returned unparseable JSON: {raw_content[:300]}")
            return None

        # ── Round 2: Self-Reflection ──
        round2_result = None
        if enable_self_reflection:
            round2_result = _run_self_reflection_round(
                client, system_message, user_content, result,
                api_retry=api_retry, temperature=temperature,
            )
            if round2_result is not None:
                result = _merge_self_reflection(result, round2_result)
                print(f"  [INFO] Self-reflection completed. "
                      f"Alignment: {result.get('alignment_score', 'N/A')}, "
                      f"Artifact: {result.get('artifact_score', 'N/A')}")
            else:
                print(f"  [WARN] Self-reflection round failed, using Round 1 scores")

        result["metadata"] = {
            "original_image": image_path,
            "class_id": class_id,
            "class_label": class_label,
            "taxonomy_available": taxonomy_info is not None,
            "auxiliary_images_included": [Path(p).name for p in aux_images],
            "ref_images_enabled": bool(ref_images),
            "ref_images_included": [r["image_name"] for r in ref_images] if ref_images else [],
            "checklist_enabled": enable_checklist,
            "self_reflection_enabled": enable_self_reflection,
            "self_reflection_succeeded": round2_result is not None if enable_self_reflection else None,
            "reflector_cost_seconds": round(cost_time, 2),
        }
        result = _calibrate_scores(result, expert_results, router_plan,
                                   pose_hard_cap=pose_hard_cap,
                                   enable_classifier_cap=enable_classifier_cap)
        if enable_checklist:
            result = _normalize_checklist_output(result, structured_taxonomy_info)

        return result

    except Exception as e:
        cost_time = time.time() - start_time
        print(f"  [ERROR] Reflector API call failed: {type(e).__name__}: {e}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Checklist Report Normalization & Saving
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_VALID_CHECKLIST_STATUSES = {"🟢 Checked", "🔴 Missing", "⚪ N/A"}


def _normalize_checklist_output(
    result: dict,
    structured_taxonomy_info: dict | None,
) -> dict:
    """Normalize the reflector's fine_grained_details to match the human annotation format.

    Ensures every checkpoint from diagnostic_checkpoints is present with a valid status.
    Fills missing entries with "⚪ N/A" and validates status values.
    """
    if structured_taxonomy_info is None:
        return result

    diagnostic_checkpoints = structured_taxonomy_info.get("diagnostic_checkpoints", {})
    if not diagnostic_checkpoints:
        return result

    raw_fine = result.get("fine_grained_details", {})
    normalized_fine: dict[str, dict[str, str]] = {}

    for category, checkpoint_list in diagnostic_checkpoints.items():
        normalized_fine[category] = {}
        raw_category = raw_fine.get(category, {}) if isinstance(raw_fine, dict) else {}

        for checkpoint_desc in checkpoint_list:
            status = raw_category.get(checkpoint_desc, "⚪ N/A")
            if status not in _VALID_CHECKLIST_STATUSES:
                # Try to coerce common variants
                status_lower = str(status).lower().strip()
                if "checked" in status_lower or "present" in status_lower or status_lower == "🟢":
                    status = "🟢 Checked"
                elif "missing" in status_lower or "absent" in status_lower or status_lower == "🔴":
                    status = "🔴 Missing"
                else:
                    status = "⚪ N/A"
            normalized_fine[category][checkpoint_desc] = status

    result["fine_grained_details"] = normalized_fine

    if "veto_activated" not in result:
        result["veto_activated"] = False
    result["veto_activated"] = bool(result["veto_activated"])

    return result


def build_checklist_annotation(
    report: dict,
    class_id: int,
    class_label: str,
    image_name: str,
) -> dict:
    """Build a human-annotation-style record from the reflector report.

    Output format matches small_scale_audit_recorrect/output_results/User_*_final_annotations.json

    Scoring rules (mimicking human annotators):
    - alignment_score: 5 * Checked / (Checked + Missing), rounded to 2 decimals
    - artifact_score: integer 0-5, derived from the reflector's artifact_score
      but quantized to the nearest integer to mimic human discrete scoring
    """
    fine_grained = report.get("fine_grained_details", {})
    checked_count = 0
    missing_count = 0
    for category_items in fine_grained.values():
        if not isinstance(category_items, dict):
            continue
        for status in category_items.values():
            if status == "\U0001f7e2 Checked":
                checked_count += 1
            elif status == "\U0001f534 Missing":
                missing_count += 1

    if checked_count + missing_count > 0:
        checklist_alignment = round(5.0 * checked_count / (checked_count + missing_count), 2)
    else:
        checklist_alignment = 0.0

    reflector_artifact = float(report.get("artifact_score", 0.0))
    checklist_artifact = min(5, max(0, round(reflector_artifact)))

    total_score = round(checklist_alignment * checklist_artifact, 2)

    return {
        "image_name": image_name,
        "class_id": class_id,
        "class_name": class_label,
        "veto_activated": bool(report.get("veto_activated", False)),
        "scores": {
            "alignment_score": checklist_alignment,
            "artifact_score": checklist_artifact,
            "total_score": total_score,
        },
        "fine_grained_details": fine_grained,
    }


def save_checklist_annotation(
    annotation: dict,
    output_dir: str | Path | None = None,
) -> str:
    """Save a single image's checklist annotation as a JSON file."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / os.environ.get("C2I_OUTPUT_DIR_NAME", "output") / "checklist_annotations"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_name = annotation.get("image_name", "unknown")
    image_id = os.path.splitext(image_name)[0]
    filename = f"checklist_{image_id}.json"
    filepath = output_dir / filename

    safe_annotation = _sanitize_evidence(annotation)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(safe_annotation, f, indent=4, ensure_ascii=False)

    print(f"  [SAVED] {filename}")
    return str(filepath)


def save_final_report(
    report: dict,
    output_dir: str | Path | None = None,
) -> str:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / os.environ.get("C2I_OUTPUT_DIR_NAME", "output") / "final_reports"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = report.get("metadata", {})
    image_id = "unknown"
    original_image = metadata.get("original_image", "")
    if original_image:
        image_id = Path(original_image).stem

    filename = f"final_evaluation_report_{image_id}.json"
    filepath = output_dir / filename

    safe_report = _sanitize_evidence(report)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(safe_report, f, indent=4, ensure_ascii=False)

    print(f"  [SAVED] {filename}")
    return str(filepath)


def print_final_summary(report: dict) -> None:
    metadata = report.get("metadata", {})
    class_label = metadata.get("class_label", "N/A")
    alignment_score = report.get("alignment_score", 0.0)
    artifact_score = report.get("artifact_score", 0.0)
    key_defects = report.get("key_defects", [])
    code_adjustments = report.get("code_adjustments", [])

    print(f"\n--- Final Evaluation Complete ---")
    print(f"Class: {class_label} | Alignment: {alignment_score:.1f}/5.0 | Artifact: {artifact_score:.1f}/5.0")
    if code_adjustments:
        for adj in code_adjustments:
            print(f"  [Code Adjustment] {adj}")
    if key_defects:
        print(f"Key Defects: {', '.join(key_defects)}")

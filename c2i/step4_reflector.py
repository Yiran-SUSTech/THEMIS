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
REFLECTOR_MODEL = "qwen3.6-plus"


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
        "You MUST independently verify areas that experts cannot cover. "
        "Do NOT assume the image is defect-free just because all experts reported success."
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


_REFLECTOR_SYSTEM_TEMPLATE = r"""You are the Supreme Judge (Reflector) of an AI image evaluation system. You must follow the 4-Stage Cognition Chain strictly: independent visual audit, evidence cross-examination, self-reflection, and final verdict. You must prioritize the Taxonomy Ground Truth as the source of truth. Output JSON only, no markdown wrapping.

**[Core Priority Laws]**
1. Your Own Eyes > Expert Silence: If YOU see a defect but no expert flagged it, trust your eyes — experts have blind spots and may simply be unable to detect that type of defect.
2. Anatomy/Topology > Aesthetics: Structure defects → penalize Artifact Score heavily regardless of visual appeal.
3. Taxonomy Compliance: Mismatch in fine-grained features or Top-1 classification → lower Alignment Score.
4. Expert Evidence for Alignment, Your Eyes for Artifact: Expert evidence is most useful for Alignment (classifier identity verification). For Artifact detection, your own visual inspection is the PRIMARY instrument — experts are supplementary.

**[Strict Adjudication Principles — GUILTY UNTIL PROVEN INNOCENT]**
You must adopt a presumption-of-defect stance. AI-generated images are assumed to contain flaws unless you can conclusively prove otherwise through your own visual inspection.

- **Subject Scrutiny:** Examine the main subject exhaustively — anatomy, proportions, limb count, digit structure, eye symmetry, fur/feather/scale texture continuity. ANY anomaly, even subtle, must be penalized.
- **Background Scrutiny:** Check for: impossible geometry, melting textures, duplicated elements, semantic inconsistencies (e.g., indoor furniture in outdoor scene), blurred or fused boundaries between subject and background.
- **Alignment Strictness:** The image must match the target class at a FINE-GRAINED level, not just superficially. If expert classifier Top-1 does not match the target class, alignment_score MUST be ≤ 2.0. If Top-3 does not contain the target class, alignment_score MUST be ≤ 1.0.
- **Artifact Strictness:** Default artifact_score starts at 2.0 (presumed flawed). You may only raise it above 2.0 if ALL of the following are met:
  (a) No expert flagged any structural/topological defect.
  (b) Pose auditor found no anatomical anomalies.
  (c) Depth/segmentation boundaries are clean and consistent.
  (d) Perceptual quality auditor confirmed no distortion.
  (e) YOUR OWN visual inspection found NO defects in subject, background, or boundaries.
  If ANY expert reports a defect, artifact_score MUST be ≤ 2.0. If multiple experts report defects, artifact_score MUST be ≤ 1.0.
- **Pose Low-Confidence Rule (CRITICAL):** When animal_pose_auditor evidence contains a `low_confidence_analysis` field, you MUST use it as a primary artifact indicator:
  (a) Low confidence keypoints (confidence < 0.5) mean the pose model is UNCERTAIN about that body part — this is a strong signal of potential artifact (melting, fusion, hallucination, or structural breakdown) in that region.
  (b) If `artifact_risk_level` is HIGH (>=40% low-confidence ratio), artifact_score MUST be ≤ 1.5.
  (c) If `artifact_risk_level` is MEDIUM (>=25% low-confidence ratio), artifact_score MUST be ≤ 2.0.
  (d) If `artifact_risk_level` is LOW (>=15% low-confidence ratio), artifact_score MUST be ≤ 3.0.
  (e) You MUST cite the affected body regions from `affected_body_regions` in your artifact_reasoning.
  (f) Do NOT dismiss low-confidence keypoints as 'model limitation' — the pose model's uncertainty IS the diagnostic signal.
- **Expert Blind Spot Awareness (CRITICAL):** Each expert has blind spots (see NOT_Capable_Of in their testimony). 'No defect reported' does NOT mean 'no defect exists'. Experts can only detect defects within their narrow specialty. You MUST independently verify:
  (a) Anatomical features that the classifier cannot verify (e.g., are barbels visible? is the lateral line continuous? are all fins correctly shaped?)
  (b) Subject-background fusion that SAM cannot detect when colors are similar
  (c) Structural artifacts in body regions where pose keypoints have low confidence
  (d) Background semantic consistency that no expert specifically checks
  (e) Subject-background boundary quality beyond what mask confidence measures
- **Independent Visual Verification:** In Stage 1, you must perform your OWN thorough visual inspection BEFORE reading expert data. In Stage 2, you must NOT let expert 'no defect' reports override your own visual findings. If you see a potential defect with your own eyes but no expert flagged it, trust your eyes — the expert may simply be blind to that type of defect.
- **Whole-Image Audit:** You must audit the ENTIRE image, not just the main subject. Background, secondary objects, and subject-background boundaries are all artifact-prone zones. Check:
  (a) Background geometry: impossible structures, melting textures, duplicated elements
  (b) Background semantics: out-of-place objects, inconsistent lighting
  (c) Subject-background boundary: fusion, melting, halo artifacts, unnatural transitions
  (d) Secondary subjects: anatomical correctness of all visible entities, not just the primary subject
- **No Free Passes:** Do NOT give high scores simply because the image 'looks nice overall' or 'most parts are fine'. A single confirmed defect in a critical area (face, limbs, subject boundary) is sufficient to cap the score.
- **Score Calibration Guide:**
  - 5.0: Flawless — no detectable defects from any expert or visual inspection (extremely rare).
  - 4.0: Minor imperfections only — no structural defects, only cosmetic issues.
  - 3.0: Moderate defects — some expert flags but not catastrophic.
  - 2.0: Notable defects — multiple expert flags or one severe structural issue.
  - 1.0: Severe defects — major anatomical/structural failure.
  - 0.0: Catastrophic — image is nonsensical or completely misaligned.

**[Output Requirements]**
Return ONLY a pure JSON object (no markdown, no extra text). Execute and document each stage inside the respective JSON fields:
{
  "stage1_independent_visual_audit": {
    "alignment_thought": "BEFORE reading expert data, independently analyze the image. Does it match 'TARGET_CLASS' at a FINE-GRAINED level? Check EVERY key feature from the taxonomy (not just overall appearance). List features that match AND features that are missing or wrong. (tentative score 0-5 & reason)",
    "artifact_thought": "BEFORE reading expert data, scan the ENTIRE image for artifacts with a FAULT-FINDING mindset. Assume the image IS flawed and your job is to find every flaw. Check THREE zones: (A) Subject — anatomy, proportions, texture continuity, limb/organ correctness, eye symmetry, skin/fur/scale quality; (B) Background — geometry, texture, semantic consistency, duplicated elements, impossible structures; (C) Subject-Background Boundary — fusion, melting, halo, unnatural transitions, color bleeding. For EACH zone, list EVERY potential defect you find, no matter how subtle. If a zone appears clean, explicitly state what you checked and confirmed. (tentative score 0-5 & reason)"
  },
  "stage2_evidence_cross_examination": {
    "expert_vs_intuition": "Compare expert hard data with your Stage 1 assessment. CRITICAL RULE for Artifact: If you found defects in Stage 1 but experts report 'no defect', do NOT let experts override your visual findings — experts have blind spots (see NOT_Capable_Of). Expert 'no defect' only means the expert's specific tool could not detect a defect, NOT that no defect exists. For Alignment, expert classifier evidence is more reliable and can adjust your score.",
    "alignment_adjustment": "How expert evidence changes your alignment assessment (if at all). Classifier evidence is strong signal for alignment.",
    "artifact_adjustment": "How expert evidence changes your artifact assessment. Remember: expert silence on artifacts does NOT override your own visual findings. Only EXPLICIT expert defect reports (e.g., low-confidence keypoints, boundary anomalies) should lower the score further. Your Stage 1 visual findings are the PRIMARY artifact evidence."
  },
  "stage3_self_reflection": {
    "critique_and_calibration": "Challenge your conclusion from the STRICT side. For each score, you must argue why it should NOT be LOWER, not why it should not be higher. If you found defects in Stage 1, did you fully account for them in the final score? Did you let expert 'no defect' reports weaken your own visual findings?",
    "bias_check": "Did you give artificially high scores because the image 'looks nice'? Did you let expert 'no defect' reports override your own eyes? List every defect you found in Stage 1 and explain whether you gave it appropriate weight in the final score. If you raised a score above Stage 1's tentative score, justify exactly why — the default direction should be downward, not upward.",
    "final_calibration": "Explicit statement of how Stage 1 tentative scores are adjusted. The final scores should be <= Stage 1 tentative scores UNLESS expert evidence provides POSITIVE proof (not just absence of negative evidence) that the image is better than Stage 1 assessment."
  },
  "stage4_final_verdict": {
    "alignment_score": 0.0,
    "artifact_score": 0.0,
    "alignment_reasoning": "Concise definitive logic explaining the final alignment score.",
    "artifact_reasoning": "Concise definitive logic explaining the final artifact score. You MUST cite: (1) defects you found with your own eyes, (2) expert evidence that supports or contradicts your findings, (3) why you trusted your eyes or the expert in case of contradiction.",
    "hard_failure_triggered": false,
    "key_defects": ["string (e.g., Extra_Limbs, Edge_Melting, Perspective_Warp, Identity_Mismatched)"]
  }
}"""


def build_reflector_prompt(
    class_label: str,
    taxonomy_info: dict | None,
    expert_results_str: str,
) -> str:
    taxonomy_desc = taxonomy_info.get("enriched_description", "No specific taxonomy found.") if taxonomy_info else "No specific taxonomy found."

    return f"""Proceed through the 4-Stage Cognition Chain defined in the system context.

**[Context]**
- Target Class: {class_label}
- Taxonomy Ground Truth: {taxonomy_desc}
- Expert Testimonies: {expert_results_str}"""


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


def _build_reflector_user_prompt_session(
    class_label: str,
    taxonomy_info: dict | None,
    expert_results_str: str,
) -> str:
    taxonomy_desc = taxonomy_info.get("enriched_description", "No specific taxonomy found.") if taxonomy_info else "No specific taxonomy found."

    return f"""[Reflector Role] Based on the expert results below, proceed through the 4-Stage Cognition Chain defined in your Reflector Role instructions.

**[Context]**
- Target Class: {class_label}
- Taxonomy Ground Truth: {taxonomy_desc}
- Expert Testimonies: {expert_results_str}"""


def run_reflector(
    client: OpenAI,
    image_path: str,
    class_id: int,
    class_label: str,
    expert_results: dict,
    experts_registry_str: str,
    session=None,
) -> dict | None:
    taxonomy_info = get_taxonomy_info(class_id)
    if taxonomy_info is None:
        print(f"  [WARN] No taxonomy info for class_id={class_id}, proceeding without prior knowledge.")

    expert_results_str = _build_expert_context_str(expert_results, experts_registry_str)

    base64_image = encode_image(image_path)

    start_time = time.time()

    if session is not None:
        prompt = _build_reflector_user_prompt_session(class_label, taxonomy_info, expert_results_str)
        user_content = [{"type": "text", "text": prompt}]

        aux_images = _collect_auxiliary_images(expert_results)
        for aux_path in aux_images:
            try:
                aux_b64 = encode_image(aux_path)
                aux_label = Path(aux_path).stem
                user_content.append({"type": "text", "text": f"[Auxiliary Expert Output Image: {aux_label}]"})
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{aux_b64}"}})
            except Exception as e:
                print(f"  [WARN] Failed to encode auxiliary image {aux_path}: {e}")

        session.add_user(user_content)
        raw_content, completion = session.call_api(
            client, REFLECTOR_MODEL, response_format={"type": "json_object"},
        )
        cost_time = time.time() - start_time

        result = parse_json_safely(raw_content)
        if result is None:
            print(f"  [ERROR] Reflector returned unparseable JSON: {raw_content[:300]}")
            return None

        result["metadata"] = {
            "original_image": image_path,
            "class_id": class_id,
            "class_label": class_label,
            "taxonomy_available": taxonomy_info is not None,
            "auxiliary_images_included": [Path(p).name for p in _collect_auxiliary_images(expert_results)],
            "reflector_cost_seconds": round(cost_time, 2),
            "session_turn_count": session.turn_count,
        }
        return result

    prompt = build_reflector_prompt(class_label, taxonomy_info, expert_results_str)

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

    system_msg = _REFLECTOR_SYSTEM_TEMPLATE

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
        completion = client.chat.completions.create(
            model=REFLECTOR_MODEL,
            messages=[
                system_message,
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw_content = completion.choices[0].message.content
        cost_time = time.time() - start_time

        result = parse_json_safely(raw_content)
        if result is None:
            print(f"  [ERROR] Reflector returned unparseable JSON: {raw_content[:300]}")
            return None

        result["metadata"] = {
            "original_image": image_path,
            "class_id": class_id,
            "class_label": class_label,
            "taxonomy_available": taxonomy_info is not None,
            "auxiliary_images_included": [Path(p).name for p in aux_images],
            "reflector_cost_seconds": round(cost_time, 2),
        }

        return result

    except Exception as e:
        cost_time = time.time() - start_time
        print(f"  [ERROR] Reflector API call failed: {e}")
        return None


def save_final_report(
    report: dict,
    output_dir: str | Path | None = None,
) -> str:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "output" / "final_reports"

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
    verdict = report.get("stage4_final_verdict", {})
    metadata = report.get("metadata", {})
    class_label = metadata.get("class_label", "N/A")
    alignment_score = verdict.get("alignment_score", 0.0)
    artifact_score = verdict.get("artifact_score", 0.0)
    hard_failure = verdict.get("hard_failure_triggered", False)
    key_defects = verdict.get("key_defects", [])

    print(f"\n--- Final Evaluation Complete ---")
    print(f"Class: {class_label} | Alignment: {alignment_score:.1f}/5.0 | Artifact: {artifact_score:.1f}/5.0 | Hard Failure: {hard_failure}")
    if key_defects:
        print(f"Key Defects: {', '.join(key_defects)}")

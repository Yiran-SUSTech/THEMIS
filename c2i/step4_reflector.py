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
TAXONOMY_STRUCTURAL_DIR = PROJECT_ROOT / "taxonomy_info_structural"
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
) -> dict:
    """Post-process Reflector output with hard rules that cannot be violated.
    These rules were previously in the prompt but are now enforced in code for reliability."""
    alignment_score = result.get("alignment_score", 0.0)
    artifact_score = result.get("artifact_score", 0.0)
    adjustments = []

    # --- Alignment calibration based on expert classifier ---
    if expert_results:
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
    if expert_results:
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


def _build_reflector_user_prompt_session(
    class_label: str,
    taxonomy_info: dict | None,
    expert_results_str: str,
    router_plan: dict | None = None,
    structured_taxonomy_info: dict | None = None,
) -> str:
    taxonomy_desc = taxonomy_info.get("enriched_description", "No specific taxonomy found.") if taxonomy_info else "No specific taxonomy found."

    # Build Router's assessment context (compact for session mode)
    router_assessment = ""
    if router_plan:
        checkpoint_verdicts = router_plan.get("checkpoint_verdicts", [])
        artifact_observations = router_plan.get("artifact_observations", [])

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

        router_assessment = f"""
**[Router's Assessment]**
- Checkpoint Verdicts: {json.dumps(checkpoint_verdicts, indent=2, ensure_ascii=False)}
- Artifact Observations: {json.dumps(artifact_observations, indent=2, ensure_ascii=False)}
- Formula Reference (starting point only): Alignment={base_alignment:.2f} ({len(present)}/{len(testable)} passed, {len(untestable)} untestable) | Artifact={base_artifact:.2f}
"""

    checkpoints_text = ""
    if structured_taxonomy_info:
        checkpoints = structured_taxonomy_info.get("diagnostic_checkpoints", {})
        if checkpoints:
            checkpoints_text = f"\n**[Diagnostic Checkpoints]**\n{json.dumps(checkpoints, indent=2, ensure_ascii=False)}\n"

    return f"""[Reflector Role] Review the Router's assessment and expert evidence.

**[Context]**
- Target Class: {class_label}
- Taxonomy: {taxonomy_desc}
{checkpoints_text}
{router_assessment}
**[Expert Testimonies]**
{expert_results_str}"""


def run_reflector(
    client: OpenAI,
    image_path: str,
    class_id: int,
    class_label: str,
    expert_results: dict,
    experts_registry_str: str,
    session=None,
    router_plan: dict | None = None,
) -> dict | None:
    taxonomy_info = get_taxonomy_info(class_id)
    if taxonomy_info is None:
        print(f"  [WARN] No taxonomy info for class_id={class_id}, proceeding without prior knowledge.")

    structured_taxonomy_info = get_structured_taxonomy_info(class_id)

    expert_results_str = _build_expert_context_str(expert_results, experts_registry_str)

    base64_image = encode_image(image_path)

    start_time = time.time()

    if session is not None:
        prompt = _build_reflector_user_prompt_session(
            class_label, taxonomy_info, expert_results_str,
            router_plan, structured_taxonomy_info,
        )
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
        result = _calibrate_scores(result, expert_results, router_plan)
        return result

    prompt = build_reflector_prompt(
        class_label, taxonomy_info, expert_results_str,
        router_plan, structured_taxonomy_info,
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
        result = _calibrate_scores(result, expert_results, router_plan)

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

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


def _build_expert_context_str(
    expert_results: dict,
    experts_registry_str: str,
) -> str:
    registry_list = json.loads(experts_registry_str)
    registry_map = {e["expert_id"]: e for e in registry_list}

    testimonies = expert_results.get("expert_testimonies", [])
    custom_prompts = expert_results.get("custom_prompts_for_reflector", "")
    focus_areas = expert_results.get("focus_areas", [])

    parts = []

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

        block = f"--- Expert: {eid} ---\nTarget: \"{target_subject}\" | Weight: {weight} | Specialty: {expert_specialty}\nDiagnostic Criteria: {json.dumps(diagnostic_criteria, ensure_ascii=False)}\nStatus: {status}"

        if status == "success":
            evidence_clean = _sanitize_evidence(evidence)
            block += f"\nEvidence:\n{json.dumps(evidence_clean, indent=2, ensure_ascii=False)}"
        else:
            block += f"\nError: {error or 'Unknown error'}"

        parts.append(block)

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


_REFLECTOR_SYSTEM_TEMPLATE = """You are the Supreme Judge (Reflector) of an AI image evaluation system. You must follow the 4-Stage Cognition Chain strictly: independent visual audit, evidence cross-examination, self-reflection, and final verdict. You must prioritize the Taxonomy Ground Truth as the source of truth. Output JSON only, no markdown wrapping.

**[Core Priority Laws]**
1. Anatomy/Topology > Aesthetics: Structure defects flagged by experts → penalize Artifact Score heavily regardless of visual appeal.
2. Taxonomy Compliance: Mismatch in fine-grained features or Top-1 classification → lower Alignment Score.
3. Expert Evidence Hierarchy: Weigh evidence proportional to expert weight in the plan.

**[Strict Adjudication Principles — GUILTY UNTIL PROVEN INNOCENT]**
You must adopt a presumption-of-defect stance. AI-generated images are assumed to contain flaws unless expert evidence conclusively proves otherwise.

- **Subject Scrutiny:** Examine the main subject exhaustively — anatomy, proportions, limb count, digit structure, eye symmetry, fur/feather/scale texture continuity. ANY anomaly, even subtle, must be penalized.
- **Background Scrutiny:** Check for: impossible geometry, melting textures, duplicated elements, semantic inconsistencies (e.g., indoor furniture in outdoor scene), blurred or fused boundaries between subject and background.
- **Alignment Strictness:** The image must match the target class at a FINE-GRAINED level, not just superficially. If expert classifier Top-1 does not match the target class, alignment_score MUST be ≤ 2.0. If Top-3 does not contain the target class, alignment_score MUST be ≤ 1.0.
- **Artifact Strictness:** Default artifact_score starts at 2.0 (presumed flawed). You may only raise it above 2.0 if ALL of the following are met:
  (a) No expert flagged any structural/topological defect.
  (b) Pose auditor found no anatomical anomalies.
  (c) Depth/segmentation boundaries are clean and consistent.
  (d) Perceptual quality auditor confirmed no distortion.
  If ANY expert reports a defect, artifact_score MUST be ≤ 2.0. If multiple experts report defects, artifact_score MUST be ≤ 1.0.
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
{{
  "stage1_independent_visual_audit": {{
    "alignment_thought": "BEFORE reading expert data, independently analyze the image. Does it match 'TARGET_CLASS'? (tentative score 0-5 & reason)",
    "artifact_thought": "Scan for visible hallucinations, melting, or blurring. (tentative score 0-5 & reason)"
  }},
  "stage2_evidence_cross_examination": {{
    "expert_vs_intuition": "Compare expert hard data with your Stage 1 assessment. Resolve contradictions — explain which you trust and why.",
    "alignment_adjustment": "How expert evidence changes your alignment assessment (if at all)",
    "artifact_adjustment": "How expert evidence changes your artifact assessment (if at all)"
  }},
  "stage3_self_reflection": {{
    "critique_and_calibration": "Challenge your blended conclusion. Did you bias toward leniency? Did you ignore an expert warning? You must explicitly justify why each score is NOT lower.",
    "bias_check": "Did you give artificially high scores because the image 'looks nice'? Are there logical contradictions? List every defect you considered giving a pass on and explain why you did or did not.",
    "final_calibration": "Explicit statement of how Stage 1 tentative scores are adjusted after Stages 2 and 3"
  }},
  "stage4_final_verdict": {{
    "alignment_score": 0.0,
    "artifact_score": 0.0,
    "alignment_reasoning": "Concise definitive logic explaining the final alignment score.",
    "artifact_reasoning": "Concise definitive logic explaining the final artifact score, citing which expert evidence was adopted or overridden.",
    "hard_failure_triggered": false,
    "key_defects": ["string (e.g., Extra_Limbs, Edge_Melting, Perspective_Warp, Identity_Mismatched)"]
  }}
}}"""


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


def run_reflector(
    client: OpenAI,
    image_path: str,
    class_id: int,
    class_label: str,
    expert_results: dict,
    experts_registry_str: str,
) -> dict | None:
    taxonomy_info = get_taxonomy_info(class_id)
    if taxonomy_info is None:
        print(f"  [WARN] No taxonomy info for class_id={class_id}, proceeding without prior knowledge.")

    expert_results_str = _build_expert_context_str(expert_results, experts_registry_str)
    prompt = build_reflector_prompt(class_label, taxonomy_info, expert_results_str)

    base64_image = encode_image(image_path)

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

    start_time = time.time()
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

        usage = getattr(completion, "usage", None)
        if usage:
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details else 0
            created = getattr(details, "cache_creation_input_tokens", 0) if details else 0
            if cached or created:
                print(f"  [CACHE] Reflector: hit={cached} tokens, created={created} tokens")

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

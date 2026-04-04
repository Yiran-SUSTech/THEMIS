from __future__ import annotations

from typing import List, Optional


def build_task_context(prompt: str | None, class_label: str | None) -> str:
    return (
        f"Prompt: {prompt or 'N/A'}\n"
        f"Class label: {class_label or 'N/A'}\n"
        "Evaluate image quality for generative model benchmarking. Treat the prompt, class label, and visible image evidence as one joint multimodal grounding problem. Focus on alignment and artifact severity."
    )


PLANNER_SYSTEM = """You are a rigorous multimodal evaluation planner for image generation assessment. Jointly reason from the image, the prompt, and the class label as one grounded task. Prefer the cheapest adequate evaluation route first and reserve stronger remote models for unresolved or genuinely hard cases. When planning structural inspection, adapt the checks to the subject and task rather than using a rigid checklist; explicitly consider likely structural or anatomical failures such as facial melting, extra or fused fingers, extra limbs, wrong tail attachment, melted hands or feet, broken joints, or impossible part boundaries when they are relevant to the image. Output only valid JSON matching the requested schema."""

JUDGE_SYSTEM = """You review evaluation plans for coverage, order, sufficiency, and whether they reason jointly from the image and the prompt/class label. Check that any structural inspection is adapted to the specific subject and likely failure modes instead of using a rote fixed checklist. Output only valid JSON matching the requested schema."""

EXPERT_SYSTEM = """You are a careful but conservative image evaluation expert. Return grounded evidence only. For structural analysis, focus on subject-specific anatomical and part-attachment integrity, and when evidence is ambiguous, prefer treating the ambiguity as a possible failure risk rather than assuming the image is correct. For artifact analysis, treat extra appendages, impossible limbs, malformed extremities, duplicated parts, broken joints, and impossible attachments as severe failures rather than minor issues whenever visible evidence supports them. Output only valid JSON matching the requested schema."""


def build_expert_reliability_context(
    expert_results: Optional[List[dict]] = None,
    include_performance_table: bool = True
) -> str:
    context_parts = []
    
    if include_performance_table:
        context_parts.append("""
## Expert Model Reliability Reference

When evaluating expert results, consider their known performance characteristics:

| Expert Type | Model | Benchmark Performance | Reliability | Weight |
|-------------|-------|----------------------|-------------|--------|
| ImageNet Classification | EfficientNetV2-S | 83.8% Top-1 | HIGH | 1.0 |
| ImageNet Classification | EfficientNetV2-L | 85.4% Top-1 | HIGH | 1.0 |
| Semantic Alignment | CLIP-ViT-B/32 | 70% zero-shot | HIGH | 1.0 |
| Semantic Alignment | CLIP-ViT-L/14 | 75% zero-shot | HIGH | 1.0 |
| Pose Estimation | YOLO11n-pose | 50.0% mAP | MEDIUM | 0.7 |
| Pose Estimation | YOLO11m-pose | 64.9% mAP | HIGH | 1.0 |
| Pose Estimation | YOLO11x-pose | 69.5% mAP | HIGH | 1.0 |
| Hand Detection | HADM-L | Specialized | MEDIUM | 0.7 |
| Scene Classification | Places365-ResNet18 | 85% Top-5 | HIGH | 1.0 |
| Image Quality | MANIQA | 0.875 SRCC | HIGH | 1.0 |
| Image Quality | MUSIQ | 0.860 SRCC | HIGH | 1.0 |
| AI Detection | DistilDIRE | 99% accuracy | HIGH | 1.0 |
| Background | RMBG-2.0 | 90.1% accuracy | HIGH | 1.0 |
| VQA | Qwen2.5-VL-3B | 96.4% DocVQA | HIGH | 1.0 |
| OCR | PaddleOCR | Industrial-grade | HIGH | 1.0 |

**Reliability Guidelines:**
- HIGH reliability (weight 1.0): Trust conclusions unless directly contradicted by visual evidence
- MEDIUM reliability (weight 0.7): Consider results but seek confirmation from other sources
- LOW reliability (weight 0.4): Use as hints only, require strong supporting evidence
- UNKNOWN reliability (weight 0.5): Treat conservatively
""")
    
    if expert_results:
        context_parts.append("\n## Current Expert Results with Reliability\n")
        
        for result in expert_results:
            expert_name = result.get("expert", "unknown")
            reliability = result.get("reliability", "unknown")
            confidence_weight = result.get("confidence_weight", 0.5)
            severity = result.get("severity", 0.0)
            confidence = result.get("confidence", 0.0)
            
            context_parts.append(
                f"- **{expert_name}**: Reliability={reliability.upper()}, "
                f"Weight={confidence_weight:.1f}, "
                f"Reported Severity={severity:.2f}, "
                f"Confidence={confidence:.2f}"
            )
    
    return "\n".join(context_parts)


def build_conflict_analysis_context(conflicts: Optional[List[dict]] = None) -> str:
    if not conflicts:
        return ""
    
    context_parts = ["\n## Detected Expert Conflicts\n"]
    context_parts.append("The following conflicts were detected between expert results. ")
    context_parts.append("When resolving conflicts, prioritize experts with higher reliability:\n")
    
    for conflict in conflicts:
        experts = conflict.get("experts", [])
        severity_diff = conflict.get("severity_difference", 0.0)
        recommended = conflict.get("recommended_trust", "")
        reason = conflict.get("reason", "")
        
        context_parts.append(f"- **Conflict**: {' vs '.join(experts)}")
        context_parts.append(f"  - Severity difference: {severity_diff:.2f}")
        context_parts.append(f"  - Recommended to trust: {recommended}")
        context_parts.append(f"  - Reason: {reason}")
    
    return "\n".join(context_parts)


REPORT_SYSTEM = """You synthesize expert evidence into an evaluation report, but you are not bound by the experts if the image itself suggests they missed a serious failure. Directly inspect the image again while reading the expert outputs. Use a conservative standard: when visible evidence suggests species mismatch, impossible anatomy, extra appendages, malformed extremities, duplicated limbs, broken joints, wrong tail attachment, or severe boundary corruption, lower the scores accordingly even if some experts were optimistic. Treat artifact_score as a quality score where 1 means minimal visible artifacts and 0 means severe visible artifacts. Set hard_failure true when the image shows severe species mismatch or severe anatomical / structural generation failure.

**IMPORTANT - Expert Reliability Weighting:**
You will receive reliability information for each expert. When synthesizing results:
1. Weight expert findings by their reliability (HIGH=1.0, MEDIUM=0.7, LOW=0.4)
2. If experts with HIGH reliability report issues, trust them strongly
3. If experts with LOW/MEDIUM reliability report issues, seek confirmation from visual evidence
4. When experts conflict, prioritize the one with higher reliability
5. Calculate weighted severity: sum(severity * weight) / sum(weights)
6. Adjust overall confidence based on expert reliability distribution

Output only valid JSON matching the requested schema."""


REFLECTOR_SYSTEM = """You are a second-pass image critic. Reinspect the image directly instead of only checking internal consistency. Your job is to find serious failures that earlier experts or the report may have missed, especially species mismatch, extra appendages, impossible limbs, malformed extremities, duplicated parts, broken joints, wrong tail attachment, impossible anatomy, and severe boundary corruption. If the report is too optimistic relative to visible evidence, reject it and request replanning or re-evaluation.

**IMPORTANT - Expert Reliability Consideration:**
When reviewing the report, consider expert model reliability:

1. **Check Reliability Weights**: Each expert result includes a reliability level and confidence weight
   - HIGH reliability (weight 1.0): EfficientNet, CLIP, MANIQA, DistilDIRE, RMBG-2.0, Qwen-VL
   - MEDIUM reliability (weight 0.7): YOLO11n-pose, HADM-L, smaller/specialized models
   - LOW reliability (weight 0.4): Experimental or unverified models
   - UNKNOWN reliability (weight 0.5): Default for unrecognized models

2. **Handle Conflicting Results**: When experts disagree:
   - Trust higher-reliability experts over lower-reliability ones
   - If a HIGH reliability expert reports severe issues, trust it even if others disagree
   - If only LOW reliability experts report issues, verify with direct visual inspection

3. **Adjust Report Confidence**: 
   - If report relies heavily on LOW reliability experts, flag for additional verification
   - If HIGH reliability experts agree, report confidence should be high
   - If HIGH reliability experts conflict, note the uncertainty

4. **Detect Potential Issues**:
   - If a HIGH reliability expert's finding was downplayed in the report, flag it
   - If the report ignores clear issues from reliable experts, reject it
   - Consider whether expert limitations (e.g., pose model on non-human subjects) affected results

5. **Recommend Actions**:
   - If LOW reliability results are critical, suggest calling HIGH reliability experts
   - If expert domain doesn't match image content, note the mismatch
   - If weighted severity significantly differs from reported severity, flag the discrepancy

Output only valid JSON matching the requested schema."""


def get_reflector_prompt_with_reliability(
    expert_results: Optional[List[dict]] = None,
    conflicts: Optional[List[dict]] = None,
    reliability_summary: Optional[dict] = None
) -> str:
    prompt_parts = [REFLECTOR_SYSTEM]
    
    prompt_parts.append(build_expert_reliability_context(expert_results))
    
    if conflicts:
        prompt_parts.append(build_conflict_analysis_context(conflicts))
    
    if reliability_summary:
        prompt_parts.append("\n## Expert Reliability Summary\n")
        prompt_parts.append(f"- HIGH reliability experts: {reliability_summary.get('high_reliability_count', 0)}")
        prompt_parts.append(f"- MEDIUM reliability experts: {reliability_summary.get('medium_reliability_count', 0)}")
        prompt_parts.append(f"- LOW reliability experts: {reliability_summary.get('low_reliability_count', 0)}")
        prompt_parts.append(f"- UNKNOWN reliability experts: {reliability_summary.get('unknown_reliability_count', 0)}")
        prompt_parts.append(f"- Weighted severity: {reliability_summary.get('weighted_severity', 0.0):.3f}")
        prompt_parts.append(f"- Overall confidence: {reliability_summary.get('overall_confidence', 0.7):.2f}")
    
    return "\n".join(prompt_parts)


def get_report_prompt_with_reliability(
    expert_results: Optional[List[dict]] = None,
    conflicts: Optional[List[dict]] = None
) -> str:
    prompt_parts = [REPORT_SYSTEM]
    
    prompt_parts.append(build_expert_reliability_context(expert_results))
    
    if conflicts:
        prompt_parts.append(build_conflict_analysis_context(conflicts))
    
    return "\n".join(prompt_parts)

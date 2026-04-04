from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class ReliabilityLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class ExpertPerformanceMetrics:
    expert_name: str
    model_name: str
    task_type: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mAP: Optional[float] = None
    srcc: Optional[float] = None
    plcc: Optional[float] = None
    benchmark: Optional[str] = None
    reliability: ReliabilityLevel = ReliabilityLevel.MEDIUM
    notes: str = ""
    source: str = ""
    
    @property
    def confidence_weight(self) -> float:
        if self.reliability == ReliabilityLevel.HIGH:
            return 1.0
        elif self.reliability == ReliabilityLevel.MEDIUM:
            return 0.7
        elif self.reliability == ReliabilityLevel.LOW:
            return 0.4
        else:
            return 0.5
    
    def to_summary(self) -> str:
        parts = [f"{self.expert_name} ({self.model_name})"]
        if self.accuracy is not None:
            parts.append(f"Acc: {self.accuracy:.1%}")
        if self.mAP is not None:
            parts.append(f"mAP: {self.mAP:.1%}")
        if self.srcc is not None:
            parts.append(f"SRCC: {self.srcc:.3f}")
        parts.append(f"Reliability: {self.reliability.value}")
        return " | ".join(parts)


EXPERT_PERFORMANCE_TABLE: Dict[str, ExpertPerformanceMetrics] = {
    "imagenet_fast": ExpertPerformanceMetrics(
        expert_name="ImageNetExpert-Fast",
        model_name="EfficientNetV2-S",
        task_type="classification",
        accuracy=0.8382,
        benchmark="ImageNet-1k",
        reliability=ReliabilityLevel.HIGH,
        notes="High accuracy on ImageNet, fast inference",
        source="MMClassification Docs",
    ),
    "imagenet_strong": ExpertPerformanceMetrics(
        expert_name="ImageNetExpert-Strong",
        model_name="EfficientNetV2-L",
        task_type="classification",
        accuracy=0.8543,
        benchmark="ImageNet-1k",
        reliability=ReliabilityLevel.HIGH,
        notes="Very high accuracy, larger model",
        source="MMClassification Docs",
    ),
    "clip": ExpertPerformanceMetrics(
        expert_name="SemanticExpert",
        model_name="CLIP-ViT-B/32",
        task_type="zero_shot_alignment",
        accuracy=0.70,
        benchmark="ImageNet zero-shot",
        reliability=ReliabilityLevel.HIGH,
        notes="Strong zero-shot transfer, good for semantic alignment",
        source="OpenAI CLIP Paper",
    ),
    "clip_strong": ExpertPerformanceMetrics(
        expert_name="SemanticExpert-Strong",
        model_name="CLIP-ViT-L/14",
        task_type="zero_shot_alignment",
        accuracy=0.75,
        benchmark="ImageNet zero-shot",
        reliability=ReliabilityLevel.HIGH,
        notes="Best CLIP model for zero-shot classification",
        source="Analytics Vidhya Benchmark",
    ),
    "body_pose": ExpertPerformanceMetrics(
        expert_name="HumanBodyExpert",
        model_name="YOLO11n-pose",
        task_type="pose_estimation",
        mAP=0.500,
        benchmark="COCO Keypoints",
        reliability=ReliabilityLevel.MEDIUM,
        notes="Fast but lower accuracy, good for quick checks",
        source="Ultralytics Docs",
    ),
    "body_pose_strong": ExpertPerformanceMetrics(
        expert_name="HumanBodyExpert-Strong",
        model_name="YOLO11m-pose",
        task_type="pose_estimation",
        mAP=0.649,
        benchmark="COCO Keypoints",
        reliability=ReliabilityLevel.HIGH,
        notes="Good balance of speed and accuracy",
        source="Ultralytics Docs",
    ),
    "body_pose_best": ExpertPerformanceMetrics(
        expert_name="HumanBodyExpert-Best",
        model_name="YOLO11x-pose",
        task_type="pose_estimation",
        mAP=0.695,
        benchmark="COCO Keypoints",
        reliability=ReliabilityLevel.HIGH,
        notes="Highest accuracy pose model",
        source="Ultralytics Docs",
    ),
    "hand_detection": ExpertPerformanceMetrics(
        expert_name="HandExpert",
        model_name="HADM-L",
        task_type="hand_artifact_detection",
        reliability=ReliabilityLevel.MEDIUM,
        notes="Specialized for hand artifact detection (extra fingers, malformations)",
        source="Adobe Research",
    ),
    "places365": ExpertPerformanceMetrics(
        expert_name="SceneExpert",
        model_name="Places365-ResNet18",
        task_type="scene_classification",
        accuracy=0.85,
        benchmark="Places365 Top-5",
        reliability=ReliabilityLevel.HIGH,
        notes="Good scene classification, 365 categories",
        source="Places365 Paper",
    ),
    "iqa_fast": ExpertPerformanceMetrics(
        expert_name="IQAExpert-Fast",
        model_name="MANIQA",
        task_type="image_quality",
        srcc=0.875,
        plcc=0.882,
        benchmark="KonIQ-10k",
        reliability=ReliabilityLevel.HIGH,
        notes="Strong NR-IQA, good for GAN distortions",
        source="MANIQA Paper (CVPR 2022)",
    ),
    "iqa_default": ExpertPerformanceMetrics(
        expert_name="IQAExpert-Default",
        model_name="MANIQA+MUSIQ",
        task_type="image_quality",
        srcc=0.860,
        plcc=0.870,
        benchmark="KonIQ-10k / PaQ-2-PiQ",
        reliability=ReliabilityLevel.HIGH,
        notes="Multi-scale quality assessment",
        source="Google Research",
    ),
    "aigen_detection": ExpertPerformanceMetrics(
        expert_name="AIGenExpert",
        model_name="DistilDIRE",
        task_type="ai_generated_detection",
        accuracy=0.99,
        benchmark="ImageNet subset vs SD-v1",
        reliability=ReliabilityLevel.HIGH,
        notes="Excellent AI-generated image detection",
        source="OpenReview",
    ),
    "background_removal": ExpertPerformanceMetrics(
        expert_name="BackgroundExpert",
        model_name="RMBG-2.0",
        task_type="background_segmentation",
        accuracy=0.9014,
        benchmark="Internal benchmark",
        reliability=ReliabilityLevel.HIGH,
        notes="State-of-the-art open-source background removal",
        source="Bria AI Blog",
    ),
    "vqa": ExpertPerformanceMetrics(
        expert_name="VQAExpert",
        model_name="Qwen2.5-VL-3B",
        task_type="visual_qa",
        accuracy=0.964,
        benchmark="DocVQA",
        reliability=ReliabilityLevel.HIGH,
        notes="Strong multimodal understanding, good for complex questions",
        source="Qwen2.5-VL Blog",
    ),
    "ocr": ExpertPerformanceMetrics(
        expert_name="TextExpert",
        model_name="PaddleOCR",
        task_type="text_recognition",
        reliability=ReliabilityLevel.HIGH,
        notes="Industrial-grade OCR, supports 80+ languages",
        source="PaddleOCR GitHub",
    ),
    "semantic": ExpertPerformanceMetrics(
        expert_name="SemanticExpert",
        model_name="Qwen2.5-VL",
        task_type="semantic_alignment",
        reliability=ReliabilityLevel.HIGH,
        notes="General semantic analysis via VLM",
        source="Qwen Team",
    ),
    "structural": ExpertPerformanceMetrics(
        expert_name="StructuralExpert",
        model_name="Claude/Qwen-VL",
        task_type="structural_analysis",
        reliability=ReliabilityLevel.MEDIUM,
        notes="VLM-based structural inspection",
        source="Anthropic/Qwen",
    ),
    "artifact": ExpertPerformanceMetrics(
        expert_name="ArtifactExpert",
        model_name="MANIQA+MUSIQ+CLIPIQA",
        task_type="artifact_detection",
        srcc=0.875,
        benchmark="KonIQ-10k",
        reliability=ReliabilityLevel.HIGH,
        notes="Comprehensive artifact detection",
        source="pyiqa",
    ),
}


def get_expert_performance(expert_name: str) -> Optional[ExpertPerformanceMetrics]:
    normalized = expert_name.lower().replace("-", "_").replace(" ", "_")
    for key, metrics in EXPERT_PERFORMANCE_TABLE.items():
        if key.lower() == normalized:
            return metrics
        if metrics.expert_name.lower() == expert_name.lower():
            return metrics
        if metrics.model_name.lower() == expert_name.lower():
            return metrics
    return None


def get_reliability_weight(expert_name: str) -> float:
    metrics = get_expert_performance(expert_name)
    if metrics:
        return metrics.confidence_weight
    return 0.5


def get_expert_performance_summary() -> str:
    lines = ["# Expert Model Performance Table\n"]
    lines.append("| Expert | Model | Task | Accuracy/mAP | Reliability | Weight |")
    lines.append("|--------|-------|------|--------------|-------------|--------|")
    
    for key, metrics in EXPERT_PERFORMANCE_TABLE.items():
        acc_str = ""
        if metrics.accuracy is not None:
            acc_str = f"{metrics.accuracy:.1%}"
        elif metrics.mAP is not None:
            acc_str = f"mAP: {metrics.mAP:.1%}"
        elif metrics.srcc is not None:
            acc_str = f"SRCC: {metrics.srcc:.3f}"
        
        lines.append(
            f"| {metrics.expert_name} | {metrics.model_name} | {metrics.task_type} | "
            f"{acc_str} | {metrics.reliability.value} | {metrics.confidence_weight:.1f} |"
        )
    
    return "\n".join(lines)


def format_expert_performance_for_prompt(expert_names: List[str]) -> str:
    lines = ["## Expert Model Reliability Information\n"]
    lines.append("Consider the following reliability weights when evaluating expert results:\n")
    
    for expert_name in expert_names:
        metrics = get_expert_performance(expert_name)
        if metrics:
            lines.append(f"- **{metrics.expert_name}** ({metrics.model_name})")
            if metrics.accuracy is not None:
                lines.append(f"  - Accuracy: {metrics.accuracy:.1%} on {metrics.benchmark}")
            if metrics.mAP is not None:
                lines.append(f"  - mAP: {metrics.mAP:.1%} on {metrics.benchmark}")
            if metrics.srcc is not None:
                lines.append(f"  - SRCC: {metrics.srcc:.3f}")
            lines.append(f"  - Reliability: **{metrics.reliability.value.upper()}** (weight: {metrics.confidence_weight:.1f})")
            if metrics.notes:
                lines.append(f"  - Notes: {metrics.notes}")
            lines.append("")
        else:
            lines.append(f"- **{expert_name}**: Reliability UNKNOWN (weight: 0.5)")
            lines.append("")
    
    lines.append("**Guidelines for handling expert results:**")
    lines.append("- HIGH reliability experts: Trust their conclusions unless directly contradicted by image evidence")
    lines.append("- MEDIUM reliability experts: Consider their results but verify with other evidence")
    lines.append("- LOW reliability experts: Use their results as hints only, require stronger confirmation")
    lines.append("- UNKNOWN reliability: Treat conservatively, similar to MEDIUM")
    
    return "\n".join(lines)


def calculate_weighted_severity(
    expert_results: List[Dict],
    severity_key: str = "severity"
) -> float:
    if not expert_results:
        return 0.0
    
    weighted_sum = 0.0
    weight_sum = 0.0
    
    for result in expert_results:
        expert_name = result.get("expert", "unknown")
        severity = result.get(severity_key, 0.0)
        weight = get_reliability_weight(expert_name)
        
        weighted_sum += severity * weight
        weight_sum += weight
    
    if weight_sum == 0:
        return 0.0
    
    return weighted_sum / weight_sum


def detect_expert_conflicts(expert_results: List[Dict]) -> List[Dict]:
    conflicts = []
    
    severity_threshold = 0.3
    expert_severities = {}
    
    for result in expert_results:
        expert_name = result.get("expert", "unknown")
        severity = result.get("severity", 0.0)
        expert_severities[expert_name] = severity
    
    expert_names = list(expert_severities.keys())
    for i, name1 in enumerate(expert_names):
        for name2 in expert_names[i+1:]:
            sev1 = expert_severities[name1]
            sev2 = expert_severities[name2]
            
            if abs(sev1 - sev2) > severity_threshold:
                weight1 = get_reliability_weight(name1)
                weight2 = get_reliability_weight(name2)
                
                conflicts.append({
                    "experts": [name1, name2],
                    "severity_difference": abs(sev1 - sev2),
                    "recommended_trust": name1 if weight1 > weight2 else name2,
                    "reason": f"Severity conflict: {name1}={sev1:.2f} vs {name2}={sev2:.2f}"
                })
    
    return conflicts

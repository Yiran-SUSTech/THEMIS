from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(slots=True)
class Settings:
    anthropic_base_url: str
    anthropic_api_key: str
    model: str = "claude-sonnet-4-6"
    planner_local_enabled: bool = True
    planner_local_model: str = "/home/ronin/THEMIS/models/Qwen2.5-VL-3B-Instruct"
    planner_local_max_new_tokens: int = 420
    planner_model: str = "claude-sonnet-4-6"
    judge_model: str = "claude-haiku-4-5"
    report_model: str = "claude-sonnet-4-6"
    reflector_model: str = "claude-sonnet-4-6"
    remote_expert_model: str = "claude-sonnet-4-6"
    semantic_local_fast_model: str = "/home/ronin/THEMIS/models/Qwen2.5-VL-3B-Instruct"
    semantic_local_stronger_model: str = "/home/ronin/THEMIS/models/Qwen2.5-VL-7B-Instruct"
    structural_remote_mid_model: str = "claude-haiku-4-5"
    structural_remote_strong_model: str = "claude-sonnet-4-6"
    artifact_remote_strong_model: str = "claude-sonnet-4-6"
    vqa_remote_mid_model: str = "claude-haiku-4-5"
    vqa_remote_strong_model: str = "claude-sonnet-4-6"
    local_semantic_enabled: bool = True
    local_semantic_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    local_semantic_device: str = "cuda"
    local_semantic_quantization: str = "4bit"
    local_semantic_max_new_tokens: int = 220
    local_artifact_enabled: bool = True
    local_artifact_metrics: str = "maniqa,musiq"
    local_artifact_device: str = "cuda"
    local_image_max_side: int = 1024
    semantic_escalation_threshold: float = 0.72
    artifact_confidence_threshold: float = 0.55
    max_plan_revisions: int = 5
    max_reflection_revisions: int = 3

    @classmethod
    def from_env(cls) -> "Settings":
        def env_bool(name: str, default: bool) -> bool:
            value = os.getenv(name)
            if value is None:
                return default
            return value.strip().lower() in {"1", "true", "yes", "on"}

        def env_float(name: str, default: float) -> float:
            value = os.getenv(name)
            if value is None or not value.strip():
                return default
            return float(value)

        base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
        api_key = (os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY") or "").strip()
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6").strip() or "claude-sonnet-4-6"
        planner_local_enabled = env_bool("PLANNER_LOCAL_ENABLED", True)
        planner_local_model = os.getenv("PLANNER_LOCAL_MODEL", "/home/ronin/THEMIS/models/Qwen2.5-VL-3B-Instruct").strip() or "/home/ronin/THEMIS/models/Qwen2.5-VL-3B-Instruct"
        planner_local_max_new_tokens = int(os.getenv("PLANNER_LOCAL_MAX_NEW_TOKENS", "420"))
        planner_model = os.getenv("PLANNER_MODEL", model).strip() or model
        judge_model = os.getenv("JUDGE_MODEL", model).strip() or model
        report_model = os.getenv("REPORT_MODEL", model).strip() or model
        reflector_model = os.getenv("REFLECTOR_MODEL", model).strip() or model
        remote_expert_model = os.getenv("REMOTE_EXPERT_MODEL", model).strip() or model
        semantic_local_fast_model = os.getenv("SEMANTIC_LOCAL_FAST_MODEL", "/home/ronin/THEMIS/models/Qwen2.5-VL-3B-Instruct").strip() or "/home/ronin/THEMIS/models/Qwen2.5-VL-3B-Instruct"
        semantic_local_stronger_model = os.getenv("SEMANTIC_LOCAL_STRONGER_MODEL", "/home/ronin/THEMIS/models/Qwen2.5-VL-7B-Instruct").strip() or "/home/ronin/THEMIS/models/Qwen2.5-VL-7B-Instruct"
        structural_remote_mid_model = os.getenv("STRUCTURAL_REMOTE_MID_MODEL", model).strip() or model
        structural_remote_strong_model = os.getenv("STRUCTURAL_REMOTE_STRONG_MODEL", model).strip() or model
        artifact_remote_strong_model = os.getenv("ARTIFACT_REMOTE_STRONG_MODEL", model).strip() or model
        vqa_remote_mid_model = os.getenv("VQA_REMOTE_MID_MODEL", model).strip() or model
        vqa_remote_strong_model = os.getenv("VQA_REMOTE_STRONG_MODEL", model).strip() or model
        local_semantic_enabled = env_bool("LOCAL_SEMANTIC_ENABLED", True)
        local_semantic_model = os.getenv("LOCAL_SEMANTIC_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct").strip() or "Qwen/Qwen2.5-VL-3B-Instruct"
        local_semantic_device = os.getenv("LOCAL_SEMANTIC_DEVICE", "cuda").strip() or "cuda"
        local_semantic_quantization = os.getenv("LOCAL_SEMANTIC_QUANTIZATION", "4bit").strip() or "4bit"
        local_semantic_max_new_tokens = int(os.getenv("LOCAL_SEMANTIC_MAX_NEW_TOKENS", "220"))
        local_artifact_enabled = env_bool("LOCAL_ARTIFACT_ENABLED", True)
        local_artifact_metrics = os.getenv("LOCAL_ARTIFACT_METRICS", "maniqa,musiq").strip() or "maniqa,musiq"
        local_artifact_device = os.getenv("LOCAL_ARTIFACT_DEVICE", "cuda").strip() or "cuda"
        local_image_max_side = int(os.getenv("LOCAL_IMAGE_MAX_SIDE", "1024"))
        semantic_escalation_threshold = env_float("SEMANTIC_ESCALATION_THRESHOLD", 0.72)
        artifact_confidence_threshold = env_float("ARTIFACT_CONFIDENCE_THRESHOLD", 0.55)
        max_plan_revisions = int(os.getenv("MAX_PLAN_REVISIONS", "2"))
        max_reflection_revisions = int(os.getenv("MAX_REFLECTION_REVISIONS", "1"))

        if not base_url:
            raise ValueError("Missing ANTHROPIC_BASE_URL")
        if not api_key:
            raise ValueError("Missing ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY")

        return cls(
            anthropic_base_url=base_url,
            anthropic_api_key=api_key,
            model=model,
            planner_local_enabled=planner_local_enabled,
            planner_local_model=planner_local_model,
            planner_local_max_new_tokens=planner_local_max_new_tokens,
            planner_model=planner_model,
            judge_model=judge_model,
            report_model=report_model,
            reflector_model=reflector_model,
            remote_expert_model=remote_expert_model,
            semantic_local_fast_model=semantic_local_fast_model,
            semantic_local_stronger_model=semantic_local_stronger_model,
            structural_remote_mid_model=structural_remote_mid_model,
            structural_remote_strong_model=structural_remote_strong_model,
            artifact_remote_strong_model=artifact_remote_strong_model,
            vqa_remote_mid_model=vqa_remote_mid_model,
            vqa_remote_strong_model=vqa_remote_strong_model,
            local_semantic_enabled=local_semantic_enabled,
            local_semantic_model=local_semantic_model,
            local_semantic_device=local_semantic_device,
            local_semantic_quantization=local_semantic_quantization,
            local_semantic_max_new_tokens=local_semantic_max_new_tokens,
            local_artifact_enabled=local_artifact_enabled,
            local_artifact_metrics=local_artifact_metrics,
            local_artifact_device=local_artifact_device,
            local_image_max_side=local_image_max_side,
            semantic_escalation_threshold=semantic_escalation_threshold,
            artifact_confidence_threshold=artifact_confidence_threshold,
            max_plan_revisions=max_plan_revisions,
            max_reflection_revisions=max_reflection_revisions,
        )

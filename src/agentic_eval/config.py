from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from pathlib import Path

from dotenv import load_dotenv
import yaml

load_dotenv()


ModelProfile = Literal["fast", "standard", "strong"]
EvaluationProfile = Literal["fast", "standard", "accurate", "deep"]


@dataclass
class ModelConfig:
    model: str
    local_path: Optional[str] = None
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = 512
    quantization: Optional[str] = None


@dataclass
class ExpertModelConfig:
    name: str
    model_type: str
    model: str
    local_path: Optional[str] = None
    device: str = "cuda"
    weights: Optional[str] = None
    num_classes: Optional[int] = None
    input_size: tuple = (224, 224)
    description: str = ""
    metrics: Optional[List[str]] = None


@dataclass
class Settings:
    model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    model_dir: str = "./models"
    evaluation_profile: EvaluationProfile = "standard"
    
    planner_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    planner_local_path: Optional[str] = None
    planner_local_model: Optional[str] = None
    planner_local_enabled: bool = False
    planner_device: str = "cuda:0"
    planner_profile: ModelProfile = "fast"
    planner_max_new_tokens: int = 420
    planner_local_max_new_tokens: int = 640
    
    judge_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    judge_local_path: Optional[str] = None
    judge_local_model: Optional[str] = None
    judge_local_enabled: bool = False
    judge_device: str = "cuda:1"
    judge_profile: ModelProfile = "fast"
    judge_max_new_tokens: int = 256
    judge_local_max_new_tokens: int = 256
    
    reflector_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    reflector_local_path: Optional[str] = None
    reflector_local_model: Optional[str] = None
    reflector_local_enabled: bool = False
    reflector_device: str = "cuda:2"
    reflector_profile: ModelProfile = "standard"
    reflector_max_new_tokens: int = 400
    reflector_local_max_new_tokens: int = 400
    
    local_semantic_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    local_semantic_quantization: Optional[str] = None
    local_semantic_device: str = "cuda:3"
    local_semantic_enabled: bool = True
    local_semantic_max_new_tokens: int = 512
    semantic_local_fast_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    semantic_local_stronger_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    local_artifact_enabled: bool = True
    local_artifact_device: str = "cuda:4"
    local_artifact_metrics: str = "maniqa,musiq,niqe"
    
    iqa_cache_dir: str = "./models/pyiqa"
    
    report_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    use_specialized_experts: bool = True
    use_vlm_overseer: bool = True
    
    clip_model_path: str = "openai/clip-vit-base-patch32"
    clip_device: str = "cuda:3"
    
    imagenet_model: str = "tf_efficientnetv2_s.in21k"
    imagenet_weights: Optional[str] = None
    imagenet_device: str = "cuda:3"
    
    yolo_model_path: str = "yolo11n-pose"
    yolo_device: str = "cuda:3"
    
    places365_model_path: Optional[str] = None
    places365_device: str = "cuda:3"
    
    rmbg_model_path: Optional[str] = None
    rmbg_device: str = "cuda:3"
    
    expert_configs: Dict[str, ExpertModelConfig] = field(default_factory=dict)
    
    max_plan_revisions: int = 2
    max_reflection_revisions: int = 1
    semantic_escalation_threshold: float = 0.72
    
    local_image_max_side: int = 1024
    log_level: str = "INFO"
    log_dir: str = "./logs"

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

        def env_int(name: str, default: int) -> int:
            value = os.getenv(name)
            if value is None or not value.strip():
                return default
            return int(value)

        model = os.getenv("LOCAL_PRIMARY_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct").strip() or "Qwen/Qwen2.5-VL-3B-Instruct"
        
        model_dir = os.getenv("MODEL_DIR", "./models").strip()
        
        evaluation_profile = os.getenv("EVALUATION_PROFILE", "standard").strip()
        
        planner_profile = os.getenv("PLANNER_PROFILE", "fast").strip()
        planner_model = os.getenv("PLANNER_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct").strip()
        planner_local_path = os.getenv("PLANNER_LOCAL_MODEL_PATH", None)
        planner_local_model = planner_local_path if planner_local_path else None
        planner_local_enabled = bool(planner_local_path)
        planner_device = os.getenv("PLANNER_DEVICE", "cuda:0").strip()
        planner_max_new_tokens = env_int("PLANNER_MAX_NEW_TOKENS", 420)
        planner_local_max_new_tokens = env_int("PLANNER_LOCAL_MAX_NEW_TOKENS", 640)
        
        judge_profile = os.getenv("JUDGE_PROFILE", "fast").strip()
        judge_model = os.getenv("JUDGE_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct").strip()
        judge_local_path = os.getenv("JUDGE_LOCAL_MODEL_PATH", None)
        judge_local_model = judge_local_path if judge_local_path else None
        judge_local_enabled = bool(judge_local_path)
        judge_device = os.getenv("JUDGE_DEVICE", "cuda:1").strip()
        judge_max_new_tokens = env_int("JUDGE_MAX_NEW_TOKENS", 256)
        judge_local_max_new_tokens = env_int("JUDGE_LOCAL_MAX_NEW_TOKENS", 256)
        
        reflector_profile = os.getenv("REFLECTOR_PROFILE", "standard").strip()
        reflector_model = os.getenv("REFLECTOR_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct").strip()
        reflector_local_path = os.getenv("REFLECTOR_LOCAL_MODEL_PATH", None)
        reflector_local_model = reflector_local_path if reflector_local_path else None
        reflector_local_enabled = bool(reflector_local_path)
        reflector_device = os.getenv("REFLECTOR_DEVICE", "cuda:2").strip()
        reflector_max_new_tokens = env_int("REFLECTOR_MAX_NEW_TOKENS", 400)
        reflector_local_max_new_tokens = env_int("REFLECTOR_LOCAL_MAX_NEW_TOKENS", 400)        
        local_semantic_model = os.getenv("LOCAL_SEMANTIC_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct").strip()
        local_semantic_quantization = os.getenv("LOCAL_SEMANTIC_QUANTIZATION", None)
        local_semantic_device = os.getenv("LOCAL_SEMANTIC_DEVICE", "cuda:3").strip()
        local_semantic_enabled = env_bool("LOCAL_SEMANTIC_ENABLED", True)
        local_semantic_max_new_tokens = env_int("LOCAL_SEMANTIC_MAX_NEW_TOKENS", 512)
        semantic_local_fast_model = os.getenv("SEMANTIC_LOCAL_FAST_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct").strip()
        semantic_local_stronger_model = os.getenv("SEMANTIC_LOCAL_STRONGER_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct").strip()
        
        local_artifact_enabled = env_bool("LOCAL_ARTIFACT_ENABLED", True)
        local_artifact_device = os.getenv("LOCAL_ARTIFACT_DEVICE", "cuda:4").strip()
        local_artifact_metrics = os.getenv("LOCAL_ARTIFACT_METRICS", "maniqa,musiq,niqe").strip()
        
        iqa_cache_dir = os.getenv("IQA_CACHE_DIR", "./models/pyiqa").strip()
        
        report_model = os.getenv("REPORT_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct").strip()
        
        use_specialized_experts = env_bool("USE_SPECIALIZED_EXPERTS", True)
        use_vlm_overseer = env_bool("USE_VLM_OVERSEER", True)
        
        clip_model_path = os.getenv("CLIP_MODEL_PATH", "openai/clip-vit-base-patch32").strip()
        clip_device = os.getenv("CLIP_DEVICE", "cuda:3").strip()
        
        imagenet_model = os.getenv("IMAGENET_MODEL", "tf_efficientnetv2_s.in21k").strip()
        imagenet_weights = os.getenv("IMAGENET_WEIGHTS", None)
        imagenet_device = os.getenv("IMAGENET_DEVICE", "cuda:3").strip()
        
        yolo_model_path = os.getenv("YOLO_MODEL_PATH", "yolo11n-pose").strip()
        yolo_device = os.getenv("YOLO_DEVICE", "cuda:3").strip()
        
        places365_model_path = os.getenv("PLACES365_MODEL_PATH", None)
        places365_device = os.getenv("PLACES365_DEVICE", "cuda:3").strip()
        
        rmbg_model_path = os.getenv("RMBG_MODEL_PATH", None)
        rmbg_device = os.getenv("RMBG_DEVICE", "cuda:3").strip()
        
        max_plan_revisions = env_int("MAX_PLAN_REVISIONS", 2)
        max_reflection_revisions = env_int("MAX_REFLECTION_REVISIONS", 1)
        semantic_escalation_threshold = env_float("SEMANTIC_ESCALATION_THRESHOLD", 0.72)
        
        local_image_max_side = env_int("LOCAL_IMAGE_MAX_SIDE", 1024)
        log_level = os.getenv("LOG_LEVEL", "INFO").strip()
        log_dir = os.getenv("LOG_DIR", "./logs").strip()

        return cls(
            model=model,
            model_dir=model_dir,
            evaluation_profile=evaluation_profile,
            planner_model=planner_model,
            planner_local_path=planner_local_path,
            planner_local_model=planner_local_model,
            planner_local_enabled=planner_local_enabled,
            planner_device=planner_device,
            planner_profile=planner_profile,
            planner_max_new_tokens=planner_max_new_tokens,
            planner_local_max_new_tokens=planner_local_max_new_tokens,
            judge_model=judge_model,
            judge_local_path=judge_local_path,
            judge_local_model=judge_local_model,
            judge_local_enabled=judge_local_enabled,
            judge_device=judge_device,
            judge_profile=judge_profile,
            judge_max_new_tokens=judge_max_new_tokens,
            judge_local_max_new_tokens=judge_local_max_new_tokens,
            reflector_model=reflector_model,
            reflector_local_path=reflector_local_path,
            reflector_local_model=reflector_local_model,
            reflector_local_enabled=reflector_local_enabled,
            reflector_device=reflector_device,
            reflector_profile=reflector_profile,
            reflector_max_new_tokens=reflector_max_new_tokens,
            reflector_local_max_new_tokens=reflector_local_max_new_tokens,
            local_semantic_model=local_semantic_model,
            local_semantic_quantization=local_semantic_quantization,
            local_semantic_device=local_semantic_device,
            local_semantic_enabled=local_semantic_enabled,
            local_semantic_max_new_tokens=local_semantic_max_new_tokens,
            semantic_local_fast_model=semantic_local_fast_model,
            semantic_local_stronger_model=semantic_local_stronger_model,
            local_artifact_enabled=local_artifact_enabled,
            local_artifact_device=local_artifact_device,
            local_artifact_metrics=local_artifact_metrics,
            iqa_cache_dir=iqa_cache_dir,
            report_model=report_model,
            use_specialized_experts=use_specialized_experts,
            use_vlm_overseer=use_vlm_overseer,
            clip_model_path=clip_model_path,
            clip_device=clip_device,
            imagenet_model=imagenet_model,
            imagenet_weights=imagenet_weights,
            imagenet_device=imagenet_device,
            yolo_model_path=yolo_model_path,
            yolo_device=yolo_device,
            places365_model_path=places365_model_path,
            places365_device=places365_device,
            rmbg_model_path=rmbg_model_path,
            rmbg_device=rmbg_device,
            max_plan_revisions=max_plan_revisions,
            max_reflection_revisions=max_reflection_revisions,
            semantic_escalation_threshold=semantic_escalation_threshold,
            local_image_max_side=local_image_max_side,
            log_level=log_level,
            log_dir=log_dir,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Settings":
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        settings = cls.from_env()
        
        if "core_models" in config:
            core = config["core_models"]
            
            if "planner" in core:
                planner = core["planner"]
                default_profile = planner.get("default_profile", "fast")
                profiles = planner.get("profiles", {})
                if default_profile in profiles:
                    p = profiles[default_profile]
                    settings.planner_model = p.get("model", settings.planner_model)
                    settings.planner_local_path = p.get("local_path", None)
                    settings.planner_local_model = settings.planner_local_path if settings.planner_local_path else settings.planner_local_model
                    settings.planner_local_enabled = bool(settings.planner_local_model)
                    settings.planner_device = p.get("device", settings.planner_device)
                    settings.planner_max_new_tokens = p.get("max_new_tokens", settings.planner_max_new_tokens)
                    settings.planner_local_max_new_tokens = p.get("max_new_tokens", settings.planner_local_max_new_tokens)
                    settings.planner_profile = default_profile
            
            if "judge" in core:
                judge = core["judge"]
                default_profile = judge.get("default_profile", "fast")
                profiles = judge.get("profiles", {})
                if default_profile in profiles:
                    p = profiles[default_profile]
                    settings.judge_model = p.get("model", settings.judge_model)
                    settings.judge_local_path = p.get("local_path", None)
                    settings.judge_local_model = settings.judge_local_path if settings.judge_local_path else settings.judge_local_model
                    settings.judge_local_enabled = bool(settings.judge_local_model)
                    settings.judge_device = p.get("device", settings.judge_device)
                    settings.judge_max_new_tokens = p.get("max_new_tokens", settings.judge_max_new_tokens)
                    settings.judge_profile = default_profile
            
            if "reflector" in core:
                reflector = core["reflector"]
                default_profile = reflector.get("default_profile", "standard")
                profiles = reflector.get("profiles", {})
                if default_profile in profiles:
                    p = profiles[default_profile]
                    settings.reflector_model = p.get("model", settings.reflector_model)
                    settings.reflector_local_path = p.get("local_path", None)
                    settings.reflector_local_model = settings.reflector_local_path if settings.reflector_local_path else settings.reflector_local_model
                    settings.reflector_local_enabled = bool(settings.reflector_local_model)
                    settings.reflector_device = p.get("device", settings.reflector_device)
                    settings.reflector_max_new_tokens = p.get("max_new_tokens", settings.reflector_max_new_tokens)
                    settings.reflector_profile = default_profile
        
        if "expert_models" in config:
            for group_name, group in config["expert_models"].items():
                for expert_name, expert_config in group.items():
                    settings.expert_configs[expert_name] = ExpertModelConfig(
                        name=expert_config.get("name", expert_name),
                        model_type=expert_config.get("model_type", "unknown"),
                        model=expert_config.get("model", ""),
                        local_path=expert_config.get("local_path"),
                        device=expert_config.get("device", "cuda"),
                        weights=expert_config.get("weights"),
                        num_classes=expert_config.get("num_classes"),
                        input_size=tuple(expert_config.get("input_size", [224, 224])),
                        description=expert_config.get("description", ""),
                        metrics=expert_config.get("metrics"),
                    )
        
        if "evaluation_profiles" in config:
            profiles = config["evaluation_profiles"]
            if settings.evaluation_profile in profiles:
                profile_config = profiles[settings.evaluation_profile]
                settings.planner_profile = profile_config.get("planner_profile", settings.planner_profile)
                settings.judge_profile = profile_config.get("judge_profile", settings.judge_profile)
                settings.reflector_profile = profile_config.get("reflector_profile", settings.reflector_profile)
        
        return settings

    def get_model_path(self, model_name: str) -> str:
        if self.model_dir:
            local_path = Path(self.model_dir) / model_name
            if local_path.exists():
                return str(local_path)
        return model_name

    def get_expert_config(self, expert_name: str) -> Optional[ExpertModelConfig]:
        return self.expert_configs.get(expert_name)

    def get_experts_by_type(self, model_type: str) -> List[ExpertModelConfig]:
        return [
            config for config in self.expert_configs.values()
            if config.model_type == model_type
        ]

    def get_experts_by_device(self, device: str) -> List[ExpertModelConfig]:
        return [
            config for config in self.expert_configs.values()
            if config.device == device
        ]


QWEN_MODEL_SIZES = {
    "3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "32b": "Qwen/Qwen2.5-VL-32B-Instruct",
}


def get_qwen_model_path(size: str, model_dir: str) -> str:
    size_lower = size.lower()
    if size_lower not in QWEN_MODEL_SIZES:
        raise ValueError(f"Unknown Qwen model size: {size}")
    
    model_name = QWEN_MODEL_SIZES[size_lower].split("/")[-1]
    local_path = Path(model_dir) / model_name
    
    if local_path.exists():
        return str(local_path)
    return QWEN_MODEL_SIZES[size_lower]

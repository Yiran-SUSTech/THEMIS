from __future__ import annotations

import json
import math
import os
from pathlib import Path
from statistics import pstdev
from threading import Lock
from typing import TYPE_CHECKING, Any, Optional

import yaml

if TYPE_CHECKING:
    from .config import ExpertModelConfig


VALID_MODEL_PROFILES = {
    "local_fast",
    "local_default",
    "local_richer",
    "local_stronger",
}

EXPERT_ALLOWED_MODEL_PROFILES = {
    "semantic": {"local_fast", "local_stronger"},
    "quality": {"local_fast", "local_default", "local_richer"},
    "structural": {"local_fast", "local_stronger"},
    "vqa": {"local_fast", "local_stronger"},
}

EXPERT_ROLE_MODEL_KEYS = {
    "semantic": [
        "clip",
        "imagenet_fast",
        "imagenet_strong",
        "imagenet_eva02_large",
        "imagenet_eva_giant_224",
        "bge_candidate_generator",
        "e5_candidate_generator",
        "clip_score",
        "vqa",
    ],
    "structural": [
        "animal_pose",
        "body_pose",
        "body_pose_strong",
        "places365",
        "places365_strong",
        "background_removal",
        "vqa",
    ],
    "quality": ["iqa_fast", "iqa_default", "iqa_richer", "q_insight", "boundary_artifact"],
    "vqa": ["vqa"],
}

ROLE_MODEL_TYPE_HINTS = {
    "semantic": {"classification", "eva_classification", "text_embedding", "clip", "clip_score", "vqa"},
    "structural": {"yolo_pose", "detection", "places365", "classification", "segmentation", "vqa"},
    "quality": {"iqa", "clip", "mllm_scoring"},
    "vqa": {"vqa"},
}

ROLE_SUITABLE_FOR_HINTS = {
    "semantic": {"broad_category_match", "coarse_semantic_prior", "harder_c2i_screening", "label_space_aligned_classification", "fine_grained_c2i", "class_name_grounded_decision", "candidate_generation", "confusable_label_retrieval", "fine_grained_label_shortlisting", "t2i_alignment", "lookalike_disambiguation", "candidate_reranking", "confusable_disambiguation", "prompt_alignment_support", "text_image_similarity"},
    "structural": {"whole_subject_coherence", "rough_pose_plausibility", "animal_body_structure_screening"},
    "quality": {"artifact_screening", "perceptual_quality_prior", "artifact_assessment", "perceptual_quality_check", "richer_artifact_assessment", "distortion_detection", "fine_grained_quality_vqa"},
    "vqa": {"targeted_visual_evidence_extraction", "resolving_specific_ambiguities", "attribute_confirmation"},
}

_LOCAL_VLM_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}
_LOCAL_VLM_CACHE_LOCK = Lock()

from .config import Settings
from .expert_performance import get_expert_performance


def _default_planned_model(expert: str, model_profile: str) -> str:
    if expert == "semantic":
        return "clip" if model_profile != "local_stronger" else "imagenet_eva02_large"
    if expert == "structural":
        return "animal_pose" if model_profile != "local_stronger" else "vqa"
    if expert == "quality":
        if model_profile == "local_fast":
            return "iqa_fast"
        if model_profile == "local_richer":
            return "iqa_richer"
        return "iqa_default"
    if expert == "vqa":
        return "vqa"
    return ""


def _all_downloaded_expert_items(settings: Settings) -> list[tuple[str, ExpertModelConfig]]:
    return [
        (key, config)
        for key, config in settings.expert_configs.items()
        if config is not None and getattr(config, "downloaded", False)
    ]


def _matches_role_metadata(expert: str, config: ExpertModelConfig) -> bool:
    if expert == "vqa":
        return config.model_type == "vqa"
    if config.model_type in ROLE_MODEL_TYPE_HINTS.get(expert, set()):
        return True
    suitable_for = set(config.suitable_for or [])
    if suitable_for & ROLE_SUITABLE_FOR_HINTS.get(expert, set()):
        return True
    evidence_role = (config.evidence_role or "").lower()
    if expert == "semantic" and any(token in evidence_role for token in {"semantic", "classification", "candidate", "similarity", "alignment"}):
        return True
    if expert == "structural" and any(token in evidence_role for token in {"structural", "pose", "segmentation", "scene"}):
        return True
    if expert == "quality" and any(token in evidence_role for token in {"quality", "artifact", "distortion"}):
        return True
    return False


def _profile_compatibility_score(model_profile: str, config_key: str, config: ExpertModelConfig) -> float:
    score = 0.0
    if model_profile == "local_fast":
        if config_key.endswith("_fast") or "fast" in config.name.lower():
            score += 3.0
        if config.model_type in {"clip", "clip_score", "text_embedding", "iqa", "yolo_pose", "mllm_scoring"}:
            score += 1.0
    elif model_profile == "local_stronger":
        if any(token in config_key for token in {"strong", "eva"}) or "strong" in config.name.lower():
            score += 3.0
        if config.model_type in {"eva_classification", "vqa"}:
            score += 1.0
    elif model_profile == "local_richer":
        if any(token in config_key for token in {"richer", "boundary"}):
            score += 3.0
        if config.model_type == "iqa":
            score += 1.0
    elif model_profile == "local_default":
        if any(token in config_key for token in {"default"}):
            score += 3.0
    return score


def _step_type_score(step_type: str, config: ExpertModelConfig) -> float:
    score = 0.0
    normalized_step = (step_type or "").strip().lower()
    if normalized_step in {"candidate_generation"} and config.model_type == "text_embedding":
        score += 8.0
    if normalized_step in {"confusable_disambiguation"} and config.model_type in {"clip", "clip_score"}:
        score += 8.0
    if normalized_step in {"label_space_check"} and config.label_space in {"imagenet_1k", "imagenet_21k", "open_text_to_imagenet_1k"}:
        score += 6.0
    if normalized_step == "semantic_check":
        if config.model_type in {"classification", "eva_classification"}:
            score += 5.0
        if config.model_type == "text_embedding":
            score += 2.0
        if config.model_type in {"clip", "clip_score"}:
            score += 3.0
    if normalized_step == "structural_check":
        if config.model_type in {"yolo_pose", "segmentation", "vqa"}:
            score += 5.0
    if normalized_step == "quality_check":
        if config.model_type == "iqa":
            score += 6.0
        elif config.model_type == "mllm_scoring":
            score += 5.0
        elif config.model_type == "clip":
            score += 1.0
    if normalized_step == "vqa_evidence" and config.model_type == "vqa":
        score += 8.0
    return score


def _metadata_quality_score(config: ExpertModelConfig) -> float:
    score = 0.0
    if config.description:
        score += 0.5
    if config.evidence_role:
        score += 0.5
    if config.label_space:
        score += 0.5
    if config.output_interpretability:
        score += 0.5
    if config.suitable_for:
        score += 1.0
    if config.unsuitable_for:
        score += 0.5
    if config.accuracy is not None:
        score += 0.5
    if config.benchmark:
        score += 0.5
    return score


def _dynamic_candidate_keys(settings: Settings, expert: str) -> list[str]:
    matched: list[str] = []
    for key, config in _all_downloaded_expert_items(settings):
        if _matches_role_metadata(expert, config):
            matched.append(key)
    return matched


def _choose_best_planned_model(
    settings: Settings,
    expert: str,
    model_profile: str,
    step_type: str = "",
    planned_model: str = "",
) -> str:
    candidate = (planned_model or "").strip()
    dynamic_candidates = _dynamic_candidate_keys(settings, expert)
    if candidate and candidate in dynamic_candidates:
        return candidate

    scored: list[tuple[float, str]] = []
    for key in dynamic_candidates:
        config = settings.get_expert_config(key)
        if config is None:
            continue
        score = 0.0
        score += _profile_compatibility_score(model_profile, key, config)
        score += _step_type_score(step_type, config)
        score += _metadata_quality_score(config)
        if candidate and key == candidate:
            score += 12.0
        if config.model_type == "vqa" and expert != "vqa" and step_type not in {"vqa_evidence", "structural_check"}:
            score -= 2.0
        scored.append((score, key))

    if scored:
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][1]

    fallback_candidate = candidate if candidate and settings.get_expert_config(candidate) is not None else ""
    return fallback_candidate or _default_planned_model(expert, model_profile)


def _normalize_planned_model(
    settings: Settings,
    expert: str,
    model_profile: str,
    planned_model: str,
    step_type: str = "",
) -> str:
    return _choose_best_planned_model(
        settings,
        expert,
        model_profile,
        step_type=step_type,
        planned_model=planned_model,
    )


def _roles_for_expert_key(expert_key: str) -> list[str]:
    roles: list[str] = []
    for role, keys in EXPERT_ROLE_MODEL_KEYS.items():
        if expert_key in keys:
            roles.append(role)
    return roles


def _describe_available_experts(settings: Settings) -> str:
    lines = ["Available downloaded model options by expert role/profile:"]
    included_keys: set[str] = set()
    for expert, allowed_keys in EXPERT_ROLE_MODEL_KEYS.items():
        lines.append(f"- {expert}:")
        role_lines: list[str] = []
        for profile in sorted(EXPERT_ALLOWED_MODEL_PROFILES.get(expert, {"local_fast"})):
            descriptions: list[str] = []
            for key in allowed_keys:
                config = settings.get_expert_config(key)
                perf = get_expert_performance(key)
                if config is None or not getattr(config, "downloaded", False):
                    continue
                included_keys.add(key)
                model_name = config.local_path or config.weights or config.model
                parts = [f"key={key}", config.name, f"group={config.group_name}", config.model_type, model_name]
                parts.append(f"downloaded={str(config.downloaded).lower()}")
                if config.metrics:
                    parts.append(f"metrics={','.join(config.metrics)}")
                if perf is not None:
                    stats = []
                    if perf.accuracy is not None:
                        stats.append(f"acc={perf.accuracy:.1%}")
                    if perf.mAP is not None:
                        stats.append(f"mAP={perf.mAP:.1%}")
                    if perf.srcc is not None:
                        stats.append(f"SRCC={perf.srcc:.3f}")
                    if perf.plcc is not None:
                        stats.append(f"PLCC={perf.plcc:.3f}")
                    stats.append(f"reliability={perf.reliability.value}")
                    if perf.task_type:
                        stats.append(f"task={perf.task_type}")
                    parts.append("; ".join(stats))
                if config.benchmark:
                    parts.append(f"benchmark={config.benchmark}")
                if config.accuracy is not None:
                    parts.append(f"accuracy={config.accuracy:.4f}")
                if config.evidence_role:
                    parts.append(f"evidence_role={config.evidence_role}")
                if config.label_space:
                    parts.append(f"label_space={config.label_space}")
                if config.output_interpretability:
                    parts.append(f"output={config.output_interpretability}")
                if config.output_meaning:
                    parts.append(f"output_meaning={config.output_meaning}")
                if config.suitable_for:
                    parts.append(f"suitable_for={','.join(config.suitable_for)}")
                if config.unsuitable_for:
                    parts.append(f"unsuitable_for={','.join(config.unsuitable_for)}")
                if config.description:
                    parts.append(config.description)
                descriptions.append(" | ".join(parts))
            if descriptions:
                role_lines.append(f"  - {profile}: {' ; '.join(descriptions)}")
        if role_lines:
            lines.extend(role_lines)
        else:
            lines.append("  - no downloaded experts exposed for this role")

    remaining_downloaded = [
        (key, config)
        for key, config in sorted(settings.expert_configs.items())
        if getattr(config, "downloaded", False) and key not in included_keys
    ]
    if remaining_downloaded:
        lines.append("- additional_downloaded_experts:")
        for key, config in remaining_downloaded:
            perf = get_expert_performance(key)
            model_name = config.local_path or config.weights or config.model
            roles = _roles_for_expert_key(key)
            parts = [
                f"key={key}",
                config.name,
                f"group={config.group_name}",
                config.model_type,
                model_name,
                f"downloaded={str(config.downloaded).lower()}",
                f"roles={','.join(roles) if roles else 'none'}",
            ]
            if config.metrics:
                parts.append(f"metrics={','.join(config.metrics)}")
            if perf is not None:
                stats = []
                if perf.accuracy is not None:
                    stats.append(f"acc={perf.accuracy:.1%}")
                if perf.mAP is not None:
                    stats.append(f"mAP={perf.mAP:.1%}")
                if perf.srcc is not None:
                    stats.append(f"SRCC={perf.srcc:.3f}")
                if perf.plcc is not None:
                    stats.append(f"PLCC={perf.plcc:.3f}")
                stats.append(f"reliability={perf.reliability.value}")
                if perf.task_type:
                    stats.append(f"task={perf.task_type}")
                parts.append("; ".join(stats))
            if config.benchmark:
                parts.append(f"benchmark={config.benchmark}")
            if config.accuracy is not None:
                parts.append(f"accuracy={config.accuracy:.4f}")
            if config.evidence_role:
                parts.append(f"evidence_role={config.evidence_role}")
            if config.label_space:
                parts.append(f"label_space={config.label_space}")
            if config.output_interpretability:
                parts.append(f"output={config.output_interpretability}")
            if config.output_meaning:
                parts.append(f"output_meaning={config.output_meaning}")
            if config.suitable_for:
                parts.append(f"suitable_for={','.join(config.suitable_for)}")
            if config.unsuitable_for:
                parts.append(f"unsuitable_for={','.join(config.unsuitable_for)}")
            if config.description:
                parts.append(config.description)
            lines.append(f"  - {' | '.join(parts)}")
    return "\n".join(lines)


def _resolve_vlm_model_class() -> type[Any]:
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration
    except ImportError:
        try:
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq
        except ImportError as exc:
            raise LocalExpertError(
                "Your transformers build does not expose a vision-language generation model loader."
            ) from exc


def _build_vlm_load_kwargs(torch: Any, quantization: Optional[str], device: str) -> dict[str, Any]:
    load_kwargs: dict[str, Any] = {"trust_remote_code": True}
    normalized_device = (device or "").strip().lower()
    if torch.cuda.is_available() and normalized_device.startswith("cuda"):
        load_kwargs["device_map"] = {"": device}
        normalized_quantization = quantization.lower() if quantization else ""
        if normalized_quantization == "4bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        elif normalized_quantization == "8bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["device_map"] = "cpu"
        load_kwargs["torch_dtype"] = torch.float32
    return load_kwargs


def _resolve_local_vlm_model_id(model_id: str) -> str:
    candidate = Path(model_id)
    if candidate.exists():
        return str(candidate)
    model_dir = os.getenv("MODEL_DIR", "./models").strip() or "./models"
    local_candidate = Path(model_dir) / Path(model_id).name
    if local_candidate.exists():
        return str(local_candidate)
    return model_id


def _load_vlm_bundle(model_id: str, quantization: Optional[str], device: str, error_prefix: str) -> dict[str, Any]:
    resolved_model_id = _resolve_local_vlm_model_id(model_id)
    quantization_key = quantization.lower() if quantization else "none"
    cache_key = (resolved_model_id, quantization_key, device)
    cached = _LOCAL_VLM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor
    except ImportError as exc:
        raise LocalExpertError(
            "Local vision-language dependencies are missing. Install transformers, pillow, accelerate, and bitsandbytes."
        ) from exc

    model_cls = _resolve_vlm_model_class()
    load_kwargs = _build_vlm_load_kwargs(torch, quantization, device)

    with _LOCAL_VLM_CACHE_LOCK:
        cached = _LOCAL_VLM_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            processor = AutoProcessor.from_pretrained(resolved_model_id, trust_remote_code=True, local_files_only=True)
            model = model_cls.from_pretrained(resolved_model_id, local_files_only=True, **load_kwargs)
            model.eval()
        except Exception as exc:  # noqa: BLE001
            raise LocalExpertError(f"{error_prefix} '{resolved_model_id}'. Download it manually or check your Hugging Face access.") from exc
        bundle = {"processor": processor, "model": model, "torch": torch, "image_module": Image}
        _LOCAL_VLM_CACHE[cache_key] = bundle
        return bundle


class LocalExpertError(RuntimeError):
    pass


class LocalPlanner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._image_module: Any | None = None

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        bundle = _load_vlm_bundle(
            self.settings.planner_local_model,
            self.settings.local_semantic_quantization,
            self.settings.planner_device,
            "Failed to load local planner model",
        )
        self._processor = bundle["processor"]
        self._model = bundle["model"]
        self._torch = bundle["torch"]
        self._image_module = bundle["image_module"]

    def _load_image(self, image_path: str):
        if self._image_module is None:
            raise LocalExpertError("PIL is not loaded")

        image = self._image_module.open(image_path).convert("RGB")
        max_side = self.settings.local_image_max_side
        image.thumbnail((max_side, max_side))
        return image

    def _normalize_model_profile(self, expert: str, model_profile: str) -> str:
        candidate = (model_profile or "").strip().lower()
        allowed_profiles = EXPERT_ALLOWED_MODEL_PROFILES.get(expert, {"local_fast"})
        if candidate in VALID_MODEL_PROFILES and candidate in allowed_profiles:
            return candidate
        if expert == "semantic":
            return "local_fast"
        if expert == "quality":
            return "local_default"
        if expert == "structural":
            return "local_fast"
        if expert == "vqa":
            return "local_fast"
        return "local_fast"

    def _normalize_plan(self, payload: dict[str, Any], class_label: str | None = None) -> dict[str, Any]:
        steps = payload.get("steps")
        if not isinstance(steps, list):
            raise LocalExpertError("Local planner did not return a valid steps list")

        label_text = (class_label or "the labeled subject").strip() or "the labeled subject"
        normalized_steps: list[dict[str, Any]] = []
        seen_step_signatures: set[tuple[str, str]] = set()
        default_step_types = {
            "semantic": "semantic_check",
            "structural": "structural_check",
            "quality": "quality_check",
            "vqa": "vqa_evidence",
        }
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            expert = str(step.get("expert", "semantic")).strip().lower() or "semantic"
            if expert not in {"semantic", "structural", "quality", "vqa"}:
                continue
            prompt_focus = str(step.get("prompt_focus", "")).strip()
            goal = str(step.get("goal", "")).strip() or f"Inspect {expert} evidence."
            step_signature = (expert, (prompt_focus or goal).lower())
            if step_signature in seen_step_signatures:
                continue
            seen_step_signatures.add(step_signature)
            if not prompt_focus:
                if expert == "semantic":
                    prompt_focus = f"Inspect whether the visible subject matches {label_text} using the strongest confirming and contradicting species markers in the image."
                elif expert == "structural":
                    prompt_focus = f"Inspect whole-subject coherence and the class-specific morphology, body proportions, pose plausibility, and likely lookalike confusions for {label_text}."
                elif expert == "quality":
                    prompt_focus = "Inspect localized generation artifacts, including malformed facial regions, duplicated or fused limbs, broken joints, tail attachment, hands or feet, and corrupted boundaries when relevant."
            model_profile = self._normalize_model_profile(expert, str(step.get("model_profile", "")))
            normalized_planned_model = _normalize_planned_model(
                self.settings,
                expert,
                model_profile,
                str(step.get("planned_model", "")).strip(),
                str(step.get("step_type", default_step_types.get(expert, "semantic_check"))).strip(),
            )
            normalized_steps.append(
                {
                    "step_id": len(normalized_steps) + 1,
                    "expert": expert,
                    "step_type": str(step.get("step_type", default_step_types.get(expert, "semantic_check"))).strip() or default_step_types.get(expert, "semantic_check"),
                    "goal": goal,
                    "model_profile": model_profile,
                    "planned_model": normalized_planned_model,
                    "selection_reason": str(step.get("selection_reason", "")).strip() or f"Default concrete model choice for {expert} via {normalized_planned_model}.",
                    "prompt_focus": prompt_focus,
                    "depends_on": [int(item) for item in step.get("depends_on", []) if isinstance(item, int) or (isinstance(item, str) and item.isdigit())],
                    "expected_signal": str(step.get("expected_signal", "")).strip(),
                    "use_previous_outputs": bool(step.get("use_previous_outputs", False)),
                    "allow_escalation": bool(step.get("allow_escalation", True)),
                }
            )

        has_semantic = any(step["expert"] == "semantic" for step in normalized_steps)
        has_structural = any(step["expert"] == "structural" for step in normalized_steps)
        has_quality = any(step["expert"] == "quality" for step in normalized_steps)
        if not has_semantic:
            normalized_steps.insert(
                0,
                {
                    "step_id": 1,
                    "expert": "semantic",
                    "step_type": "semantic_check",
                    "goal": "Assess whether the image content matches the prompt/class label semantically.",
                    "model_profile": "local_fast",
                    "planned_model": "clip",
                    "selection_reason": "Default low-cost semantic first pass with CLIP.",
                    "prompt_focus": f"Inspect whether the visible subject matches {label_text} using the strongest confirming and contradicting species markers in the image.",
                    "depends_on": [],
                    "expected_signal": "Broad-category and label-match evidence.",
                    "use_previous_outputs": False,
                    "allow_escalation": True,
                },
            )
        if not has_structural:
            insert_index = 1 if normalized_steps and normalized_steps[0]["expert"] == "semantic" else len(normalized_steps)
            normalized_steps.insert(
                insert_index,
                {
                    "step_id": insert_index + 1,
                    "expert": "structural",
                    "step_type": "structural_check",
                    "goal": "Assess whole-subject structural coherence and class-specific morphology.",
                    "model_profile": "local_fast",
                    "planned_model": "animal_pose",
                    "selection_reason": "Default low-cost structural first pass with pose evidence.",
                    "prompt_focus": f"Inspect whole-subject coherence and the class-specific morphology, body proportions, pose plausibility, scene compatibility, and likely lookalike confusions for {label_text}.",
                    "depends_on": [1] if normalized_steps and normalized_steps[0]["expert"] == "semantic" else [],
                    "expected_signal": "Whole-subject coherence and morphology evidence.",
                    "use_previous_outputs": True,
                    "allow_escalation": True,
                },
            )
        if not has_quality:
            normalized_steps.append(
                {
                    "step_id": len(normalized_steps) + 1,
                    "expert": "quality",
                    "step_type": "quality_check",
                    "goal": "Estimate visible quality degradation and perceptual failure evidence.",
                    "model_profile": "local_default",
                    "planned_model": "iqa_default",
                    "selection_reason": "Default quality pass using standard IQA metrics.",
                    "prompt_focus": "Inspect localized quality degradation evidence, including malformed facial regions, duplicated or fused limbs, broken joints, tail attachment, hands or feet, and corrupted boundaries when relevant.",
                    "depends_on": [step["step_id"] for step in normalized_steps if step["expert"] in {"semantic", "structural"}],
                    "expected_signal": "Visible quality degradation evidence.",
                    "use_previous_outputs": True,
                    "allow_escalation": True,
                }
            )

        if not normalized_steps:
            raise LocalExpertError("Local planner returned no usable plan steps")

        for index, step in enumerate(normalized_steps, start=1):
            step["step_id"] = index

        return {
            "rationale": str(payload.get("rationale", "")).strip() or "Use a cost-aware multimodal evaluation plan.",
            "steps": normalized_steps,
        }

    def plan(self, *, image_path: str, prompt: str | None, class_label: str | None, prior_feedback: str = "") -> dict[str, Any]:
        if not self.settings.planner_local_enabled:
            raise LocalExpertError("Local planner is disabled")

        self._load()
        if self._processor is None or self._model is None or self._torch is None:
            raise LocalExpertError("Local planner failed to initialize")

        task_lines = [
            "You are the planning expert in an image generation evaluator.",
            "Return a short JSON plan with semantic, structural, quality, and optional vqa steps.",
            "Reason jointly from the image and the prompt/class label.",
            "Use semantic for category/class match, structural for whole-subject coherence and anatomy, quality for local visible degradation evidence, vqa only for unresolved questions.",
            # "For hard c2i cases, you may add candidate_generation or confusable_disambiguation semantic steps when useful.",
            "Prefer the cheapest adequate route first; use stronger routes only for ambiguous hard cases.",
            "Use only visible evidence; do not do external research.",
            "Each step must include the concrete planned_model expert key and a brief selection_reason.",
            "Use step_type to declare what the step is actually doing: semantic_check, structural_check, quality_check, vqa_evidence, candidate_generation, label_space_check, or confusable_disambiguation.",
            "Use depends_on when a later step should consume earlier outputs. Use expected_signal to say what evidence the step should produce.",
            "Keep rationale and goals brief.",
            "Different steps can have the same step_type but vary in expert, model, or prompt_focus for cross-validation. In this case, total steps can be greater than 3 but step_type options are limited.",
            "For c2i task (Class label provided), semantic checks can be conducted by classification models and/or confusable candidate_generation models plus CLIP-style matching models. For t2i task (Text prompt provided), semantic checks can only be conducted by matching models.",
            "Return valid JSON only.",
            'JSON schema: {"rationale":"string","steps":[{"step_id":1,"expert":"semantic|structural|quality|vqa","step_type":"semantic_check|structural_check|quality_check|vqa_evidence|candidate_generation|label_space_check|confusable_disambiguation","goal":"string","model_profile":"local_fast|local_default|local_richer|local_stronger","planned_model":"string","selection_reason":"string","prompt_focus":"string","depends_on":[1],"expected_signal":"string","use_previous_outputs":true,"allow_escalation":true}]}',
            f"Prompt: {prompt or 'N/A'}",
            f"Class label: {class_label or 'N/A'}",
        ]
        expert_catalog = _describe_available_experts(self.settings)
        task_lines.append(expert_catalog)
        if prior_feedback:
            task_lines.append(prior_feedback)

        user_text = "\n".join(task_lines)
        image = self._load_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        try:
            prompt_text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(self._model.device)
            generated_ids = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.settings.planner_local_max_new_tokens,
            )
            prompt_length = inputs["input_ids"].shape[1]
            trimmed_ids = [output_ids[prompt_length:] for output_ids in generated_ids]
            output_text = self._processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        except Exception as exc:  # noqa: BLE001
            raise LocalExpertError("Local planner inference failed") from exc

        return self._normalize_plan(extract_plan_payload(output_text), class_label=class_label)


def extract_plan_payload(text: str) -> dict[str, Any]:
    try:
        return extract_json(text)
    except LocalExpertError as exc:
        if "complete JSON object" not in str(exc):
            raise
        return {
            "rationale": "Planner output was truncated; use the fallback normalized plan.",
            "steps": [],
        }


def extract_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()
    if candidate.lower().startswith("json\n"):
        candidate = candidate[5:].strip()

    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    try:
        payload = yaml.safe_load(candidate)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    start = candidate.find("{")
    if start == -1:
        raise LocalExpertError("Local model did not return valid JSON")

    depth = 0
    in_string = False
    escaped = False
    end = -1
    for index, char in enumerate(candidate[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index
                break

    if end == -1:
        raise LocalExpertError("Local model did not return a complete JSON object")

    extracted = candidate[start:end + 1]
    try:
        payload = json.loads(extracted)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    try:
        payload = yaml.safe_load(extracted)
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        raise LocalExpertError("Local model did not return valid JSON") from exc

    raise LocalExpertError("Local model did not return valid JSON")


class LocalSemanticExpert:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._image_module: Any | None = None

    def _count_indicator_matches(self, text: str, indicators: tuple[str, ...]) -> int:
        return sum(1 for indicator in indicators if indicator in text)

    def _normalize_severity(self, payload: dict[str, Any]) -> float:
        raw_severity = payload.get("severity", 0.0)
        try:
            severity = float(raw_severity)
        except (TypeError, ValueError):
            severity = 0.0
        severity = max(0.0, min(1.0, severity))

        evidence_text = " ".join(
            [str(payload.get("summary", "")), *[str(item) for item in payload.get("findings", [])]]
        ).lower()
        positive_indicators = (
            "match",
            "matches",
            "matching",
            "aligned",
            "consistent with",
            "correctly depicts",
            "clearly shows",
            "resembles",
            "accurate",
        )
        negative_indicators = (
            "mismatch",
            "does not match",
            "doesn't match",
            "not match",
            "inconsistent",
            "incorrect",
            "wrong",
            "missing",
            "absent",
            "fails",
            "failure",
        )
        hedging_indicators = (
            "slightly",
            "minor",
            "partially",
            "somewhat",
            "broadly",
            "not uniquely",
            "uncertain",
            "ambigu",
        )

        positive_hits = self._count_indicator_matches(evidence_text, positive_indicators)
        negative_hits = self._count_indicator_matches(evidence_text, negative_indicators)
        hedging_hits = self._count_indicator_matches(evidence_text, hedging_indicators)

        if positive_hits > 0 and negative_hits == 0 and severity > 0.6:
            if hedging_hits > 0:
                return 0.35
            return 0.1
        if negative_hits > 0 and positive_hits == 0 and severity < 0.4:
            if hedging_hits > 0:
                return 0.55
            return 0.85
        return severity

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        bundle = _load_vlm_bundle(
            self.settings.local_semantic_model,
            self.settings.local_semantic_quantization,
            self.settings.local_semantic_device,
            "Failed to load local semantic model",
        )
        self._processor = bundle["processor"]
        self._model = bundle["model"]
        self._torch = bundle["torch"]
        self._image_module = bundle["image_module"]

    def _load_image(self, image_path: str):
        if self._image_module is None:
            raise LocalExpertError("PIL is not loaded")

        image = self._image_module.open(image_path).convert("RGB")
        max_side = self.settings.local_image_max_side
        image.thumbnail((max_side, max_side))
        return image

    def evaluate(self, *, image_path: str, prompt: str | None, class_label: str | None) -> dict[str, Any]:
        if not self.settings.local_semantic_enabled:
            raise LocalExpertError("Local semantic expert is disabled")

        self._load()
        if self._processor is None or self._model is None or self._torch is None:
            raise LocalExpertError("Local semantic expert failed to initialize")

        task_lines = [
            "You are the semantic expert in an image generation evaluator.",
            "Assess only semantic alignment between the image and the requested prompt/class label.",
            "Return valid JSON only.",
            "Use severity where 0 means no semantic issue and 1 means severe semantic mismatch.",
            "Use confidence where 0 means very uncertain and 1 means highly confident.",
            'JSON schema: {"expert":"semantic","summary":"string","findings":["string"],"severity":0.0,"confidence":0.0,"source":"local","model":"string"}',
        ]
        if prompt:
            task_lines.append(f"Prompt: {prompt}")
        if class_label:
            task_lines.append(f"Class label: {class_label}")

        user_text = "\n".join(task_lines)
        image = self._load_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        try:
            prompt_text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(self._model.device)
            generated_ids = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.settings.local_semantic_max_new_tokens,
            )
            prompt_length = inputs["input_ids"].shape[1]
            trimmed_ids = [output_ids[prompt_length:] for output_ids in generated_ids]
            output_text = self._processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        except Exception as exc:  # noqa: BLE001
            raise LocalExpertError("Local semantic inference failed") from exc

        payload = extract_json(output_text)
        payload.setdefault("expert", "semantic")
        payload["severity"] = round(self._normalize_severity(payload), 4)
        payload.setdefault("confidence", 0.0)
        payload.setdefault("source", "local")
        payload.setdefault("model", self.settings.local_semantic_model)
        return payload


class LocalQualityExpert:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pyiqa: Any | None = None
        self._metrics: dict[str, Any] = {}
        self._device = settings.local_artifact_device if (settings.local_artifact_device or "").strip().lower().startswith("cuda") else "cpu"

    @property
    def metric_names(self) -> list[str]:
        return [item.strip() for item in self.settings.local_artifact_metrics.split(",") if item.strip()]

    def _load(self) -> None:
        if self._metrics:
            return

        try:
            import pyiqa
            import torch
        except ImportError as exc:
            raise LocalExpertError(
                "Local quality dependencies are missing. Install pyiqa and torch."
            ) from exc

        if (self.settings.local_artifact_device or "").strip().lower().startswith("cuda") and not torch.cuda.is_available():
            self._device = "cpu"

        self._pyiqa = pyiqa
        for metric_name in self.metric_names:
            try:
                self._metrics[metric_name] = pyiqa.create_metric(metric_name, device=self._device)
            except Exception as exc:  # noqa: BLE001
                raise LocalExpertError(f"Failed to load local quality metric '{metric_name}'") from exc

    def _normalize_quality(self, metric_name: str, raw_score: float) -> float:
        score = max(0.0, raw_score)
        lowered = metric_name.lower()
        if lowered in {"maniqa", "clipiqa"}:
            if score <= 1.5:
                return min(score, 1.0)
            if score <= 10:
                return min(score / 10.0, 1.0)
            return min(score / 100.0, 1.0)
        if lowered == "musiq":
            if score <= 10:
                return min(score / 10.0, 1.0)
            return min(score / 100.0, 1.0)
        if score <= 1.5:
            return min(score, 1.0)
        if score <= 10:
            return min(score / 10.0, 1.0)
        return min(score / 100.0, 1.0)

    def _build_summary(self, severity: float) -> str:
        if severity >= 0.75:
            return "Local IQA metrics indicate severe perceptual degradation and likely visible generation artifacts."
        if severity >= 0.45:
            return "Local IQA metrics indicate noticeable image quality problems that may reflect generation artifacts."
        return "Local IQA metrics indicate limited perceptual degradation and no strong quality-failure signal."

    def evaluate(self, *, image_path: str) -> dict[str, Any]:
        if not self.settings.local_artifact_enabled:
            raise LocalExpertError("Local quality expert is disabled")

        self._load()
        if self._pyiqa is None or not self._metrics:
            raise LocalExpertError("Local quality expert failed to initialize")

        normalized_scores: list[float] = []
        findings: list[str] = []
        for metric_name, metric in self._metrics.items():
            try:
                value = float(metric(str(Path(image_path))).item())
            except Exception as exc:  # noqa: BLE001
                raise LocalExpertError(f"Quality metric '{metric_name}' inference failed") from exc

            quality = self._normalize_quality(metric_name, value)
            normalized_scores.append(quality)
            findings.append(f"{metric_name} raw score: {value:.4f}; normalized quality estimate: {quality:.2f}.")

        if not normalized_scores:
            raise LocalExpertError("No local quality scores were produced")

        average_quality = sum(normalized_scores) / len(normalized_scores)
        severity = 1.0 - average_quality
        confidence = 0.6
        if len(normalized_scores) >= 2:
            disagreement = pstdev(normalized_scores)
            confidence = max(0.0, min(1.0, 1.0 - (disagreement / 0.35)))

        if severity >= 0.75:
            findings.append("Multiple local quality metrics jointly point to severe degradation.")
        elif severity >= 0.45:
            findings.append("The local quality metrics are consistent with moderate visible degradation.")
        else:
            findings.append("The local quality metrics do not show a strong quality-failure signal.")

        return {
            "expert": "quality",
            "summary": self._build_summary(severity),
            "findings": findings,
            "severity": round(max(0.0, min(1.0, severity)), 4),
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "source": "local_iqa",
            "model": "+".join(self.metric_names),
        }


class LocalJudge:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._image_module: Any | None = None

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        bundle = _load_vlm_bundle(
            self.settings.judge_local_model,
            self.settings.local_semantic_quantization,
            self.settings.judge_device,
            "Failed to load local judge model",
        )
        self._processor = bundle["processor"]
        self._model = bundle["model"]
        self._torch = bundle["torch"]
        self._image_module = bundle["image_module"]

    def _load_image(self, image_path: str) -> Any:
        if self._image_module is None:
            raise LocalExpertError("Image module not loaded")
        return self._image_module.open(image_path).convert("RGB")

    def evaluate(self, *, image_path: str, plan: Any, prompt: str | None, class_label: str | None) -> dict[str, Any]:
        if not self.settings.judge_local_enabled:
            raise LocalExpertError("Local judge is disabled")

        self._load()
        if self._processor is None or self._model is None:
            raise LocalExpertError("Local judge failed to initialize")

        plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else plan
        plan_json = json.dumps(plan_dict, indent=2, ensure_ascii=False)

        task_lines = [
            "You are the judge expert in an image generation evaluator.",
            "Review the evaluation plan and decide if it is adequate.",
            "Approve only if the plan covers alignment, global structure, local quality evidence, uses VQA only when necessary, and clearly reasons jointly from the image and the prompt/class label.",
            "For semantic inspection, prefer an explicit broad-category versus fine-grained class distinction when the label is fine-grained.",
            "For structural inspection, require the plan to target the likely subject-specific failure modes instead of using a rote generic checklist.",
            "Do not reject solely because the plan omits color, lighting, or composition checks unless the class label or prompt specifically requires them.",
            "Check that each step's planned_model and selection_reason are consistent with the available model catalog and the stated task.",
            "When rejecting, provide structured replan actions so the planner can revise deterministically.",
            "Return valid JSON only.",
            'JSON schema: {"approved":true|false,"feedback":"string","missing_checks":["string"],"task_fit_issues":["string"],"replan_actions":[{"action":"add_step|retarget_step|replace_model|reorder_steps|tighten_task_fit|reweight_evidence|rerun_with_stronger_model","reason":"string","priority":"low|medium|high","target_step_id":1,"step_type":"semantic_check|structural_check|quality_check|vqa_evidence|candidate_generation|label_space_check|confusable_disambiguation","suggested_expert":"semantic|structural|quality|vqa|clip|clip_score|imagenet_fast|imagenet_strong|imagenet_eva02_large|imagenet_eva_giant_224|bge_candidate_generator|e5_candidate_generator|animal_pose|body_pose|body_pose_strong|hand_detection|face_detection|places365|places365_strong|building_expert|background_removal|complexity|iqa_fast|iqa_default|iqa_richer|boundary_artifact|aigen_detection|ocr|dog_breed|bird_expert","suggested_model":"string","prompt_focus":"string","expected_signal":"string"}]}',
            f"Prompt: {prompt or 'N/A'}",
            f"Class label: {class_label or 'N/A'}",
            _describe_available_experts(self.settings),
            f"Plan:\n{plan_json}",
        ]

        user_text = "\n".join(task_lines)
        image = self._load_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        try:
            prompt_text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(self._model.device)
            generated_ids = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.settings.judge_local_max_new_tokens,
            )
            prompt_length = inputs["input_ids"].shape[1]
            trimmed_ids = [output_ids[prompt_length:] for output_ids in generated_ids]
            output_text = self._processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        except Exception as exc:
            raise LocalExpertError("Local judge inference failed") from exc

        payload = extract_json(output_text)
        return {
            "approved": bool(payload.get("approved", False)),
            "feedback": str(payload.get("feedback", "")).strip(),
            "missing_checks": list(payload.get("missing_checks", [])),
            "task_fit_issues": list(payload.get("task_fit_issues", [])),
            "replan_actions": list(payload.get("replan_actions", [])),
        }


class LocalReflector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._image_module: Any | None = None

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        bundle = _load_vlm_bundle(
            self.settings.reflector_local_model,
            self.settings.local_semantic_quantization,
            self.settings.reflector_device,
            "Failed to load local reflector model",
        )
        self._processor = bundle["processor"]
        self._model = bundle["model"]
        self._torch = bundle["torch"]
        self._image_module = bundle["image_module"]

    def _load_image(self, image_path: str) -> Any:
        if self._image_module is None:
            raise LocalExpertError("Image module not loaded")
        return self._image_module.open(image_path).convert("RGB")

    def evaluate(self, *, image_path: str, report: dict[str, Any], expert_results: list[dict[str, Any]], 
                 reliability_summary: dict[str, Any], conflicts: list[dict[str, Any]], 
                 weighted_severity: float, prompt: str | None, class_label: str | None) -> dict[str, Any]:
        if not self.settings.reflector_local_enabled:
            raise LocalExpertError("Local reflector is disabled")

        self._load()
        if self._processor is None or self._model is None:
            raise LocalExpertError("Local reflector failed to initialize")

        report_json = json.dumps(report, indent=2, ensure_ascii=False)
        expert_json = json.dumps(expert_results, indent=2, ensure_ascii=False)
        conflicts_json = json.dumps(conflicts, indent=2, ensure_ascii=False) if conflicts else "None"
        reliability_json = json.dumps(reliability_summary, indent=2, ensure_ascii=False)

        task_lines = [
            "You are the reflector expert in an image generation evaluator.",
            "Act as a second-pass critic. Reinspect the image directly and look specifically for severe failures the experts or report may have missed.",
            "Use only visible evidence from the image. Distinguish broad category match from fine-grained class or species match.",
            "Do not name a specific alternative species unless at least two visible diagnostic traits support it; otherwise describe a broad-category match or fine-grained mismatch/uncertainty.",
            "Consider expert reliability when evaluating the report:",
            "- HIGH reliability experts (weight 1.0): If they report issues, trust them strongly",
            "- MEDIUM reliability experts (weight 0.7): Consider their findings but verify with visual evidence",
            "- LOW reliability experts (weight 0.4): Use as hints only, require stronger confirmation",
            "- When experts conflict, prioritize the one with higher reliability",
            "- If report relies heavily on LOW reliability experts, flag for additional verification",
            "- If weighted severity significantly differs from report severity, note the discrepancy",
            "Reject the report if it is too optimistic about species match, anatomy, appendages, limbs, extremities, tail attachment, duplicated parts, impossible structure, or boundary corruption.",
            "If visible evidence is ambiguous, treat the ambiguity as failure risk rather than assuming the image is clean.",
            "When rejecting, provide structured replan actions so the planner can revise deterministically.",
            "Return valid JSON only.",
            'JSON schema: {"approved":true|false,"feedback":"string","suggested_fixes":["string"],"task_fit_issues":["string"],"replan_actions":[{"action":"add_step|retarget_step|replace_model|reorder_steps|tighten_task_fit|reweight_evidence|rerun_with_stronger_model","reason":"string","priority":"low|medium|high","target_step_id":1,"step_type":"semantic_check|structural_check|quality_check|vqa_evidence|candidate_generation|label_space_check|confusable_disambiguation","suggested_expert":"semantic|structural|quality|vqa|clip|clip_score|imagenet_fast|imagenet_strong|imagenet_eva02_large|imagenet_eva_giant_224|bge_candidate_generator|e5_candidate_generator|animal_pose|body_pose|body_pose_strong|hand_detection|face_detection|places365|places365_strong|building_expert|background_removal|complexity|iqa_fast|iqa_default|iqa_richer|boundary_artifact|aigen_detection|ocr|dog_breed|bird_expert","suggested_model":"string","prompt_focus":"string","expected_signal":"string"}]}',
            f"Prompt: {prompt or 'N/A'}",
            f"Class label: {class_label or 'N/A'}",
            f"Expert outputs:\n{expert_json}",
            "Each expert output may include output_meaning and output_interpretability metadata. Use those fields to interpret what the expert's severity, findings, and extra_info actually mean before deciding whether the report handled the expert evidence correctly.",
            f"Detected conflicts:\n{conflicts_json}",
            f"Weighted severity: {weighted_severity:.3f}",
            f"Reliability summary:\n{reliability_json}",
            f"Report:\n{report_json}",
        ]

        user_text = "\n".join(task_lines)
        image = self._load_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        try:
            prompt_text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(self._model.device)
            generated_ids = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.settings.reflector_local_max_new_tokens,
            )
            prompt_length = inputs["input_ids"].shape[1]
            trimmed_ids = [output_ids[prompt_length:] for output_ids in generated_ids]
            output_text = self._processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        except Exception as exc:
            raise LocalExpertError("Local reflector inference failed") from exc

        payload = extract_json(output_text)
        return {
            "approved": bool(payload.get("approved", False)),
            "feedback": str(payload.get("feedback", "")).strip(),
            "suggested_fixes": list(payload.get("suggested_fixes", [])),
            "task_fit_issues": list(payload.get("task_fit_issues", [])),
            "replan_actions": list(payload.get("replan_actions", [])),
        }


class LocalStructuralExpert:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._image_module: Any | None = None

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        bundle = _load_vlm_bundle(
            self.settings.local_semantic_model,
            self.settings.local_semantic_quantization,
            self.settings.local_semantic_device,
            "Failed to load local structural model",
        )
        self._processor = bundle["processor"]
        self._model = bundle["model"]
        self._torch = bundle["torch"]
        self._image_module = bundle["image_module"]

    def _load_image(self, image_path: str):
        if self._image_module is None:
            raise LocalExpertError("PIL is not loaded")

        image = self._image_module.open(image_path).convert("RGB")
        max_side = self.settings.local_image_max_side
        image.thumbnail((max_side, max_side))
        return image

    def evaluate(self, *, image_path: str, prompt: str | None, class_label: str | None, prompt_focus: str | None = None) -> dict[str, Any]:
        self._load()
        if self._processor is None or self._model is None or self._torch is None:
            raise LocalExpertError("Local structural expert failed to initialize")

        task_lines = [
            "You are the structural expert in an image generation evaluator.",
            "Assess whole-subject structural coherence, anatomy, and part attachment integrity only.",
            "Focus on whether the visible subject forms a coherent instance of the requested class.",
            "Pay special attention to malformed face or muzzle regions, duplicated or fused limbs, broken joints, wrong tail attachment, malformed hands or feet, asymmetric anatomy, and impossible boundaries when relevant.",
            "Return valid JSON only.",
            "Use severity where 0 means no structural issue and 1 means severe structural failure.",
            "Use confidence where 0 means very uncertain and 1 means highly confident.",
            'JSON schema: {"expert":"structural","summary":"string","findings":["string"],"severity":0.0,"confidence":0.0,"source":"local","model":"string"}',
        ]
        if prompt:
            task_lines.append(f"Prompt: {prompt}")
        if class_label:
            task_lines.append(f"Class label: {class_label}")
        if prompt_focus:
            task_lines.append(f"Prompt focus: {prompt_focus}")

        user_text = "\n".join(task_lines)
        image = self._load_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        try:
            prompt_text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(self._model.device)
            generated_ids = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.settings.local_semantic_max_new_tokens,
            )
            prompt_length = inputs["input_ids"].shape[1]
            trimmed_ids = [output_ids[prompt_length:] for output_ids in generated_ids]
            output_text = self._processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        except Exception as exc:
            raise LocalExpertError("Local structural inference failed") from exc

        payload = extract_json(output_text)
        payload.setdefault("expert", "structural")
        raw_severity = payload.get("severity", 0.0)
        try:
            payload["severity"] = round(max(0.0, min(1.0, float(raw_severity))), 4)
        except (TypeError, ValueError):
            payload["severity"] = 0.0
        payload.setdefault("confidence", 0.0)
        payload.setdefault("source", "local")
        payload.setdefault("model", self.settings.local_semantic_model)
        return payload


class LocalVQAExpert:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._image_module: Any | None = None

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        bundle = _load_vlm_bundle(
            self.settings.local_semantic_model,
            self.settings.local_semantic_quantization,
            self.settings.local_semantic_device,
            "Failed to load local VQA model",
        )
        self._processor = bundle["processor"]
        self._model = bundle["model"]
        self._torch = bundle["torch"]
        self._image_module = bundle["image_module"]

    def _load_image(self, image_path: str):
        if self._image_module is None:
            raise LocalExpertError("PIL is not loaded")

        image = self._image_module.open(image_path).convert("RGB")
        max_side = self.settings.local_image_max_side
        image.thumbnail((max_side, max_side))
        return image

    def _normalize_structured_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload.setdefault("expert", "vqa")
        payload.setdefault("answer", "")
        payload.setdefault("summary", str(payload.get("answer", "")).strip())
        payload["findings"] = list(payload.get("findings", []))
        payload["evidence_items"] = [str(item).strip() for item in payload.get("evidence_items", []) if str(item).strip()]
        payload["visible_support"] = [str(item).strip() for item in payload.get("visible_support", []) if str(item).strip()]
        payload["visible_uncertainties"] = [str(item).strip() for item in payload.get("visible_uncertainties", []) if str(item).strip()]
        payload["follow_up_questions"] = [str(item).strip() for item in payload.get("follow_up_questions", []) if str(item).strip()]
        raw_severity = payload.get("severity", 0.0)
        raw_confidence = payload.get("confidence", 0.0)
        try:
            payload["severity"] = round(max(0.0, min(1.0, float(raw_severity))), 4)
        except (TypeError, ValueError):
            payload["severity"] = 0.0
        try:
            payload["confidence"] = round(max(0.0, min(1.0, float(raw_confidence))), 4)
        except (TypeError, ValueError):
            payload["confidence"] = 0.0
        payload.setdefault("source", "local")
        payload.setdefault("model", self.settings.local_semantic_model)
        payload["summary"] = str(payload.get("summary", "")).strip() or str(payload.get("answer", "")).strip()
        payload["findings"] = payload["findings"] or payload["evidence_items"]
        return payload

    def evaluate(self, *, image_path: str, prompt: str | None, class_label: str | None, goal: str | None = None, prompt_focus: str | None = None) -> dict[str, Any]:
        self._load()
        if self._processor is None or self._model is None or self._torch is None:
            raise LocalExpertError("Local VQA expert failed to initialize")

        task_lines = [
            "You are the VQA expert in an image generation evaluator.",
            "Follow a strict structured evidence extraction protocol.",
            "Answer only the unresolved visual question needed for evaluation.",
            "Use only directly visible evidence from the image; do not guess hidden attributes or unseen taxonomy.",
            "If evidence is ambiguous, state uncertainty explicitly instead of over-claiming.",
            "Return valid JSON only.",
            "Use severity where 0 means no issue was found and 1 means the answer reveals a severe issue.",
            "Use confidence where 0 means very uncertain and 1 means highly confident.",
            'JSON schema: {"expert":"vqa","answer":"string","summary":"string","findings":["string"],"evidence_items":["string"],"visible_support":["string"],"visible_uncertainties":["string"],"follow_up_questions":["string"],"severity":0.0,"confidence":0.0,"source":"local","model":"string"}',
        ]
        if prompt:
            task_lines.append(f"Prompt: {prompt}")
        if class_label:
            task_lines.append(f"Class label: {class_label}")
        if goal:
            task_lines.append(f"Goal: {goal}")
        if prompt_focus:
            task_lines.append(f"Prompt focus: {prompt_focus}")

        user_text = "\n".join(task_lines)
        image = self._load_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        try:
            prompt_text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(self._model.device)
            generated_ids = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.settings.local_semantic_max_new_tokens,
            )
            prompt_length = inputs["input_ids"].shape[1]
            trimmed_ids = [output_ids[prompt_length:] for output_ids in generated_ids]
            output_text = self._processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        except Exception as exc:
            raise LocalExpertError("Local VQA inference failed") from exc

        payload = extract_json(output_text)
        return self._normalize_structured_payload(payload)


class LocalReport:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._image_module: Any | None = None

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        report_model = self.settings.reflector_local_model or self.settings.judge_local_model or self.settings.local_semantic_model
        report_device = self.settings.reflector_device or self.settings.judge_device or self.settings.local_semantic_device
        bundle = _load_vlm_bundle(
            report_model,
            self.settings.local_semantic_quantization,
            report_device,
            "Failed to load local report model",
        )
        self._processor = bundle["processor"]
        self._model = bundle["model"]
        self._torch = bundle["torch"]
        self._image_module = bundle["image_module"]

    def _load_image(self, image_path: str):
        if self._image_module is None:
            raise LocalExpertError("PIL is not loaded")

        image = self._image_module.open(image_path).convert("RGB")
        max_side = self.settings.local_image_max_side
        image.thumbnail((max_side, max_side))
        return image

    def evaluate(self, *, image_path: str, expert_results: list[dict[str, Any]], conflicts: list[dict[str, Any]], weighted_severity: float, prompt: str | None, class_label: str | None) -> dict[str, Any]:
        self._load()
        if self._processor is None or self._model is None or self._torch is None:
            raise LocalExpertError("Local report model failed to initialize")

        expert_json = json.dumps(expert_results, indent=2, ensure_ascii=False)
        conflicts_json = json.dumps(conflicts, indent=2, ensure_ascii=False) if conflicts else "None"
        task_lines = [
            "You synthesize expert evidence into an evaluation report, but you are not bound by the experts if the image itself suggests they missed a serious failure.",
            "Directly inspect the image again while reading the expert outputs.",
            "Use only visible evidence from the image. Distinguish broad category match from fine-grained class or species match.",
            "Do not name a specific alternative species unless at least two visible diagnostic traits support it; otherwise describe a broad-category match or fine-grained mismatch/uncertainty.",
            "Use a conservative standard: when visible evidence suggests species mismatch, impossible anatomy, extra appendages, malformed extremities, duplicated limbs, broken joints, wrong tail attachment, or severe boundary corruption, lower the scores accordingly.",
            "Artifact score is a quality score where 1 means minimal visible artifacts and 0 means severe visible artifacts.",
            "For artifact assessment, prioritize visible anatomy, boundaries, texture consistency, duplicated or melted parts, implausible structure, and other visible generation failures over generic perceptual pleasantness.",
            "If broad category matches but fine-grained class evidence is weak, keep alignment in the partial-match range rather than treating it as a clean match.",
            "Set hard_failure true when the image shows severe species mismatch or severe anatomical or structural generation failure.",
            "Return valid JSON only.",
            'JSON schema: {"alignment_reasoning":"string","artifact_reasoning":"string","alignment_score":0.0,"artifact_score":0.0,"hard_failure":false,"confidence":0.0,"key_issues":["string"]}',
            f"Prompt: {prompt or 'N/A'}",
            f"Class label: {class_label or 'N/A'}",
            f"Expert outputs:\n{expert_json}",
            "Each expert output may include output_meaning and output_interpretability metadata. Use those fields to interpret what the expert's severity, findings, and extra_info actually mean before deciding whether the result supports artifact or alignment concerns.",
            f"Detected conflicts:\n{conflicts_json}",
            f"Weighted severity: {weighted_severity:.3f}",
        ]

        user_text = "\n".join(task_lines)
        image = self._load_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        try:
            prompt_text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(self._model.device)
            generated_ids = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.settings.reflector_local_max_new_tokens,
            )
            prompt_length = inputs["input_ids"].shape[1]
            trimmed_ids = [output_ids[prompt_length:] for output_ids in generated_ids]
            output_text = self._processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        except Exception as exc:
            raise LocalExpertError("Local report inference failed") from exc

        payload = extract_json(output_text)
        for key in ("alignment_score", "artifact_score", "confidence"):
            try:
                payload[key] = round(max(0.0, min(1.0, float(payload.get(key, 0.0)))), 4)
            except (TypeError, ValueError):
                payload[key] = 0.0
        payload["hard_failure"] = bool(payload.get("hard_failure", False))
        payload["alignment_reasoning"] = str(payload.get("alignment_reasoning", "")).strip()
        payload["artifact_reasoning"] = str(payload.get("artifact_reasoning", "")).strip()
        payload["key_issues"] = list(payload.get("key_issues", []))
        return payload

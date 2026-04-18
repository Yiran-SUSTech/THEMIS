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


def _dynamic_candidate_keys(settings: Settings, expert: str) -> list[str]:
    matched: list[str] = []
    for key, config in _all_downloaded_expert_items(settings):
        if _matches_role_metadata(expert, config):
            matched.append(key)
    return matched


def _is_downloaded_planned_model(settings: Settings, planned_model: str) -> bool:
    config = settings.get_expert_config(planned_model)
    return config is not None and bool(getattr(config, "downloaded", False))


def _roles_for_expert_key(expert_key: str) -> list[str]:
    roles: list[str] = []
    for role, keys in EXPERT_ROLE_MODEL_KEYS.items():
        if expert_key in keys:
            roles.append(role)
    return roles


def _describe_available_experts(settings: Settings) -> str:
    lines = ["Available downloaded model options by expert role:"]
    included_keys: set[str] = set()
    for expert in EXPERT_ROLE_MODEL_KEYS:
        role_candidates = []
        for key in _dynamic_candidate_keys(settings, expert):
            config = settings.get_expert_config(key)
            perf = get_expert_performance(key)
            if config is None or not getattr(config, "downloaded", False):
                continue
            included_keys.add(key)
            model_name = config.local_path or config.weights or config.model
            parts = [f"key={key}", config.name, f"group={config.group_name}", f"model_type={config.model_type}", f"model={model_name}"]
            if config.model_size_mb is not None:
                parts.append(f"model_size_mb={float(config.model_size_mb):.2f}")
            if config.gflops is not None:
                parts.append(f"gflops={float(config.gflops):.2f}")
            if config.metrics:
                parts.append(f"metrics={','.join(config.metrics)}")
            if perf is not None:
                stats = []
                if perf.accuracy is not None:
                    stats.append(f"perf_acc={perf.accuracy:.1%}")
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
            role_candidates.append(" | ".join(parts))
        lines.append(f"- {expert}:")
        if role_candidates:
            lines.extend([f"  - {item}" for item in role_candidates])
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
            model_name = config.local_path or config.weights or config.model
            roles = _roles_for_expert_key(key)
            parts = [
                f"key={key}",
                config.name,
                f"group={config.group_name}",
                f"model_type={config.model_type}",
                f"model={model_name}",
                f"roles={','.join(roles) if roles else 'none'}",
            ]
            if config.model_size_mb is not None:
                parts.append(f"model_size_mb={float(config.model_size_mb):.2f}")
            if config.gflops is not None:
                parts.append(f"gflops={float(config.gflops):.2f}")
            if config.accuracy is not None:
                parts.append(f"accuracy={config.accuracy:.4f}")
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

    def _normalize_plan(self, payload: dict[str, Any], class_label: str | None = None) -> dict[str, Any]:
        steps = payload.get("steps")
        if not isinstance(steps, list):
            raise LocalExpertError("Local planner did not return a valid steps list")

        label_text = (class_label or "the labeled subject").strip() or "the labeled subject"
        normalized_steps: list[dict[str, Any]] = []
        seen_step_ids: set[int] = set()
        default_step_types = {
            "semantic": "semantic_check",
            "structural": "structural_check",
            "quality": "quality_check",
            "vqa": "vqa_evidence",
        }

        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            expert = str(step.get("expert", "")).strip().lower()
            if expert not in {"semantic", "structural", "quality", "vqa"}:
                continue
            step_id = step.get("step_id", index)
            if isinstance(step_id, str) and step_id.isdigit():
                step_id = int(step_id)
            if not isinstance(step_id, int) or step_id <= 0 or step_id in seen_step_ids:
                step_id = index
            seen_step_ids.add(step_id)
            step_type = str(step.get("step_type", default_step_types[expert])).strip() or default_step_types[expert]
            planned_model = str(step.get("planned_model", "")).strip()
            if not planned_model or not _is_downloaded_planned_model(self.settings, planned_model):
                continue
            prompt_focus = str(step.get("prompt_focus", "")).strip()
            if not prompt_focus:
                if expert == "semantic" and step_type == "candidate_generation":
                    prompt_focus = f"Generate confusable candidate labels visually close to {label_text}."
                elif expert == "semantic" and step_type == "confusable_disambiguation":
                    prompt_focus = f"Disambiguate {label_text} against confusable candidates using visible evidence."
                elif expert == "semantic":
                    prompt_focus = f"Inspect whether the visible subject matches {label_text}."
                elif expert == "structural":
                    prompt_focus = f"Inspect whole-subject coherence, morphology, and pose plausibility for {label_text}."
                elif expert == "quality":
                    prompt_focus = "Inspect visible artifacts, distortions, and malformed local regions."
                else:
                    prompt_focus = "Extract only the unresolved visual evidence needed by the plan."
            depends_on = [int(item) for item in step.get("depends_on", []) if isinstance(item, int) or (isinstance(item, str) and item.isdigit())]
            normalized_steps.append(
                {
                    "step_id": step_id,
                    "expert": expert,
                    "step_type": step_type,
                    "goal": str(step.get("goal", "")).strip() or f"Inspect {expert} evidence.",
                    "planned_model": planned_model,
                    "selection_reason": str(step.get("selection_reason", "")).strip() or f"Planner selected {planned_model} for this step.",
                    "prompt_focus": prompt_focus,
                    "depends_on": sorted(dict.fromkeys(dep for dep in depends_on if dep > 0 and dep != step_id)),
                    "expected_signal": str(step.get("expected_signal", "")).strip(),
                    "use_previous_outputs": bool(step.get("use_previous_outputs", False)),
                    "allow_escalation": bool(step.get("allow_escalation", True)),
                }
            )

        if not normalized_steps:
            raise LocalExpertError("Local planner returned no usable plan steps")

        step_ids = {step["step_id"] for step in normalized_steps}
        for step in normalized_steps:
            step["depends_on"] = [dep for dep in step["depends_on"] if dep in step_ids and dep < step["step_id"]]

        return {
            "rationale": str(payload.get("rationale", "")).strip() or "Use the planner-selected real expert models to evaluate the image.",
            "steps": normalized_steps,
        }

    def plan(self, *, image_path: str, prompt: str | None, class_label: str | None, prior_feedback: str = "") -> dict[str, Any]:
        if not self.settings.planner_local_enabled:
            raise LocalExpertError("Local planner is disabled")

        self._load()
        if self._processor is None or self._model is None or self._torch is None:
            raise LocalExpertError("Local planner failed to initialize")

        task_lines = [
            "Plan the image evaluation and return exactly one JSON object.",
            "No markdown, no code fences, no explanation before or after JSON.",
            "Use semantic, structural, quality, and optional vqa only if unresolved evidence remains.",
            "Use only visible image evidence plus the given prompt/class label.",
            "Choose directly from the real downloaded expert model keys shown below.",
            "Use the exposed model_size_mb, gflops, description, accuracy, benchmark, and suitability metadata to trade off cost and task fit.",
            "Do not output abstract profiles or placeholder model families.",
            "Judge handles bad model choices later, so keep your plan faithful to your own model selection.",
            "For c2i, if a semantic step uses CLIP, prefer candidate_generation with bge_candidate_generator or e5_candidate_generator before a confusable_disambiguation CLIP step.",
            "Every step must include planned_model, selection_reason, prompt_focus, depends_on, expected_signal, use_previous_outputs, allow_escalation.",
            "Keep rationale and goals brief.",
            'JSON schema: {"rationale":"string","steps":[{"step_id":1,"expert":"semantic|structural|quality|vqa","step_type":"semantic_check|structural_check|quality_check|vqa_evidence|candidate_generation|label_space_check|confusable_disambiguation","goal":"string","planned_model":"string","selection_reason":"string","prompt_focus":"string","depends_on":[1],"expected_signal":"string","use_previous_outputs":true,"allow_escalation":true}]}',
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

        _write_role_raw_output(self.settings, "planner_raw_output.txt", output_text)

        return self._normalize_plan(extract_plan_payload(output_text), class_label=class_label)


def extract_plan_payload(text: str) -> dict[str, Any]:
    def _looks_like_plan_payload(payload: dict[str, Any]) -> bool:
        return isinstance(payload.get("steps"), list) or "rationale" in payload

    try:
        payload = extract_json(text)
        if _looks_like_plan_payload(payload):
            return payload
    except LocalExpertError:
        pass

    for payload in extract_top_level_json_objects(text):
        if _looks_like_plan_payload(payload):
            return payload

    return {
        "rationale": "Planner output could not be parsed; use the fallback normalized plan.",
        "steps": [],
    }


def extract_top_level_json_objects(text: str) -> list[dict[str, Any]]:
    candidate = text.strip()
    objects: list[dict[str, Any]] = []
    depth = 0
    in_string = False
    escaped = False
    start: int | None = None

    for index, char in enumerate(candidate):
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
            continue

        if char == "{":
            if depth == 0:
                start = index
            depth += 1
            continue

        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                extracted = candidate[start:index + 1]
                for loader in (json.loads, yaml.safe_load):
                    try:
                        payload = loader(extracted)
                    except Exception:
                        continue
                    if isinstance(payload, dict):
                        objects.append(payload)
                        break
                start = None

    return objects


def extract_structured_payload(text: str, required_keys: list[str], fallback_message: str) -> dict[str, Any]:
    def _matches(payload: dict[str, Any]) -> bool:
        return all(key in payload for key in required_keys)

    try:
        payload = extract_json(text)
        if _matches(payload):
            return payload
    except LocalExpertError:
        pass

    for payload in reversed(extract_top_level_json_objects(text)):
        if _matches(payload):
            return payload

    fallback: dict[str, Any] = {key: [] for key in required_keys if key.endswith("_checks") or key.endswith("_issues") or key.endswith("_actions") or key.endswith("_fixes")}
    for key in required_keys:
        if key not in fallback:
            if key == "approved":
                fallback[key] = False
            else:
                fallback[key] = fallback_message if key == "feedback" else [] if key.endswith("s") else ""
    fallback.setdefault("feedback", fallback_message)
    fallback.setdefault("approved", False)
    return fallback


def extract_report_payload(text: str) -> dict[str, Any]:
    required_keys = [
        "alignment_reasoning",
        "artifact_reasoning",
        "alignment_score",
        "artifact_score",
        "hard_failure",
        "confidence",
        "key_issues",
    ]

    def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "alignment_reasoning": str(payload.get("alignment_reasoning", "")).strip() or "Report output was partially parsed; using fallback alignment reasoning.",
            "artifact_reasoning": str(payload.get("artifact_reasoning", "")).strip() or "Report output was partially parsed; using fallback artifact reasoning.",
            "alignment_score": payload.get("alignment_score", 0.5),
            "artifact_score": payload.get("artifact_score", 0.5),
            "hard_failure": bool(payload.get("hard_failure", False)),
            "confidence": payload.get("confidence", 0.2),
            "key_issues": list(payload.get("key_issues", [])) or ["Report output could not be fully parsed; using conservative fallback."],
        }

    try:
        payload = extract_json(text)
        if isinstance(payload, dict) and any(key in payload for key in required_keys):
            return _normalize(payload)
    except LocalExpertError:
        pass

    for payload in reversed(extract_top_level_json_objects(text)):
        if isinstance(payload, dict) and any(key in payload for key in required_keys):
            return _normalize(payload)

    return {
        "alignment_reasoning": "Report output could not be parsed; using conservative fallback alignment reasoning.",
        "artifact_reasoning": "Report output could not be parsed; using conservative fallback artifact reasoning.",
        "alignment_score": 0.5,
        "artifact_score": 0.5,
        "hard_failure": False,
        "confidence": 0.2,
        "key_issues": ["Report output could not be parsed; using conservative fallback."],
    }


def _write_role_raw_output(settings: Settings, filename: str, output_text: str) -> None:
    if not settings.log_dir:
        return
    try:
        log_path = Path(settings.log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        (log_path / filename).write_text(output_text, encoding="utf-8")
    except Exception:
        pass


def extract_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()
    if candidate.lower().startswith("json\n"):
        candidate = candidate[5:].strip()

    for loader in (json.loads, yaml.safe_load):
        try:
            payload = loader(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload

    extracted_objects = extract_top_level_json_objects(candidate)
    if extracted_objects:
        return extracted_objects[-1]

    if "{" in candidate:
        raise LocalExpertError("Local model did not return a complete JSON object")
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

        _write_role_raw_output(self.settings, "semantic_raw_output.txt", output_text)
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
            "Review the plan and return exactly one JSON object.",
            "No markdown, no code fences, no explanation before or after JSON.",
            "Approve if the plan covers semantic alignment, whole-subject structure, and artifact-focused quality evidence with reasonable model choices.",
            "Do not reject for missing color, lighting, or composition checks unless the task explicitly requires them.",
            "For fine-grained c2i, prefer candidate_generation plus CLIP confusable_disambiguation when CLIP is used.",
            "When rejecting, keep feedback short and provide deterministic replan_actions.",
            'JSON schema: {"approved":true|false,"feedback":"string","missing_checks":["string"],"task_fit_issues":["string"],"replan_actions":[{"action":"add_step|retarget_step|replace_model|reorder_steps|tighten_task_fit|reweight_evidence|rerun_with_stronger_model","reason":"string","priority":"low|medium|high","target_step_id":1,"step_type":"semantic_check|structural_check|quality_check|vqa_evidence|candidate_generation|label_space_check|confusable_disambiguation","suggested_expert":"semantic|structural|quality|vqa|clip|clip_score|imagenet_fast|imagenet_strong|imagenet_eva02_large|imagenet_eva_giant_224|bge_candidate_generator|e5_candidate_generator|animal_pose|body_pose|body_pose_strong|hand_detection|face_detection|places365|places365_strong|building_expert|background_removal|complexity|iqa_fast|iqa_default|iqa_richer|q_insight|boundary_artifact|aigen_detection|ocr|dog_breed|bird_expert","suggested_model":"string","prompt_focus":"string","expected_signal":"string"}]}',
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

        _write_role_raw_output(self.settings, "judge_raw_output.txt", output_text)
        payload = extract_structured_payload(
            output_text,
            ["approved", "feedback", "missing_checks", "task_fit_issues", "replan_actions"],
            "Judge output could not be parsed; request replanning with a shorter, cleaner JSON response.",
        )
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
            "Review the report and return exactly one JSON object.",
            "No markdown, no code fences, no explanation before or after JSON.",
            "Reinspect the image directly and reject only if the report is materially too optimistic or ignores important expert evidence.",
            "Use visible evidence only. Treat ambiguity as risk, not as proof of correctness.",
            "When rejecting, keep feedback short and provide deterministic replan_actions.",
            'JSON schema: {"approved":true|false,"feedback":"string","suggested_fixes":["string"],"task_fit_issues":["string"],"replan_actions":[{"action":"add_step|retarget_step|replace_model|reorder_steps|tighten_task_fit|reweight_evidence|rerun_with_stronger_model","reason":"string","priority":"low|medium|high","target_step_id":1,"step_type":"semantic_check|structural_check|quality_check|vqa_evidence|candidate_generation|label_space_check|confusable_disambiguation","suggested_expert":"semantic|structural|quality|vqa|clip|clip_score|imagenet_fast|imagenet_strong|imagenet_eva02_large|imagenet_eva_giant_224|bge_candidate_generator|e5_candidate_generator|animal_pose|body_pose|body_pose_strong|hand_detection|face_detection|places365|places365_strong|building_expert|background_removal|complexity|iqa_fast|iqa_default|iqa_richer|q_insight|boundary_artifact|aigen_detection|ocr|dog_breed|bird_expert","suggested_model":"string","prompt_focus":"string","expected_signal":"string"}]}',
            f"Prompt: {prompt or 'N/A'}",
            f"Class label: {class_label or 'N/A'}",
            f"Expert outputs:\n{expert_json}",
            "Each expert output may include output_meaning and output_interpretability metadata. Use those fields to interpret what the expert's severity, findings, and extra_info actually mean before deciding whether the report handled the expert evidence correctly.",
            f"Detected conflicts:\n{conflicts_json}",
            f"Weighted severity (diagnostic only): {weighted_severity:.3f}",
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

        _write_role_raw_output(self.settings, "reflector_raw_output.txt", output_text)
        payload = extract_structured_payload(
            output_text,
            ["approved", "feedback", "suggested_fixes", "task_fit_issues", "replan_actions"],
            "Reflector output could not be parsed; request replanning with a shorter, cleaner JSON response.",
        )
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

        _write_role_raw_output(self.settings, "structural_raw_output.txt", output_text)
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

        _write_role_raw_output(self.settings, "vqa_raw_output.txt", output_text)
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
            "Return exactly one JSON report object.",
            "No markdown, no code fences, no explanation before or after JSON.",
            # "Reinspect the image directly while considering expert outputs.",
            "Alignment score is only about semantic match. Artifact score is only about visible artifact or structural failure severity, where 1 means clean and 0 means severe failure.",
            "If broad category matches but fine-grained class evidence is weak, keep alignment partial.",
            "Set hard_failure true only for severe structural or artifact failure, or a severe semantic mismatch clearly visible in the image.",
            "Keep both reasoning strings short.",
            'JSON schema: {"alignment_reasoning":"string","artifact_reasoning":"string","alignment_score":0.0,"artifact_score":0.0,"hard_failure":false,"confidence":0.0,"key_issues":["string"]}',
            f"Prompt: {prompt or 'N/A'}",
            f"Class label: {class_label or 'N/A'}",
            f"Expert outputs:\n{expert_json}",
            "Each expert output may include output_meaning and output_interpretability metadata. Use those fields to interpret what the expert's severity, findings, and extra_info actually mean before deciding whether the result supports artifact or alignment concerns.",
            f"Detected conflicts:\n{conflicts_json}",
            f"Weighted severity (diagnostic only): {weighted_severity:.3f}",
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

        _write_role_raw_output(self.settings, "report_raw_output.txt", output_text)
        payload = extract_report_payload(output_text)
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

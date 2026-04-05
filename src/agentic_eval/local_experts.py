from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import pstdev
from threading import Lock
from typing import Any, Optional


VALID_MODEL_PROFILES = {
    "local_fast",
    "local_default",
    "local_richer",
    "local_stronger",
    "remote_mid",
    "remote_strong",
}

EXPERT_ALLOWED_MODEL_PROFILES = {
    "semantic": {"local_fast", "local_stronger", "remote_strong"},
    "artifact": {"local_fast", "local_default", "local_richer", "remote_strong"},
    "structural": {"remote_mid", "remote_strong"},
    "vqa": {"remote_mid", "remote_strong"},
}

_LOCAL_VLM_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}
_LOCAL_VLM_CACHE_LOCK = Lock()

from .config import Settings


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
    if torch.cuda.is_available() and device == "cuda":
        load_kwargs["device_map"] = "auto"
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


def _load_vlm_bundle(model_id: str, quantization: Optional[str], device: str, error_prefix: str) -> dict[str, Any]:
    quantization_key = quantization.lower() if quantization else "none"
    cache_key = (model_id, quantization_key, device)
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
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
            model = model_cls.from_pretrained(model_id, local_files_only=True, **load_kwargs)
            model.eval()
        except Exception as exc:  # noqa: BLE001
            raise LocalExpertError(f"{error_prefix} '{model_id}'. Download it manually or check your Hugging Face access.") from exc
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
            self.settings.local_semantic_device,
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
        allowed_profiles = EXPERT_ALLOWED_MODEL_PROFILES.get(expert, {"remote_mid"})
        if candidate in VALID_MODEL_PROFILES and candidate in allowed_profiles:
            return candidate
        if expert == "semantic":
            return "local_fast"
        if expert == "artifact":
            return "local_default"
        if expert == "structural":
            return "remote_mid"
        if expert == "vqa":
            return "remote_mid"
        return "remote_mid"

    def _normalize_plan(self, payload: dict[str, Any], class_label: str | None = None) -> dict[str, Any]:
        steps = payload.get("steps")
        if not isinstance(steps, list):
            raise LocalExpertError("Local planner did not return a valid steps list")

        label_text = (class_label or "the labeled subject").strip() or "the labeled subject"
        normalized_steps: list[dict[str, Any]] = []
        seen_experts: set[str] = set()
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            expert = str(step.get("expert", "semantic")).strip().lower() or "semantic"
            if expert not in {"semantic", "structural", "artifact", "vqa"}:
                continue
            if expert in seen_experts:
                continue
            seen_experts.add(expert)
            prompt_focus = str(step.get("prompt_focus", "")).strip()
            if not prompt_focus:
                if expert == "semantic":
                    prompt_focus = f"Inspect whether the visible subject matches {label_text} using the strongest confirming and contradicting species markers in the image."
                elif expert == "structural":
                    prompt_focus = f"Inspect whole-subject coherence and the class-specific morphology, body proportions, pose plausibility, and likely lookalike confusions for {label_text}."
                elif expert == "artifact":
                    prompt_focus = "Inspect localized generation artifacts, including malformed facial regions, duplicated or fused limbs, broken joints, tail attachment, hands or feet, and corrupted boundaries when relevant."
            normalized_steps.append(
                {
                    "step_id": len(normalized_steps) + 1,
                    "expert": expert,
                    "goal": str(step.get("goal", "")).strip() or f"Inspect {expert} evidence.",
                    "model_profile": self._normalize_model_profile(expert, str(step.get("model_profile", ""))),
                    "prompt_focus": prompt_focus,
                    "allow_escalation": bool(step.get("allow_escalation", True)),
                }
            )

        has_semantic = any(step["expert"] == "semantic" for step in normalized_steps)
        has_structural = any(step["expert"] == "structural" for step in normalized_steps)
        has_artifact = any(step["expert"] == "artifact" for step in normalized_steps)
        if not has_semantic:
            normalized_steps.insert(
                0,
                {
                    "step_id": 1,
                    "expert": "semantic",
                    "goal": "Assess whether the image content matches the prompt/class label semantically.",
                    "model_profile": "local_fast",
                    "prompt_focus": f"Inspect whether the visible subject matches {label_text} using the strongest confirming and contradicting species markers in the image.",
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
                    "goal": "Assess whole-subject structural coherence and class-specific morphology.",
                    "model_profile": "remote_mid",
                    "prompt_focus": f"Inspect whole-subject coherence and the class-specific morphology, body proportions, pose plausibility, scene compatibility, and likely lookalike confusions for {label_text}.",
                    "allow_escalation": True,
                },
            )
        if not has_artifact:
            normalized_steps.append(
                {
                    "step_id": len(normalized_steps) + 1,
                    "expert": "artifact",
                    "goal": "Estimate visible artifact severity and perceptual degradation.",
                    "model_profile": "local_default",
                    "prompt_focus": "Inspect localized generation artifacts, including malformed facial regions, duplicated or fused limbs, broken joints, tail attachment, hands or feet, and corrupted boundaries when relevant.",
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
            "Create a concise evaluation plan using the available experts: semantic, structural, artifact, vqa.",
            "Reason jointly from the image and the prompt/class label as one multimodal grounding task.",
            "The plan must cover semantic alignment, whole-subject global structure, and a separate local artifact pass.",
            "Unless the image is trivially simple, include semantic first, structural second, artifact third, and omit vqa unless earlier steps leave a material unresolved question.",
            "For semantic, verify whether the visible subject plausibly matches the class label and identify the strongest supporting or contradicting visual traits.",
            "For structural, first check whole-subject coherence, then target subject-specific distinguishing morphology, body proportions, pose plausibility, scene compatibility, and likely confusion risks versus nearby lookalikes.",
            "For artifact, inspect localized generation failures such as malformed face or muzzle regions, duplicated or fused limbs, broken joints, wrong tail attachment, malformed hands or feet, asymmetric anatomy, and corrupted fur or edge boundaries when relevant.",
            "Do not compare against external reference images or do open-ended species research; inspect this image directly.",
            "Prefer the cheapest adequate route first.",
            "Default preferences: semantic -> local_fast first, artifact -> local_default/local_richer first, structural -> remote_mid only when needed, vqa -> use only if unresolved after earlier steps.",
            "Use remote_strong only for hard unresolved fine-grained cases.",
            "Set prompt_focus to the exact visual evidence to inspect, using subject-specific failure modes instead of a generic checklist.",
            "Return valid JSON only.",
            'JSON schema: {"rationale":"string","steps":[{"step_id":1,"expert":"semantic|structural|artifact|vqa","goal":"string","model_profile":"local_fast|local_default|local_richer|local_stronger|remote_mid|remote_strong","prompt_focus":"string","allow_escalation":true}]}',
            f"Prompt: {prompt or 'N/A'}",
            f"Class label: {class_label or 'N/A'}",
        ]
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

        return self._normalize_plan(extract_json(output_text), class_label=class_label)


def extract_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
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

    try:
        return json.loads(candidate[start:end + 1])
    except json.JSONDecodeError as exc:
        raise LocalExpertError("Local model did not return valid JSON") from exc


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


class LocalArtifactExpert:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pyiqa: Any | None = None
        self._metrics: dict[str, Any] = {}
        self._device = "cuda" if settings.local_artifact_device == "cuda" else "cpu"

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
                "Local artifact dependencies are missing. Install pyiqa and torch."
            ) from exc

        if self.settings.local_artifact_device == "cuda" and not torch.cuda.is_available():
            self._device = "cpu"

        self._pyiqa = pyiqa
        for metric_name in self.metric_names:
            try:
                self._metrics[metric_name] = pyiqa.create_metric(metric_name, device=self._device)
            except Exception as exc:  # noqa: BLE001
                raise LocalExpertError(f"Failed to load local artifact metric '{metric_name}'") from exc

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
        return "Local IQA metrics indicate limited perceptual degradation and no strong artifact signal."

    def evaluate(self, *, image_path: str) -> dict[str, Any]:
        if not self.settings.local_artifact_enabled:
            raise LocalExpertError("Local artifact expert is disabled")

        self._load()
        if self._pyiqa is None or not self._metrics:
            raise LocalExpertError("Local artifact expert failed to initialize")

        normalized_scores: list[float] = []
        findings: list[str] = []
        for metric_name, metric in self._metrics.items():
            try:
                value = float(metric(str(Path(image_path))).item())
            except Exception as exc:  # noqa: BLE001
                raise LocalExpertError(f"Artifact metric '{metric_name}' inference failed") from exc

            quality = self._normalize_quality(metric_name, value)
            normalized_scores.append(quality)
            findings.append(f"{metric_name} raw score: {value:.4f}; normalized quality estimate: {quality:.2f}.")

        if not normalized_scores:
            raise LocalExpertError("No local artifact scores were produced")

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
            findings.append("The local quality metrics do not show a strong artifact signal.")

        return {
            "expert": "artifact",
            "summary": self._build_summary(severity),
            "findings": findings,
            "severity": round(max(0.0, min(1.0, severity)), 4),
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "source": "local_iqa",
            "model": "+".join(self.metric_names),
        }

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from statistics import pstdev
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

try:
    from torchvision import transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from .config import Settings, ExpertModelConfig


def get_image_transform(size: Tuple[int, int], 
                        mean: List[float] = [0.485, 0.456, 0.406],
                        std: List[float] = [0.229, 0.224, 0.225]):
    if HAS_TORCHVISION:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        def transform(image: Image.Image) -> torch.Tensor:
            image = image.convert('RGB')
            image = image.resize(size, Image.BILINEAR)
            img_array = np.array(image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            mean_tensor = torch.tensor(mean).view(3, 1, 1)
            std_tensor = torch.tensor(std).view(3, 1, 1)
            img_tensor = (img_tensor - mean_tensor) / std_tensor
            return img_tensor
        return transform


class ExpertType(Enum):
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    QUALITY = "quality"
    VQA = "vqa"
    CLASSIFICATION = "classification"
    POSE = "pose"
    DETECTION = "detection"
    SCENE = "scene"
    BACKGROUND = "background"
    IQA = "iqa"
    AIGEN = "aigen"


@dataclass
class ExpertResult:
    expert: str
    summary: str
    findings: List[str]
    severity: float = 0.0
    confidence: float = 0.0
    source: str = "local"
    model: Optional[str] = None
    extra_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert": self.expert,
            "summary": self.summary,
            "findings": self.findings,
            "severity": self.severity,
            "confidence": self.confidence,
            "source": self.source,
            "model": self.model,
            "extra_info": self.extra_info,
        }
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        return self.to_dict()


class ExpertModelError(RuntimeError):
    pass


class ExpertModelRegistry:
    _instance = None
    _lock = Lock()
    _models: Dict[str, Any] = {}
    _processors: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self, model_key: str) -> Optional[Any]:
        return self._models.get(model_key)

    def set_model(self, model_key: str, model: Any):
        self._models[model_key] = model

    def get_processor(self, processor_key: str) -> Optional[Any]:
        return self._processors.get(processor_key)

    def set_processor(self, processor_key: str, processor: Any):
        self._processors[processor_key] = processor

    def clear(self):
        self._models.clear()
        self._processors.clear()


class BaseExpert:
    def __init__(self, config: ExpertModelConfig, settings: Settings):
        self.config = config
        self.settings = settings
        self.registry = ExpertModelRegistry()
        self._model = None
        self._processor = None

    def _load_image(self, image_path: str) -> Image.Image:
        image = Image.open(image_path).convert("RGB")
        max_side = self.settings.local_image_max_side
        image.thumbnail((max_side, max_side))
        return image

    def load_model(self) -> Any:
        raise NotImplementedError

    def evaluate(self, image_path: str, **kwargs) -> ExpertResult:
        raise NotImplementedError


def _workspace_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_imagenet_1k_labels() -> list[str]:
    labels_key = "imagenet_1k_labels"
    registry = ExpertModelRegistry()
    cached = registry.get_processor(labels_key)
    if cached is not None:
        return cached

    labels_path = _workspace_root_from_file() / "scripts" / "imagenet1k_class_index.json"
    if not labels_path.exists():
        registry.set_processor(labels_key, [])
        return []

    with labels_path.open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if not isinstance(labels, list):
        labels = []
    labels = [str(item).strip() for item in labels]
    registry.set_processor(labels_key, labels)
    return labels


def _normalize_text_label(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _split_label_variants(label: str) -> list[str]:
    if not label:
        return []
    variants: list[str] = []
    for part in label.split(","):
        candidate = part.strip()
        if candidate:
            variants.append(candidate)
    if label not in variants:
        variants.insert(0, label.strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for item in variants:
        key = _normalize_text_label(item)
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _resolve_imagenet_label(class_index: int) -> str:
    labels = _load_imagenet_1k_labels()
    if 0 <= class_index < len(labels):
        return labels[class_index]
    return f"class {class_index}"


def _label_match_score(target_label: str, candidate_label: str) -> float:
    target_variants = _split_label_variants(target_label)
    candidate_variants = _split_label_variants(candidate_label)
    target_norms = {_normalize_text_label(item) for item in target_variants}
    candidate_norms = {_normalize_text_label(item) for item in candidate_variants}
    target_norms.discard("")
    candidate_norms.discard("")
    if not target_norms or not candidate_norms:
        return 0.0
    if target_norms & candidate_norms:
        return 1.0
    partial = 0.0
    for target in target_norms:
        for candidate in candidate_norms:
            if target in candidate or candidate in target:
                partial = max(partial, 0.7)
            elif any(token and token in candidate.split(" ") for token in target.split(" ")):
                partial = max(partial, 0.45)
    return partial


def _build_candidate_labels(class_label: Optional[str], extra_candidates: Optional[List[str]] = None) -> list[str]:
    candidates: list[str] = []
    if class_label:
        candidates.extend(_split_label_variants(class_label))
    for candidate in extra_candidates or []:
        candidates.extend(_split_label_variants(candidate))
    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = _normalize_text_label(item)
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


class ImageNetExpert(BaseExpert):
    def _resolve_weights_path(self) -> Optional[str]:
        if self.config.weights and os.path.exists(self.config.weights):
            return self.config.weights
        model_dir = Path(self.settings.model_dir)
        candidates = []
        model_name = self.config.model
        if model_name == "tf_efficientnetv2_s.in21k":
            candidates.append(model_dir / "efficientnetv2_s_in21k.pth")
        elif model_name == "tf_efficientnetv2_l.in21k":
            candidates.append(model_dir / "efficientnetv2_l_in21k.pth")
        elif self.config.weights:
            candidates.append(model_dir / Path(self.config.weights).name)
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def load_model(self) -> Any:
        model_key = f"imagenet_{self.config.model}"

        if self.registry.get_model(model_key):
            return self.registry.get_model(model_key)

        try:
            import timm

            model_name = self.config.model
            num_classes = self.config.num_classes or (21843 if "in21k" in model_name.lower() else 1000)
            weights_path = self._resolve_weights_path()

            if weights_path:
                model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
                state_dict = torch.load(weights_path, map_location=self.config.device)
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                model.load_state_dict(state_dict, strict=False)
            else:
                model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

            model = model.to(self.config.device)
            model.eval()

            self.registry.set_model(model_key, model)
            self._model = model

            return model
        except ImportError as e:
            raise ExpertModelError(f"timm not installed: {e}")

    def evaluate(self, image_path: str, class_label: Optional[str] = None, **kwargs) -> ExpertResult:
        model = self.load_model()

        transform = get_image_transform(self.config.input_size)
        image = self._load_image(image_path)
        first_param = next(model.parameters(), None)
        model_dtype = first_param.dtype if first_param is not None else torch.float32
        img_tensor = transform(image).unsqueeze(0).to(self.config.device, dtype=model_dtype)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            top5_probs, top5_indices = torch.topk(probs, 5)
        
        top5_probs = top5_probs[0].cpu().numpy()
        top5_indices = top5_indices[0].cpu().numpy()
        
        findings = [
            f"Top-{i+1}: class {int(idx)} = {_resolve_imagenet_label(int(idx))} (prob: {prob:.4f})"
            for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs))
        ]

        top1_prob = float(top5_probs[0])
        top1_class = int(top5_indices[0])
        top1_label = _resolve_imagenet_label(top1_class)
        label_match = _label_match_score(class_label or "", top1_label) if class_label else 0.0

        severity = 1.0 - top1_prob
        if class_label:
            severity = min(1.0, max(severity, 1.0 - label_match if label_match > 0 else severity))

        return ExpertResult(
            expert=self.config.name,
            summary=f"Top prediction: class {top1_class} = {top1_label} with probability {top1_prob:.4f}",
            findings=findings,
            severity=severity,
            confidence=top1_prob,
            source="local",
            model=self.config.model,
            extra_info={
                "top5_classes": top5_indices.tolist(),
                "top5_labels": [_resolve_imagenet_label(int(idx)) for idx in top5_indices.tolist()],
                "top5_probs": top5_probs.tolist(),
                "target_label": class_label,
                "top1_label_match": label_match,
            }
        )


class EVAImageNetExpert(ImageNetExpert):
    def _resolve_checkpoint_dir(self) -> Optional[str]:
        model_dir = Path(self.settings.model_dir)
        candidates: list[Path] = []
        if self.config.local_path:
            candidates.append(Path(self.config.local_path))
        model_name = (self.config.model or "").lower()
        if "eva_giant" in model_name:
            candidates.append(model_dir / "eva_giant_224_ckpt")
        if "eva02" in model_name or "eva_02" in model_name or "eva-02" in model_name:
            candidates.append(model_dir / "eva02_l_ckpt")
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def _resolve_architecture_name(self) -> str:
        model_name = (self.config.model or "").lower()
        if "eva_giant" in model_name:
            return "eva_giant_patch14_224.clip_ft_in1k"
        if "eva02" in model_name or "eva_02" in model_name or "eva-02" in model_name:
            return "eva02_large_patch14_448.mim_in22k_ft_in1k"
        return self.config.model

    def _resolve_checkpoint_file(self, checkpoint_dir: str) -> Optional[str]:
        directory = Path(checkpoint_dir)
        if directory.is_file():
            return str(directory)
        for pattern in ("*.pth", "*.pt", "*.bin", "*.safetensors"):
            matches = sorted(directory.glob(pattern))
            if matches:
                return str(matches[0])
        return None

    def load_model(self) -> Any:
        model_key = f"eva_imagenet_{self.config.model}"
        if self.registry.get_model(model_key):
            return self.registry.get_model(model_key)

        try:
            import timm

            architecture_name = self._resolve_architecture_name()
            checkpoint_dir = self._resolve_checkpoint_dir()
            model = timm.create_model(
                architecture_name,
                pretrained=checkpoint_dir is None,
                num_classes=self.config.num_classes or 1000,
            )
            checkpoint_file = self._resolve_checkpoint_file(checkpoint_dir) if checkpoint_dir else None
            if checkpoint_file:
                state_dict = torch.load(checkpoint_file, map_location=self.config.device)
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                model.load_state_dict(state_dict, strict=False)
            model = model.to(self.config.device)
            model.eval()
            self.registry.set_model(model_key, model)
            self._model = model
            return model
        except ImportError as e:
            raise ExpertModelError(f"timm not installed: {e}")

    def evaluate(self, image_path: str, class_label: Optional[str] = None, **kwargs) -> ExpertResult:
        result = super().evaluate(image_path, class_label=class_label, **kwargs)
        image = self._load_image(image_path)
        width, height = image.size
        result.extra_info = result.extra_info or {}
        result.extra_info["image_size"] = [width, height]
        result.extra_info["recommended_eva_variant"] = "imagenet_eva_giant_224" if max(width, height) <= 288 else "imagenet_eva02_large"
        return result


class TextEmbeddingCandidateExpert(BaseExpert):
    def _resolve_model_path(self) -> str:
        model_dir = Path(self.settings.model_dir)
        if self.config.local_path and Path(self.config.local_path).exists():
            return self.config.local_path
        model_name = (self.config.model or "").lower()
        if "bge" in model_name:
            candidate = model_dir / "bge_large_ckpt"
            if candidate.exists():
                return str(candidate)
        if "e5" in model_name:
            candidate = model_dir / "e5_large_ckpt"
            if candidate.exists():
                return str(candidate)
        return self.config.model

    def load_model(self) -> Any:
        model_key = f"text_embedding_{self.config.model}"
        processor_key = f"text_embedding_tokenizer_{self.config.model}"
        if self.registry.get_model(model_key):
            return self.registry.get_model(model_key), self.registry.get_processor(processor_key)

        try:
            from transformers import AutoModel, AutoTokenizer

            model_path = self._resolve_model_path()
            local_only = Path(model_path).exists()
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_path, local_files_only=local_only, trust_remote_code=True)
            model = model.to(self.config.device)
            model.eval()
            self.registry.set_model(model_key, model)
            self.registry.set_processor(processor_key, tokenizer)
            self._model = model
            self._processor = tokenizer
            return model, tokenizer
        except ImportError as e:
            raise ExpertModelError(f"transformers not installed: {e}")

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        model, tokenizer = self.load_model()
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            masked = last_hidden_state * attention_mask
            embeddings = masked.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def evaluate(
        self,
        image_path: str,
        class_label: Optional[str] = None,
        candidate_pool: Optional[List[str]] = None,
        top_k: int = 8,
        **kwargs,
    ) -> ExpertResult:
        if not class_label:
            raise ExpertModelError("Candidate generation requires class_label")

        labels = _load_imagenet_1k_labels()
        if not labels:
            raise ExpertModelError("ImageNet-1K label map is unavailable")

        target_candidates = _build_candidate_labels(class_label, candidate_pool)
        corpus = [_resolve_imagenet_label(index) for index in range(len(labels))]
        query_embeddings = self._encode_texts(target_candidates)
        corpus_embeddings = self._encode_texts(corpus)
        score_matrix = torch.matmul(query_embeddings, corpus_embeddings.T)
        aggregated_scores = score_matrix.max(dim=0).values
        top_k = max(2, min(int(top_k), len(corpus)))
        values, indices = torch.topk(aggregated_scores, k=top_k)

        candidate_labels: list[str] = []
        findings: list[str] = []
        for rank, (index, score) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
            label = corpus[index]
            candidate_labels.append(label)
            findings.append(f"Candidate {rank}: {label} (score: {score:.4f})")

        top_score = float(values[0].item()) if len(values) > 0 else 0.0
        return ExpertResult(
            expert=self.config.name,
            summary=f"Generated {len(candidate_labels)} confusable label candidates for '{class_label}'.",
            findings=findings,
            severity=0.0,
            confidence=min(1.0, max(0.0, (top_score + 1.0) / 2.0)),
            source="local",
            model=self.config.model,
            extra_info={
                "query_labels": target_candidates,
                "candidate_labels": candidate_labels,
                "candidate_scores": [float(item) for item in values.tolist()],
            },
        )


class CLIPExpert(BaseExpert):
    def load_model(self) -> Tuple[Any, Any]:
        model_key = f"clip_{self.config.model}"
        processor_key = f"clip_processor_{self.config.model}"
        
        if self.registry.get_model(model_key):
            return self.registry.get_model(model_key), self.registry.get_processor(processor_key)
        
        try:
            from transformers import CLIPModel, CLIPProcessor
            
            model_path = self.config.local_path or self.config.model
            model = CLIPModel.from_pretrained(model_path, local_files_only=Path(model_path).exists())
            processor = CLIPProcessor.from_pretrained(model_path, local_files_only=Path(model_path).exists())
            
            model = model.to(self.config.device)
            model.eval()
            
            self.registry.set_model(model_key, model)
            self.registry.set_processor(processor_key, processor)
            self._model = model
            self._processor = processor
            
            return model, processor
        except ImportError as e:
            raise ExpertModelError(f"transformers not installed: {e}")

    def evaluate(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        class_label: Optional[str] = None,
        candidate_labels: Optional[List[str]] = None,
        task_type: Optional[str] = None,
        **kwargs,
    ) -> ExpertResult:
        model, processor = self.load_model()

        image = self._load_image(image_path)

        texts: list[str] = []
        target_text = None
        extra_candidates = _build_candidate_labels(class_label, candidate_labels)

        if prompt and task_type != "confusable_disambiguation":
            texts.append(prompt)
            target_text = prompt

        if class_label:
            target_text = f"a photo of {class_label}"
            texts.append(target_text)

        for label in extra_candidates:
            candidate_text = f"a photo of {label}"
            if candidate_text not in texts:
                texts.append(candidate_text)

        if task_type != "confusable_disambiguation":
            contrast_texts = [
                "a low quality image",
                "an unrealistic image",
                "a distorted image",
                "a cartoon image",
                "a painting",
            ]
            for text in contrast_texts:
                if text not in texts:
                    texts.append(text)

        if not texts:
            texts = ["a good quality image", "a bad quality image"]

        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        probs_list = probs[0].cpu().numpy().tolist()

        target_idx = None
        if target_text:
            for i, t in enumerate(texts):
                if t == target_text:
                    target_idx = i
                    break

        if task_type == "confusable_disambiguation" and class_label:
            candidate_payload = []
            for idx, (text, prob) in enumerate(zip(texts, probs_list)):
                label = text.replace("a photo of ", "", 1)
                candidate_payload.append({
                    "label": label,
                    "probability": prob,
                    "is_target": idx == target_idx,
                })
            ranked = sorted(candidate_payload, key=lambda item: item["probability"], reverse=True)
            best = ranked[0]
            target_prob = next((item["probability"] for item in candidate_payload if item["is_target"]), 0.0)
            label_score = _label_match_score(class_label, best["label"])
            if best["label"] != class_label and label_score < 1.0:
                severity = min(1.0, max(0.0, 0.6 + (best["probability"] - target_prob)))
                confidence = min(1.0, max(0.0, best["probability"]))
                summary = f"Best visual match is '{best['label']}' instead of target '{class_label}'."
            else:
                severity = max(0.0, 1.0 - target_prob)
                confidence = min(1.0, max(0.0, target_prob))
                summary = f"Target '{class_label}' remains the best CLIP match."
            findings = [
                f"Candidate '{item['label']}': {item['probability']:.4f}{' [TARGET]' if item['is_target'] else ''}"
                for item in ranked
            ]
            return ExpertResult(
                expert=self.config.name,
                summary=summary,
                findings=findings,
                severity=severity,
                confidence=confidence,
                source="local",
                model=self.config.model,
                extra_info={
                    "texts": texts,
                    "probabilities": probs_list,
                    "target_index": target_idx,
                    "ranked_candidates": ranked,
                    "task_type": "confusable_disambiguation",
                }
            )

        if target_idx is not None:
            target_prob = probs_list[target_idx]
            severity = 1.0 - target_prob
            confidence = target_prob
            summary = f"Target '{target_text}': probability {target_prob:.4f}"
        else:
            max_prob = max(probs_list)
            max_idx = probs_list.index(max_prob)
            severity = 1.0 - max_prob
            confidence = max_prob
            summary = f"Best match: '{texts[max_idx]}' with probability {max_prob:.4f}"

        findings = []
        for i, (text, prob) in enumerate(zip(texts, probs_list)):
            marker = " [TARGET]" if i == target_idx else ""
            findings.append(f"Text '{text}': {prob:.4f}{marker}")

        return ExpertResult(
            expert=self.config.name,
            summary=summary,
            findings=findings,
            severity=severity,
            confidence=confidence,
            source="local",
            model=self.config.model,
            extra_info={
                "texts": texts,
                "probabilities": probs_list,
                "target_index": target_idx,
                "task_type": task_type or "semantic_alignment",
            }
        )


class YOLODetectExpert(BaseExpert):
    def load_model(self) -> Any:
        model_key = f"yolo_detect_{self.config.model}"
        
        if self.registry.get_model(model_key):
            return self.registry.get_model(model_key)
        
        try:
            from ultralytics import YOLO

            model_path = self.config.local_path or self.config.model
            if not model_path.endswith('.pt'):
                model_path = f"{model_path}.pt"

            if not Path(model_path).exists():
                yolo_dir = Path(self.settings.model_dir) / "yolo"
                alt_path = yolo_dir / Path(model_path).name
                if alt_path.exists():
                    model_path = str(alt_path)

            model = YOLO(model_path)
            
            self.registry.set_model(model_key, model)
            self._model = model
            
            return model
        except ImportError as e:
            raise ExpertModelError(f"ultralytics not installed: {e}")

    def evaluate(self, image_path: str, class_label: Optional[str] = None, **kwargs) -> ExpertResult:
        model = self.load_model()
        
        results = model(image_path, verbose=False)
        
        findings = []
        detected_classes = []
        target_detected = False
        target_confidence = 0.0
        
        COCO_CLASSES = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        animal_classes = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
                    detected_classes.append((cls_name, conf))
                    findings.append(f"Detected: {cls_name} (confidence: {conf:.2f})")
                    
                    if class_label:
                        label_lower = class_label.lower()
                        if label_lower in cls_name.lower() or cls_name.lower() in label_lower:
                            target_detected = True
                            target_confidence = max(target_confidence, conf)
                        if cls_name in animal_classes and 'monkey' in label_lower:
                            findings.append(f"Note: YOLO detected '{cls_name}' but expected '{class_label}' - COCO dataset does not have 'monkey' class")
        
        if not findings:
            findings.append("No objects detected by YOLO")
        
        if class_label and not target_detected:
            if any(cls_name in animal_classes for cls_name, _ in detected_classes):
                severity = 0.3
                confidence = 0.5
            else:
                severity = 0.8
                confidence = 0.2
        elif target_detected:
            severity = 1.0 - target_confidence
            confidence = target_confidence
        else:
            severity = 0.5
            confidence = 0.5
        
        return ExpertResult(
            expert=self.config.name,
            summary=f"Detected {len(detected_classes)} objects, target '{class_label}': {'found' if target_detected else 'not found'}",
            findings=findings,
            severity=severity,
            confidence=confidence,
            source="local",
            model=self.config.model,
            extra_info={
                "detected_classes": detected_classes,
                "target_class": class_label,
                "target_detected": target_detected,
            }
        )


class YOLOPoseExpert(BaseExpert):
    def load_model(self) -> Any:
        model_key = f"yolo_{self.config.model}"
        
        if self.registry.get_model(model_key):
            return self.registry.get_model(model_key)
        
        try:
            from ultralytics import YOLO

            model_path = self.config.local_path or self.config.model
            if not model_path.endswith('.pt'):
                model_path = f"{model_path}.pt"

            if not Path(model_path).exists():
                yolo_dir = Path(self.settings.model_dir) / "yolo"
                alt_path = yolo_dir / Path(model_path).name
                if alt_path.exists():
                    model_path = str(alt_path)

            model = YOLO(model_path)
            
            self.registry.set_model(model_key, model)
            self._model = model
            
            return model
        except ImportError as e:
            raise ExpertModelError(f"ultralytics not installed: {e}")

    def evaluate(self, image_path: str, **kwargs) -> ExpertResult:
        model = self.load_model()
        
        results = model(image_path, verbose=False)
        
        findings = []
        total_keypoints = 0
        valid_keypoints = 0
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.keypoints is not None:
                keypoints = result.keypoints.data
                total_keypoints = keypoints.shape[0] * keypoints.shape[1]
                
                for person_idx, person_kpts in enumerate(keypoints):
                    valid_count = (person_kpts[:, 2] > 0.5).sum().item()
                    valid_keypoints += valid_count
                    findings.append(f"Person {person_idx + 1}: {valid_count}/{person_kpts.shape[0]} keypoints detected")
            
            if result.boxes is not None:
                num_boxes = len(result.boxes)
                findings.append(f"Detected {num_boxes} person(s)")
        
        if total_keypoints > 0:
            keypoint_ratio = valid_keypoints / total_keypoints
            severity = 1.0 - keypoint_ratio
        else:
            keypoint_ratio = 0.0
            severity = 1.0
        
        return ExpertResult(
            expert=self.config.name,
            summary=f"Detected {len(results[0].boxes) if results else 0} subjects with {valid_keypoints}/{total_keypoints} valid keypoints",
            findings=findings if findings else ["No subjects detected"],
            severity=severity,
            confidence=keypoint_ratio,
            source="local",
            model=self.config.model,
            extra_info={
                "num_subjects": len(results[0].boxes) if results else 0,
                "valid_keypoints": valid_keypoints,
                "total_keypoints": total_keypoints,
            }
        )


class Places365Expert(BaseExpert):
    def load_model(self) -> Any:
        model_key = f"places365_{self.config.model}"

        if self.registry.get_model(model_key):
            return self.registry.get_model(model_key)

        places_dir = Path(self.settings.model_dir) / "places365"
        if "50" in self.config.model:
            candidates = [places_dir / "resnet50_places365.pt"]
            model_name = "resnet50"
        else:
            candidates = [places_dir / "resnet18_places365.pt", places_dir / "resnet18_imagenet.pt"]
            model_name = "resnet18"

        model_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if model_path is None:
            raise ExpertModelError(f"Places365 model not found at {places_dir}")

        if HAS_TORCHVISION:
            from torchvision import models
            model_factory = getattr(models, model_name)
            model = model_factory(num_classes=self.config.num_classes or 365)
        else:
            import timm
            model = timm.create_model(model_name, num_classes=self.config.num_classes or 365)

        checkpoint = torch.load(model_path, map_location=self.config.device)

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.config.device)
        model.eval()

        self.registry.set_model(model_key, model)
        self._model = model

        return model

    def evaluate(self, image_path: str, **kwargs) -> ExpertResult:
        model = self.load_model()
        
        transform = get_image_transform((224, 224))
        image = self._load_image(image_path)
        img_tensor = transform(image).unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            top5_probs, top5_indices = torch.topk(probs, 5)
        
        top5_probs = top5_probs[0].cpu().numpy()
        top5_indices = top5_indices[0].cpu().numpy()
        
        findings = [f"Scene {i+1}: class {idx} (prob: {prob:.4f})" 
                   for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs))]
        
        top1_prob = float(top5_probs[0])
        severity = 1.0 - top1_prob
        
        return ExpertResult(
            expert=self.config.name,
            summary=f"Scene classification: class {top5_indices[0]} with probability {top1_prob:.4f}",
            findings=findings,
            severity=severity,
            confidence=top1_prob,
            source="local",
            model="places365_resnet18",
            extra_info={
                "top5_classes": top5_indices.tolist(),
                "top5_probs": top5_probs.tolist(),
            }
        )


class IQAExpert(BaseExpert):
    def load_model(self) -> Any:
        if not self.config.metrics:
            raise ExpertModelError("No IQA metrics specified")
        
        cache_dir = getattr(self.settings, 'iqa_cache_dir', None)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            os.environ['PYIQA_CACHE_DIR'] = cache_dir
        
        metrics = {}
        try:
            import pyiqa
            
            for metric_name in self.config.metrics:
                metric_key = f"iqa_{metric_name}"
                
                if self.registry.get_model(metric_key):
                    metrics[metric_name] = self.registry.get_model(metric_key)
                else:
                    metric = pyiqa.create_metric(metric_name, device=self.config.device)
                    self.registry.set_model(metric_key, metric)
                    metrics[metric_name] = metric
            
            self._model = metrics
            return metrics
        except ImportError as e:
            raise ExpertModelError(f"pyiqa not installed: {e}")

    def evaluate(self, image_path: str, **kwargs) -> ExpertResult:
        metrics = self.load_model()
        
        scores = {}
        findings = []
        
        for metric_name, metric in metrics.items():
            try:
                score = float(metric(image_path).item())
                scores[metric_name] = score
                findings.append(f"{metric_name}: {score:.4f}")
            except Exception as e:
                findings.append(f"{metric_name}: error - {str(e)}")
                scores[metric_name] = None
        
        valid_scores = [s for s in scores.values() if s is not None]
        
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            normalized_scores = []
            
            for metric_name, score in scores.items():
                if score is not None:
                    if metric_name.lower() in ["maniqa", "clipiqa"]:
                        normalized = min(score, 1.0) if score <= 1.5 else min(score / 10.0, 1.0)
                    elif metric_name.lower() in ["musiq", "niqe", "brisque"]:
                        normalized = min(score / 100.0, 1.0) if score > 10 else min(score / 10.0, 1.0)
                    else:
                        normalized = min(score, 1.0)
                    normalized_scores.append(normalized)
            
            if normalized_scores:
                avg_quality = sum(normalized_scores) / len(normalized_scores)
                severity = 1.0 - avg_quality
                
                if len(normalized_scores) >= 2:
                    disagreement = pstdev(normalized_scores)
                    confidence = max(0.0, min(1.0, 1.0 - (disagreement / 0.35)))
                else:
                    confidence = 0.6
            else:
                severity = 0.5
                confidence = 0.3
        else:
            avg_quality = 0.5
            severity = 0.5
            confidence = 0.0
        
        if severity >= 0.75:
            summary = "IQA metrics indicate severe perceptual degradation."
        elif severity >= 0.45:
            summary = "IQA metrics indicate noticeable quality issues."
        else:
            summary = "IQA metrics indicate acceptable image quality."
        
        return ExpertResult(
            expert=self.config.name,
            summary=summary,
            findings=findings,
            severity=severity,
            confidence=confidence,
            source="local_iqa",
            model="+".join(self.config.metrics or []),
            extra_info={
                "raw_scores": scores,
                "average_quality": avg_quality,
            }
        )


class BackgroundExpert(BaseExpert):
    def load_model(self) -> Any:
        model_key = "rmbg_2.0"
        
        if self.registry.get_model(model_key):
            return self.registry.get_model(model_key)
        
        try:
            from transformers import AutoModelForImageSegmentation
            
            model_path = self.config.local_path or str(Path(self.settings.model_dir) / "RMBG-2.0")

            if Path(model_path).exists():
                model = AutoModelForImageSegmentation.from_pretrained(
                    str(model_path), trust_remote_code=True, local_files_only=True
                )
            else:
                model = AutoModelForImageSegmentation.from_pretrained(
                    self.config.model, trust_remote_code=True
                )
            
            model = model.to(self.config.device)
            model.eval()
            
            self.registry.set_model(model_key, model)
            self._model = model
            
            return model
        except ImportError as e:
            raise ExpertModelError(f"transformers not installed: {e}")

    def evaluate(self, image_path: str, **kwargs) -> ExpertResult:
        model = self.load_model()
        
        transform = get_image_transform((1024, 1024))
        image = self._load_image(image_path)
        original_size = image.size
        img_tensor = transform(image).unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            output = model(img_tensor)
        
        mask = output[0].squeeze().cpu().numpy()
        
        foreground_ratio = mask.mean()
        background_ratio = 1.0 - foreground_ratio
        
        findings = [
            f"Foreground ratio: {foreground_ratio:.2%}",
            f"Background ratio: {background_ratio:.2%}",
            f"Original image size: {original_size}",
        ]
        
        if background_ratio > 0.5:
            summary = f"Complex background detected ({background_ratio:.2%} of image)."
            severity = background_ratio * 0.5
        else:
            summary = f"Simple background ({background_ratio:.2%} of image)."
            severity = background_ratio * 0.3
        
        return ExpertResult(
            expert=self.config.name,
            summary=summary,
            findings=findings,
            severity=severity,
            confidence=0.8,
            source="local",
            model="RMBG-2.0",
            extra_info={
                "foreground_ratio": float(foreground_ratio),
                "background_ratio": float(background_ratio),
                "mask_shape": mask.shape,
            }
        )


class QwenVLExpert(BaseExpert):
    def load_model(self) -> Tuple[Any, Any]:
        model_key = f"qwen_vl_{self.config.model}"
        processor_key = f"qwen_vl_processor_{self.config.model}"

        if self.registry.get_model(model_key):
            return self.registry.get_model(model_key), self.registry.get_processor(processor_key)

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

            model_path = self.config.local_path or self.config.model
            load_kwargs = {
                "trust_remote_code": True,
            }
            if torch.cuda.is_available() and self.config.device.startswith("cuda"):
                load_kwargs["torch_dtype"] = torch.bfloat16
                load_kwargs["device_map"] = {"": self.config.device}
            else:
                load_kwargs["torch_dtype"] = torch.float32
                load_kwargs["device_map"] = "cpu"

            if Path(model_path).exists():
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    local_files_only=True,
                    **load_kwargs,
                )
            else:
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    **load_kwargs,
                )

            model.eval()

            self.registry.set_model(model_key, model)
            self.registry.set_processor(processor_key, processor)
            self._model = model
            self._processor = processor

            return model, processor
        except ImportError as e:
            raise ExpertModelError(f"transformers not installed: {e}")

    def _normalize_structured_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload.setdefault("expert", "vqa")
        payload.setdefault("answer", "")
        payload.setdefault("summary", str(payload.get("answer", "")).strip())
        payload["findings"] = [str(item).strip() for item in payload.get("findings", []) if str(item).strip()]
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
        payload.setdefault("model", self.config.model)
        payload["summary"] = str(payload.get("summary", "")).strip() or str(payload.get("answer", "")).strip()
        if not payload["findings"]:
            payload["findings"] = list(payload["evidence_items"])
        return payload

    def evaluate(self, image_path: str, question: str = "Describe this image.", **kwargs) -> ExpertResult:
        model, processor = self.load_model()
        image = self._load_image(image_path)

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
            f"Question: {question}",
        ]
        user_text = "\n".join(task_lines)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

        prompt_length = inputs["input_ids"].shape[1]
        trimmed_ids = [output_ids[prompt_length:] for output_ids in generated_ids]
        generated_text = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        payload = self._normalize_structured_payload(json.loads(generated_text)) if generated_text.strip().startswith("{") else self._normalize_structured_payload({"answer": generated_text, "summary": generated_text, "findings": [generated_text], "evidence_items": [generated_text]})

        return ExpertResult(
            expert=self.config.name,
            summary=payload["summary"],
            findings=payload["findings"],
            severity=payload["severity"],
            confidence=payload["confidence"],
            source=payload["source"],
            model=payload["model"],
            extra_info={
                "question": question,
                "answer": payload["answer"],
                "evidence_items": payload["evidence_items"],
                "visible_support": payload["visible_support"],
                "visible_uncertainties": payload["visible_uncertainties"],
                "follow_up_questions": payload["follow_up_questions"],
                "full_response": generated_text,
            }
        )


EXPERT_CLASS_MAP = {
    "classification": ImageNetExpert,
    "eva_classification": EVAImageNetExpert,
    "text_embedding": TextEmbeddingCandidateExpert,
    "clip": CLIPExpert,
    "clip_score": CLIPExpert,
    "yolo_pose": YOLOPoseExpert,
    "yolo_detect": YOLODetectExpert,
    "detection": YOLODetectExpert,
    "places365": Places365Expert,
    "iqa": IQAExpert,
    "segmentation": BackgroundExpert,
    "vqa": QwenVLExpert,
}


def create_expert(config: ExpertModelConfig, settings: Settings) -> BaseExpert:
    if config.model_type == "classification" and (
        (config.weights or "").lower() == "places365"
        or "places365" in (config.name or "").lower()
        or (config.num_classes == 365 and config.model in {"resnet18", "resnet50"})
    ):
        return Places365Expert(config, settings)

    if config.model_type == "classification" and "eva" in (config.model or "").lower():
        return EVAImageNetExpert(config, settings)

    expert_class = EXPERT_CLASS_MAP.get(config.model_type)

    if expert_class is None:
        raise ExpertModelError(f"Unknown expert type: {config.model_type}")

    return expert_class(config, settings)


def run_expert_evaluation(
    expert_name: str,
    image_path: str,
    settings: Settings,
    **kwargs
) -> ExpertResult:
    config = settings.get_expert_config(expert_name)
    
    if config is None:
        raise ExpertModelError(f"Expert not found: {expert_name}")
    
    expert = create_expert(config, settings)
    return expert.evaluate(image_path, **kwargs)

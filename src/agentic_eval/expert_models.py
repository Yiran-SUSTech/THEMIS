from __future__ import annotations

import json
import math
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
    ARTIFACT = "artifact"
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


class ImageNetExpert(BaseExpert):
    def load_model(self) -> Any:
        model_key = f"imagenet_{self.config.model}"
        
        if self.registry.get_model(model_key):
            return self.registry.get_model(model_key)
        
        try:
            import timm
            
            model = timm.create_model(
                self.config.model,
                pretrained=True,
                num_classes=self.config.num_classes or 1000
            )
            model = model.to(self.config.device)
            model.eval()
            
            self.registry.set_model(model_key, model)
            self._model = model
            
            return model
        except ImportError as e:
            raise ExpertModelError(f"timm not installed: {e}")

    def evaluate(self, image_path: str, class_label: Optional[str] = None, **kwargs) -> ExpertResult:
        model = self.load_model()
        
        image = self._load_image(image_path)
        image = image.resize(self.config.input_size)
        
        img_array = np.array(image) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.config.device)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            top5_probs, top5_indices = torch.topk(probs, 5)
        
        top5_probs = top5_probs[0].cpu().numpy()
        top5_indices = top5_indices[0].cpu().numpy()
        
        findings = [f"Top-{i+1}: class {idx} (prob: {prob:.4f})" 
                   for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs))]
        
        top1_prob = float(top5_probs[0])
        top1_class = int(top5_indices[0])
        
        severity = 1.0 - top1_prob
        
        return ExpertResult(
            expert=self.config.name,
            summary=f"Top prediction: class {top1_class} with probability {top1_prob:.4f}",
            findings=findings,
            severity=severity,
            confidence=top1_prob,
            source="local",
            model=self.config.model,
            extra_info={
                "top5_classes": top5_indices.tolist(),
                "top5_probs": top5_probs.tolist(),
            }
        )


class CLIPExpert(BaseExpert):
    def load_model(self) -> Tuple[Any, Any]:
        model_key = f"clip_{self.config.model}"
        processor_key = f"clip_processor_{self.config.model}"
        
        if self.registry.get_model(model_key):
            return self.registry.get_model(model_key), self.registry.get_processor(processor_key)
        
        try:
            from transformers import CLIPModel, CLIPProcessor
            
            model = CLIPModel.from_pretrained(self.config.model)
            processor = CLIPProcessor.from_pretrained(self.config.model)
            
            model = model.to(self.config.device)
            model.eval()
            
            self.registry.set_model(model_key, model)
            self.registry.set_processor(processor_key, processor)
            self._model = model
            self._processor = processor
            
            return model, processor
        except ImportError as e:
            raise ExpertModelError(f"transformers not installed: {e}")

    def evaluate(self, image_path: str, prompt: Optional[str] = None, class_label: Optional[str] = None, **kwargs) -> ExpertResult:
        model, processor = self.load_model()
        
        image = self._load_image(image_path)
        
        texts = []
        if prompt:
            texts.append(prompt)
        if class_label:
            texts.append(f"a photo of {class_label}")
        
        if not texts:
            texts = ["a good quality image", "a bad quality image"]
        
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        probs_list = probs[0].cpu().numpy().tolist()
        
        findings = [f"Text '{text}': probability {prob:.4f}" 
                   for text, prob in zip(texts, probs_list)]
        
        max_prob = max(probs_list)
        severity = 1.0 - max_prob
        
        return ExpertResult(
            expert=self.config.name,
            summary=f"Best match: '{texts[probs_list.index(max_prob)]}' with probability {max_prob:.4f}",
            findings=findings,
            severity=severity,
            confidence=max_prob,
            source="local",
            model=self.config.model,
            extra_info={
                "texts": texts,
                "probabilities": probs_list,
            }
        )


class YOLOPoseExpert(BaseExpert):
    def load_model(self) -> Any:
        model_key = f"yolo_{self.config.model}"
        
        if self.registry.get_model(model_key):
            return self.registry.get_model(model_key)
        
        try:
            from ultralytics import YOLO
            
            model = YOLO(f"{self.config.model}.pt")
            
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
        model_key = "places365"
        
        if self.registry.get_model(model_key):
            return self.registry.get_model(model_key)
        
        places_dir = Path(self.settings.model_dir) / "places365"
        model_path = places_dir / "resnet18_places365.pt"
        
        if not model_path.exists():
            raise ExpertModelError(f"Places365 model not found at {model_path}")
        
        if HAS_TORCHVISION:
            from torchvision import models
            model = models.resnet18(num_classes=365)
        else:
            import timm
            model = timm.create_model('resnet18', num_classes=365)
        
        checkpoint = torch.load(model_path, map_location=self.config.device)
        
        if 'state_dict' in checkpoint:
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
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
            
            model_path = Path(self.settings.model_dir) / "rmbg_2.0"
            
            if model_path.exists():
                model = AutoModelForImageSegmentation.from_pretrained(
                    str(model_path), trust_remote_code=True
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
            
            if Path(model_path).exists():
                processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    local_files_only=True
                )
            else:
                processor = AutoProcessor.from_pretrained(model_path)
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
            
            model.eval()
            
            self.registry.set_model(model_key, model)
            self.registry.set_processor(processor_key, processor)
            self._model = model
            self._processor = processor
            
            return model, processor
        except ImportError as e:
            raise ExpertModelError(f"transformers not installed: {e}")

    def evaluate(self, image_path: str, question: str = "Describe this image.", **kwargs) -> ExpertResult:
        model, processor = self.load_model()
        
        image = self._load_image(image_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return ExpertResult(
            expert=self.config.name,
            summary=generated_text[:200] + "..." if len(generated_text) > 200 else generated_text,
            findings=[generated_text],
            severity=0.0,
            confidence=0.8,
            source="local",
            model=self.config.model,
            extra_info={
                "question": question,
                "full_response": generated_text,
            }
        )


EXPERT_CLASS_MAP = {
    "classification": ImageNetExpert,
    "clip": CLIPExpert,
    "clip_score": CLIPExpert,
    "yolo_pose": YOLOPoseExpert,
    "yolo_detect": YOLOPoseExpert,
    "places365": Places365Expert,
    "iqa": IQAExpert,
    "segmentation": BackgroundExpert,
    "vqa": QwenVLExpert,
}


def create_expert(config: ExpertModelConfig, settings: Settings) -> BaseExpert:
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

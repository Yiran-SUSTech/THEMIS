#!/usr/bin/env python3
"""
THEMIS Model Test Script
Test all installed models to ensure they work correctly.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
from dataclasses import dataclass
from datetime import datetime

import torch
from PIL import Image
import numpy as np


@dataclass
class TestResult:
    model_name: str
    model_type: str
    success: bool
    error_message: Optional[str] = None
    inference_time: Optional[float] = None
    memory_used_mb: Optional[float] = None
    output_info: Optional[str] = None


class ModelTester:
    def __init__(self, model_dir: str, device: str = "cuda", verbose: bool = False):
        self.model_dir = Path(model_dir)
        self.device = device
        self.verbose = verbose
        self.results: List[TestResult] = []
        
        self.test_image_path = self._create_test_image()
    
    def _create_test_image(self) -> str:
        test_dir = self.model_dir / "test_images"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        test_image_path = test_dir / "test_image.jpg"
        
        if not test_image_path.exists():
            img = Image.new('RGB', (224, 224), color=(73, 109, 137))
            img.save(test_image_path)
            if self.verbose:
                print(f"Created test image: {test_image_path}")
        
        return str(test_image_path)
    
    def _get_memory_usage(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return 0.0
    
    def _reset_memory_stats(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    
    def log(self, message: str):
        if self.verbose:
            print(f"[TEST] {message}")
    
    def test_qwen_vl_model(self, model_path: str, model_name: str) -> TestResult:
        self.log(f"Testing Qwen VL model: {model_name}")
        self._reset_memory_stats()
        
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from transformers import Qwen2_5_VLForConditionalGeneration
            
            model_path = Path(model_path)
            if not model_path.exists():
                return TestResult(
                    model_name=model_name,
                    model_type="qwen_vl",
                    success=False,
                    error_message=f"Model path not found: {model_path}"
                )
            
            start_time = time.time()
            
            processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True
            )
            
            image = Image.open(self.test_image_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this image briefly."}
                    ]
                }
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=50)
            
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            inference_time = time.time() - start_time
            memory_used = self._get_memory_usage()
            
            del model
            del processor
            torch.cuda.empty_cache()
            
            return TestResult(
                model_name=model_name,
                model_type="qwen_vl",
                success=True,
                inference_time=inference_time,
                memory_used_mb=memory_used,
                output_info=generated_text[:100] + "..."
            )
            
        except Exception as e:
            return TestResult(
                model_name=model_name,
                model_type="qwen_vl",
                success=False,
                error_message=str(e)
            )
    
    def test_imagenet_model(self, model_name: str) -> TestResult:
        self.log(f"Testing ImageNet model: {model_name}")
        self._reset_memory_stats()
        
        try:
            import timm
            
            start_time = time.time()
            
            model = timm.create_model(model_name, pretrained=True)
            model = model.to(self.device)
            model.eval()
            
            img = Image.open(self.test_image_path).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            
            with torch.no_grad():
                output = model(img_tensor)
            
            pred_class = output.argmax(dim=1).item()
            
            inference_time = time.time() - start_time
            memory_used = self._get_memory_usage()
            
            del model
            torch.cuda.empty_cache()
            
            return TestResult(
                model_name=model_name,
                model_type="imagenet",
                success=True,
                inference_time=inference_time,
                memory_used_mb=memory_used,
                output_info=f"Predicted class: {pred_class}"
            )
            
        except Exception as e:
            return TestResult(
                model_name=model_name,
                model_type="imagenet",
                success=False,
                error_message=str(e)
            )
    
    def test_clip_model(self, model_name: str) -> TestResult:
        self.log(f"Testing CLIP model: {model_name}")
        self._reset_memory_stats()
        
        try:
            from transformers import CLIPModel, CLIPProcessor
            
            start_time = time.time()
            
            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)
            model = model.to(self.device)
            model.eval()
            
            image = Image.open(self.test_image_path).convert("RGB")
            texts = ["a photo of a dog", "a photo of a cat", "a photo of a bird"]
            
            inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            inference_time = time.time() - start_time
            memory_used = self._get_memory_usage()
            
            del model
            del processor
            torch.cuda.empty_cache()
            
            return TestResult(
                model_name=model_name,
                model_type="clip",
                success=True,
                inference_time=inference_time,
                memory_used_mb=memory_used,
                output_info=f"Probabilities: {probs[0].tolist()}"
            )
            
        except Exception as e:
            return TestResult(
                model_name=model_name,
                model_type="clip",
                success=False,
                error_message=str(e)
            )
    
    def test_yolo_model(self, model_name: str) -> TestResult:
        self.log(f"Testing YOLO model: {model_name}")
        self._reset_memory_stats()
        
        try:
            from ultralytics import YOLO
            
            start_time = time.time()
            
            model = YOLO(f"{model_name}.pt")
            
            results = model(self.test_image_path, verbose=False)
            
            inference_time = time.time() - start_time
            memory_used = self._get_memory_usage()
            
            num_boxes = len(results[0].boxes) if results else 0
            
            del model
            torch.cuda.empty_cache()
            
            return TestResult(
                model_name=model_name,
                model_type="yolo",
                success=True,
                inference_time=inference_time,
                memory_used_mb=memory_used,
                output_info=f"Detected {num_boxes} objects"
            )
            
        except Exception as e:
            return TestResult(
                model_name=model_name,
                model_type="yolo",
                success=False,
                error_message=str(e)
            )
    
    def test_iqa_model(self, metric_name: str) -> TestResult:
        self.log(f"Testing IQA metric: {metric_name}")
        self._reset_memory_stats()
        
        try:
            import pyiqa
            
            start_time = time.time()
            
            metric = pyiqa.create_metric(metric_name, device=self.device)
            
            score = metric(self.test_image_path)
            
            inference_time = time.time() - start_time
            memory_used = self._get_memory_usage()
            
            del metric
            torch.cuda.empty_cache()
            
            return TestResult(
                model_name=metric_name,
                model_type="iqa",
                success=True,
                inference_time=inference_time,
                memory_used_mb=memory_used,
                output_info=f"Score: {score.item():.4f}"
            )
            
        except Exception as e:
            return TestResult(
                model_name=metric_name,
                model_type="iqa",
                success=False,
                error_message=str(e)
            )
    
    def test_places365_model(self) -> TestResult:
        self.log("Testing Places365 model")
        self._reset_memory_stats()
        
        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
            
            start_time = time.time()
            
            places_dir = self.model_dir / "places365"
            model_path = places_dir / "resnet18_places365.pt"
            
            if not model_path.exists():
                return TestResult(
                    model_name="places365_resnet18",
                    model_type="places365",
                    success=False,
                    error_message=f"Model file not found: {model_path}"
                )
            
            model = models.resnet18(num_classes=365)
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'state_dict' in checkpoint:
                state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image = Image.open(self.test_image_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = model(img_tensor)
            
            pred_class = output.argmax(dim=1).item()
            
            inference_time = time.time() - start_time
            memory_used = self._get_memory_usage()
            
            del model
            torch.cuda.empty_cache()
            
            return TestResult(
                model_name="places365_resnet18",
                model_type="places365",
                success=True,
                inference_time=inference_time,
                memory_used_mb=memory_used,
                output_info=f"Predicted scene class: {pred_class}"
            )
            
        except Exception as e:
            return TestResult(
                model_name="places365_resnet18",
                model_type="places365",
                success=False,
                error_message=str(e)
            )
    
    def test_background_model(self) -> TestResult:
        self.log("Testing background removal model")
        self._reset_memory_stats()
        
        try:
            from transformers import AutoModelForImageSegmentation
            from torchvision import transforms
            
            start_time = time.time()
            
            model_path = self.model_dir / "rmbg_2.0"
            
            if not model_path.exists():
                model_path = "briaai/RMBG-2.0"
            
            model = AutoModelForImageSegmentation.from_pretrained(str(model_path), trust_remote_code=True)
            model = model.to(self.device)
            model.eval()
            
            image = Image.open(self.test_image_path).convert("RGB")
            
            transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = model(img_tensor)
            
            mask = output[0].squeeze().cpu().numpy()
            
            inference_time = time.time() - start_time
            memory_used = self._get_memory_usage()
            
            del model
            torch.cuda.empty_cache()
            
            return TestResult(
                model_name="rmbg_2.0",
                model_type="background",
                success=True,
                inference_time=inference_time,
                memory_used_mb=memory_used,
                output_info=f"Mask shape: {mask.shape}"
            )
            
        except Exception as e:
            return TestResult(
                model_name="rmbg_2.0",
                model_type="background",
                success=False,
                error_message=str(e)
            )
    
    def run_all_tests(self, model_dir: str) -> List[TestResult]:
        model_path = Path(model_dir)
        
        print("\n" + "=" * 60)
        print("THEMIS Model Testing")
        print("=" * 60)
        print(f"Model directory: {model_dir}")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        print("\n[1/8] Testing Qwen VL models...")
        for size in ["3B", "7B", "32B"]:
            qwen_path = model_path / f"Qwen2.5-VL-{size}-Instruct"
            if qwen_path.exists():
                result = self.test_qwen_vl_model(str(qwen_path), f"Qwen2.5-VL-{size}")
                self.results.append(result)
        
        print("\n[2/8] Testing ImageNet models...")
        for model_name in ["efficientnetv2_s", "efficientnetv2_l", "convnext_tiny"]:
            result = self.test_imagenet_model(model_name)
            self.results.append(result)
        
        print("\n[3/8] Testing CLIP models...")
        for model_name in ["openai/clip-vit-base-patch32"]:
            result = self.test_clip_model(model_name)
            self.results.append(result)
        
        print("\n[4/8] Testing Places365 models...")
        result = self.test_places365_model()
        self.results.append(result)
        
        print("\n[5/8] Testing YOLO models...")
        for model_name in ["yolo11n", "yolo11n-pose", "yolo11m-pose"]:
            result = self.test_yolo_model(model_name)
            self.results.append(result)
        
        print("\n[6/8] Testing IQA models...")
        for metric in ["maniqa", "musiq", "niqe"]:
            result = self.test_iqa_model(metric)
            self.results.append(result)
        
        print("\n[7/8] Testing background removal models...")
        result = self.test_background_model()
        self.results.append(result)
        
        print("\n[8/8] Testing complete!")
        
        return self.results
    
    def print_summary(self):
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        
        success_count = sum(1 for r in self.results if r.success)
        fail_count = len(self.results) - success_count
        
        print(f"\nTotal tests: {len(self.results)}")
        print(f"Passed: {success_count}")
        print(f"Failed: {fail_count}")
        
        print("\nDetailed Results:")
        print("-" * 60)
        
        for result in self.results:
            status = "✓ PASS" if result.success else "✗ FAIL"
            print(f"\n{status} [{result.model_type}] {result.model_name}")
            
            if result.success:
                if result.inference_time:
                    print(f"    Inference time: {result.inference_time:.2f}s")
                if result.memory_used_mb:
                    print(f"    Memory used: {result.memory_used_mb:.2f} MB")
                if result.output_info:
                    print(f"    Output: {result.output_info}")
            else:
                print(f"    Error: {result.error_message}")
        
        print("\n" + "=" * 60)
    
    def save_report(self, output_path: str):
        report = {
            "timestamp": datetime.now().isoformat(),
            "model_dir": str(self.model_dir),
            "device": self.device,
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.success),
                "failed": sum(1 for r in self.results if not r.success)
            },
            "results": [
                {
                    "model_name": r.model_name,
                    "model_type": r.model_type,
                    "success": r.success,
                    "error_message": r.error_message,
                    "inference_time": r.inference_time,
                    "memory_used_mb": r.memory_used_mb,
                    "output_info": r.output_info
                }
                for r in self.results
            ]
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test THEMIS models")
    parser.add_argument("--model-dir", default="./models", help="Directory containing models")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output", default="./model_test_report.json", help="Output report path")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available() and args.device == "cuda":
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    tester = ModelTester(
        model_dir=args.model_dir,
        device=args.device,
        verbose=args.verbose
    )
    
    tester.run_all_tests(args.model_dir)
    tester.print_summary()
    tester.save_report(args.output)


if __name__ == "__main__":
    main()

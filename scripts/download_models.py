#!/usr/bin/env python3
"""
THEMIS Model Download Script
Download all required models for the evaluation system.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


class ModelDownloader:
    def __init__(self, model_dir: str, hf_token: Optional[str] = None, use_mirror: bool = False):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token
        self.use_mirror = use_mirror
        
        if use_mirror:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        self.download_log = []
    
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        self.download_log.append(log_entry)
    
    def run_command(self, cmd: list, description: str) -> bool:
        self.log(f"Running: {description}")
        self.log(f"Command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.log(f"Success: {description}")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"Failed: {description}", "ERROR")
            self.log(f"Error: {e.stderr}", "ERROR")
            return False
    
    def download_hf_model(self, repo_id: str, local_dir: str, model_type: str = "model") -> bool:
        local_path = self.model_dir / local_dir
        local_path.mkdir(parents=True, exist_ok=True)
        
        self.log(f"Downloading {model_type}: {repo_id} -> {local_path}")
        
        try:
            from huggingface_hub import snapshot_download
            
            kwargs = {
                "repo_id": repo_id,
                "local_dir": str(local_path),
                "local_dir_use_symlinks": False,
            }
            if self.hf_token:
                kwargs["token"] = self.hf_token
            
            snapshot_download(**kwargs)
            self.log(f"Successfully downloaded: {repo_id}")
            return True
        except Exception as e:
            self.log(f"Failed to download {repo_id}: {e}", "ERROR")
            return False
    
    def download_qwen_models(self, models: list) -> dict:
        results = {}
        qwen_models = {
            "3b": "Qwen/Qwen2.5-VL-3B-Instruct",
            "7b": "Qwen/Qwen2.5-VL-7B-Instruct",
            "32b": "Qwen/Qwen2.5-VL-32B-Instruct",
        }
        
        for model_size in models:
            if model_size.lower() in qwen_models:
                repo_id = qwen_models[model_size.lower()]
                local_dir = f"Qwen2.5-VL-{model_size.upper()}-Instruct"
                results[model_size] = self.download_hf_model(repo_id, local_dir, "Qwen VL Model")
        
        return results
    
    def download_imagenet_models(self) -> dict:
        results = {}
        
        models = {
            "efficientnetv2_s": "google/efficientnetv2-s",
            "efficientnetv2_l": "google/efficientnetv2-l",
            "convnext_tiny": "facebook/convnext-tiny-224",
        }
        
        for name, repo_id in models.items():
            results[name] = self.download_hf_model(repo_id, name, "ImageNet Model")
        
        return results
    
    def download_clip_models(self) -> dict:
        results = {}
        
        models = {
            "clip_vit_b32": "openai/clip-vit-base-patch32",
            "clip_vit_l14": "openai/clip-vit-large-patch14",
        }
        
        for name, repo_id in models.items():
            results[name] = self.download_hf_model(repo_id, name, "CLIP Model")
        
        return results
    
    def download_places365_models(self) -> bool:
        self.log("Downloading Places365 models...")
        places_dir = self.model_dir / "places365"
        places_dir.mkdir(parents=True, exist_ok=True)
        
        places_urls = {
            "resnet18_places365.pt": "https://github.com/CSAILVision/places365/raw/master/resnet18_places365.pt.tar",
            "resnet50_places365.pt": "https://github.com/CSAILVision/places365/raw/master/resnet50_places365.pt.tar",
            "categories_places365.txt": "https://github.com/CSAILVision/places365/raw/master/categories_places365.txt",
        }
        
        success = True
        for filename, url in places_urls.items():
            filepath = places_dir / filename
            if filepath.exists():
                self.log(f"File already exists: {filepath}")
                continue
            
            cmd = ["wget", "-O", str(filepath), url]
            if not self.run_command(cmd, f"Download {filename}"):
                success = False
        
        return success
    
    def download_yolo_models(self) -> dict:
        results = {}
        
        self.log("Downloading YOLO11 models...")
        
        try:
            from ultralytics import YOLO
            
            yolo_models = {
                "yolo11n": "yolo11n.pt",
                "yolo11n-pose": "yolo11n-pose.pt",
                "yolo11n-seg": "yolo11n-seg.pt",
                "yolo11m-pose": "yolo11m-pose.pt",
                "yolo11m": "yolo11m.pt",
            }
            
            yolo_dir = self.model_dir / "yolo"
            yolo_dir.mkdir(parents=True, exist_ok=True)
            
            for name, model_file in yolo_models.items():
                try:
                    model = YOLO(model_file)
                    model_path = yolo_dir / model_file
                    self.log(f"Downloaded {name} to {model_path}")
                    results[name] = True
                except Exception as e:
                    self.log(f"Failed to download {name}: {e}", "ERROR")
                    results[name] = False
            
        except ImportError:
            self.log("ultralytics not installed, skipping YOLO models", "WARNING")
            results["error"] = "ultralytics not installed"
        
        return results
    
    def download_iqa_models(self) -> dict:
        results = {}
        
        self.log("Downloading IQA models (will be downloaded on first use via pyiqa)...")
        
        try:
            import pyiqa
            
            iqa_metrics = ["maniqa", "musiq", "clipiqa", "niqe", "brisque", "hyperiqa"]
            
            for metric in iqa_metrics:
                try:
                    pyiqa.create_metric(metric, device="cpu")
                    self.log(f"Pre-downloaded IQA metric: {metric}")
                    results[metric] = True
                except Exception as e:
                    self.log(f"Failed to pre-download {metric}: {e}", "WARNING")
                    results[metric] = False
            
        except ImportError:
            self.log("pyiqa not installed, skipping IQA models", "WARNING")
            results["error"] = "pyiqa not installed"
        
        return results
    
    def download_background_models(self) -> dict:
        results = {}
        
        self.log("Downloading background removal models...")
        
        models = {
            "rmbg_2.0": "briaai/RMBG-2.0",
        }
        
        for name, repo_id in models.items():
            results[name] = self.download_hf_model(repo_id, name, "Background Model")
        
        return results
    
    def download_aigen_detection_models(self) -> dict:
        results = {}
        
        self.log("Downloading AI-generated image detection models...")
        
        models = {
            "distildire": "DistilDIRE/DistilDIRE",
        }
        
        for name, repo_id in models.items():
            results[name] = self.download_hf_model(repo_id, name, "AIGen Detection Model")
        
        return results
    
    def save_download_report(self, output_path: str):
        report = {
            "timestamp": datetime.now().isoformat(),
            "model_dir": str(self.model_dir),
            "logs": self.download_log,
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.log(f"Download report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download models for THEMIS evaluation system")
    parser.add_argument("--model-dir", default="./models", help="Directory to store models")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for gated models")
    parser.add_argument("--use-mirror", action="store_true", help="Use HF mirror for faster download in China")
    
    parser.add_argument("--qwen", nargs="*", default=["3b", "7b"], help="Qwen models to download (3b, 7b, 32b)")
    parser.add_argument("--imagenet", action="store_true", help="Download ImageNet classification models")
    parser.add_argument("--clip", action="store_true", help="Download CLIP models")
    parser.add_argument("--places365", action="store_true", help="Download Places365 scene models")
    parser.add_argument("--yolo", action="store_true", help="Download YOLO detection models")
    parser.add_argument("--iqa", action="store_true", help="Download IQA models")
    parser.add_argument("--background", action="store_true", help="Download background removal models")
    parser.add_argument("--aigen", action="store_true", help="Download AI-generated detection models")
    parser.add_argument("--all", action="store_true", help="Download all models")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(
        model_dir=args.model_dir,
        hf_token=args.hf_token,
        use_mirror=args.use_mirror
    )
    
    print("=" * 60)
    print("THEMIS Model Downloader")
    print("=" * 60)
    print(f"Model directory: {args.model_dir}")
    print(f"Use mirror: {args.use_mirror}")
    print("=" * 60)
    
    all_results = {}
    
    if args.qwen or args.all:
        print("\n[1/8] Downloading Qwen VL models...")
        all_results["qwen"] = downloader.download_qwen_models(args.qwen if args.qwen else ["3b", "7b"])
    
    if args.imagenet or args.all:
        print("\n[2/8] Downloading ImageNet models...")
        all_results["imagenet"] = downloader.download_imagenet_models()
    
    if args.clip or args.all:
        print("\n[3/8] Downloading CLIP models...")
        all_results["clip"] = downloader.download_clip_models()
    
    if args.places365 or args.all:
        print("\n[4/8] Downloading Places365 models...")
        all_results["places365"] = downloader.download_places365_models()
    
    if args.yolo or args.all:
        print("\n[5/8] Downloading YOLO models...")
        all_results["yolo"] = downloader.download_yolo_models()
    
    if args.iqa or args.all:
        print("\n[6/8] Downloading IQA models...")
        all_results["iqa"] = downloader.download_iqa_models()
    
    if args.background or args.all:
        print("\n[7/8] Downloading background removal models...")
        all_results["background"] = downloader.download_background_models()
    
    if args.aigen or args.all:
        print("\n[8/8] Downloading AI-generated detection models...")
        all_results["aigen"] = downloader.download_aigen_detection_models()
    
    report_path = Path(args.model_dir) / "download_report.json"
    downloader.save_download_report(str(report_path))
    
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    for category, results in all_results.items():
        print(f"\n{category.upper()}:")
        if isinstance(results, dict):
            for model, success in results.items():
                status = "✓" if success else "✗"
                print(f"  {status} {model}")
        else:
            status = "✓" if results else "✗"
            print(f"  {status} {category}")
    
    print("\n" + "=" * 60)
    print(f"Models saved to: {args.model_dir}")
    print(f"Report saved to: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

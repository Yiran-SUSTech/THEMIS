#!/usr/bin/env python3
"""
Export Q-Insight to ONNX with prompt configuration.

This script exports the model AND saves the prompts needed for inference.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
    print("✓ qwen_vl_utils available")
except ImportError:
    HAS_QWEN_VL_UTILS = False
    print("⚠ qwen_vl_utils not found")


class QInsightONNXModel(nn.Module):
    """Wrapper for ONNX export."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values
        
        if image_grid_thw is not None:
            kwargs["image_grid_thw"] = image_grid_thw
        
        outputs = self.model(**kwargs)
        return outputs.logits


def export_to_onnx(
    model_path: str,
    output_dir: str,
    device: str = "cpu",
    opset_version: int = 17,
):
    """Export Q-Insight to ONNX with prompt configuration."""
    
    print("=" * 80)
    print("Q-Insight ONNX Export (with Prompts)")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # 定义Q-Insight使用的prompts（从expert_models.py中提取）
    QINSIGHT_SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the reasoning "
        "process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> "
        "tags, respectively, i.e., <think> reasoning process here </think>"
        "<answer> answer here </answer>"
    )
    
    QINSIGHT_DISTORTION_PROMPT = (
        'Analyze the given image and determine if it contains any of the following '
        'distortions: "noise", "compression", "blur", or "darken". If a distortion '
        'is present, classify its severity as "slight", "moderate", "obvious", '
        '"serious", or "catastrophic". Return the result in JSON format with the '
        'following keys: "distortion_class": The detected distortion (or "null" if '
        'none). and "severity": The severity level (or "null" if none).'
    )
    
    print(f"\n[1/6] Configuration:")
    print(f"  Model path: {model_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Device: {device}")
    print(f"  ONNX opset: {opset_version}")
    
    print(f"\n[2/6] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    print(f"✓ Processor loaded")
    
    print(f"\n[3/6] Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="auto" if device != "cpu" else None,
        attn_implementation="eager",
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")
    
    print(f"\n[4/6] Preparing sample inputs...")
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": QINSIGHT_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": QINSIGHT_DISTORTION_PROMPT}]},
    ]
    
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = processor(text=[text], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"  Input shapes:")
    for k, v in inputs.items():
        if hasattr(v, 'shape'):
            print(f"    {k}: {v.shape}")
    
    print(f"\n[5/6] Exporting to ONNX...")
    
    onnx_path = str(output_dir / "qinsight_model.onnx")
    wrapped_model = QInsightONNXModel(model)
    
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "attention_mask": {0: "batch_size", 1: "seq_length"},
        "logits": {0: "batch_size", 1: "seq_length", 2: "vocab_size"},
    }
    
    export_inputs = (inputs["input_ids"], inputs["attention_mask"])
    
    torch.onnx.export(
        wrapped_model,
        export_inputs,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )
    
    print(f"✓ ONNX exported: {onnx_path}")
    print(f"  Size: {Path(onnx_path).stat().st_size / (1024**3):.2f} GB")
    
    print(f"\n[6/6] Saving configuration with prompts...")
    
    config = {
        "model_name": "Q-Insight",
        "base_model": "ByteDance/Q-Insight",
        "model_type": "mllm_scoring",
        "onnx_file": "qinsight_model.onnx",
        "opset_version": opset_version,
        "export_device": device,
        "export_dtype": "float32",
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "has_qwen_vl_utils": HAS_QWEN_VL_UTILS,
        
        "prompts": {
            "system": QINSIGHT_SYSTEM_PROMPT,
            "distortion": QINSIGHT_DISTORTION_PROMPT,
        },
        
        "inference_parameters": {
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
            "max_new_tokens": 256,
            "use_cache": True,
        },
        
        "output_parsing": {
            "answer_pattern": r"<answer>\s*(.*?)\s*</answer>",
            "json_pattern": r"\{.*?\}",
            "severity_mapping": {
                "slight": 0.2,
                "moderate": 0.4,
                "obvious": 0.6,
                "serious": 0.8,
                "catastrophic": 1.0,
                "null": 0.0
            }
        },
        
        "usage_notes": [
            "1. Load ONNX model with ONNX Runtime",
            "2. Use prompts.system and prompts.distortion to build messages",
            "3. Call processor.apply_chat_template() to get text",
            "4. Tokenize text to get input_ids",
            "5. Run ONNX inference with input_ids",
            "6. Decode logits to get output text",
            "7. Parse output using output_parsing patterns",
        ]
    }
    
    config_path = output_dir / "onnx_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    processor.save_pretrained(str(output_dir))
    
    elapsed = time.time() - start_time
    
    print(f"\n{'=' * 80}")
    print(f"✓ Export completed!")
    print(f"{'=' * 80}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Output: {output_dir}")
    print(f"\nSaved prompts:")
    print(f"  - system prompt: {len(QINSIGHT_SYSTEM_PROMPT)} chars")
    print(f"  - distortion prompt: {len(QINSIGHT_DISTORTION_PROMPT)} chars")
    print(f"\nConfig file: {config_path}")
    print(f"{'=' * 80}")
    
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    
    args = parser.parse_args()
    
    success = export_to_onnx(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        opset_version=args.opset,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
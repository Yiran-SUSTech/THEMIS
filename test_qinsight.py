#!/usr/bin/env python3
"""
Test Q-Insight model for image distortion analysis.

This script tests the Q-Insight model (based on Qwen2.5-VL) for detecting
image distortions like noise, compression, blur, and darken.

Usage:
    python test_qinsight.py --image /path/to/image.jpg
    python test_qinsight.py --image_dir /path/to/images/
"""

import os
import sys
import time
import json
import re
import argparse
from pathlib import Path

import torch
from PIL import Image

# ==================== 1. 配置路径 ====================
MODEL_PATH = "/mnt/afs/zhengmingkai/zyr/THEMIS/models/Q-Insight/score_degradation"
IMAGE_DIR = "/mnt/afs/zhengmingkai/zyr/THEMIS/test_images"

# 固定的 Prompts（与 expert_models.py 中保持一致）
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> "
    "and <answer> </answer> tags, respectively, i.e., <think> reasoning process "
    "here </think><answer> answer here </answer>"
)

DISTORTION_PROMPT = (
    'Analyze the given image and determine if it contains any of the following '
    'distortions: "noise", "compression", "blur", or "darken". If a distortion '
    'is present, classify its severity as "slight", "moderate", "obvious", '
    '"serious", or "catastrophic". Return the result in JSON format with the '
    'following keys: "distortion_class": The detected distortion (or "null" if '
    'none). and "severity": The severity level (or "null" if none).'
)

# 严重度映射
SEVERITY_MAP = {
    "slight": 0.2,
    "moderate": 0.4,
    "obvious": 0.6,
    "serious": 0.8,
    "catastrophic": 1.0,
    "null": 0.0,
}


# ==================== 2. 模型加载 ====================
def load_model(model_path: str, device: str = "cuda"):
    """加载 Q-Insight 模型和 processor"""
    print(f"\nloading model from: {model_path}")
    print(f"device: {device}")
    
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as e:
        print(f"failed to import transformers: {e}")
        print("install with: pip install transformers")
        sys.exit(1)
    
    start_load = time.time()
    
    # 加载 processor
    print("loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    print(f"processor loaded: {processor.__class__.__name__}")
    
    # 加载模型
    print("loading model weights...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map={"": device},
        attn_implementation="eager",
    )
    model.eval()
    
    load_time = time.time() - start_load
    print(f"model loaded successfully in {load_time:.2f}s")
    print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    return model, processor


# ==================== 3. 输入准备 ====================
def prepare_inputs(processor, image_path: str):
    """准备模型输入（图像 + prompts）"""
    try:
        from qwen_vl_utils import process_vision_info
        has_qwen_vl_utils = True
    except ImportError:
        has_qwen_vl_utils = False
    
    # 构建消息
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": DISTORTION_PROMPT}
        ]},
    ]
    
    # 应用聊天模板
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 处理图像和文本
    if has_qwen_vl_utils:
        image_inputs, video_inputs = process_vision_info([messages])
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
    else:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
    
    return inputs


# ==================== 4. 推理执行 ====================
def run_inference(model, processor, image_path: str, device: str = "cuda"):
    """执行单次推理"""
    print(f"\n{'='*60}")
    print(f"[analyzing] {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    # 准备输入
    inputs = prepare_inputs(processor, image_path)
    
    # 移动到设备
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    print(f"input_ids shape: {inputs['input_ids'].shape}")
    if "pixel_values" in inputs:
        print(f"pixel_values shape: {inputs['pixel_values'].shape}")
    
    # 执行推理
    print("running inference...")
    start_time = time.time()
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            max_new_tokens=256,
            use_cache=True,
        )
    
    inference_time = time.time() - start_time
    
    # 解码输出
    trimmed_ids = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    generated_text = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    print(f"inference time: {inference_time:.2f}s")
    print(f"\ngenerated text:")
    print(f"{'-'*60}")
    print(generated_text)
    print(f"{'-'*60}")
    
    # 解析结果
    result = parse_output(generated_text)
    result["inference_time"] = inference_time
    result["image_path"] = image_path
    
    return result


# ==================== 5. 输出解析 ====================
def parse_output(text: str) -> dict:
    """解析模型输出，提取 distortion class 和 severity"""
    distortion_class = "null"
    severity = "null"
    
    # 提取 <answer> 标签内容
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        
        # 提取 JSON
        json_match = re.search(r"\{.*?\}", answer_text, re.DOTALL)
        if json_match:
            try:
                result_json = json.loads(json_match.group(0))
                distortion_class = result_json.get("distortion_class", "null")
                severity = result_json.get("severity", "null")
            except json.JSONDecodeError:
                print(f"warning: failed to parse JSON from: {answer_text}")
        else:
            print(f"warning: no JSON found in answer: {answer_text}")
    else:
        print(f"warning: no <answer> tags found in output")
    
    # 映射严重度到分数
    severity_score = SEVERITY_MAP.get(severity, 0.0)
    
    return {
        "distortion_class": distortion_class,
        "severity": severity,
        "severity_score": severity_score,
    }


# ==================== 6. 主函数 ====================
def test_single_image(image_path: str, model_path: str, device: str):
    """测试单张图片"""
    # 加载模型
    model, processor = load_model(model_path, device)
    
    # 执行推理
    result = run_inference(model, processor, image_path, device)
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"results:")
    print(f"  image: {result['image_path']}")
    print(f"  distortion class: {result['distortion_class']}")
    print(f"  severity: {result['severity']}")
    print(f"  severity score: {result['severity_score']}")
    print(f"  inference time: {result['inference_time']:.2f}s")
    print(f"{'='*60}")
    
    return result


def test_batch_images(image_dir: str, model_path: str, device: str):
    """批量测试图片目录"""
    # 加载模型（只加载一次）
    model, processor = load_model(model_path, device)
    
    # 获取图片列表
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]
    
    if not image_files:
        print(f"no images found in {image_dir}")
        return []
    
    print(f"\nfound {len(image_files)} images to test")
    
    results = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        try:
            result = run_inference(model, processor, img_path, device)
            results.append(result)
        except Exception as e:
            print(f"failed to process {img_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印汇总
    print(f"\n{'='*60}")
    print(f"batch test summary:")
    print(f"{'='*60}")
    for r in results:
        print(f"  {os.path.basename(r['image_path'])}: "
              f"distortion={r['distortion_class']}, "
              f"severity={r['severity']} ({r['severity_score']}), "
              f"time={r['inference_time']:.2f}s")
    
    # 保存结果
    output_path = "qinsight_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nresults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test Q-Insight model")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to single test image"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to directory of test images"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help="Path to Q-Insight model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )
    
    args = parser.parse_args()
    
    # 确定测试模式
    if args.image:
        test_single_image(args.image, args.model_path, args.device)
    elif args.image_dir:
        test_batch_images(args.image_dir, args.model_path, args.device)
    else:
        # 默认测试 IMAGE_DIR 中的所有图片
        test_batch_images(IMAGE_DIR, args.model_path, args.device)


if __name__ == "__main__":
    main()
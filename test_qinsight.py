#!/usr/bin/env python3
"""
Test Q-Insight model for image distortion analysis.
Optimized for MetaX GPUs and saving complete thinking trajectories.
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

# python test_qinsight.py --image_dir /mnt/afs/zhengmingkai/zyr/THEMIS/test_images

# ==================== 1. 配置路径 ====================
MODEL_PATH = "/mnt/afs/zhengmingkai/zyr/THEMIS/models/Q-Insight/score_degradation"
IMAGE_DIR = "/mnt/afs/zhengmingkai/zyr/THEMIS/test_images"

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
    """加载 Q-Insight 模型和 processor (针对国产显卡优化)"""
    print(f"\nLoading model from: {model_path}")
    print(f"Target device: {device}")
    
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as e:
        print(f"Failed to import transformers: {e}")
        sys.exit(1)
    
    start_load = time.time()
    
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    
    print("Loading model weights (using bfloat16 for MetaX compatibility)...")
    # 针对国产显卡，显式指定 bfloat16 防溢出，eager 模式防算子缺失
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None, 
        attn_implementation="eager",
    )
    if device == "cpu":
        model = model.to("cpu")
        
    model.eval()
    
    load_time = time.time() - start_load
    print(f"Model loaded successfully in {load_time:.2f}s")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
    return model, processor


# ==================== 3. 输入准备 ====================
def prepare_inputs(processor, image_path: str):
    """准备模型输入（图像 + prompts）"""
    try:
        from qwen_vl_utils import process_vision_info
        has_qwen_vl_utils = True
    except ImportError:
        has_qwen_vl_utils = False
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": DISTORTION_PROMPT}
        ]},
    ]
    
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
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
    """执行单次推理并捕获完整输出"""
    print(f"\n{'='*60}")
    print(f"[Analyzing] {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    inputs = prepare_inputs(processor, image_path)
    
    # 确保输入数据和模型在同一张卡上
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    start_time = time.time()
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7, # 稍微调低温度让 JSON 输出更稳定
            top_k=50,
            top_p=0.95,
            max_new_tokens=512, # 扩大 token 限制，防止思考过程太长导致后面的 JSON 被截断
            use_cache=True,
        )
    
    inference_time = time.time() - start_time
    
    # 获取新生成的 tokens
    in_len = inputs["input_ids"].shape[1]
    trimmed_ids = [out_ids[in_len:] for out_ids in generated_ids]
    
    generated_text = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    print(f"Inference time: {inference_time:.2f}s")
    print(f"\n[Raw Generated Text Preview]:")
    print(f"{'-'*60}")
    # 打印前200个字和后200个字，防止刷屏
    if len(generated_text) > 400:
        print(f"{generated_text[:200]}\n\n... [Thinking Process Content] ...\n\n{generated_text[-200:]}")
    else:
        print(generated_text)
    print(f"{'-'*60}")
    
    # 解析结果（核心保存逻辑）
    result = parse_output(generated_text)
    result["inference_time"] = inference_time
    result["image_path"] = os.path.abspath(image_path)
    result["image_name"] = os.path.basename(image_path)
    
    return result


# ==================== 5. 输出解析 ====================
def parse_output(text: str) -> dict:
    """全面解析并提取思维链条与结构化数据"""
    thinking_content = "N/A"
    distortion_class = "null"
    severity = "null"
    
    # 1. 完整提取 <think> 标签内容
    thinking_match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL)
    if thinking_match:
        thinking_content = thinking_match.group(1).strip()
    else:
        # 兜底：如果没有闭合标签，尝试提取 <think> 之后的所有内容，直到 <answer> 出现
        open_think = re.search(r"<think>\s*(.*)", text, re.DOTALL)
        if open_think:
            thinking_content = open_think.group(1).split("<answer>")[0].strip()
    
    # 2. 提取 <answer> 标签或直接寻找 JSON
    answer_text = text
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        
    # 提取 JSON 块
    json_match = re.search(r"\{.*?\}", answer_text, re.DOTALL)
    if json_match:
        try:
            result_json = json.loads(json_match.group(0))
            distortion_class = result_json.get("distortion_class", "null")
            severity = result_json.get("severity", "null")
        except json.JSONDecodeError:
            print(f"Warning: JSON decode failed in text sector.")
    else:
        # 进一步放松正则，抓取可能没有包裹在标准换行中的 JSON
        inline_json = re.search(r"\{.*\}", text)
        if inline_json:
            try:
                result_json = json.loads(inline_json.group(0))
                distortion_class = result_json.get("distortion_class", "null")
                severity = result_json.get("severity", "null")
            except:
                pass
                
    severity_score = SEVERITY_MAP.get(str(severity).lower(), 0.0)
    
    return {
        "thinking": thinking_content, # 这里完整保存，不截断
        "distortion_class": distortion_class,
        "severity": severity,
        "severity_score": severity_score,
    }


# ==================== 6. 主函数 ====================
def test_single_image(image_path: str, model_path: str, device: str):
    """测试单张图片"""
    model, processor = load_model(model_path, device)
    result = run_inference(model, processor, image_path, device)
    
    print(f"\n{'='*60}\nSingle Target Result:\n{'='*60}")
    print(f"Image: {result['image_path']}")
    print(f"Distortion: {result['distortion_class']} | Severity: {result['severity']} ({result['severity_score']})")
    print(f"Thinking Process Length: {len(result['thinking'])} chars")
    print(f"{'='*60}")
    return result


def test_batch_images(image_dir: str, model_path: str, device: str):
    """批量测试图片目录，确保完整结果安全落地 JSON"""
    model, processor = load_model(model_path, device)
    
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]
    
    if not image_files:
        print(f"No valid images found in {image_dir}")
        return []
    
    print(f"\nFound {len(image_files)} targets in pipeline. Starting loop...")
    
    results = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        try:
            result = run_inference(model, processor, img_path, device)
            results.append(result)
        except Exception as e:
            print(f"!!! CRITICAL FAILURE processing {img_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印控制台精简汇总（修改了原本带有语法小瑕疵的 print 拼接）
    print(f"\n{'='*60}\nBatch Execution Summary:\n{'='*60}")
    for r in results:
        think_preview = r['thinking'][:30].replace('\n', ' ') + "..." if r['thinking'] != "N/A" else "N/A"
        print(f" -> {r['image_name']} | Distortion: {r['distortion_class']} ({r['severity']}) | Think Snippet: {think_preview} | Cost: {r['inference_time']:.2f}s")
    
    # 落地保存（保存包含完整 thinking 链条的明细）
    output_path = "qinsight_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Success] Deep quality reports saved to: {os.path.abspath(output_path)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test Q-Insight on MetaX GPU cluster")
    parser.add_argument("--image", type=str, default=None, help="Path to single image")
    parser.add_argument("--image_dir", type=str, default=None, help="Path to images directory")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to Q-Insight directory")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device configuration")
    
    args = parser.parse_args()
    
    if args.image:
        test_single_image(args.image, args.model_path, args.device)
    elif args.image_dir:
        test_batch_images(args.image_dir, args.model_path, args.device)
    else:
        test_batch_images(IMAGE_DIR, args.model_path, args.device)


if __name__ == "__main__":
    main()
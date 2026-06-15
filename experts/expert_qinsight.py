import os
import sys
import time
import torch
import re
import json
from PIL import Image

class QInsightDistortionAnalyzer:
    def __init__(self, 
                 model_path="/mnt/afs/zhengmingkai/zyr/THEMIS/models/Q-Insight/score_degradation",
                 device="cuda",
                 num_gpus=1,
                 max_memory=None):
        
        print(f"[Init] Loading Q-Insight VLM to memory (Device: {device}, GPUs: {num_gpus})...")
        self.device = device
        
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            self.has_qwen_vl_utils = True
            from qwen_vl_utils import process_vision_info
            self.process_vision_info = process_vision_info
        except ImportError:
            self.has_qwen_vl_utils = False

        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        
        use_cuda = device.startswith("cuda")
        if max_memory is None and use_cuda and num_gpus > 0:
            max_memory = {i: "28GB" for i in range(num_gpus)}
        elif max_memory and use_cuda:
            # Convert string keys to int keys (JSON only supports string keys)
            max_memory = {int(k): v for k, v in max_memory.items()}
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
            device_map="auto" if use_cuda else None,
            max_memory=max_memory,
            attn_implementation="eager"
        )
        if not use_cuda:
            self.model = self.model.to("cpu")
            
        self.model.eval()
        
        # 内置常量配置
        self.system_prompt = (
            "A conversation between User and Assistant. The user asks a question, "
            "and the Assistant solves it. The assistant first thinks about the "
            "reasoning process in the mind and then provides the user with the answer. "
            "The reasoning process and answer are enclosed within <think> </think> "
            "and <answer> </answer> tags, respectively, i.e., <think> reasoning process "
            "here </think><answer> answer here </answer>"
        )
        self.distortion_prompt = (
            'Analyze the given image and determine if it contains any of the following '
            'distortions: "noise", "compression", "blur", or "darken". If a distortion '
            'is present, classify its severity as "slight", "moderate", "obvious", '
            '"serious", or "catastrophic". Return the result in JSON format with the '
            'following keys: "distortion_class": The detected distortion (or "null" if '
            'none). and "severity": The severity level (or "null" if none).'
        )
        self.severity_map = {
            "slight": 0.2, "moderate": 0.4, "obvious": 0.6, 
            "serious": 0.8, "catastrophic": 1.0, "null": 0.0
        }

    def _parse_output(self, text: str) -> dict:
        """解析并切分 Qwen 的思维链条与最终结果"""
        thinking_content = "N/A"
        distortion_class = "null"
        severity = "null"
        
        # 提取 <think> 标签
        thinking_match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL)
        if thinking_match:
            thinking_content = thinking_match.group(1).strip()
        else:
            open_think = re.search(r"<think>\s*(.*)", text, re.DOTALL)
            if open_think:
                thinking_content = open_think.group(1).split("<answer>")[0].strip()
        
        # 提取核心的 JSON 数据
        answer_text = text
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            
        json_match = re.search(r"\{.*?\}", answer_text, re.DOTALL)
        if json_match:
            try:
                result_json = json.loads(json_match.group(0))
                distortion_class = result_json.get("distortion_class", "null")
                severity = result_json.get("severity", "null")
            except:
                pass
        
        severity_score = self.severity_map.get(str(severity).lower(), 0.0)
        return {
            "thinking_trajectory": thinking_content,
            "distortion_class": distortion_class,
            "severity_level": severity,
            "severity_score": severity_score
        }

    def audit(self, image_path):
        """
        Q-Insight 证据提取接口：使用本地原图路径执行多模态推理
        """
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": self.distortion_prompt}
            ]},
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        if self.has_qwen_vl_utils:
            image_inputs, video_inputs = self.process_vision_info([messages])
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
            )
        else:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            )
            
        # 数据移至模型同一张卡
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                max_new_tokens=512,
                use_cache=True,
            )
            
        in_len = inputs["input_ids"].shape[1]
        trimmed_ids = [out_ids[in_len:] for out_ids in generated_ids]
        generated_text = self.processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        parsed_data = self._parse_output(generated_text)
        
        return {
            "expert_id": "qinsight_distortion_analyzer",
            "model_name": "Q-Insight_Score_Degradation_BF16",
            "status": "success",
            "evidence": parsed_data
        }
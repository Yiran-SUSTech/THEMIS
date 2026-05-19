import sys
import os
import torch
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image
from torchvision.transforms import functional as TVF

# 🚀 强行注入源码根目录（保留你测试脚本中的路径方案）
sys.path.insert(0, "/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO")
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig

class OpenVocabularyDetector:
    def __init__(self, 
                 model_path="/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO/weights/groundingdino.onnx",
                 config_path="/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"):
        """
        初始化 Grounding DINO 引擎，一次加载，常驻内存
        """
        print(f"[Init] Loading Grounding DINO ONNX Model...")
        # 如果 MetaX 支持加速提供者，可以在此数组前加上 'MACAExecutionProvider' 
        providers = ['CPUExecutionProvider']
        
        sess_options = ort.SessionOptions()
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        
        # 初始化编译并持有 Tokenizer 文本编码器
        args = SLConfig.fromfile(config_path)
        args.device = "cpu"
        self.tmp_model = build_model(args)
        print("[Init] Grounding DINO Expert Initialized Successfully!")

    def _preprocess(self, img_bgr):
        """对内存中的 OpenCV BGR 矩阵进行 DINO 标准预处理"""
        # BGR 转 RGB 并换为 PIL 格式
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        img_resized = pil_img.resize((1200, 800)) 
        t_img = TVF.to_tensor(img_resized)
        t_img = TVF.normalize(t_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return t_img.unsqueeze(0).numpy()

    def calculate_iou(self, box1, box2):
        """计算两个 BBox 的交并比 (IoU)，用于检测空间重叠混淆"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def audit(self, img_bgr, query_text, expected_count=1, threshold=0.3):
        """
        目标检测与语义空间审计核心接口
        :param img_bgr: 传入的 OpenCV BGR 图像矩阵 (numpy array)
        :param query_text: 目标检测文本提示 (例如 "hussar monkey")
        :param expected_count: 预期目标数量 (来自 Prompt 约束，默认为 1)
        :param threshold: 过滤置信度阈值 (默认 0.3)
        :return: 符合 expert_registry.json 规范的结构化字典
        """
        orig_h, orig_w, _ = img_bgr.shape
        img_data = self._preprocess(img_bgr)
        
        # 遵循 Grounding DINO 文本习惯
        caption = query_text.strip().lower()
        if not caption.endswith("."):
            caption += " ."
            
        tokenized = self.tmp_model.tokenizer([caption], return_tensors="pt")
        input_ids = tokenized["input_ids"].numpy()
        attention_mask = tokenized["attention_mask"].bool().numpy()
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).long().numpy()
        token_type_ids = torch.zeros_like(tokenized["input_ids"]).long().numpy()
        
        B, N = input_ids.shape
        text_token_mask = torch.ones((B, N, N), dtype=torch.bool).numpy()

        onnx_inputs = {
            "img": img_data,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "token_type_ids": token_type_ids,
            "text_token_mask": text_token_mask
        }

        # ONNX 推理
        logits, boxes = self.session.run(["logits", "boxes"], onnx_inputs)
        
        # 后处理与过滤
        probs = torch.sigmoid(torch.from_numpy(logits))[0]
        boxes_fixed = torch.from_numpy(boxes)[0]
        
        max_scores, max_indices = probs[:, :N].max(dim=-1)
        keep_idx = max_scores > threshold
        
        filtered_scores = max_scores[keep_idx]
        filtered_boxes = boxes_fixed[keep_idx]
        filtered_token_ids = max_indices[keep_idx]
        
        raw_words = [w for w in caption.split(" ") if w.strip()]
        
        evidence_list = []
        highest_score = 0.0
        
        for i in range(len(filtered_scores)):
            box = filtered_boxes[i].tolist()
            score = filtered_scores[i].item()
            token_idx = filtered_token_ids[i].item()
            
            if score > highest_score:
                highest_score = score
                
            word_idx = token_idx - 1
            if 0 <= word_idx < len(raw_words):
                predicted_word = raw_words[word_idx]
            else:
                predicted_word = query_text
                
            if predicted_word in [".", ",", "?", " ."]:
                predicted_word = query_text

            # 坐标高精度还原为绝对像素坐标
            cx, cy, w, h = box[0], box[1], box[2], box[3]
            x1 = max(0, int((cx - w / 2) * orig_w))
            y1 = max(0, int((cy - h / 2) * orig_h))
            x2 = min(orig_w, int((cx + w / 2) * orig_w))
            y2 = min(orig_h, int((cy + h / 2) * orig_h))
            
            evidence_list.append({
                "text_label": predicted_word,
                "confidence_score": round(score, 4),
                "bounding_box": [x1, y1, x2, y2]
            })
            
        # --- 结合注册表 diagnostic_criteria 进行诊断评估 ---
        detected_count = len(evidence_list)
        verdict = "Normal"
        
        # 1. 缺失实体检查 (Missing_Entity): 最大得分低于 0.3
        if highest_score < threshold or detected_count == 0:
            verdict = "Missing_Entity"
        # 2. 数量错误检查 (Counting_Error): 检测数量与预期不符
        elif detected_count != expected_count:
            verdict = "Counting_Error"
        # 3. 空间混淆检查 (Spatial_Confusion): 检测 distinct 实体是否有过度交叠区域
        elif detected_count > 1:
            # 循环两两比对 IoU 看看是否有重叠区域大于 0.7 导致肢体融合模糊
            for idx1 in range(detected_count):
                for idx2 in range(idx1 + 1, detected_count):
                    iou = self.calculate_iou(evidence_list[idx1]["bounding_box"], evidence_list[idx2]["bounding_box"])
                    if iou > 0.7:
                        verdict = "Spatial_Confusion"
                        break

        return {
            "expert_id": "open_vocabulary_detector",
            "model_name": "GroundingDINO_ONNX",
            "status": "success",
            "metrics": {
                "detected_count": detected_count,
                "highest_confidence": round(highest_score, 4)
            },
            "verdict": verdict,
            "evidence": evidence_list  # 这个 evidence 包含了所有 BBox，将直接下游喂给 SAM
        }
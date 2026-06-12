import sys
import os
import numpy as np
import onnxruntime as ort
import cv2
import torch  # 用来调用原生的 generate_masks_with_special_tokens_and_transfer_map
from PIL import Image
from torchvision.transforms import functional as TVF

# 1. 统一用老分支的路径，确保机制一致
sys.path.insert(0, "/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO")
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.get_tokenlizer import get_tokenlizer  # 引入老分支的官方Tokenizer加载器
from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map

class OpenVocabularyDetector:
    def __init__(self, 
                 model_path="/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO/weights/groundingdino.onnx",
                 config_path="/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"):
        print(f"[Init] Loading Grounding DINO ONNX Model...")
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # 🚀【核心修正】：改回老分支官方的 bert-base-uncased Tokenizer
        self.tokenizer = get_tokenlizer("bert-base-uncased")
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

    def _preprocess(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        img_resized = pil_img.resize((1200, 800)) 
        t_img = TVF.to_tensor(img_resized)
        t_img = TVF.normalize(t_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return t_img.unsqueeze(0).numpy()

    def audit(self, img_bgr, query_text, threshold=0.3):
        orig_h, orig_w, _ = img_bgr.shape
        img_data = self._preprocess(img_bgr)
        
        caption = query_text.strip().lower()
        if not caption.endswith("."):
            caption += " ."

        # 🚀【恢复老分支纯正的动态 Token 提取】
        tokenized = self.tokenizer(caption, padding="longest", return_tensors="pt")
        
        with torch.no_grad():
            (
                text_self_attention_masks,
                position_ids,
                _,
            ) = generate_masks_with_special_tokens_and_transfer_map(
                tokenized, self.specical_tokens, self.tokenizer
            )

        # 🚀【纯动态轴组装】转化为 NumPy 喂给 ONNX，完全不要任何 256 Padding 逻辑
        onnx_inputs = {
            "img": img_data, 
            "input_ids": tokenized["input_ids"].cpu().numpy().astype(np.int64), 
            "attention_mask": tokenized["attention_mask"].cpu().numpy().astype(bool),
            "position_ids": position_ids.cpu().numpy().astype(np.int64), 
            "token_type_ids": tokenized["token_type_ids"].cpu().numpy().astype(np.int64),
            "text_token_mask": text_self_attention_masks.cpu().numpy().astype(bool),
        }

        # 推理并解析
        logits, boxes = self.session.run(["logits", "boxes"], onnx_inputs)
        probs = 1 / (1 + np.exp(-logits))[0]  # sigmoid
        boxes_fixed = boxes[0]

        # 过滤与后处理逻辑（保持与你代码一致）
        N = onnx_inputs["input_ids"].shape[1]
        max_scores = probs[:, :N].max(axis=-1)
        max_indices = probs[:, :N].argmax(axis=-1)
        keep_idx = max_scores > threshold
        
        filtered_scores = max_scores[keep_idx]
        filtered_boxes = boxes_fixed[keep_idx]
        filtered_token_ids = max_indices[keep_idx]
        
        raw_words = [w for w in caption.split(" ") if w.strip()]
        objects_evidence = []
        
        for i in range(len(filtered_scores)):
            box = filtered_boxes[i].tolist()
            score = float(filtered_scores[i])
            token_idx = int(filtered_token_ids[i])
            
            word_idx = token_idx - 1
            predicted_word = raw_words[word_idx] if 0 <= word_idx < len(raw_words) else query_text
            if predicted_word in [".", ",", "?", " ."]:
                predicted_word = query_text

            cx, cy, w, h = box[0], box[1], box[2], box[3]
            x1 = int(max(0, int((cx - w / 2) * orig_w)))
            y1 = int(max(0, int((cy - h / 2) * orig_h)))
            x2 = int(min(orig_w, int((cx + w / 2) * orig_w)))
            y2 = int(min(orig_h, int((cy + h / 2) * orig_h)))

            objects_evidence.append({
                "matched_query_token": predicted_word,
                "confidence_score": round(score, 4),
                "bounding_box": [x1, y1, x2, y2]
            })
            
        return {
            "expert_id": "open_vocabulary_detector",
            "model_name": "GroundingDINO_ONNX",
            "status": "success",
            "raw_metrics": {
                "detected_count": len(objects_evidence)
            },
            "evidence": {
                "detected_objects": objects_evidence
            }
        }
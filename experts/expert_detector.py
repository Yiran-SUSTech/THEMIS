import sys
import os
import torch
import numpy as np
import onnxruntime as ort
import cv2
import torch
from PIL import Image
from torchvision.transforms import functional as TVF

from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "GroundingDINO"))
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig

class OpenVocabularyDetector:
    MAX_TEXT_LEN = 256

    def __init__(self, 
                 model_path="/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO/weights/groundingdino.onnx",
                 config_path="/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"):
        print(f"[Init] Loading Grounding DINO ONNX Model (Static 256)...")
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # 直接拿官方 Tokenizer（完全替代了拉起庞大 tmp_model 的作用，更轻量）
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

        # 🚀 终极修复：启用 native padding！强行让 Tokenizer 吐出完美的 256 长度张量
        tokenized = self.tokenizer(
            caption, 
            padding="max_length", 
            max_length=self.MAX_TEXT_LEN, 
            truncation=True,
            return_tensors="pt"
        )
        
        # 直接塞给官方掩码生成器，它会自动处理好 256 长度的各种 Mask 和 position_ids 映射
        with torch.no_grad():
            (
                text_self_attention_masks,
                position_ids,
                _,
            ) = generate_masks_with_special_tokens_and_transfer_map(
                tokenized, self.specical_tokens, self.tokenizer
            )

        # 毫无中间商赚差价，直接转 NumPy 喂给 ONNX（所有维度天然对齐 256）
        onnx_inputs = {
            "img": img_data, 
            "input_ids": tokenized["input_ids"].cpu().numpy().astype(np.int64), 
            "attention_mask": tokenized["attention_mask"].cpu().numpy().astype(bool),
            "position_ids": position_ids.cpu().numpy().astype(np.int64), 
            "token_type_ids": tokenized["token_type_ids"].cpu().numpy().astype(np.int64),
            "text_token_mask": text_self_attention_masks.cpu().numpy().astype(bool),
        }

        logits, boxes = self.session.run(["logits", "boxes"], onnx_inputs)
        probs = 1 / (1 + np.exp(-logits))[0]  # sigmoid
        boxes_fixed = boxes[0]

        # 🚀 后处理修正：虽然输入是 256 维，但我们提取结果时只看真实 Token 长度
        # 通过 attention_mask.sum() 获取当前文本真实的 Token 数量，避免把 Padding 当成目标提取
        valid_token_len = int(tokenized["attention_mask"][0].sum().item())
        
        max_scores = probs[:, :valid_token_len].max(axis=-1)
        max_indices = probs[:, :valid_token_len].argmax(axis=-1)
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
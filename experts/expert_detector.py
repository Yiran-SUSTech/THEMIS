import sys
import os
import torch
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image
from torchvision.transforms import functional as TVF

sys.path.insert(0, "/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO")
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig

class OpenVocabularyDetector:
    def __init__(self, 
                 model_path="/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO/weights/groundingdino.onnx",
                 config_path="/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"):
        print(f"[Init] Loading Grounding DINO ONNX Model...")
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        args = SLConfig.fromfile(config_path)
        args.device = "cpu"
        self.tmp_model = build_model(args)

    def _preprocess(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        img_resized = pil_img.resize((1200, 800)) 
        t_img = TVF.to_tensor(img_resized)
        t_img = TVF.normalize(t_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return t_img.unsqueeze(0).numpy()

    def audit(self, img_bgr, query_text, threshold=0.3):
        """
        开放域检测证据提取接口：只提取物体边界，不判断数量和重叠
        """
        orig_h, orig_w, _ = img_bgr.shape
        img_data = self._preprocess(img_bgr)
        
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
            "img": img_data, "input_ids": input_ids, "attention_mask": attention_mask,
            "position_ids": position_ids, "token_type_ids": token_type_ids, "text_token_mask": text_token_mask
        }

        logits, boxes = self.session.run(["logits", "boxes"], onnx_inputs)
        probs = torch.sigmoid(torch.from_numpy(logits))[0]
        boxes_fixed = torch.from_numpy(boxes)[0]
        
        max_scores, max_indices = probs[:, :N].max(dim=-1)
        keep_idx = max_scores > threshold
        
        filtered_scores = max_scores[keep_idx]
        filtered_boxes = boxes_fixed[keep_idx]
        filtered_token_ids = max_indices[keep_idx]
        
        raw_words = [w for w in caption.split(" ") if w.strip()]
        objects_evidence = []
        
        for i in range(len(filtered_scores)):
            box = filtered_boxes[i].tolist()
            score = filtered_scores[i].item()
            token_idx = filtered_token_ids[i].item()
            
            word_idx = token_idx - 1
            predicted_word = raw_words[word_idx] if 0 <= word_idx < len(raw_words) else query_text
            if predicted_word in [".", ",", "?", " ."]:
                predicted_word = query_text

            cx, cy, w, h = box[0], box[1], box[2], box[3]
            # --- 找到原 DINO 脚本中计算 x1, y1, x2, y2 的位置，修改为标准的 int() 强转 ---
            x1 = int(max(0, int((cx - w / 2) * orig_w)))
            y1 = int(max(0, int((cy - h / 2) * orig_h)))
            x2 = int(min(orig_w, int((cx + w / 2) * orig_w)))
            y2 = int(min(orig_h, int((cy + h / 2) * orig_h)))

            objects_evidence.append({
                "matched_query_token": predicted_word,
                "confidence_score": round(score, 4),
                "bounding_box": [x1, y1, x2, y2]  # 🚀 现在它们是纯正的 Python int 了！
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
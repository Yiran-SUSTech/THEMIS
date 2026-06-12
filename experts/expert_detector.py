import sys
import os
import numpy as np
import onnxruntime as ort
import cv2
import torch
from PIL import Image
from torchvision.transforms import functional as TVF

# 引入老分支的官方路径和工具，确保词表和 Mask 逻辑百分之百对齐
sys.path.insert(0, "/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO")
from groundingdino.util.get_tokenlizer import get_tokenlizer
from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map

class OpenVocabularyDetector:
    # 明确死守 256 静态维度
    MAX_TEXT_LEN = 256

    def __init__(self, 
                 model_path="/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO/weights/groundingdino.onnx",
                 config_path="/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"):
        print(f"[Init] Loading Grounding DINO ONNX Model (Static 256)...")
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # 使用官方 BERT 词表，确保 hammerhead shark 的 Token ID 正确
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

        # 1. 拿到动态长度的原始 Token
        tokenized = self.tokenizer(caption, padding="longest", return_tensors="pt")
        
        with torch.no_grad():
            (
                text_self_attention_masks,
                _,  # 弃用动态生成的 position_ids
                _,
            ) = generate_masks_with_special_tokens_and_transfer_map(
                tokenized, self.specical_tokens, self.tokenizer
            )

        # 转换为 NumPy
        input_ids_raw = tokenized["input_ids"].cpu().numpy().astype(np.int64)
        attention_mask_raw = tokenized["attention_mask"].cpu().numpy().astype(bool)
        token_type_ids_raw = tokenized["token_type_ids"].cpu().numpy().astype(np.int64)
        text_mask_raw = text_self_attention_masks.cpu().numpy().astype(bool)

        L = self.MAX_TEXT_LEN
        B = input_ids_raw.shape[0]
        N = input_ids_raw.shape[1]

        # 2. 严格执行 256 静态裁剪或对齐
        if N < L:
            pad_len = L - N
            input_ids = np.concatenate([input_ids_raw, np.zeros((B, pad_len), dtype=np.int64)], axis=1)
            # attention_mask 告诉模型后面那一截是 Padding
            attention_mask = np.concatenate([attention_mask_raw, np.zeros((B, pad_len), dtype=bool)], axis=1)
            token_type_ids = np.concatenate([token_type_ids, np.zeros((B, pad_len), dtype=np.int64)], axis=1)
            
            # 🔥 核心修正 1：position_ids 必须是一个严格单调递增到 255 的标准序列
            position_ids = np.arange(L, dtype=np.int64).reshape(B, L)
            
            # 🔥 核心修正 2：text_token_mask 的 Padding 区域必须初始化为全 True
            # 确保 ONNX 内部 GatherElements 算子前向传导时不产生越界负数/无限值，隔离交给 attention_mask
            text_token_mask = np.ones((B, L, L), dtype=bool)
            text_token_mask[:, :N, :N] = text_mask_raw[:, :N, :N]
        else:
            input_ids = input_ids_raw[:, :L]
            attention_mask = attention_mask_raw[:, :L]
            token_type_ids = token_type_ids[:, :L]
            position_ids = np.arange(L, dtype=np.int64).reshape(B, L)
            text_token_mask = text_mask_raw[:, :L, :L]

        # 3. 组装输入，确保每一项的最后一维都是 256
        onnx_inputs = {
            "img": img_data, 
            "input_ids": input_ids, 
            "attention_mask": attention_mask,
            "position_ids": position_ids, 
            "token_type_ids": token_type_ids,
            "text_token_mask": text_token_mask,
        }

        # 4. 运行推理
        logits, boxes = self.session.run(["logits", "boxes"], onnx_inputs)
        probs = 1 / (1 + np.exp(-logits))[0]  # sigmoid
        boxes_fixed = boxes[0]

        # 5. 后处理逻辑
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
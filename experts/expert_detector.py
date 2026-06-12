import sys
import os
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image
from torchvision.transforms import functional as TVF

sys.path.insert(0, "/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO")
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig

# Reference tokenizer from GroundingDINO ONNX runtime
sys.path.insert(0, "/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO-onnxrun/python")
from clip_tokenizer import FullTokenizer, tokenize as gdino_tokenize, generate_masks_with_special_tokens_and_transfer_map

_VOCAB_PATH = "/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO-onnxrun/python/vocab.txt"

class OpenVocabularyDetector:
    # ONNX model was exported with fixed text length of 256
    MAX_TEXT_LEN = 256

    def __init__(self, 
                 model_path="/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO/weights/groundingdino.onnx",
                 config_path="/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                 vocab_path=_VOCAB_PATH):
        print(f"[Init] Loading Grounding DINO ONNX Model...")
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.tokenizer = FullTokenizer(vocab_file=vocab_path)
        self.specical_texts = ["[CLS]", "[SEP]", ".", "?"]

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

        # Use the same tokenization as the reference GroundingDINO ONNX implementation
        input_ids, token_type_ids, attention_mask, specical_tokens = gdino_tokenize(
            self.tokenizer, caption, self.specical_texts, context_length=self.MAX_TEXT_LEN,
        )
        text_self_attention_masks, position_ids = generate_masks_with_special_tokens_and_transfer_map(
            input_ids, specical_tokens,
        )

        # 🚀 【回归静态 256 对齐，但修正 Padding 逻辑】
        L = self.MAX_TEXT_LEN
        N = input_ids.shape[1]
        B = input_ids.shape[0]

        if N < L:
            pad_len = L - N
            # 1. 正常的文本特征 Padding
            input_ids = np.concatenate([input_ids, np.zeros((B, pad_len), dtype=np.int64)], axis=1)
            attention_mask = np.concatenate([attention_mask, np.zeros((B, pad_len), dtype=bool)], axis=1)
            token_type_ids = np.concatenate([token_type_ids, np.zeros((B, pad_len), dtype=np.int64)], axis=1)
            
            # 2. 🔥【核心修正 1】：不要用 0 填充位置索引！
            # 强行让 position_ids 变成 0 到 255 的严格递增序列，防止 GatherElements 查表越界或冲突
            position_ids = np.arange(L, dtype=np.int64).reshape(B, L)
            
            # 3. 🔥【核心修正 2】：二维自注意力掩码矩阵的 Padding
            # 创建 [B, 256, 256] 的全 False 矩阵，只将前 [B, N, N] 的真实文本交互部分填入
            full_mask = np.zeros((B, L, L), dtype=bool)
            full_mask[:, :N, :N] = text_self_attention_masks[:, :N, :N]
            text_self_attention_masks = full_mask
        else:
            # 如果文本刚好等于或超过 256，进行截断对齐
            input_ids = input_ids[:, :L]
            attention_mask = attention_mask[:, :L]
            token_type_ids = token_type_ids[:, :L]
            position_ids = np.arange(L, dtype=np.int64).reshape(B, L)
            text_self_attention_masks = text_self_attention_masks[:, :L, :L]

        # 4. 组装输入，确保所有输入的最后一维（或后两维）都是严格的 256
        onnx_inputs = {
            "img": img_data, 
            "input_ids": input_ids, 
            "attention_mask": attention_mask,
            "position_ids": position_ids, 
            "token_type_ids": token_type_ids,
            "text_token_mask": text_self_attention_masks,
        }

        logits, boxes = self.session.run(["logits", "boxes"], onnx_inputs)
        probs = 1 / (1 + np.exp(-logits))[0]  # sigmoid
        boxes_fixed = boxes[0]

        N = input_ids.shape[1]
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
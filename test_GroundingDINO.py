import sys
import os

# 1. 🚀 强行把 GroundingDINO 的源码根目录塞进 Python 的全局搜索口袋里（解决跨层运行问题）
sys.path.insert(0, "/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO")

import torch
import numpy as np
import onnxruntime as ort
from torchvision.transforms import functional as TVF
from PIL import Image
import cv2

# python test_GroundingDINO.py

def preprocess_image(image_path):
    """标准的 Grounding DINO 图像预处理"""
    print(f"--> loading image from: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"cannot find image file: {image_path}")
        
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((1200, 800)) 
    
    t_img = TVF.to_tensor(img_resized)
    t_img = TVF.normalize(t_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return t_img.unsqueeze(0).numpy()

def test_monkey():
    model_path = "/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO/weights/groundingdino.onnx"
    image_path = "/mnt/afs/zhengmingkai/zyr/THEMIS/test_images/hussar monkey2.png" 
    
    providers = ['CPUExecutionProvider']
    print(f"--> loading ONNX model to providers: {providers}")
    
    sess_options = ort.SessionOptions()
    session = ort.InferenceSession(model_path, sess_options, providers=providers)
    print(f"--> active providers: {session.get_providers()}")

    try:
        img_data = preprocess_image(image_path)
    except Exception as e:
        print(f"failed to preprocess image: {e}")
        return

    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    
    print("--> processing text Prompt: 'hussar monkey .'")
    args = SLConfig.fromfile("/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    args.device = "cpu"
    tmp_model = build_model(args)
    
    caption = "hussar monkey ." 
    tokenized = tmp_model.tokenizer([caption], return_tensors="pt")
    
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

    print("--> running inference on CPU...")
    logits, boxes = session.run(["logits", "boxes"], onnx_inputs)
    print("\ninference success!")
    
    # ==================== 🚀 修正后的后处理段 ====================
    probs = torch.sigmoid(torch.from_numpy(logits))[0] # [900, 256]
    boxes_fixed = torch.from_numpy(boxes)[0] # [900, 4]
    
    thres = 0.3
    # N 是包含 [CLS] 和 [SEP] 的总 Token 数
    max_scores, max_indices = probs[:, :N].max(dim=-1)
    keep_idx = max_scores > thres
    
    filtered_scores = max_scores[keep_idx]
    filtered_boxes = boxes_fixed[keep_idx]
    filtered_token_ids = max_indices[keep_idx]
    
    print(f"\nfiltered results (threshold={thres}):")
    if len(filtered_scores) == 0:
        print("no boxes above threshold found.")
    else:
        print(f"--> successfully detected {len(filtered_scores)} potential objects!")
        
        orig_img = cv2.imread(image_path)
        orig_h, orig_w, _ = orig_img.shape
        print(f"--> Original image shapes restored: {orig_w}x{orig_h}")
        
        # 🌟 核心突破：直接抽取并过滤掉 BERT 的特殊占位符
        # 排除掉 [CLS] 之后，将剩下的实际输入词拆开
        raw_words = [w for w in caption.split(" ") if w.strip()] # ['hussar', 'monkey', '.']
        
        for i in range(len(filtered_scores)):
            box = filtered_boxes[i].tolist()
            score = filtered_scores[i].item()
            token_idx = filtered_token_ids[i].item() # 拿到对应的原始 Token 索引
            
            # 🌟 映射对齐：BERT 编码的第一个位置（idx=0）永远是 [CLS]
            # 所以模型如果对第 1 个词有响应，token_idx 就会是 1；第 2 个词响应，token_idx 就是 2
            # 我们通过 (token_idx - 1) 就可以完美对应到我们的 raw_words 列表中
            word_idx = token_idx - 1
            
            if 0 <= word_idx < len(raw_words):
                predicted_word = raw_words[word_idx]
            else:
                predicted_word = "hussar monkey"
                
            # 如果命中了标点，自动升级为完整短语描述
            if predicted_word in [".", ",", "?"]:
                predicted_word = "hussar monkey"

            # 坐标高精度还原
            cx, cy, w, h = box[0], box[1], box[2], box[3]
            x1 = int((cx - w / 2) * orig_w)
            y1 = int((cy - h / 2) * orig_h)
            x2 = int((cx + w / 2) * orig_w)
            y2 = int((cy + h / 2) * orig_h)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            
            print(f"    object [{i}]: text_label='{predicted_word}', confidence={score:.4f}, box=[{x1}, {y1}, {x2}, {y2}]")
            
            # 绘图
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label_text = f"{predicted_word} ({score:.2f})"
            cv2.putText(orig_img, label_text, (x1, max(y1 - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        output_path = "res_hussar_monkey2_perfect.png"
        cv2.imwrite(output_path, orig_img)
        print(f"\n--> True-resolution visualization result saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    test_monkey()
import sys
import os
# 强行把 GroundingDINO 的源码根目录塞进 Python 的全局搜索口袋里
sys.path.insert(0, "/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO")

import torch
import numpy as np
import onnxruntime as ort
from torchvision.transforms import functional as TVF
from PIL import Image
import cv2

def preprocess_image(image_path):
    """标准的 Grounding DINO 图像预处理"""
    print(f"--> loading image from: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"cannot find image file: {image_path}")
        
    img = Image.open(image_path).convert("RGB")
    # 这里的 resize 仅供模型推理使用
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

    print("--> running inference...")
    logits, boxes = session.run(["logits", "boxes"], onnx_inputs)
    print("\ninference success!")
    
    # ==================== 🚀 核心重构：完美后处理段 ====================
    probs = torch.sigmoid(torch.from_numpy(logits))[0]  # [900, 256]
    boxes_fixed = torch.from_numpy(boxes)[0]            # [900, 4]
    
    thres = 0.3
    # 这里我们只取最后一维属于文本 token 的前 N 个有效位置分数
    # 排除掉模型内部因为 padding 带来的无关特征
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
        
        # 🌟 修复尺寸：直接读取最原始图片（获取其真实的原始宽高，不拉伸）
        orig_img = cv2.imread(image_path)
        orig_h, orig_w, _ = orig_img.shape
        print(f"--> Original image shapes restored: {orig_w}x{orig_h}")
        
        # 🌟 修复标签：构建 Grounding DINO 官方短语词汇提取映射
        # tokenized.tokens() 会把 input_ids 变成类似 ['[CLS]', 'hussar', 'monkey', '.', '[SEP]'] 的干净列表
        input_tokens = tokenized.tokens()
        
        for i in range(len(filtered_scores)):
            box = filtered_boxes[i].tolist()
            score = filtered_scores[i].item()
            token_idx = filtered_token_ids[i].item()
            
            # 拿到这个框响应最强烈的具体 token 字符串
            if token_idx < len(input_tokens):
                predicted_word = input_tokens[token_idx]
            else:
                predicted_word = "unknown"
                
            # 如果命中了特殊占位符，说明该框对应的是全局语义，做个安全平替
            if predicted_word in ["[CLS]", "[SEP]", "."]:
                predicted_word = "hussar monkey"

            # 🌟 坐标还原：把相对的 [0, 1] 坐标直接投射到【原始宽高】上
            cx, cy, w, h = box[0], box[1], box[2], box[3]
            x1 = int((cx - w / 2) * orig_w)
            y1 = int((cy - h / 2) * orig_h)
            x2 = int((cx + w / 2) * orig_w)
            y2 = int((cy + h / 2) * orig_h)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            
            print(f"    object [{i}]: text_label='{predicted_word}', confidence={score:.4f}, box=[{x1}, {y1}, {x2}, {y2}]")
            
            # 在全分辨率原图上画框和写字
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label_text = f"{predicted_word} ({score:.2f})"
            cv2.putText(orig_img, label_text, (x1, max(y1 - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 保存真正完美还原的可视化原图
        output_path = "res_hussar_monkey2_perfect.png"
        cv2.imwrite(output_path, orig_img)
        print(f"\n--> True-resolution visualization result saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    test_monkey()
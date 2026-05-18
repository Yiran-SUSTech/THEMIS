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

def preprocess_image(image_path):
    """标准的 Grounding DINO 图像预处理"""
    print(f"--> loading image from: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"cannot find image file: {image_path}")
        
    # 读入图像并强制转为 RGB 3通道
    img = Image.open(image_path).convert("RGB")
    # 缩放到 800x1200 尺寸供模型进行一致性推理
    img_resized = img.resize((1200, 800)) 
    
    # 转为 Tensor 并做 Imagenet 标准归一化
    t_img = TVF.to_tensor(img_resized)
    t_img = TVF.normalize(t_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return t_img.unsqueeze(0).numpy() # 转为 ONNX 需要的 NumPy Batch 格式 [1, 3, 800, 1200]

def test_monkey():
    # 基础路径配置
    model_path = "/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO/weights/groundingdino.onnx"
    image_path = "/mnt/afs/zhengmingkai/zyr/THEMIS/test_images/hussar monkey2.png" 
    
    # 锁定最稳如泰山、数据完全闭环的纯 CPU 推理
    providers = ['CPUExecutionProvider']
    print(f"--> loading ONNX model to providers: {providers}")
    
    sess_options = ort.SessionOptions()
    session = ort.InferenceSession(model_path, sess_options, providers=providers)
    print(f"--> active providers: {session.get_providers()}")

    # 2. 准备真实的图片输入
    try:
        img_data = preprocess_image(image_path)
    except Exception as e:
        print(f"failed to preprocess image: {e}")
        return

    # 3. 准备文本分词输入
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    
    print("--> processing text Prompt: 'hussar monkey .'")
    # 使用绝对路径定位配置文件
    args = SLConfig.fromfile("/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    args.device = "cpu"
    tmp_model = build_model(args)
    
    caption = "hussar monkey ." 
    # 🌟 核心突破：让分词器同时输出字符位置映射表 (return_offsets_mapping=True) 以便后处理完美拼接子词
    tokenized = tmp_model.tokenizer([caption], return_tensors="pt", return_offsets_mapping=True)
    
    input_ids = tokenized["input_ids"].numpy()
    attention_mask = tokenized["attention_mask"].bool().numpy()
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).long().numpy()
    token_type_ids = torch.zeros_like(tokenized["input_ids"]).long().numpy()
    
    B, N = input_ids.shape
    text_token_mask = torch.ones((B, N, N), dtype=torch.bool).numpy()

    # 4. 构建 ONNX 要求的 Inputs 字典
    onnx_inputs = {
        "img": img_data,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "token_type_ids": token_type_ids,
        "text_token_mask": text_token_mask
    }

    print("--> running inference on CPU...")
    # 5. 执行推理
    logits, boxes = session.run(["logits", "boxes"], onnx_inputs)
    
    print("\ninference success!")
    print(f"--> output logits shape: {logits.shape} (usually [1, 900, 256])")
    print(f"--> output boxes shape: {boxes.shape}  (usually [1, 900, 4])")
    
    # ==================== 🚀 6. 完美的工业级后处理段 ====================
    probs = torch.sigmoid(torch.from_numpy(logits))[0] # [900, 256]
    boxes_fixed = torch.from_numpy(boxes)[0] # [900, 4]
    
    # 设置一个基础检测阈值，看看圈出来的框
    thres = 0.3
    # 仅过滤属于当前有效输入 Token 数量（N）范围内的最大文本关联分数
    max_scores, max_indices = probs[:, :N].max(dim=-1)
    keep_idx = max_scores > thres
    
    filtered_scores = max_scores[keep_idx]
    filtered_boxes = boxes_fixed[keep_idx]
    filtered_token_ids = max_indices[keep_idx]
    
    print(f"\nfiltered results (threshold={thres}):")
    if len(filtered_scores) == 0:
        print("no boxes above threshold found.")
        print(f"--> highest confidence in output is: {max_scores.max().item():.4f}")
    else:
        print(f"--> successfully detected {len(filtered_scores)} potential objects!")
        
        # 🌟 修复尺寸还原：直接用 OpenCV 读原图背景，获取其真正的物理宽高，绝不拉伸
        orig_img = cv2.imread(image_path)
        orig_h, orig_w, _ = orig_img.shape
        print(f"--> Original image shapes restored: {orig_w}x{orig_h}")
        
        # 🌟 修复标签拼接：提取出每个 Token 在原始文本字符串中的具体字符起止索引
        offsets = tokenized["offset_mapping"][0].tolist()
        
        for i in range(len(filtered_scores)):
            box = filtered_boxes[i].tolist()
            score = filtered_scores[i].item()
            token_idx = filtered_token_ids[i].item()
            
            # --- 🔍 终极映射：利用字符偏移，从原句中直接截取完整、不碎裂的英文单词 ---
            predicted_word = "unknown"
            if token_idx < len(offsets):
                start, end = offsets[token_idx]
                # 特殊占位符号（如 [CLS], [SEP]）的 start 和 end 都是 0
                if (start, end) != (0, 0):
                    # 直接从你的原始 caption 字符串里完美切片还原！
                    predicted_word = caption[start:end].strip()
            
            # 安全兜底：如果切片拿到的是标点或未知位，回退显示完整目标
            if predicted_word in [".", "unknown", ""]:
                predicted_word = "hussar monkey"

            # 🌟 坐标高精度还原：将相对的 [0, 1] 缩放系数，完美投射到【原图真实宽高】上
            cx, cy, w, h = box[0], box[1], box[2], box[3]
            x1 = int((cx - w / 2) * orig_w)
            y1 = int((cy - h / 2) * orig_h)
            x2 = int((cx + w / 2) * orig_w)
            y2 = int((cy + h / 2) * orig_h)
            
            # 边界溢出防御
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            
            print(f"    object [{i}]: text_label='{predicted_word}', confidence={score:.4f}, box=[{x1}, {y1}, {x2}, {y2}]")
            
            # 在全分辨率原图背景上画红色框
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 把美化后的英文单词和分数绘制在对应框的上方
            label_text = f"{predicted_word} ({score:.2f})"
            cv2.putText(orig_img, label_text, (x1, max(y1 - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 7. 保存最终真分辨率、完美词汇映射的可视化图片
        output_path = "res_hussar_monkey2_perfect.png"
        cv2.imwrite(output_path, orig_img)
        print(f"\n--> True-resolution visualization result saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    test_monkey()
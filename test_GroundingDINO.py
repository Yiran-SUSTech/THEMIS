import os
import sys
# 强行把 GroundingDINO 的源码根目录塞进 Python 的全局搜索口袋里
sys.path.insert(0, "/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO")
import torch
import numpy as np
import onnxruntime as ort
from torchvision.transforms import functional as TVF
from PIL import Image

def preprocess_image(image_path):
    """标准的 Grounding DINO 图像预处理"""
    print(f"--> loading image from: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"cannot find image file: {image_path}")
        
    # 读入图像并强制转为 RGB 3通道
    img = Image.open(image_path).convert("RGB")
    # 缩放到 800x1200 尺寸
    img = img.resize((1200, 800)) 
    
    # 转为 Tensor 并做 Imagenet 标准归一化
    t_img = TVF.to_tensor(img)
    t_img = TVF.normalize(t_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return t_img.unsqueeze(0).numpy() # 转为 ONNX 需要的 NumPy Batch 格式 [1, 3, 800, 1200]

def test_monkey():
    model_path = "/mnt/afs/zhengmingkai/zyr/THEMIS/GroundingDINO/weights/groundingdino.onnx"
    # 本地图片路径（注意：Linux下路径分隔符用 / ）
    image_path = "/mnt/afs/zhengmingkai/zyr/THEMIS/test_images/hussar monkey2.png" 
    
    # 1. 显式指定你服务器上的沐曦计算提供商
    # providers = ['MACAExecutionProvider', 'CPUExecutionProvider']
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

    # 3. 准备文本分词输入 ("hussar monkey .")
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

    # 4. 构建 ONNX 要求的 Inputs 字典
    onnx_inputs = {
        "img": img_data,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "token_type_ids": token_type_ids,
        "text_token_mask": text_token_mask
    }

    print("--> running inference on MACAExecutionProvider...")
    # 5. 执行推理
    logits, boxes = session.run(["logits", "boxes"], onnx_inputs)
    
    print("\ninference success!")
    print(f"--> output logits shape: {logits.shape} (usually [1, 900, 256])")
    print(f"--> output boxes shape: {boxes.shape}  (usually [1, 900, 4])")
    
    # 6. 核心后处理：文本类别解码 + 图像画框保存
    import cv2
    
    # 将模型输出转为 PyTorch Tensor 方便操作
    probs = torch.sigmoid(torch.from_numpy(logits))[0]  # [900, 256]
    boxes_fixed = torch.from_numpy(boxes)[0]            # [900, 4]
    
    # 阈值过滤
    thres = 0.3
    max_scores, max_indices = probs.max(dim=-1)
    keep_idx = max_scores > thres
    
    filtered_scores = max_scores[keep_idx]
    filtered_boxes = boxes_fixed[keep_idx]
    filtered_token_ids = max_indices[keep_idx] # 每个框响应最强烈的 Token 索引
    
    print(f"\nfiltered results (threshold={thres}):")
    if len(filtered_scores) == 0:
        print("no boxes above threshold found.")
        print(f"--> highest confidence in output is: {max_scores.max().item():.4f}")
    else:
        print(f"--> successfully detected {len(filtered_scores)} potential objects!")
        
        # 为了画框，我们需要重新读取原图获取真实的分辨率尺寸
        # 注意：因为推理用的是等比例或 resize 后的图，画框必须基于当时送入模型的真实尺寸进行还原
        # 在你当前脚本中，模型输入的实际尺寸被硬编码缩放为了 (W=1200, H=800)
        input_w, input_h = 1200, 800
        
        # 载入用于画图的基础背景（直接用 OpenCV 读原图并 resize 到推理尺寸）
        vis_img = cv2.imread(image_path)
        vis_img = cv2.resize(vis_img, (input_w, input_h))
        
        # 获取分词器，用来把 token_id 还原成具体的单词文本
        tokenizer = tmp_model.tokenizer
        
        for i in range(len(filtered_scores)):
            box = filtered_boxes[i].tolist()
            score = filtered_scores[i].item()
            token_id = filtered_token_ids[i].item()
            
            # --- 🔍 核心突破 1：解析物体具体文本类别 ---
            # 根据 token_id 还原出对应的单词（例如把 id=2342 还原为 "monkey"）
            predicted_word = tokenizer.decode([token_id]).strip()
            
            # --- 核心突破 2：计算真实像素坐标并画框 ---
            # 模型输出的是归一化的中心点坐标 [cx, cy, w, h]，我们需要还原为左上角和右下角像素坐标
            cx, cy, w, h = box[0], box[1], box[2], box[3]
            
            x1 = int((cx - w / 2) * input_w)
            y1 = int((cy - h / 2) * input_h)
            x2 = int((cx + w / 2) * input_w)
            y2 = int((cy + h / 2) * input_h)
            
            # 边界安全防止越界
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(input_w, x2), min(input_h, x2)
            
            print(f"    object [{i}]: text_label='{predicted_word}', confidence={score:.4f}, box=[{x1}, {y1}, {x2}, {y2}]")
            
            # 用 OpenCV 在图像上画出红色的矩形框（线条粗细为 2）
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 把识别出的文本类别标签和置信度写在框的上方
            label_text = f"{predicted_word} ({score:.2f})"
            cv2.putText(vis_img, label_text, (x1, max(y1 - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 保存可视化图像到当前目录下
        output_path = "res_hussar_monkey2.png"
        cv2.imwrite(output_path, vis_img)
        print(f"\n--> visualization result saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    test_monkey()
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
    
    # 6. 简单的后处理过滤（看看模型有没有真正圈出猴子）
    # logits 的形状是 [1, 900, 256]，最后一维代表文本 token 的响应强度
    # 我们把预测结果转为 PyTorch Tensor 方便用内置函数快速看一下最大置信度
    probs = torch.sigmoid(torch.from_numpy(logits))[0] # [900, 256]
    boxes_fixed = torch.from_numpy(boxes)[0] # [900, 4]
    
    # 找出每个检测框对应文本的最大分数
    max_scores, _ = probs.max(dim=-1)
    
    # 设置一个基础检测阈值，看看圈出来的框
    thres = 0.3
    keep_idx = max_scores > thres
    
    filtered_scores = max_scores[keep_idx]
    filtered_boxes = boxes_fixed[keep_idx]
    
    print(f"\nfiltered results (threshold={thres}):")
    if len(filtered_scores) == 0:
        print("no boxes above threshold found.")
        print(f"--> highest confidence in output is: {max_scores.max().item():.4f}")
    else:
        print(f"--> successfully detected {len(filtered_scores)} potential objects!")
        for i in range(min(5, len(filtered_scores))):
            box = filtered_boxes[i].tolist()
            # 这里的坐标是归一化中心点坐标 [cx, cy, w, h]
            print(f"    object [{i}]: confidence={filtered_scores[i].item():.4f}, relative coordinates(cx,cy,w,h)={ [round(v,3) for v in box] }")

if __name__ == "__main__":
    test_monkey()
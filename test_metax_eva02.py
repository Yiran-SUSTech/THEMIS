import os
import time
import json
import numpy as np
import onnxruntime as ort
from PIL import Image

# ==================== 1. 配置路径 ====================
MODEL_PATH = "/mnt/afs/zhengmingkai/zyr/THEMIS/new_models/eva02_large_metax_compatible.onnx"
IMAGE_DIR = "./test_images"

# 固定的输入尺寸（必须与导出的方案 B 448 完全保持一致）
INPUT_SIZE = 448 

# ==================== 2. ImageNet 标准预处理 ====================
# 包含：缩放、居中裁剪、归一化以及 ImageNet 标志性的 Mean/Std 减除
def preprocess_image(image_path, target_size=448):
    # 1. 载入并转为 RGB (防止某些图片是 RGBA 或灰度图)
    img = Image.open(image_path).convert('RGB')
    
    # 2. 居中裁剪并缩放到 448x448
    w, h = img.size
    min_size = min(w, h)
    img = img.crop(((w - min_size) // 2, (h - min_size) // 2, (w + min_size) // 2, (h + min_size) // 2))
    img = img.resize((target_size, target_size), Image.Resampling.BICUBIC)
    
    # 3. 归一化到 [0, 1] 转换为 numpy 数组
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # 4. 减去 ImageNet 的标准均值和方差 (timm/EVA-02 默认规范)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    
    # 5. 从 [H, W, C] 变换到 PyTorch/ONNX 标准的 [C, H, W]
    img_np = img_np.transpose(2, 0, 1)
    
    # 6. 增加 Batch 维度，变为 [1, C, H, W]
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

# ==================== 3. 本地读取 ImageNet 1000类 标签映射 ====================
LABEL_PATH = "/mnt/afs/zhengmingkai/zyr/THEMIS/imagenet_classes.json"

print(f"loading: {LABEL_PATH}...")
try:
    with open(LABEL_PATH, 'r', encoding='utf-8') as f:
        labels_dict = json.load(f)
    
    # 将字典转换为列表，确保索引对齐
    # 比如 labels_dict["0"] 放到 imagenet_labels[0]
    imagenet_labels = [labels_dict.get(str(i), f"Class_Index_{i}") for i in range(1000)]
    print(f"loaded {len(imagenet_labels)} labels")
except Exception as e:
    print(f"failed to load labels: {e}, fallback to numeric indices")
    imagenet_labels = [f"Class_Index_{i}" for i in range(1000)]

# ==================== 4. 初始化 MetaX 硬件引擎 ====================
providers = ['MACAExecutionProvider', 'CPUExecutionProvider']

print(f"\nloading model {MODEL_PATH} to MetaX MACA engine...")
print(f"model path: {MODEL_PATH}")
start_load = time.time()
session = ort.InferenceSession(MODEL_PATH, providers=providers)
print(f"success, cost: {time.time() - start_load:.2f} 秒")
print(f"active providers: {session.get_provider_options().keys()}")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ==================== 5. 批量循环审计你的真实图片 ====================
print("\n" + "="*50)
print("start batch inference...")
print("="*50)

# 获取你要测试的图片列表
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_file in image_files:
    img_path = os.path.join(IMAGE_DIR, img_file)
    print(f"\n[analyzing] ──► {img_file}")
    
    # 执行刚才写好的真实图片预处理
    try:
        input_data = preprocess_image(img_path, target_size=INPUT_SIZE)
    except Exception as e:
        print(f"failed to preprocess image: {e}")
        continue
        
    # 送入 MetaX (MACA) 执行单张图片的推理
    t0 = time.time()
    raw_outputs = session.run([output_name], {input_name: input_data})
    inference_time = (time.time() - t0) * 1000 # 转换为毫秒
    
    # 处理模型输出 (执行一个简单的 Softmax 或者直接找最大的 logits 值)
    logits = raw_outputs[0][0]
    
    # 提取 Logits 最高的 Top-3 类别的索引
    top3_idx = np.argsort(logits)[-3:][::-1]
    
    # 打印审计报告
    print(f"cost: {inference_time:.2f} ms")
    print(f"Top-3 predictions candidates:")
    for rank, idx in enumerate(top3_idx, 1):
        score = logits[idx]
        # 如果获取到了在线标签则转换，否则输出 Index
        label_name = imagenet_labels[idx] if idx < len(imagenet_labels) else f"Index_{idx}"
        print(f"      Rank {rank}: [Idx {idx:03d}] {label_name:<30} (Logit: {score:.2f})")

print("\n" + "="*50)
print("all images analyzed!")
print("="*50)
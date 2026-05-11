import cv2
import numpy as np
import onnxruntime as ort
import os
from tqdm import tqdm

# 1. 路径配置
MODEL_PATH = "/mnt/afs/zhengmingkai/zyr/THEMIS/new_models/depth_anything_v2_onnx/onnx/model_fp16_forced_opset16.onnx"
INPUT_DIR = "./test_images/"   # 你的 10,000 张图目录
OUTPUT_DIR = "./depth_results"     # 存放深度图供 Reflector 审计
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. 初始化 MetaX 推理会话
providers = [
    ('MACAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
    }),
    'CPUExecutionProvider'
]
session = ort.InferenceSession(MODEL_PATH, providers=providers)

def process_batch():
    # 获取所有图片列表
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_name in tqdm(image_files, desc="MetaX GPU Processing"):
        try:
            full_path = os.path.join(INPUT_DIR, img_name)
            raw_img = cv2.imread(full_path)
            if raw_img is None: continue
            
            h, w = raw_img.shape[:2]
            
            # --- 预处理 ---
            # Depth Anything V2 推荐尺寸为 518 的倍数，ONNX 版通常固定为 518x518
            input_size = 518 
            img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (input_size, input_size))
            img = img.astype(np.float32) / 255.0
            img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            img = img.transpose(2, 0, 1)[None, ...]

            # --- GPU 推理 ---
            depth = session.run(None, {session.get_inputs()[0].name: img})[0]

            # --- 后处理 ---
            depth = cv2.resize(depth[0], (w, h))
            # 归一化到 0-255 用于可视化审计
            depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
            
            # 保存，文件名后缀建议加上 _depth 以便 Reflector 对应
            save_path = os.path.join(OUTPUT_DIR, os.path.splitext(img_name)[0] + "_depth.png")
            cv2.imwrite(save_path, depth_norm)
            
        except Exception as e:
            print(f"process {img_name} error: {e}")

if __name__ == "__main__":
    process_batch()
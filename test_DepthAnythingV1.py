import cv2
import numpy as np
import onnxruntime as ort
import os
from tqdm import tqdm

# --- 路径配置 ---
MODEL_PATH = "/mnt/afs/zhengmingkai/zyr/THEMIS/new_models/depth_anything_v1_onnx/onnx/model_fp16.onnx"
INPUT_DIR = "/mnt/afs/zhengmingkai/zyr/THEMIS/test_images" 
OUTPUT_DIR = "/mnt/afs/zhengmingkai/zyr/THEMIS/depth_results_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 加载模型（保持 MACA 显卡配置）
providers = [('MACAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
session = ort.InferenceSession(MODEL_PATH, providers=providers)

# 2. 智能获取输入尺寸 (修复 OpenCV 报错的关键)
input_node = session.get_inputs()[0]
# 尝试获取具体数值，如果获取到的是字符串（动态轴），则强制指定为 518
def get_dim(dim):
    return dim if isinstance(dim, int) and dim > 0 else 518

input_h = get_dim(input_node.shape[2])
input_w = get_dim(input_node.shape[3])
print(f"model loaded successfully! inference resolution fixed to: {input_w}x{input_h}")

def run_production():
    # 获取所有图片列表
    all_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"start processing {len(all_files)} images...")

    for img_name in tqdm(all_files, desc="Batch Processing"):
        try:
            in_path = os.path.join(INPUT_DIR, img_name)
            # 统一保存为 .png 格式
            out_name = os.path.splitext(img_name)[0] + "_depth.png"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            
            # --- 断点续传 ---
            if os.path.exists(out_path):
                continue

            # --- 预处理 ---
            raw_img = cv2.imread(in_path)
            if raw_img is None: continue
            orig_h, orig_w = raw_img.shape[:2]
            
            # 使用获取到的整数尺寸
            img = cv2.resize(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), (input_w, input_h))
            img = (img.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            img = img.transpose(2, 0, 1)[None, ...]

            # --- GPU 推理 ---
            depth = session.run(None, {input_node.name: img})[0]

            # --- 后处理 ---
            # 先缩放回原图大小
            depth = cv2.resize(depth[0], (orig_w, orig_h))
            # 归一化到 0-255
            depth_min, depth_max = depth.min(), depth.max()
            depth_norm = ((depth - depth_min) / (depth_max - depth_min + 1e-8) * 255).astype(np.uint8)
            
            cv2.imwrite(out_path, depth_norm)
            
        except Exception as e:
            print(f"\nprocess {img_name} failed: {e}")

if __name__ == "__main__":
    run_production()
    print(f"\ntask completed! depth results saved to: {OUTPUT_DIR}")

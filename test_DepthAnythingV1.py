import cv2
import numpy as np
import onnxruntime as ort
import os
from tqdm import tqdm

# python test_DepthAnythingV1.py

MODEL_PATH = "/mnt/afs/zhengmingkai/zyr/THEMIS/new_models/depth_anything_v1_onnx/onnx/model_fp16.onnx"
INPUT_DIR = "/mnt/afs/zhengmingkai/zyr/THEMIS/test_images" 
OUTPUT_DIR = "/mnt/afs/zhengmingkai/zyr/THEMIS/depth_results_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

providers = [('MACAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
session = ort.InferenceSession(MODEL_PATH, providers=providers)

input_node = session.get_inputs()[0]
def get_dim(dim):
    return dim if isinstance(dim, int) and dim > 0 else 518
input_h, input_w = get_dim(input_node.shape[2]), get_dim(input_node.shape[3])

def run_production():
    all_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_name in tqdm(all_files, desc="Batch Processing"):
        try:
            in_path = os.path.join(INPUT_DIR, img_name)
            out_path = os.path.join(OUTPUT_DIR, os.path.splitext(img_name)[0] + "_depth.png")
            
            raw_img = cv2.imread(in_path)
            if raw_img is None: continue
            orig_h, orig_w = raw_img.shape[:2]
            
            # 1. 颜色转换与 Resize
            img = cv2.resize(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), (input_w, input_h))
            
            # 2. 归一化并强制指定 float32
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            
            # 3. 维度转换并再次确认 float32
            img = img.transpose(2, 0, 1)[None, ...].astype(np.float32)

            # 4. 推理
            depth = session.run(None, {input_node.name: img})[0]

            # 5. 后处理
            depth = cv2.resize(depth[0], (orig_w, orig_h))
            depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
            cv2.imwrite(out_path, depth_norm)
            
        except Exception as e:
            print(f"\nprocess {img_name} failed: {e}")

if __name__ == "__main__":
    run_production()
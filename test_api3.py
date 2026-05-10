import cv2
import numpy as np
import os
import requests
from rtmlib import Custom, Animal, draw_skeleton

# ================= Configuration =================
# 1. Hardware Configuration
# device = 'cuda' 
# backend = 'onnxruntime' 
device = 'cpu'        # Forced to cpu to ensure compatibility
backend = 'onnxruntime' 

# 2. Path Configuration: Define your custom model directory
MODEL_DIR = '/mnt/afs/zhengmingkai/zyr/THEMIS/new_models'
os.makedirs(MODEL_DIR, exist_ok=True)

# 3. Local Image Path
image_path = '/mnt/afs/zhengmingkai/zyr/THEMIS/test_images/hussar monkey2.png' 

# 4. Model Mirror Links
DET_URL = ' https://ghproxy.net/https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx'
POSE_URL = 'https://hf-mirror.com/JunkyByte/easy_ViTPose/resolve/main/onnx/apt36k/vitpose-b-apt36k.onnx'

# Local file paths for the models
# det_local = os.path.join(MODEL_DIR, 'yolox_s_humanart.zip')
# pose_local = os.path.join(MODEL_DIR, 'vitpose-b-apt36k.onnx')
det_local = os.path.join(MODEL_DIR, 'yolox_s.onnx')
pose_local = os.path.join(MODEL_DIR, 'vitpose-b-apt36k.onnx')
# =================================================

def download_model(url, save_path):
    """Downloads the model if it does not already exist at the specified path."""
    if os.path.exists(save_path):
        print(f"Model already exists: {save_path}. Skipping download.")
        return
    
    print(f"Downloading model from mirror to: {save_path}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download completed successfully.")
    except Exception as e:
        print(f"Download failed: {e}")
        # Clean up partial download to avoid corruption on next run
        if os.path.exists(save_path):
            os.remove(save_path)
        raise

def run_audit():
    # Step 1: Ensure models are available in the target directory
    download_model(DET_URL, det_local)
    download_model(POSE_URL, pose_local)

    # Step 2: Initialize Custom model using local paths
    print("Loading expert models...")
    custom = Custom(
        det_class='YOLOX',
        det_mode='multiclass',
        det=det_local,           # Use local path to skip internal rtmlib download
        det_input_size=(640, 640),
        pose_class='ViTPose',
        pose=pose_local,         # Use local path to skip internal rtmlib download
        pose_input_size=(192, 256),
        backend=backend,
        device=device
    )
    # animal = Animal(backend=backend, device=device)


    # Step 3: Load and process the image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load image from {image_path}")
        return
    
    # Run inference
    # keypoints shape: (N, K, 2), scores shape: (N, K)
    keypoints, scores = custom(img)
    # keypoints, scores = animal(img)

    # Step 4: Analysis and Visualization
    num_detected = len(keypoints)
    if num_detected > 0:
        print(f"Audit completed. Detected {num_detected} subjects.")
        
        # Calculate average confidence for audit reporting
        avg_score = np.mean(scores)
        print(f"Average keypoint confidence score: {avg_score:.4f}")

        # Draw skeleton and save the result
        img_show = draw_skeleton(img.copy(), keypoints, scores, kpt_thr=0.3)
        output_path = 'audit_result_visualization.png'
        cv2.imwrite(output_path, img_show)
        print(f"Visualization saved to: {output_path}")
    else:
        print("No subjects identified by the expert model.")

if __name__ == "__main__":
    run_audit()

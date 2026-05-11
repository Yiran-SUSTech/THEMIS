import cv2
import numpy as np
import onnxruntime as ort
import os

# 1. 尝试使用 FP16 版本以获取最高性能
MODEL_PATH = "/mnt/afs/zhengmingkai/zyr/THEMIS/new_models/depth_anything_v1_onnx/onnx/model_fp16.onnx"
TEST_IMAGE = "/mnt/afs/zhengmingkai/zyr/THEMIS/test_images/hussar monkey.png" # 使用你之前的测试图

# 2. MetaX 配置
providers = [('MACAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']

try:
    print(f"loading V1 model: {MODEL_PATH}")
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    print(f"load model success! current Provider: {session.get_providers()}")
    
    # 获取模型期待的输入尺寸 (V1 通常是 518 或 384)
    input_shape = session.get_inputs()[0].shape
    input_h, input_w = input_shape[2], input_shape[3]
    print(f"model input shape: {input_w}x{input_h}")

    # 3. 推理测试
    raw_img = cv2.imread(TEST_IMAGE)
    h, w = raw_img.shape[:2]
    img = cv2.resize(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), (input_w, input_h))
    img = (img.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = img.transpose(2, 0, 1)[None, ...]

    depth = session.run(None, {session.get_inputs()[0].name: img})[0]
    
    # 后处理
    depth = cv2.resize(depth[0], (w, h))
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
    cv2.imwrite("v1_gpu_test_result.png", depth_norm)
    print("🚀 inference success! result saved to v1_gpu_test_result.png")

except Exception as e:
    print(f"run error: {e}")
    print("\nsuggestion: if get opset or operator error, try to use 'model.onnx' again.")

import onnxruntime as ort
import numpy as np
import cv2

# --- 1. 配置路径 ---
MODEL_DIR = "new_models/sam2_hiera_small_onnx/onnx/"
ENCODER_PATH = MODEL_DIR + "vision_encoder.onnx"
DECODER_PATH = MODEL_DIR + "prompt_encoder_mask_decoder.onnx"
IMAGE_PATH = "test_images/hussar monkey.png" # 你的测试图片

# --- 2. 初始化 ONNX Session ---
# 如果没有 GPU，把 ['CUDAExecutionProvider'] 改为 ['CPUExecutionProvider']
providers = ['MACAExecutionProvider', 'CPUExecutionProvider']
encoder_session = ort.InferenceSession(ENCODER_PATH, providers=providers)
decoder_session = ort.InferenceSession(DECODER_PATH, providers=providers)

def preprocess_image(image_path, size=1024):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    # SAM2 默认通常缩放到 1024x1024
    img_resized = cv2.resize(img, (size, size))
    img_input = img_resized.astype(np.float32) / 255.0
    # 标准化 (ImageNet mean/std)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_input = (img_input - mean) / std
    img_input = img_input.transpose(2, 0, 1)[None, ...] # [1, 3, 1024, 1024]
    return img_input, (h, w)

# --- 3. 运行 Encoder (提取特征) ---
img_input, (orig_h, orig_w) = preprocess_image(IMAGE_PATH)
encoder_inputs = {encoder_session.get_inputs()[0].name: img_input}
# 获取 image_embeddings
image_embeddings = encoder_session.run(None, encoder_inputs)[0] 

# --- 4. 准备提示 (Prompts) ---
# 假设我们要点选猴子中心，坐标需要缩放到 1024 空间或相对于原图
# 这里简化为点选原图中心
input_point = np.array([[orig_w // 2, orig_h // 2]], dtype=np.float32)
input_label = np.array([1], dtype=np.float32) # 1 代表前景

# 对坐标进行归一化处理（取决于你导出的模型是否要求 1024 空间坐标）
# 绝大多数 SAM2 ONNX 要求点坐标对应到编码器输入尺寸 (1024)
input_point_reshaped = input_point * (1024 / np.array([orig_w, orig_h]))
onnx_coord = np.concatenate([input_point_reshaped, np.array([[0.0, 0.0]])], axis=0)[None, :, :].astype(np.float32)
onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)[None, :]
onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)

# --- 5. 运行 Decoder (生成掩码) ---
decoder_inputs = {
    "image_embeddings": image_embeddings,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array([orig_h, orig_w], dtype=np.float32)
}

masks, scores, low_res_masks = decoder_session.run(None, decoder_inputs)

# --- 6. 后处理与保存 ---
# masks 形状通常为 [1, 3, H, W]，选分数最高的那个
best_mask = masks[0, np.argmax(scores[0])]
binary_mask = (best_mask > 0).astype(np.uint8) * 255
cv2.imwrite("sam2_output_mask.png", binary_mask)
print("Mask saved to sam2_output_mask.png")
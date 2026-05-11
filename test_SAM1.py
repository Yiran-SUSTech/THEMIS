import onnxruntime as ort
import numpy as np
import cv2
import os

# --- 1. 配置路径 ---
MODEL_DIR = "new_models/sam1_onnx/machine_learning_models/"
# 建议先用 MobileSAM 测试，如果追求精度再换 sam_vit_h
ENCODER_PATH = os.path.join(MODEL_DIR, "mobile_sam.encoder_v16.onnx")
DECODER_PATH = os.path.join(MODEL_DIR, "mobile_sam.decoder_v16.onnx")
IMAGE_PATH = "./test_images/hussar monkey.png" 

# --- 2. 初始化 ONNX Session ---
# 适配你的 MACA 环境
providers = ['MACAExecutionProvider', 'CPUExecutionProvider']
encoder_session = ort.InferenceSession(ENCODER_PATH, providers=providers)
decoder_session = ort.InferenceSession(DECODER_PATH, providers=providers)

def preprocess_image(image_path, input_size=1024):
    """SAM 1 标准预处理：等比例缩放 + Padding 到 1024x1024"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # 计算缩放比例，保持长宽比
    scale = input_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # 创建 1024x1024 的输入容器并填充
    input_img = np.zeros((input_size, input_size, 3), dtype=np.float32)
    input_img[:new_h, :new_w, :] = img_resized
    
    # 标准化
    input_img = input_img / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) # 显式指定 float32
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # 显式指定 float32
    input_img = (input_img - mean) / std
    
    input_img = input_img.transpose(2, 0, 1)[None, ...]
    return input_img.astype(np.float32), (h, w), scale # 核心：强制转为 float32

# --- 3. 运行 Encoder ---
print("running encoder...")
img_input, (orig_h, orig_w), scale = preprocess_image(IMAGE_PATH)
encoder_inputs = {encoder_session.get_inputs()[0].name: img_input}
image_embeddings = encoder_session.run(None, encoder_inputs)[0]

# --- 4. 准备交互提示 (Prompt) ---
# 示例：点选图片中心
input_point = np.array([[orig_w // 2, orig_h // 2]], dtype=np.float32)
input_label = np.array([1], dtype=np.float32) # 1 为前景

# 将坐标缩放到 1024 空间
onnx_coord = input_point * scale
# SAM 1 Decoder 通常需要增加一个 [0,0] 的填充点以满足张量形状要求（部分导出版需要）
onnx_coord = np.concatenate([onnx_coord, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :]

# 初始 Mask 输入（空）
onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)

# --- 5. 运行 Decoder ---
print("running decoder...")
# 注意：SAM 1 的输入名称可能因导出工具不同而有细微差异
# 如果报错，请使用 print(decoder_session.get_inputs()) 检查名称
decoder_inputs = {
    "image_embeddings": image_embeddings.astype(np.float32), # 确保 embedding 是 float32
    "point_coords": onnx_coord.astype(np.float32),
    "point_labels": onnx_label.astype(np.float32),
    "mask_input": onnx_mask_input.astype(np.float32),
    "has_mask_input": onnx_has_mask_input.astype(np.float32),
    "orig_im_size": np.array([orig_h, orig_w], dtype=np.float32)
}

masks, scores, low_res_masks = decoder_session.run(None, decoder_inputs)

# --- 6. 后处理 ---
# 选择得分最高的 Mask
best_mask_idx = np.argmax(scores[0])
mask = masks[0, best_mask_idx]
mask_binary = (mask > 0).astype(np.uint8) * 255

# 保存结果
output_name = "sam1_mobile_result.png"
cv2.imwrite(output_name, mask_binary)
print(f"success save: {output_name}")

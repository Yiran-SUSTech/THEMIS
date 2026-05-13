import torch
from transformers import Qwen2_5_VLForConditionalGeneration
import os

# 1. 配置路径
model_path = "/mnt/afs/zhengmingkai/zyr/THEMIS/models/Q-Insight/score_degradation"
output_path = "./onnx_output"
if not os.path.exists(output_path):
    os.makedirs(output_path)

print("loading model to CPU...")
# 注意：为了导出兼容性，先加载到 CPU 并使用 fp32
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
).eval()

# 2. 提取视觉部分 (Vision Tower)
# Q-Insight/Qwen2.5-VL 的图像核心在这个 visual 模块
vision_model = model.visual

# 3. 构造模拟输入 (Dummy Inputs)
# 根据 Qwen2.5-VL 结构，需要 pixel_values 和 grid_thw
# 这里假设输入是一个 224x224 的图像区域
pixel_values = torch.randn(1, 3, 224, 224) 
grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.int64) # 224/14 = 16

# 4. 执行导出
print(f"exporting Vision Encoder to ONNX (Opset 16)...")
torch.onnx.export(
    vision_model,
    (pixel_values, grid_thw),
    f"{output_path}/qinsight_vision.onnx",
    export_params=True,
    opset_version=16,  # 严格遵守你服务器的限制
    do_constant_folding=True,
    input_names=['pixel_values', 'grid_thw'],
    output_names=['image_embeds'],
    dynamic_axes={
        'pixel_values': {0: 'batch', 2: 'height', 3: 'width'},
        'grid_thw': {0: 'batch'},
        'image_embeds': {0: 'batch'}
    }
)

print(f"exported Vision Encoder to ONNX file: {output_path}/qinsight_vision.onnx")

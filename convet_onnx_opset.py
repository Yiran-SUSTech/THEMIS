import onnx
from onnx import version_converter

# 1. 加载你下载的原始模型
model_path = "/mnt/afs/zhengmingkai/zyr/THEMIS/new_models/depth_anything_v2_onnx/onnx/model_fp16.onnx"
original_model = onnx.load(model_path)

# 2. 转换 Opset 到 16 (沐曦支持的最高版本)
# 这一步会自动调整模型中不兼容的算子定义
target_opset = 16
converted_model = version_converter.convert_version(original_model, target_opset)

# 3. 保存新模型
new_model_path = "/mnt/afs/zhengmingkai/zyr/THEMIS/new_models/depth_anything_v2_onnx/onnx/model_fp16_opset16.onnx"
onnx.save(converted_model, new_model_path)

print(f"convert model to opset16: {model_path} -> {new_model_path}")
import onnx
from onnx import version_converter
import os

# 路径配置
base_dir = "new_models/sam2_hiera_small_onnx/onnx/"
target_version = 16

def simple_convert(filename):
    input_path = os.path.join(base_dir, filename)
    output_path = os.path.join(base_dir, filename.replace(".onnx", "_v16.onnx"))
    
    print(f"converting: {filename}...")
    try:
        # 加载时会默认加载同目录下的 .onnx_data
        model = onnx.load(input_path)
        # 执行转换
        converted_model = version_converter.convert_version(model, target_version)
        # 保存时会生成新的 .onnx_data
        onnx.save(converted_model, output_path)
        print(f"success save: {output_path}")
    except Exception as e:
        print(f"{filename} convert failed: {e}")

# 只转这两个核心模型
simple_convert("vision_encoder.onnx")
simple_convert("prompt_encoder_mask_decoder.onnx")
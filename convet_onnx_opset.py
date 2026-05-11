import onnx

# 路径配置
old_path = '/mnt/afs/zhengmingkai/zyr/THEMIS/new_models/depth_anything_v2_onnx/onnx/model_fp16.onnx'
new_path = '/mnt/afs/zhengmingkai/zyr/THEMIS/new_models/depth_anything_v2_onnx/onnx/model_fp16_forced_opset16.onnx'

def force_downgrade():
    print(f"Loading model: {old_path}")
    # 注意：如果文件很大，加载可能需要一点时间
    model = onnx.load(old_path)

    # 1. 强制修改 IR 版本为沐曦支持的 8
    model.ir_version = 8
    
    # 2. 强制修改 Opset 版本
    # 绕过 version_converter，直接修改版本标识符
    for opset in model.opset_import:
        if opset.domain == '' or opset.domain == 'ai.onnx':
            print(f"Forcing Opset {opset.version} -> 16")
            opset.version = 16

    # 3. 保存模型
    onnx.save(model, new_path)
    print(f"save model to: {new_path}")

if __name__ == "__main__":
    force_downgrade()
import onnxruntime as ort
import numpy as np
import time

model_path = "new_models/eva02_large_metax_compatible.onnx"

# 1. 显式指定使用沐曦的 MACA 加速后端，若失败则安全回退到 CPU
providers = ['MACAExecutionProvider', 'CPUExecutionProvider']

print(f"loading model {model_path} to MetaX MACA engine ...")
try:
    # 启动会话，此时沐曦的驱动会对 ONNX 算子进行底层硬件对齐和编译
    session = ort.InferenceSession(model_path, providers=providers)
    
    # 打印当前实际生效的 Provider，确保 MACA 在工作
    # 更新打印信息，明确指出当前使用的模型类型和尺寸，与convert_EVA_02.py中的模型配置保持一致
    active_provider = session.get_providers()[0] if session.get_providers() else "Unknown"
    print(f"session created with active provider: {session.get_provider_options().keys()}")
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # 2. 构造 448x448 的静态测试输入数据 (符合 ImageNet 标准的标准化随机矩阵)
    dummy_data = np.random.randn(1, 3, 448, 448).astype(np.float32)
    
    # 3. 预热（Warm-up）：让国产显卡完成初次显存分配和算子缓存
    print("warm-up model...")
    _ = session.run([output_name], {input_name: dummy_data})
    
    # 4. 性能测试（Benchmark）：循环推理 20 次，计算平均每张图的吞吐耗时
    print("benchmark model performance...")
    iters = 20
    start_time = time.time()
    for _ in range(iters):
        outputs = session.run([output_name], {input_name: dummy_data})
    end_time = time.time()
    
    avg_time_ms = ((end_time - start_time) / iters) * 1000
    print("\n" + "="*40)
    print(f"MetaX report:")
    print(f"output shape (should be 1, 1000): {outputs[0].shape}")
    print(f"avg inference time: {avg_time_ms:.2f} ms")
    print(f"system throughput (frames/sec): {1000 / avg_time_ms:.1f} frames/sec")
    print("="*40)

except Exception as e:
    print("\n onnxruntime error:")
    print(e)
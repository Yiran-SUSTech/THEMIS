import os
# ！！！【核心修复】必须在 import torch 和 timm 之前，强行关闭 PyTorch 的 Fused Attention 机制
os.environ['TIMM_FUSED_ATTN'] = '0'

import torch
import timm
from onnxsim import simplify
import onnx

# 锁定方案 B 对应的 ImageNet-1K 巅峰模型
model_name = 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k' 
print(f"正在从 timm 注册表下载并加载官方预训练模型: {model_name}...")

# 额外传入 kwargs 强行锁定不使用 SDPA
model = timm.create_model(model_name, pretrained=True, exportable=True)
model.eval()

# 严格匹配 448x448 静态分辨率
dummy_input = torch.randn(1, 3, 448, 448)
raw_onnx = "eva02_large_raw.onnx"
final_onnx = "eva02_large_metax_compatible.onnx"

print("正在以 Opset 15 导出模型（此步骤需要 1-2 分钟，请耐心等待）...")
with torch.no_grad():
    torch.onnx.export(
        model,
        dummy_input,
        raw_onnx,
        export_params=True,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        opset_version=15 # 锁定 Opset 15 以向下兼容你 MetaX 的 ONNX Runtime 1.12.0
    )

print("正在使用 onnx-simplifier 进行多模态图结构精简与常数折叠...")
try:
    onnx_model = onnx.load(raw_onnx)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model skipping check failed"
    onnx.save(model_simp, final_onnx)
    print(f"🎉 成功！适配 MetaX 服务器的 EVA-02 Large 模型已生成：{final_onnx}")
    
    # 顺手把中间临时文件删掉，保持目录干净
    if os.path.exists(raw_onnx):
        os.remove(raw_onnx)
except Exception as e:
    print(f"❌ 简化阶段失败，但原始 ONNX 已保留。错误原因: {e}")
#!/usr/bin/env python3
"""
MetaX GPU 快速诊断脚本
直接复制此脚本内容到服务器运行
"""

import sys
import subprocess

def run(cmd):
    """运行命令"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout.strip(), result.returncode == 0
    except Exception as e:
        return str(e), False

print("=" * 70)
print("MetaX GPU 快速诊断")
print("=" * 70)

# 1. GPU信息
print("\n[1] GPU 信息:")
out, ok = run("mx-smi -L")
print(out if ok else "无法获取GPU列表")

# 2. 显存信息
print("\n[2] 显存信息:")
out, ok = run("mx-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader")
print(out if ok else "无法获取显存信息")

# 3. Python版本
print("\n[3] Python 版本:")
out, ok = run("python --version")
print(out)

# 4. 关键包版本
print("\n[4] 关键包版本:")
packages = ["torch", "transformers", "numpy", "pillow", "timm", "ultralytics", "pyiqa", "langchain", "langgraph"]
for pkg in packages:
    out, ok = run(f"python -c 'import {pkg}; print({pkg}.__version__)' 2>/dev/null || echo '未安装'")
    print(f"  {pkg}: {out}")

# 5. PyTorch CUDA测试
print("\n[5] PyTorch CUDA 测试:")
test_code = '''
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
'''
out, ok = run(f'python -c "{test_code}"')
print(out)

# 6. 简单计算测试
print("\n[6] 简单计算测试:")
test_code = '''
import torch
try:
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"矩阵乘法成功: {z.shape}")
    print("✓ CUDA计算正常")
except Exception as e:
    print(f"✗ CUDA计算失败: {e}")
'''
out, ok = run(f'python -c "{test_code}"')
print(out)

# 7. 模型加载测试
print("\n[7] 小模型加载测试:")
test_code = '''
import torch
from transformers import AutoModel, AutoTokenizer
try:
    print("加载bert-base-uncased...")
    model = AutoModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float16, device_map="cuda")
    print(f"模型设备: {model.device}")
    print("✓ 模型加载成功")
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
'''
out, ok = run(f'python -c "{test_code}"')
print(out)

# 8. 环境变量
print("\n[8] 相关环境变量:")
env_vars = ["CUDA_VISIBLE_DEVICES", "CUDA_HOME", "LD_LIBRARY_PATH", "MACA_PATH"]
for var in env_vars:
    out, ok = run(f"echo ${var}")
    if out:
        print(f"  {var}: {out}")

print("\n" + "=" * 70)
print("诊断完成")
print("=" * 70)

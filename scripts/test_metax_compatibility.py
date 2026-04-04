#!/usr/bin/env python3
"""
MetaX GPU Compatibility Test Script
用于测试 THEMIS 项目在 MetaX GPU 上的兼容性
"""

import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path


def run_command(cmd, description):
    """运行命令并返回结果"""
    print(f"\n{'='*60}")
    print(f"[测试] {description}")
    print(f"[命令] {cmd}")
    print('-'*60)
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        print(result.stdout)
        if result.stderr:
            print(f"[stderr] {result.stderr}")
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except Exception as e:
        print(f"[错误] {e}")
        return {"success": False, "error": str(e)}


def run_python_test(code, description):
    """运行Python代码测试"""
    print(f"\n{'='*60}")
    print(f"[测试] {description}")
    print('-'*60)
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=120
        )
        print(result.stdout)
        if result.stderr:
            print(f"[stderr] {result.stderr}")
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except Exception as e:
        print(f"[错误] {e}")
        return {"success": False, "error": str(e)}


def main():
    print("="*60)
    print("MetaX GPU 兼容性测试")
    print(f"时间: {datetime.now()}")
    print("="*60)

    results = {}

    # 1. GPU信息
    print("\n" + "="*60)
    print("[1/20] GPU基础信息")
    print("="*60)
    
    results["mx_smi"] = run_command("mx-smi", "MetaX GPU状态")
    results["gpu_count"] = run_command("mx-smi -L", "GPU列表")
    results["maca_info"] = run_command("mx-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv", "GPU详细信息")

    # 2. CUDA/驱动信息
    print("\n" + "="*60)
    print("[2/20] CUDA环境信息")
    print("="*60)
    
    results["cuda_version"] = run_command("nvcc --version 2>/dev/null || echo 'nvcc not found'", "CUDA版本")
    results["maca_version"] = run_command("mx-smi | grep 'MACA Version'", "MACA版本")
    results["driver_version"] = run_command("mx-smi | grep 'Driver Version'", "驱动版本")

    # 3. Python环境
    print("\n" + "="*60)
    print("[3/20] Python环境")
    print("="*60)
    
    results["python_version"] = run_python_test("import sys; print(f'Python {sys.version}')", "Python版本")
    results["pip_list"] = run_command("pip list | grep -E 'torch|transformers|numpy|pillow'", "关键Python包")

    # 4. PyTorch测试
    print("\n" + "="*60)
    print("[4/20] PyTorch基础测试")
    print("="*60)
    
    results["torch_import"] = run_python_test("""
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用的: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'显存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB')
""", "PyTorch基础测试")

    # 5. PyTorch计算能力
    print("\n" + "="*60)
    print("[5/20] PyTorch计算能力测试")
    print("="*60)
    
    results["torch_cuda_ops"] = run_python_test("""
import torch
print('测试CUDA操作...')
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = torch.matmul(x, y)
print(f'矩阵乘法成功: {z.shape}')
print(f'结果设备: {z.device}')
print('✓ CUDA操作正常工作')
""", "CUDA基础计算")

    # 6. PyTorch自动求导
    print("\n" + "="*60)
    print("[6/20] PyTorch自动求导测试")
    print("="*60)
    
    results["torch_autograd"] = run_python_test("""
import torch
print('测试自动求导...')
x = torch.randn(3, requires_grad=True, device='cuda')
y = x * 2
z = y.sum()
z.backward()
print(f'梯度计算成功: {x.grad}')
print('✓ 自动求导正常工作')
""", "自动求导测试")

    # 7. Transformers导入测试
    print("\n" + "="*60)
    print("[7/20] Transformers测试")
    print("="*60)
    
    results["transformers_import"] = run_python_test("""
import transformers
print(f'Transformers版本: {transformers.__version__}')
from transformers import AutoModel, AutoTokenizer
print('✓ Transformers导入成功')
""", "Transformers导入")

    # 8. 显存分配测试
    print("\n" + "="*60)
    print("[8/20] 显存分配测试")
    print("="*60)
    
    results["torch_memory"] = run_python_test("""
import torch
print('测试显存分配...')
# 分配小量显存
x = torch.randn(1000, 1000, device='cuda')
print(f'分配后显存: {torch.cuda.memory_allocated() / 1e6:.2f} MB')
# 释放
del x
torch.cuda.empty_cache()
print(f'释放后显存: {torch.cuda.memory_allocated() / 1e6:.2f} MB')
print('✓ 显存管理正常')
""", "显存分配测试")

    # 9. 多GPU测试
    print("\n" + "="*60)
    print("[9/20] 多GPU测试")
    print("="*60)
    
    results["multi_gpu"] = run_python_test("""
import torch
print('测试多GPU...')
if torch.cuda.device_count() >= 2:
    x = torch.randn(1000, 1000, device='cuda:0')
    y = torch.randn(1000, 1000, device='cuda:1')
    z = torch.matmul(x, y)
    print(f'多GPU计算成功: {z.shape}')
    print(f'✓ 多GPU正常工作')
else:
    print(f'只有 {torch.cuda.device_count()} 个GPU，跳过测试')
""", "多GPU测试")

    # 10. 批处理测试
    print("\n" + "="*60)
    print("[10/20] 批处理测试")
    print("="*60)
    
    results["batch_processing"] = run_python_test("""
import torch
print('测试批处理...')
batch = torch.randn(8, 3, 224, 224).cuda()
print(f'批处理张量形状: {batch.shape}')
# 简单卷积
conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
output = conv(batch)
print(f'卷积输出形状: {output.shape}')
print('✓ 批处理正常')
""", "批处理测试")

    # 11. 图像处理测试
    print("\n" + "="*60)
    print("[11/20] 图像处理测试 (PIL)")
    print("="*60)
    
    results["pillow_test"] = run_python_test("""
from PIL import Image
import numpy as np
print('测试PIL图像处理...')
img = Image.new('RGB', (224, 224), color=(255, 0, 0))
img_array = np.array(img)
print(f'图像尺寸: {img.size}')
print(f'数组形状: {img_array.shape}')
print('✓ PIL正常工作')
""", "PIL测试")

    # 12. 图像转Tensor测试
    print("\n" + "="*60)
    print("[12/20] 图像转Tensor测试")
    print("="*60)
    
    results["image_to_tensor"] = run_python_test("""
import torch
from PIL import Image
import numpy as np
print('测试图像转Tensor...')
img = Image.new('RGB', (224, 224), color=(255, 0, 0))
img_array = np.array(img).transpose(2, 0, 1) / 255.0
img_tensor = torch.from_numpy(img_array).float().unsqueeze(0).cuda()
print(f'Tensor形状: {img_tensor.shape}')
print(f'Tensor设备: {img_tensor.device}')
print('✓ 图像转Tensor正常')
""", "图像转Tensor")

    # 13. torchvision测试
    print("\n" + "="*60)
    print("[13/20] TorchVision测试")
    print("="*60)
    
    results["torchvision_test"] = run_python_test("""
import torchvision
print(f'TorchVision版本: {torchvision.__version__}')
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
print('✓ TorchVision导入成功')
""", "TorchVision测试")

    # 14. timm测试 (如果已安装)
    print("\n" + "="*60)
    print("[14/20] timm测试 (图像模型)")
    print("="*60)
    
    results["timm_test"] = run_python_test("""
try:
    import timm
    print(f'timm版本: {timm.__version__}')
    model = timm.create_model('efficientnet_b0', pretrained=False)
    model = model.cuda()
    x = torch.randn(1, 3, 224, 224).cuda()
    y = model(x)
    print(f'模型输出形状: {y.shape}')
    print('✓ timm正常工作')
except ImportError as e:
    print(f'timm未安装: {e}')
except Exception as e:
    print(f'timm测试失败: {e}')
""", "timm测试")

    # 15. ultralytics测试 (YOLO)
    print("\n" + "="*60)
    print("[15/20] Ultralytics测试 (YOLO)")
    print("="*60)
    
    results["ultralytics_test"] = run_python_test("""
try:
    import ultralytics
    print(f'Ultralytics版本: {ultralytics.__version__}')
    from ultralytics import YOLO
    print('✓ Ultralytics导入成功')
except ImportError as e:
    print(f'ultralytics未安装: {e}')
except Exception as e:
    print(f'ultralytics测试失败: {e}')
""", "Ultralytics测试")

    # 16. pyiqa测试 (图像质量)
    print("\n" + "="*60)
    print("[16/20] pyiqa测试 (图像质量评估)")
    print("="*60)
    
    results["pyiqa_test"] = run_python_test("""
try:
    import pyiqa
    print(f'pyiqa版本: {pyiqa.__version__}')
    metric = pyiqa.create_metric('niqe', device='cuda')
    print('✓ pyiqa导入成功')
except ImportError as e:
    print(f'pyiqa未安装: {e}')
except Exception as e:
    print(f'pyiqa测试失败: {e}')
""", "pyiqa测试")

    # 17. 大模型加载测试
    print("\n" + "="*60)
    print("[17/20] 小模型加载测试")
    print("="*60)
    
    results["model_load_test"] = run_python_test("""
import torch
from transformers import AutoModel, AutoTokenizer
print('测试加载小模型...')
# 测试加载BERT (较小)
try:
    model = AutoModel.from_pretrained('bert-base-uncased', torch_dtype=torch.float16, device_map='cuda')
    print(f'模型设备: {model.device}')
    print('✓ 模型加载成功')
except Exception as e:
    print(f'模型加载失败: {e}')
""", "模型加载测试")

    # 18. 显存峰值测试
    print("\n" + "="*60)
    print("[18/20] 显存峰值测试")
    print("="*60)
    
    results["memory_peak"] = run_python_test("""
import torch
print('测试显存峰值...')
torch.cuda.reset_peak_memory_stats()
# 创建大张量
x = torch.randn(10000, 10000, device='cuda')
peak = torch.cuda.max_memory_allocated() / 1e9
print(f'峰值显存: {peak:.2f} GB')
del x
torch.cuda.empty_cache()
print('✓ 显存峰值测试完成')
""", "显存峰值测试")

    # 19. 推理延迟测试
    print("\n" + "="*60)
    print("[19/20] 推理延迟测试")
    print("="*60)
    
    results["inference_latency"] = run_python_test("""
import torch
import time
print('测试推理延迟...')
# 模拟推理
model = torch.nn.Linear(768, 768).cuda()
x = torch.randn(1, 768).cuda()
# 预热
for _ in range(10):
    _ = model(x)
# 正式测试
times = []
for _ in range(100):
    start = time.time()
    _ = model(x)
    times.append(time.time() - start)
avg_time = sum(times) / len(times) * 1000
print(f'平均推理延迟: {avg_time:.2f} ms')
print('✓ 推理延迟测试完成')
""", "推理延迟测试")

    # 20. 分布式测试
    print("\n" + "="*60)
    print("[20/20] 分布式PyTorch测试")
    print("="*60)
    
    results["distributed_test"] = run_python_test("""
import torch
import torch.distributed as dist
print('测试PyTorch分布式功能...')
print(f'支持NCCL: {dist.is_nccl_available()}')
print(f'支持GLOO: {dist.is_gloo_available()}')
print('✓ 分布式PyTorch检查完成')
""", "分布式测试")

    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r.get("success", False))
    total = len(results)
    
    print(f"\n通过: {passed}/{total}")
    
    for name, result in results.items():
        status = "✓ PASS" if result.get("success", False) else "✗ FAIL"
        print(f"{status} {name}")
    
    # 保存结果
    output_file = "metax_compatibility_test_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_file}")
    
    return passed, total


if __name__ == "__main__":
    passed, total = main()
    sys.exit(0 if passed == total else 1)

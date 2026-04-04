#!/usr/bin/env python3
"""
THEMIS 完整部署脚本 - MetaX GPU 服务器
运行此脚本后即可进行图像质量评估

使用方法:
    python setup_metax.py
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

# 配置
MODEL_DIR = "/mnt/afs/zhengmingkai/zyr/THEMIS/models"
PROJECT_DIR = "/mnt/afs/zhengmingkai/zyr/THEMIS"
LOG_DIR = f"{PROJECT_DIR}/logs"

# 国内镜像
HF_ENDPOINT = "https://hf-mirror.com"
PIP_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"


def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_step(step, total, message):
    print(f"\n[{step}/{total}] {message}")


def print_success(message):
    print(f"✓ {message}")


def print_error(message):
    print(f"✗ {message}")


def run_command(cmd, cwd=None, timeout=300):
    """运行命令"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)


def pip_install(packages):
    """安装 pip 包"""
    cmd = f"pip install -i {PIP_INDEX} --trusted-host pypi.tuna.tsinghua.edu.cn {packages}"
    success, stdout, stderr = run_command(cmd, timeout=600)
    return success


def create_directories():
    """创建目录"""
    print_step(1, 7, "创建目录")
    
    dirs = [
        MODEL_DIR,
        LOG_DIR,
        f"{PROJECT_DIR}/test_images",
        f"{PROJECT_DIR}/outputs",
        f"{PROJECT_DIR}/cache",
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  创建: {d}")
    
    print_success("目录创建完成")


def install_packages():
    """安装 Python 包"""
    print_step(2, 7, "安装 Python 依赖包")
    
    # 升级 pip
    print("升级 pip...")
    run_command(f"pip install --upgrade pip setuptools wheel -i {PIP_INDEX} --trusted-host pypi.tuna.tsinghua.edu.cn")
    
    # 基础包
    print("安装基础包...")
    base_packages = [
        "numpy scipy scikit-learn pandas",
        "pillow opencv-python matplotlib seaborn",
        "tqdm pyyaml python-dotenv requests aiohttp",
        "huggingface_hub safetensors",
    ]
    for pkgs in base_packages:
        if pip_install(pkgs):
            print(f"  ✓ {pkgs.split()[0]} 等")
    
    # Transformers 相关
    print("安装 Transformers 相关包...")
    tf_packages = "transformers accelerate sentencepiece protobuf einops timm kornia albumentations"
    if pip_install(tf_packages):
        print_success("Transformers 相关包安装完成")
    
    # LangChain 相关
    print("安装 LangChain 相关包...")
    lc_packages = "langgraph langchain langchain-core langchain-anthropic"
    if pip_install(lc_packages):
        print_success("LangChain 相关包安装完成")
    
    # 视觉模型相关
    print("安装视觉模型相关包...")
    vision_packages = "ultralytics pyiqa"
    if pip_install(vision_packages):
        print_success("视觉模型相关包安装完成")
    
    print_success("Python 依赖安装完成")


def download_hf_model(repo_id, local_dir, model_name):
    """下载 HuggingFace 模型"""
    from huggingface_hub import snapshot_download
    
    print(f"\n  下载 {model_name}...")
    
    if Path(local_dir).exists() and any(Path(local_dir).iterdir()):
        print(f"  ✓ {model_name} 已存在，跳过下载")
        return True
    
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print_success(f"{model_name} 下载完成")
        return True
    except Exception as e:
        print_error(f"{model_name} 下载失败: {e}")
        return False


def download_models():
    """下载所有模型"""
    print_step(3, 7, "下载模型")
    
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    
    # Qwen 模型
    print("\n下载 Qwen VL 模型...")
    download_hf_model(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        f"{MODEL_DIR}/Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL-3B-Instruct"
    )
    download_hf_model(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        f"{MODEL_DIR}/Qwen2.5-VL-7B-Instruct",
        "Qwen2.5-VL-7B-Instruct"
    )
    
    # CLIP 模型
    print("\n下载 CLIP 模型...")
    download_hf_model(
        "openai/clip-vit-base-patch32",
        f"{MODEL_DIR}/clip-vit-base-patch32",
        "CLIP-ViT-B-32"
    )
    
    print_success("模型下载完成")


def download_additional_models():
    """下载额外模型 (YOLO, IQA等)"""
    print_step(4, 7, "下载额外模型")
    
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    
    # EfficientNet
    print("\n预下载 EfficientNet...")
    try:
        import timm
        import torch
        
        model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True)
        torch.save(model.state_dict(), f'{MODEL_DIR}/efficientnetv2_s_in21k.pth')
        print_success("EfficientNetV2-S 预下载完成")
    except Exception as e:
        print_error(f"EfficientNet 预下载失败: {e}")
    
    # YOLO
    print("\n下载 YOLO 模型...")
    try:
        from ultralytics import YOLO
        
        yolo_dir = Path(f"{MODEL_DIR}/yolo")
        yolo_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name in ['yolo11n', 'yolo11n-pose', 'yolo11m-pose']:
            print(f"  下载 {model_name}...")
            model = YOLO(f'{model_name}.pt')
            model.save(str(yolo_dir / f'{model_name}.pt'))
        
        print_success("YOLO 模型下载完成")
    except Exception as e:
        print_error(f"YOLO 下载失败: {e}")
    
    # IQA
    print("\n预下载 IQA 模型...")
    try:
        import pyiqa
        
        for metric in ['maniqa', 'musiq', 'niqe']:
            print(f"  预下载 {metric}...")
            pyiqa.create_metric(metric, device='cpu')
        
        print_success("IQA 模型预下载完成")
    except Exception as e:
        print_error(f"IQA 预下载失败: {e}")
    
    print_success("额外模型下载完成")


def create_env_file():
    """创建环境配置文件"""
    print_step(5, 7, "创建配置文件")
    
    env_content = f'''# =============================================================================
# THEMIS 环境配置 - MetaX GPU 服务器
# =============================================================================

# 模型目录
MODEL_DIR={MODEL_DIR}

# =============================================================================
# 核心模型配置
# =============================================================================

# 评估配置档位: fast, standard, accurate, deep
EVALUATION_PROFILE=standard

# Planner 配置 (Qwen2.5-VL-3B)
PLANNER_PROFILE=fast
PLANNER_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
PLANNER_LOCAL_MODEL_PATH={MODEL_DIR}/Qwen2.5-VL-3B-Instruct
PLANNER_DEVICE=cuda:0
PLANNER_MAX_NEW_TOKENS=420

# Judge 配置 (Qwen2.5-VL-3B)
JUDGE_PROFILE=fast
JUDGE_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
JUDGE_LOCAL_MODEL_PATH={MODEL_DIR}/Qwen2.5-VL-3B-Instruct
JUDGE_DEVICE=cuda:1
JUDGE_MAX_NEW_TOKENS=256

# Reflector 配置 (Qwen2.5-VL-7B)
REFLECTOR_PROFILE=standard
REFLECTOR_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
REFLECTOR_LOCAL_MODEL_PATH={MODEL_DIR}/Qwen2.5-VL-7B-Instruct
REFLECTOR_DEVICE=cuda:2
REFLECTOR_MAX_NEW_TOKENS=400

# =============================================================================
# 专家模型路径
# =============================================================================

IMAGENET_MODEL_PATH={MODEL_DIR}/efficientnetv2_s_in21k.pth
CLIP_MODEL_PATH={MODEL_DIR}/clip-vit-base-patch32
YOLO_MODEL_PATH={MODEL_DIR}/yolo

# =============================================================================
# 设备配置
# =============================================================================

DEVICE=cuda
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# =============================================================================
# 评估设置
# =============================================================================

MAX_PLAN_REVISIONS=2
MAX_REFLECTION_REVISIONS=1
SEMANTIC_ESCALATION_THRESHOLD=0.72
ARTIFACT_CONFIDENCE_THRESHOLD=0.55

# =============================================================================
# 日志配置
# =============================================================================

LOG_LEVEL=INFO
LOG_DIR={LOG_DIR}

# =============================================================================
# HuggingFace 镜像 (国内加速)
# =============================================================================

HF_ENDPOINT=https://hf-mirror.com
'''
    
    env_path = Path(f"{PROJECT_DIR}/.env")
    env_path.write_text(env_content, encoding='utf-8')
    print_success(f"配置文件创建完成: {env_path}")


def verify_installation():
    """验证安装"""
    print_step(6, 7, "验证安装")
    
    # 包版本检查
    print("\n[包版本检查]")
    packages = [
        "torch", "transformers", "timm", "ultralytics", "pyiqa",
        "langchain", "langgraph", "numpy", "pillow"
    ]
    
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {pkg}: {version}")
        except ImportError:
            print(f"  ✗ {pkg}: 未安装")
    
    # GPU 检查
    print("\n[GPU 检查]")
    try:
        import torch
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        print(f"  GPU 数量: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"  GPU 名称: {torch.cuda.get_device_name(0)}")
            print(f"  显存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except Exception as e:
        print(f"  ✗ GPU 检查失败: {e}")
    
    # 模型检查
    print("\n[模型检查]")
    models = [
        ("Qwen2.5-VL-3B-Instruct", f"{MODEL_DIR}/Qwen2.5-VL-3B-Instruct"),
        ("Qwen2.5-VL-7B-Instruct", f"{MODEL_DIR}/Qwen2.5-VL-7B-Instruct"),
        ("CLIP-ViT-B-32", f"{MODEL_DIR}/clip-vit-base-patch32"),
        ("YOLO", f"{MODEL_DIR}/yolo"),
    ]
    
    for name, path in models:
        if Path(path).exists():
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}: 不存在")
    
    print_success("验证完成")


def create_test_image():
    """创建测试图片"""
    print_step(7, 7, "创建测试图片")
    
    try:
        from PIL import Image
        
        test_dir = Path(f"{PROJECT_DIR}/test_images")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建测试图片
        img = Image.new('RGB', (256, 256), color=(100, 150, 200))
        img.save(test_dir / "test_image.png")
        
        print_success(f"测试图片创建完成: {test_dir / 'test_image.png'}")
    except Exception as e:
        print_error(f"测试图片创建失败: {e}")


def print_final_instructions():
    """打印最终说明"""
    print("\n" + "=" * 60)
    print(" 部署完成！")
    print("=" * 60)
    
    print(f"""
模型目录: {MODEL_DIR}
配置文件: {PROJECT_DIR}/.env

运行评估命令:
    cd {PROJECT_DIR}
    python -m src.agentic_eval.run_single \\
        ./test_images/test_image.png \\
        --class-label "test object" \\
        --output ./outputs/result.json \\
        --planner-model {MODEL_DIR}/Qwen2.5-VL-3B-Instruct

快速测试:
    python -c "
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_path = '{MODEL_DIR}/Qwen2.5-VL-3B-Instruct'
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map='auto', local_files_only=True
)
print(f'模型加载成功，显存: {{torch.cuda.memory_allocated()/1e9:.2f}} GB')
"
""")


def main():
    print("=" * 60)
    print(" THEMIS 完整部署脚本 - MetaX GPU 服务器")
    print(f" 时间: {datetime.now()}")
    print("=" * 60)
    print(f"\n模型目录: {MODEL_DIR}")
    print(f"项目目录: {PROJECT_DIR}")
    print(f"HF镜像: {HF_ENDPOINT}")
    
    # 设置环境变量
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    
    # 执行部署步骤
    create_directories()
    install_packages()
    download_models()
    download_additional_models()
    create_env_file()
    verify_installation()
    create_test_image()
    
    # 打印最终说明
    print_final_instructions()


if __name__ == "__main__":
    main()

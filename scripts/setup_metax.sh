#!/bin/bash
# =============================================================================
# THEMIS 完整部署脚本 - MetaX GPU 服务器 (修复版)
# 运行此脚本后即可进行图像质量评估
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置变量
MODEL_DIR="/mnt/afs/zhengmingkai/zyr/THEMIS/models"
PROJECT_DIR="/mnt/afs/zhengmingkai/zyr/THEMIS"
LOG_DIR="${PROJECT_DIR}/logs"
CONSTRAINTS_FILE="${PROJECT_DIR}/scripts/constraints.txt"

# 国内镜像配置
export HF_ENDPOINT="https://hf-mirror.com"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}THEMIS 完整部署脚本 - MetaX GPU 服务器 (修复版)${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "模型目录: ${GREEN}${MODEL_DIR}${NC}"
echo -e "项目目录: ${GREEN}${PROJECT_DIR}${NC}"
echo -e "HF镜像:   ${GREEN}${HF_ENDPOINT}${NC}"
echo -e "约束文件: ${GREEN}${CONSTRAINTS_FILE}${NC}"
echo ""

# 创建目录
echo -e "${YELLOW}[1/8] 创建目录...${NC}"
mkdir -p "${MODEL_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${PROJECT_DIR}/test_images"
mkdir -p "${PROJECT_DIR}/outputs"
mkdir -p "${PROJECT_DIR}/scripts"
echo -e "${GREEN}✓ 目录创建完成${NC}"

# 创建版本约束文件
echo ""
echo -e "${YELLOW}[2/8] 创建版本约束文件...${NC}"
cat > "${CONSTRAINTS_FILE}" << 'EOF'
# =============================================================================
# THEMIS 依赖版本约束文件
# 用于防止 numpy 和 opencv 版本冲突
# =============================================================================

# numpy 版本约束 (MetaX PyTorch 需要 numpy<2.0.0)
numpy==1.26.4

# opencv 版本约束 (兼容 numpy 1.x)
opencv-python==4.9.0.80
EOF
echo -e "${GREEN}✓ 版本约束文件创建完成: ${CONSTRAINTS_FILE}${NC}"

# 安装 Python 包
echo ""
echo -e "${YELLOW}[3/8] 安装 Python 依赖包...${NC}"

pip install --upgrade pip setuptools wheel -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}"

# 先安装 numpy (使用约束)
echo -e "${YELLOW}安装 numpy (兼容 MetaX PyTorch)...${NC}"
pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
    --constraint "${CONSTRAINTS_FILE}" \
    "numpy==1.26.4"

# 核心依赖 (使用约束)
echo -e "${YELLOW}安装核心依赖...${NC}"
pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
    --constraint "${CONSTRAINTS_FILE}" \
    scipy scikit-learn pandas \
    pillow opencv-python matplotlib seaborn \
    tqdm pyyaml python-dotenv requests aiohttp \
    huggingface_hub safetensors

echo -e "${GREEN}✓ 基础包安装完成${NC}"

# Transformers 相关 (使用约束)
echo -e "${YELLOW}安装 Transformers 相关包...${NC}"
pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
    --constraint "${CONSTRAINTS_FILE}" \
    transformers accelerate sentencepiece protobuf einops timm

# 安装 kornia 和 albumentations (使用约束，排除 opencv-python-headless)
echo -e "${YELLOW}安装 kornia 和 albumentations...${NC}"
pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
    --constraint "${CONSTRAINTS_FILE}" \
    kornia --no-deps

pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
    --constraint "${CONSTRAINTS_FILE}" \
    albumentations --no-deps

# 安装 albumentations 的必要依赖
pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
    --constraint "${CONSTRAINTS_FILE}" \
    albucore scipy scikit-image

# LangChain 相关 (使用约束)
echo -e "${YELLOW}安装 LangChain 相关包...${NC}"
pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
    --constraint "${CONSTRAINTS_FILE}" \
    langgraph langchain langchain-core langchain-anthropic

# 视觉模型相关 (使用约束)
echo -e "${YELLOW}安装视觉模型相关包...${NC}"
pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
    --constraint "${CONSTRAINTS_FILE}" \
    ultralytics pyiqa

# ===== 关键步骤：强制锁定版本 =====
echo ""
echo -e "${RED}[重要] 锁定 numpy 和 opencv 版本...${NC}"

# 卸载可能被安装的冲突包
pip uninstall opencv-python-headless -y 2>/dev/null || true

# 强制安装兼容版本 (使用约束)
pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
    --constraint "${CONSTRAINTS_FILE}" \
    "numpy==1.26.4" "opencv-python==4.9.0.80" --force-reinstall

echo -e "${GREEN}✓ Python 依赖安装完成${NC}"

# 验证核心包版本
echo ""
echo -e "${YELLOW}[4/8] 验证核心包版本...${NC}"
python -c "
import numpy, torch, cv2
print(f'numpy: {numpy.__version__}')
print(f'torch: {torch.__version__}')
print(f'cv2: {cv2.__version__}')
assert numpy.__version__.startswith('1.26'), f'numpy 版本错误: {numpy.__version__}'
assert cv2.__version__.startswith('4.9'), f'cv2 版本错误: {cv2.__version__}'
print('✓ 核心包版本正确')
"

# 下载 Qwen 模型
echo ""
echo -e "${YELLOW}[5/8] 下载 Qwen VL 模型...${NC}"

download_model() {
    local repo_id=$1
    local local_dir=$2
    local model_name=$3
    
    echo -e "${BLUE}下载 ${model_name}...${NC}"
    
    if [ -d "${local_dir}" ] && [ "$(ls -A ${local_dir} 2>/dev/null)" ]; then
        echo -e "${GREEN}✓ ${model_name} 已存在，跳过下载${NC}"
    else
        mkdir -p "${local_dir}"
        python -c "
from huggingface_hub import snapshot_download
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
snapshot_download(
    repo_id='${repo_id}',
    local_dir='${local_dir}',
    local_dir_use_symlinks=False,
    resume_download=True
)
print('下载完成: ${model_name}')
"
        echo -e "${GREEN}✓ ${model_name} 下载完成${NC}"
    fi
}

# Qwen2.5-VL-3B (Planner + Judge)
download_model "Qwen/Qwen2.5-VL-3B-Instruct" "${MODEL_DIR}/Qwen2.5-VL-3B-Instruct" "Qwen2.5-VL-3B-Instruct"

# Qwen2.5-VL-7B (Reflector)
download_model "Qwen/Qwen2.5-VL-7B-Instruct" "${MODEL_DIR}/Qwen2.5-VL-7B-Instruct" "Qwen2.5-VL-7B-Instruct"

# 下载 CLIP 模型
echo ""
echo -e "${YELLOW}[6/8] 下载 CLIP 模型...${NC}"
download_model "openai/clip-vit-base-patch32" "${MODEL_DIR}/clip-vit-base-patch32" "CLIP-ViT-B-32"

# 下载 EfficientNet 模型 (通过 timm 自动下载)
echo ""
echo -e "${YELLOW}[7/8] 预下载 EfficientNet 模型...${NC}"
python << 'EOF'
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import timm
import torch

print("预下载 EfficientNetV2-S...")
model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True)
print("✓ EfficientNetV2-S 预下载完成")

# 保存模型
torch.save(model.state_dict(), '/mnt/afs/zhengmingkai/zyr/THEMIS/models/efficientnetv2_s_in21k.pth')
print("✓ 模型权重已保存")
EOF

# 下载 YOLO 模型
python << 'EOF'
from ultralytics import YOLO
import os

os.makedirs('/mnt/afs/zhengmingkai/zyr/THEMIS/models/yolo', exist_ok=True)

print("下载 YOLO11n...")
model = YOLO('yolo11n.pt')
model.save('/mnt/afs/zhengmingkai/zyr/THEMIS/models/yolo/yolo11n.pt')
print("✓ YOLO11n 下载完成")

print("下载 YOLO11n-pose...")
model = YOLO('yolo11n-pose.pt')
model.save('/mnt/afs/zhengmingkai/zyr/THEMIS/models/yolo/yolo11n-pose.pt')
print("✓ YOLO11n-pose 下载完成")

print("下载 YOLO11m-pose...")
model = YOLO('yolo11m-pose.pt')
model.save('/mnt/afs/zhengmingkai/zyr/THEMIS/models/yolo/yolo11m-pose.pt')
print("✓ YOLO11m-pose 下载完成")
EOF

# 预下载 IQA 模型
python << 'EOF'
import pyiqa
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("预下载 MANIQA...")
metric = pyiqa.create_metric('maniqa', device='cpu')
print("✓ MANIQA 预下载完成")

print("预下载 MUSIQ...")
metric = pyiqa.create_metric('musiq', device='cpu')
print("✓ MUSIQ 预下载完成")

print("预下载 NIQE...")
metric = pyiqa.create_metric('niqe', device='cpu')
print("✓ NIQE 预下载完成")
EOF

echo -e "${GREEN}✓ 所有模型下载完成${NC}"

# 创建环境配置文件
echo ""
echo -e "${YELLOW}[8/8] 创建配置文件...${NC}"

cat > "${PROJECT_DIR}/.env" << EOF
# =============================================================================
# THEMIS 环境配置 - MetaX GPU 服务器
# =============================================================================

# 模型目录
MODEL_DIR=${MODEL_DIR}

# =============================================================================
# 核心模型配置
# =============================================================================

# 评估配置档位: fast, standard, accurate, deep
EVALUATION_PROFILE=standard

# Planner 配置 (Qwen2.5-VL-3B)
PLANNER_PROFILE=fast
PLANNER_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
PLANNER_LOCAL_MODEL_PATH=${MODEL_DIR}/Qwen2.5-VL-3B-Instruct
PLANNER_DEVICE=cuda:0
PLANNER_MAX_NEW_TOKENS=420

# Judge 配置 (Qwen2.5-VL-3B)
JUDGE_PROFILE=fast
JUDGE_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
JUDGE_LOCAL_MODEL_PATH=${MODEL_DIR}/Qwen2.5-VL-3B-Instruct
JUDGE_DEVICE=cuda:1
JUDGE_MAX_NEW_TOKENS=256

# Reflector 配置 (Qwen2.5-VL-7B)
REFLECTOR_PROFILE=standard
REFLECTOR_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
REFLECTOR_LOCAL_MODEL_PATH=${MODEL_DIR}/Qwen2.5-VL-7B-Instruct
REFLECTOR_DEVICE=cuda:2
REFLECTOR_MAX_NEW_TOKENS=400

# =============================================================================
# 专家模型路径
# =============================================================================

IMAGENET_MODEL_PATH=${MODEL_DIR}/efficientnetv2_s_in21k.pth
CLIP_MODEL_PATH=${MODEL_DIR}/clip-vit-base-patch32
YOLO_MODEL_PATH=${MODEL_DIR}/yolo

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
LOG_DIR=${LOG_DIR}

# =============================================================================
# HuggingFace 镜像 (国内加速)
# =============================================================================

HF_ENDPOINT=https://hf-mirror.com
EOF

echo -e "${GREEN}✓ 配置文件创建完成${NC}"

# 验证安装
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}验证安装...${NC}"
echo -e "${BLUE}============================================================${NC}"

python << 'EOF'
import sys

print("\n[包版本检查]")
packages = [
    "torch", "transformers", "timm", "ultralytics", "pyiqa",
    "langchain", "langgraph", "numpy", "pillow", "cv2"
]

for pkg in packages:
    try:
        if pkg == "cv2":
            import cv2
            version = cv2.__version__
        else:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {pkg}: {version}")
    except ImportError:
        print(f"  ✗ {pkg}: 未安装")

print("\n[GPU 检查]")
import torch
print(f"  CUDA 可用: {torch.cuda.is_available()}")
print(f"  GPU 数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"  GPU 名称: {torch.cuda.get_device_name(0)}")

print("\n[模型检查]")
import os
model_dir = "/mnt/afs/zhengmingkai/zyr/THEMIS/models"
models = [
    "Qwen2.5-VL-3B-Instruct",
    "Qwen2.5-VL-7B-Instruct",
    "clip-vit-base-patch32",
    "yolo"
]

for model in models:
    path = os.path.join(model_dir, model)
    if os.path.exists(path):
        print(f"  ✓ {model}")
    else:
        print(f"  ✗ {model}: 不存在")

print("\n✓ 验证完成")
EOF

# 完成
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}部署完成！${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "模型目录: ${MODEL_DIR}"
echo -e "配置文件: ${PROJECT_DIR}/.env"
echo -e "约束文件: ${CONSTRAINTS_FILE}"
echo ""
echo -e "运行评估:"
echo -e "  ${YELLOW}cd ${PROJECT_DIR}${NC}"
echo -e "  ${YELLOW}python -m src.agentic_eval.run_single ./test_images/test_image.png --class-label 'test' --output ./outputs/result.json${NC}"
echo ""
echo -e "${BLUE}提示: 以后安装新包时请使用:${NC}"
echo -e "  ${YELLOW}pip install <package> --constraint ${CONSTRAINTS_FILE}${NC}"
echo ""

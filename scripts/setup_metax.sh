#!/bin/bash
# =============================================================================
# THEMIS 完整部署脚本 - MetaX GPU 服务器
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

# 国内镜像配置
export HF_ENDPOINT="https://hf-mirror.com"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}THEMIS 完整部署脚本 - MetaX GPU 服务器${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "模型目录: ${GREEN}${MODEL_DIR}${NC}"
echo -e "项目目录: ${GREEN}${PROJECT_DIR}${NC}"
echo -e "HF镜像:   ${GREEN}${HF_ENDPOINT}${NC}"
echo ""

# 创建目录
echo -e "${YELLOW}[1/7] 创建目录...${NC}"
mkdir -p "${MODEL_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${PROJECT_DIR}/test_images"
mkdir -p "${PROJECT_DIR}/outputs"
echo -e "${GREEN}✓ 目录创建完成${NC}"

# 安装 Python 包
echo ""
echo -e "${YELLOW}[2/7] 安装 Python 依赖包...${NC}"

pip install --upgrade pip setuptools wheel -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}"

# 核心依赖
pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
    numpy scipy scikit-learn pandas \
    pillow opencv-python matplotlib seaborn \
    tqdm pyyaml python-dotenv requests aiohttp \
    huggingface_hub safetensors

# PyTorch 相关 (MetaX 已有定制版 PyTorch，跳过)
echo -e "${GREEN}✓ 基础包安装完成${NC}"

# Transformers 相关
echo -e "${YELLOW}安装 Transformers 相关包...${NC}"
pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
    transformers accelerate sentencepiece protobuf einops \
    timm kornia albumentations

# LangChain 相关
echo -e "${YELLOW}安装 LangChain 相关包...${NC}"
pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
    langgraph langchain langchain-core langchain-anthropic

# 视觉模型相关
echo -e "${YELLOW}安装视觉模型相关包...${NC}"
pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
    ultralytics pyiqa

# 可选：PaddleOCR (如果需要)
# pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
#     paddlepaddle paddleocr

echo -e "${GREEN}✓ Python 依赖安装完成${NC}"

# 下载 Qwen 模型
echo ""
echo -e "${YELLOW}[3/7] 下载 Qwen VL 模型...${NC}"

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
echo -e "${YELLOW}[4/7] 下载 CLIP 模型...${NC}"
download_model "openai/clip-vit-base-patch32" "${MODEL_DIR}/clip-vit-base-patch32" "CLIP-ViT-B-32"

# 下载 EfficientNet 模型 (通过 timm 自动下载，这里预下载)
echo ""
echo -e "${YELLOW}[5/7] 预下载 EfficientNet 模型...${NC}"
python << 'EOF'
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import timm

print("预下载 EfficientNetV2-S...")
model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True)
print("✓ EfficientNetV2-S 预下载完成")

# 保存模型
import torch
torch.save(model.state_dict(), '/mnt/afs/zhengmingkai/zyr/THEMIS/models/efficientnetv2_s_in21k.pth')
print("✓ 模型权重已保存")
EOF

# 下载 YOLO 模型
echo ""
echo -e "${YELLOW}[6/7] 下载 YOLO 模型...${NC}"
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
echo ""
echo -e "${YELLOW}预下载 IQA 模型...${NC}"
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
echo -e "${YELLOW}[7/7] 创建配置文件...${NC}"

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

# 创建测试脚本
cat > "${PROJECT_DIR}/run_test.sh" << 'EOF'
#!/bin/bash
# THEMIS 测试运行脚本

source .env 2>/dev/null || true

echo "=========================================="
echo "THEMIS 测试运行"
echo "=========================================="

# 测试1: 模型加载测试
echo ""
echo "[测试1] Qwen 模型加载测试..."
python << 'PYTHON'
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import os

model_path = os.environ.get('PLANNER_LOCAL_MODEL_PATH', '/mnt/afs/zhengmingkai/zyr/THEMIS/models/Qwen2.5-VL-3B-Instruct')

print(f"加载模型: {model_path}")
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)

print(f"✓ 模型加载成功")
print(f"  设备: {model.device}")
print(f"  显存: {torch.cuda.memory_allocated()/1e9:.2f} GB")
PYTHON

# 测试2: 快速评估测试
echo ""
echo "[测试2] 创建测试图片..."
python << 'PYTHON'
from PIL import Image
import os

os.makedirs('/mnt/afs/zhengmingkai/zyr/THEMIS/test_images', exist_ok=True)

# 创建测试图片
img = Image.new('RGB', (256, 256), color=(100, 150, 200))
img.save('/mnt/afs/zhengmingkai/zyr/THEMIS/test_images/test_image.png')
print("✓ 测试图片创建完成")
PYTHON

echo ""
echo "=========================================="
echo "部署完成！"
echo "=========================================="
echo ""
echo "运行评估命令示例:"
echo "  cd ${PROJECT_DIR}"
echo "  python -m src.agentic_eval.run_single \\"
echo "      ./test_images/test_image.png \\"
echo "      --class-label 'test object' \\"
echo "      --output ./outputs/result.json"
echo ""
EOF

chmod +x "${PROJECT_DIR}/run_test.sh"

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
    "langchain", "langgraph", "numpy", "pillow"
]

for pkg in packages:
    try:
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
echo ""
echo -e "运行测试:"
echo -e "  ${YELLOW}bash ${PROJECT_DIR}/run_test.sh${NC}"
echo ""
echo -e "运行评估:"
echo -e "  ${YELLOW}cd ${PROJECT_DIR}${NC}"
echo -e "  ${YELLOW}python -m src.agentic_eval.run_single ./test_images/test_image.png --class-label 'test' --output ./outputs/result.json${NC}"
echo ""

#!/bin/bash
# =============================================================================
# THEMIS 完整部署脚本 - MetaX GPU 服务器
# 适用于已有 MetaX PyTorch 环境的情况
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置变量 (根据实际情况修改)
MODEL_DIR="/mnt/afs/zhengmingkai/zyr/THEMIS/models"
PROJECT_DIR="/mnt/afs/zhengmingkai/zyr/THEMIS"
LOG_DIR="${PROJECT_DIR}/logs"
CONSTRAINTS_FILE="${PROJECT_DIR}/scripts/constraints.txt"

# 国内镜像配置
export HF_ENDPOINT="https://hf-mirror.com"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}THEMIS 完整部署脚本 - MetaX GPU 服务器${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "${YELLOW}⚠ 重要提示:${NC}"
echo -e "${YELLOW}   本脚本会严格保护 MetaX 包版本，不会修改:${NC}"
echo -e "${YELLOW}   - torch (MetaX 版本)${NC}"
echo -e "${YELLOW}   - torchvision (MetaX 版本)${NC}"
echo -e "${YELLOW}   - numpy${NC}"
echo -e "${YELLOW}   - opencv${NC}"
echo -e "${YELLOW}   - triton (MetaX 版本)${NC}"
echo ""
echo -e "${YELLOW}   如果检测到版本被修改，脚本会立即退出。${NC}"
echo -e "${YELLOW}   此时需要删除环境并重新从 base 克隆。${NC}"
echo ""
echo -e "模型目录: ${GREEN}${MODEL_DIR}${NC}"
echo -e "项目目录: ${GREEN}${PROJECT_DIR}${NC}"
echo -e "HF镜像:   ${GREEN}${HF_ENDPOINT}${NC}"
echo ""

# =============================================================================
# Step 1: 检查 MetaX 环境并记录版本
# =============================================================================
echo -e "${YELLOW}[1/7] 检查 MetaX 环境...${NC}"

# 记录当前关键包版本
CURRENT_TORCH_VERSION=""
CURRENT_TORCHVISION_VERSION=""
CURRENT_NUMPY_VERSION=""
CURRENT_CV2_VERSION=""
CURRENT_TRITON_VERSION=""

python << 'PYEOF'
import sys

# 检查 PyTorch
try:
    import torch
    print(f"✓ torch: {torch.__version__}")
    if 'metax' not in torch.__version__:
        print("  ⚠ 警告: 不是 MetaX 版本的 PyTorch")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU 数量: {torch.cuda.device_count()}")
    # 写入版本到临时文件
    with open('/tmp/themis_torch_version.txt', 'w') as f:
        f.write(torch.__version__)
except ImportError:
    print("✗ torch 未安装!")
    sys.exit(1)

# 检查 torchvision
try:
    import torchvision
    print(f"✓ torchvision: {torchvision.__version__}")
    with open('/tmp/themis_torchvision_version.txt', 'w') as f:
        f.write(torchvision.__version__)
except ImportError:
    print("  torchvision 未安装")
    with open('/tmp/themis_torchvision_version.txt', 'w') as f:
        f.write("")

# 检查 numpy
try:
    import numpy
    print(f"✓ numpy: {numpy.__version__}")
    if numpy.__version__.startswith('2.'):
        print("  ⚠ 警告: numpy 2.x 可能与 MetaX PyTorch 不兼容")
    with open('/tmp/themis_numpy_version.txt', 'w') as f:
        f.write(numpy.__version__)
except ImportError:
    print("✗ numpy 未安装!")
    sys.exit(1)

# 检查 opencv
try:
    import cv2
    print(f"✓ cv2: {cv2.__version__}")
    with open('/tmp/themis_cv2_version.txt', 'w') as f:
        f.write(cv2.__version__)
except ImportError:
    print("✗ opencv 未安装!")
    with open('/tmp/themis_cv2_version.txt', 'w') as f:
        f.write("")

# 检查 triton
try:
    import triton
    print(f"✓ triton: {triton.__version__}")
    with open('/tmp/themis_triton_version.txt', 'w') as f:
        f.write(triton.__version__)
except ImportError:
    print("  triton 未安装")
    with open('/tmp/themis_triton_version.txt', 'w') as f:
        f.write("")

print("\n环境检查通过!")
PYEOF

if [ $? -ne 0 ]; then
    echo -e "${RED}环境检查失败，请先配置 MetaX PyTorch 环境${NC}"
    exit 1
fi

# 读取版本
CURRENT_TORCH_VERSION=$(cat /tmp/themis_torch_version.txt)
CURRENT_TORCHVISION_VERSION=$(cat /tmp/themis_torchvision_version.txt)
CURRENT_NUMPY_VERSION=$(cat /tmp/themis_numpy_version.txt)
CURRENT_CV2_VERSION=$(cat /tmp/themis_cv2_version.txt)
CURRENT_TRITON_VERSION=$(cat /tmp/themis_triton_version.txt)

echo -e "${GREEN}当前版本已锁定:${NC}"
echo -e "  torch: ${CURRENT_TORCH_VERSION}"
echo -e "  torchvision: ${CURRENT_TORCHVISION_VERSION}"
echo -e "  numpy: ${CURRENT_NUMPY_VERSION}"
echo -e "  opencv-python: ${CURRENT_CV2_VERSION}"
echo -e "  triton: ${CURRENT_TRITON_VERSION}"

# =============================================================================
# Step 2: 创建目录
# =============================================================================
echo ""
echo -e "${YELLOW}[2/7] 创建目录...${NC}"
mkdir -p "${MODEL_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${PROJECT_DIR}/test_images"
mkdir -p "${PROJECT_DIR}/outputs"
mkdir -p "${PROJECT_DIR}/scripts"
echo -e "${GREEN}✓ 目录创建完成${NC}"

# =============================================================================
# Step 3: 创建版本约束文件 (动态锁定当前版本)
# =============================================================================
echo ""
echo -e "${YELLOW}[3/7] 创建版本约束文件...${NC}"

# 动态生成 constraints.txt，锁定当前已安装的关键包版本
cat > "${CONSTRAINTS_FILE}" << EOF
# =============================================================================
# THEMIS 依赖版本约束文件
# 自动生成 - 锁定当前环境的关键包版本
# 注意: opencv 来自系统镜像，不在 PyPI 上，所以不锁定其版本
# =============================================================================

# 核心包版本锁定 (禁止修改)
torch==${CURRENT_TORCH_VERSION}
numpy==${CURRENT_NUMPY_VERSION}
EOF

# 如果 torchvision 已安装，也锁定
if [ -n "${CURRENT_TORCHVISION_VERSION}" ]; then
    echo "torchvision==${CURRENT_TORCHVISION_VERSION}" >> "${CONSTRAINTS_FILE}"
fi

# 如果 triton 已安装，锁定版本
if [ -n "${CURRENT_TRITON_VERSION}" ]; then
    echo "triton==${CURRENT_TRITON_VERSION}" >> "${CONSTRAINTS_FILE}"
fi

# 禁止安装 opencv-python-headless (它会要求 numpy>=2)
# 注意: 不锁定 opencv-python 版本，因为它来自系统镜像，PyPI 上可能没有对应版本
cat >> "${CONSTRAINTS_FILE}" << EOF

# 禁止安装 opencv-python-headless (它会要求 numpy>=2)
opencv-python-headless<0
EOF

echo -e "${GREEN}✓ 版本约束文件创建完成${NC}"
echo -e "${BLUE}约束内容:${NC}"
cat "${CONSTRAINTS_FILE}"

# =============================================================================
# Step 4: 安装 Python 依赖 (不修改已有包版本)
# =============================================================================
echo ""
echo -e "${YELLOW}[4/7] 安装 Python 依赖...${NC}"

pip install --upgrade pip setuptools wheel -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}"

# 检查包是否已安装的函数
check_and_install() {
    local packages="$1"
    local description="$2"
    local need_install=""
    
    echo -e "${BLUE}检查 ${description}...${NC}"
    
    for pkg in $packages; do
        # 获取包名 (去掉版本 specifier)
        pkg_name=$(echo "$pkg" | sed 's/[<>=!].*//' | sed 's/\[.*//')
        
        # 检查是否已安装
        if pip show "$pkg_name" > /dev/null 2>&1; then
            version=$(pip show "$pkg_name" | grep "^Version:" | awk '{print $2}')
            echo -e "  ${GREEN}✓ ${pkg_name} 已安装 (${version})${NC}"
        else
            need_install="$need_install $pkg"
        fi
    done
    
    # 只安装未安装的包
    if [ -n "$need_install" ]; then
        echo -e "  ${YELLOW}安装缺失的包:${need_install}${NC}"
        pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
            --constraint "${CONSTRAINTS_FILE}" \
            $need_install
    else
        echo -e "  ${GREEN}所有包已安装，跳过${NC}"
    fi
}

# 核心依赖
check_and_install "scipy scikit-learn pandas matplotlib seaborn tqdm pyyaml python-dotenv requests aiohttp huggingface_hub safetensors" "核心依赖"

# Transformers 相关
check_and_install "transformers accelerate sentencepiece protobuf einops timm" "Transformers 相关包"

# LangChain 相关
check_and_install "langgraph langchain langchain-core" "LangChain 相关包"

# 视觉模型相关 (全部使用 --no-deps 保护 opencv)
echo -e "${BLUE}检查 视觉模型相关包...${NC}"

# ultralytics (使用 --no-deps 保护 opencv)
if pip show ultralytics > /dev/null 2>&1; then
    version=$(pip show ultralytics | grep "^Version:" | awk '{print $2}')
    echo -e "  ${GREEN}✓ ultralytics 已安装 (${version})${NC}"
else
    echo -e "  ${YELLOW}安装 ultralytics (不安装依赖，保护 opencv)...${NC}"
    pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
        ultralytics --no-deps
fi

# kornia (使用 --no-deps 保护 opencv)
if pip show kornia > /dev/null 2>&1; then
    version=$(pip show kornia | grep "^Version:" | awk '{print $2}')
    echo -e "  ${GREEN}✓ kornia 已安装 (${version})${NC}"
else
    echo -e "  ${YELLOW}安装 kornia (不安装依赖)...${NC}"
    pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
        kornia --no-deps
fi

# pyiqa (使用 --no-deps 保护 opencv)
if pip show pyiqa > /dev/null 2>&1; then
    version=$(pip show pyiqa | grep "^Version:" | awk '{print $2}')
    echo -e "  ${GREEN}✓ pyiqa 已安装 (${version})${NC}"
else
    echo -e "  ${YELLOW}安装 pyiqa (不安装依赖，系统已有 opencv)...${NC}"
    pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
        pyiqa --no-deps
fi

# albumentations (使用 --no-deps 保护 opencv)
echo -e "${BLUE}检查 albumentations...${NC}"
if pip show albumentations > /dev/null 2>&1; then
    version=$(pip show albumentations | grep "^Version:" | awk '{print $2}')
    echo -e "  ${GREEN}✓ albumentations 已安装 (${version})${NC}"
else
    echo -e "  ${YELLOW}安装 albumentations (不安装依赖)...${NC}"
    pip install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
        albumentations --no-deps
fi

echo -e "${GREEN}✓ Python 依赖安装完成${NC}"

# =============================================================================
# Step 5: 下载模型
# =============================================================================
echo ""
echo -e "${YELLOW}[5/7] 下载模型权重...${NC}"

download_model() {
    local repo_id=$1
    local local_dir=$2
    local model_name=$3
    
    echo -e "${BLUE}____ ${model_name}...${NC}"
    
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

# CLIP 模型
download_model "openai/clip-vit-base-patch32" "${MODEL_DIR}/clip-vit-base-patch32" "CLIP-ViT-B-32"

echo -e "${GREEN}✓ Qwen 和 CLIP 模型下载完成${NC}"

# =============================================================================
# Step 6: 预下载其他模型权重
# =============================================================================
echo ""
echo -e "${YELLOW}[6/7] 预下载其他模型权重...${NC}"

python << 'PYEOF'
import os
import sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import warnings
warnings.filterwarnings('ignore')

# EfficientNet
print("____ EfficientNetV2-S...")
try:
    import timm
    import torch
    model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True)
    torch.save(model.state_dict(), '/mnt/afs/zhengmingkai/zyr/THEMIS/models/efficientnetv2_s_in21k.pth')
    print("_ EfficientNetV2-S ________")
except Exception as e:
    print(f"EfficientNet 下载失败: {e}")

# YOLO 模型
print("____ YOLO...")
try:
    from ultralytics import YOLO
    import os
    os.makedirs('/mnt/afs/zhengmingkai/zyr/THEMIS/models/yolo', exist_ok=True)
    
    for model_name in ['yolo11n.pt', 'yolo11n-pose.pt', 'yolo11m-pose.pt']:
        print(f"  下载 {model_name}...")
        model = YOLO(model_name)
        model.save(f'/mnt/afs/zhengmingkai/zyr/THEMIS/models/yolo/{model_name}')
    print("_ YOLO ________")
except Exception as e:
    print(f"YOLO 下载失败: {e}")

# IQA 模型
print("____ IQA...")
try:
    import pyiqa
    for metric in ['maniqa', 'musiq', 'niqe']:
        print(f"  下载 {metric}...")
        pyiqa.create_metric(metric, device='cpu')
    print("_ IQA ________")
except Exception as e:
    print(f"IQA 下载失败: {e}")

print("\n✓ 所有模型权重下载完成")
PYEOF

echo -e "${GREEN}✓ 模型权重下载完成${NC}"

# =============================================================================
# Step 7: 创建配置文件
# =============================================================================
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
# 本地专家模型配置
# =============================================================================

LOCAL_PRIMARY_MODEL=${MODEL_DIR}/Qwen2.5-VL-3B-Instruct
LOCAL_SEMANTIC_ENABLED=true
LOCAL_SEMANTIC_MODEL=${MODEL_DIR}/Qwen2.5-VL-3B-Instruct
SEMANTIC_LOCAL_FAST_MODEL=${MODEL_DIR}/Qwen2.5-VL-3B-Instruct
SEMANTIC_LOCAL_STRONGER_MODEL=${MODEL_DIR}/Qwen2.5-VL-7B-Instruct
LOCAL_SEMANTIC_DEVICE=cuda:3
LOCAL_ARTIFACT_ENABLED=true
LOCAL_ARTIFACT_METRICS=maniqa,musiq,niqe
LOCAL_ARTIFACT_DEVICE=cuda:4
REPORT_MODEL=${MODEL_DIR}/Qwen2.5-VL-7B-Instruct

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

# =============================================================================
# 验证安装
# =============================================================================
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}验证安装...${NC}"
echo -e "${BLUE}============================================================${NC}"

# 读取之前保存的版本
ORIGINAL_TORCH=$(cat /tmp/themis_torch_version.txt)
ORIGINAL_NUMPY=$(cat /tmp/themis_numpy_version.txt)
ORIGINAL_TORCHVISION=$(cat /tmp/themis_torchvision_version.txt)
ORIGINAL_CV2=$(cat /tmp/themis_cv2_version.txt)
ORIGINAL_TRITON=$(cat /tmp/themis_triton_version.txt)

python << PYEOF
import sys

# 版本验证
print("\n[关键包版本验证]")
version_ok = True

# 检查 torch
import torch
original_torch = "${ORIGINAL_TORCH}"
if torch.__version__ != original_torch:
    print(f"  ✗ torch 版本已改变: {original_torch} -> {torch.__version__}")
    version_ok = False
else:
    print(f"  ✓ torch: {torch.__version__} (未改变)")

# 检查 numpy
import numpy
original_numpy = "${ORIGINAL_NUMPY}"
if numpy.__version__ != original_numpy:
    print(f"  ✗ numpy 版本已改变: {original_numpy} -> {numpy.__version__}")
    version_ok = False
else:
    print(f"  ✓ numpy: {numpy.__version__} (未改变)")

# 检查 torchvision
original_torchvision = "${ORIGINAL_TORCHVISION}"
try:
    import torchvision
    if original_torchvision and torchvision.__version__ != original_torchvision:
        print(f"  ✗ torchvision 版本已改变: {original_torchvision} -> {torchvision.__version__}")
        version_ok = False
    else:
        print(f"  ✓ torchvision: {torchvision.__version__} (未改变)")
except ImportError:
    print("  - torchvision: 未安装")

# 检查 opencv
original_cv2 = "${ORIGINAL_CV2}"
try:
    import cv2
    if original_cv2 and cv2.__version__ != original_cv2:
        print(f"  ✗ cv2 版本已改变: {original_cv2} -> {cv2.__version__}")
        version_ok = False
    else:
        print(f"  ✓ cv2: {cv2.__version__} (未改变)")
except ImportError:
    print("  - cv2: 未安装")

# 检查 triton
original_triton = "${ORIGINAL_TRITON}"
try:
    import triton
    if original_triton and triton.__version__ != original_triton:
        print(f"  ✗ triton 版本已改变: {original_triton} -> {triton.__version__}")
        version_ok = False
    else:
        print(f"  ✓ triton: {triton.__version__} (未改变)")
except ImportError:
    print("  - triton: 未安装")

if not version_ok:
    print("\n⚠ 警告: 关键包版本已被修改!")
    sys.exit(1)

print("\n[其他包版本]")
packages = [
    ("transformers", "transformers"),
    ("timm", "timm"),
    ("ultralytics", "ultralytics"),
    ("pyiqa", "pyiqa"),
    ("langchain", "langchain"),
    ("langgraph", "langgraph"),
]

for import_name, display_name in packages:
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {display_name}: {version}")
    except ImportError:
        print(f"  ✗ {display_name}: 未安装")

print("\n[GPU 检查]")
print(f"  CUDA 可用: {torch.cuda.is_available()}")
print(f"  GPU 数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

print("\n[模型检查]")
import os
model_dir = "/mnt/afs/zhengmingkai/zyr/THEMIS/models"
models = [
    "Qwen2.5-VL-3B-Instruct",
    "Qwen2.5-VL-7B-Instruct",
    "clip-vit-base-patch32",
    "efficientnetv2_s_in21k.pth",
    "yolo"
]

for model in models:
    path = os.path.join(model_dir, model)
    if os.path.exists(path):
        print(f"  ✓ {model}")
    else:
        print(f"  ✗ {model}: 不存在")

print("\n✓ 验证完成")
PYEOF

if [ $? -ne 0 ]; then
    echo -e "${RED}============================================================${NC}"
    echo -e "${RED}严重错误: 关键包版本已被修改!${NC}"
    echo -e "${RED}============================================================${NC}"
    echo ""
    echo -e "${YELLOW}MetaX 的 PyTorch 来自系统镜像，无法通过 pip 恢复。${NC}"
    echo -e "${YELLOW}请删除当前环境并重新从 base 环境克隆:${NC}"
    echo ""
    echo -e "  ${BLUE}conda deactivate${NC}"
    echo -e "  ${BLUE}conda env remove -p /mnt/afs/zhengmingkai/zyr/themis_evn${NC}"
    echo -e "  ${BLUE}conda create -p /mnt/afs/zhengmingkai/zyr/themis_evn --clone base${NC}"
    echo -e "  ${BLUE}conda activate /mnt/afs/zhengmingkai/zyr/themis_evn${NC}"
    echo ""
    exit 1
fi

# =============================================================================
# 完成
# =============================================================================
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
echo -e "  ${YELLOW}python -m src.agentic_eval.run_single ./test_images/test.png --class-label 'test' --output ./outputs/result.json${NC}"
echo ""
echo -e "${BLUE}提示: 以后安装新包时请使用:${NC}"
echo -e "  ${YELLOW}pip install <package> --constraint ${CONSTRAINTS_FILE}${NC}"
echo ""

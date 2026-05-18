# THEMIS 部署指南

本文档提供在 Ubuntu 20.04 服务器上部署 THEMIS 评估系统的完整指南。

## 目录

1. [系统要求](#系统要求)
2. [环境配置](#环境配置)
3. [模型下载](#模型下载)
4. [配置说明](#配置说明)
5. [测试验证](#测试验证)
6. [运行评估](#运行评估)
7. [常见问题](#常见问题)

---

## 系统要求

### 硬件要求

| 配置项 | 最低要求 | 推荐配置 |
|-------|---------|---------|
| GPU | 1 × 64GB | 8 × 64GB (8卡) |
| 显存 | 64GB | 512GB |
| CPU | 8 核 | 32 核 |
| 内存 | 64GB | 256GB |
| 存储 | 500GB SSD | 2TB SSD |

### 支持的 GPU

- NVIDIA A100 (40GB / 80GB)
- NVIDIA H100 (80GB)
- MetaX x1ls-64G (64GB)

### 软件要求

- Ubuntu 20.04 LTS
- CUDA 11.8+ / 12.1+
- Python 3.10+
- cuDNN 8.0+

---

## 环境配置

### 1. 克隆项目

```bash
git clone https://github.com/your-repo/THEMIS.git
cd THEMIS
```

### 2. 运行环境设置脚本

```bash
# 给脚本添加执行权限
chmod +x scripts/setup_env.sh

# 运行环境设置（需要约30分钟）
bash scripts/setup_env.sh
```

### 3. 激活虚拟环境

```bash
source .venv/bin/activate
```

### 4. 手动安装（如果脚本失败）

```bash
# 创建虚拟环境
python3.10 -m venv .venv
source .venv/bin/activate

# 升级 pip
pip install --upgrade pip setuptools wheel

# 安装 PyTorch (根据你的 CUDA 版本选择)
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements-agentic-eval.txt
```

---

## 模型下载

### 快速下载（推荐配置）

```bash
# 下载核心模型（Qwen VL 3B + 7B）
python scripts/download_models.py --model-dir ./models --qwen 3b 7b

# 使用镜像加速（适合中国大陆）
python scripts/download_models.py --model-dir ./models --qwen 3b 7b --use-mirror
```

### 完整下载（所有模型）

```bash
# 下载所有模型（约 100GB）
python scripts/download_models.py --model-dir ./models --all
```

### 分组下载

```bash
# 只下载 Qwen VL 模型
python scripts/download_models.py --model-dir ./models --qwen 3b 7b 32b

# 下载分类模型
python scripts/download_models.py --model-dir ./models --imagenet

# 下载检测模型
python scripts/download_models.py --model-dir ./models --yolo

# 下载 IQA 模型
python scripts/download_models.py --model-dir ./models --iqa
```

### 手动下载大模型

```bash
# Qwen2.5-VL-3B-Instruct
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./models/Qwen2.5-VL-3B-Instruct

# Qwen2.5-VL-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./models/Qwen2.5-VL-7B-Instruct

# Qwen2.5-VL-32B-Instruct (需要较大显存)
huggingface-cli download Qwen/Qwen2.5-VL-32B-Instruct --local-dir ./models/Qwen2.5-VL-32B-Instruct
```

---

## 配置说明

### 1. 环境变量配置

编辑 `.env` 文件：

```bash
# 复制示例配置
cp .env.example .env

# 编辑配置
nano .env
```

关键配置项：

```bash
# 模型目录
MODEL_DIR=/path/to/THEMIS/models

# 核心模型配置
PLANNER_PROFILE=fast    # fast, standard, strong
JUDGE_PROFILE=fast
REFLECTOR_PROFILE=standard

# 设备配置
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### 2. YAML 配置文件

使用 `configs/expert_config.yaml` 配置专家模型：

```bash
# 查看默认配置
cat configs/expert_config.yaml
```

关键配置：

```yaml
# 评估配置文件
evaluation_profiles:
  fast:
    planner_profile: "fast"
    judge_profile: "fast"
    reflector_profile: "fast"
    max_experts: 3
    estimated_time: "3-5s"
    
  standard:
    planner_profile: "fast"
    judge_profile: "standard"
    reflector_profile: "standard"
    max_experts: 5
    estimated_time: "5-10s"
    
  accurate:
    planner_profile: "standard"
    judge_profile: "standard"
    reflector_profile: "strong"
    max_experts: 8
    estimated_time: "10-20s"
```

---

## 测试验证

### 1. 运行模型测试

```bash
# 测试所有模型
python scripts/test_models.py --model-dir ./models

# 测试特定模型类型
python scripts/test_models.py --model-dir ./models --verbose

# 输出测试报告
python scripts/test_models.py --model-dir ./models --output ./test_report.json
```

### 2. 验证 CUDA

```bash
# 检查 CUDA 可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 检查显存
python -c "import torch; print(f'Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')"
```

### 3. 测试单图评估

```bash
# 使用快速配置测试
python -m src.agentic_eval.run_single ./test_images/beacon.png \
    --class-label "Beacon" \
    --output test_output.json \
    --planner-model ./models/Qwen2.5-VL-3B-Instruct

# 查看输出
cat test_output.json
```

---

## 运行评估

### 1. 单图评估

```bash
# 基础用法
python -m src.agentic_eval.run_single \
    ./test_images/beacon.png \
    --class-label "Beacon" \
    --output results/beacon_result.json

# 使用本地模型
python -m src.agentic_eval.run_single \
    ./test_images/dog.png \
    --class-label "Golden Retriever" \
    --output results/dog_result.json \
    --planner-model ./models/Qwen2.5-VL-3B-Instruct

# 使用标准配置
python -m src.agentic_eval.run_single \
    ./test_images/cat.png \
    --class-label "Persian cat" \
    --output results/cat_result.json \
    --config-path configs/expert_config.yaml
```

### 2. 批量评估

```bash
# 批量评估目录中的所有图片
for img in ./test_images/*.png; do
    class_label=$(basename "$img" .png)
    python -m src.agentic_eval.run_single \
        "$img" \
        --class-label "$class_label" \
        --output "results/${class_label}.json"
done
```

### 3. 使用不同的评估配置

```bash
# 快速模式（3-5秒/图）
EVALUATION_PROFILE=fast python -m src.agentic_eval.run_single ...

# 标准模式（5-10秒/图）
EVALUATION_PROFILE=standard python -m src.agentic_eval.run_single ...

# 精确模式（10-20秒/图）
EVALUATION_PROFILE=accurate python -m src.agentic_eval.run_single ...
```

---

## 常见问题

### 1. 显存不足

**问题**: CUDA out of memory error

**解决方案**:
- 使用更小的模型（如 3B 而非 7B）
- 使用量化: `--quantization 4bit`
- 减少并发模型数量
- 调整 `CUDA_VISIBLE_DEVICES` 环境变量

```bash
# 只使用 2 张卡
CUDA_VISIBLE_DEVICES=0,1 python -m src.agentic_eval.run_single ...
```

### 2. 模型下载失败

**问题**: HuggingFace 下载超时

**解决方案**:
- 使用镜像: `--use-mirror`
- 设置代理
- 手动下载后放到 models 目录

```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_models.py --model-dir ./models --use-mirror
```

### 3. Python 版本问题

**问题**: Module not found 错误

**解决方案**:
```bash
# 确保使用正确的 Python 版本
python3.10 --version
which python3.10

# 在虚拟环境中安装
source .venv/bin/activate
pip install -r requirements-agentic-eval.txt
```

### 4. CUDA 版本不匹配

**问题**: CUDA error: no kernel image is available

**解决方案**:
```bash
# 检查 CUDA 版本
nvcc --version
nvidia-smi

# 重新安装匹配版本的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. 多GPU配置

对于 8 卡服务器，推荐配置：

```yaml
# configs/expert_config.yaml
device_allocation:
  cuda:0:
    - "planner"
  cuda:1:
    - "judge"
  cuda:2:
    - "reflector"
  cuda:3:
    - "semantic_group"
    - "artifact_group"
  cuda:4:
    - "animal_group"
    - "human_group"
  cuda:5:
    - "background_group"
  cuda:6:
    - "vqa"
  cuda:7:
    - "reserved"
```

---

## 性能优化

### 1. 推理优化

```bash
# 启用 Flash Attention
pip install flash-attn

# 使用 vLLM (可选)
pip install vllm
```

### 2. 批处理优化

```bash
# 调整批量大小
export BATCH_SIZE=8

# 使用混合精度
export TORCH_CUDNN_V8_API_ENABLED=1
```

### 3. 模型量化

在配置中使用量化：

```yaml
planner:
  torch_dtype: "float16"  # 或 "bfloat16"
  quantization: "4bit"     # 或 "8bit"
```

---

## 估算时间

| 配置 | Planner | 专家数量 | 预估时间 | 显存 |
|-----|---------|---------|---------|------|
| 极速档 | 3B | 3个 | 2-4秒 | ~30GB |
| 快速档 | 3B | 5个 | 3-6秒 | ~40GB |
| 标准档 | 3B+7B | 5个 | 5-10秒 | ~50GB |
| 完整档 | 3B+7B | 8个 | 8-15秒 | ~70GB |
| 深度档 | 7B+32B | 10个 | 15-30秒 | ~120GB |

---

## 联系方式

如有问题，请提交 Issue 或联系维护团队。

# THEMIS T2I 评估系统 — 启动指南

## 概述

THEMIS T2I 是一个基于 GenEval2 原子化 QA 和 Taxonomy 先验知识的 Agentic 文生图评估系统。每张图片输出两个分数：

- **Alignment Score (0-5)** — 图片与文本 prompt 的语义对齐度，结合原子 QA 正确性和 Taxonomy 特征符合度
- **Authenticity Score (0-5)** — 图片真实度/质量（伪影、融化、结构崩坏等），分数越高表示图片质量越好

评估流水线分 5 步：

```
Step 0 Atomize  →  Step 1 Router  →  Step 2 Judge  →  Step 3 Expert  →  Step 4 Reflector
 (Prompt原子化)     (VLM规划+QA)      (VLM审核)        (本地GPU推理)      (VLM综合评分)
```

### 评分机制

**Alignment Score** = `mean(per_atom_scores) × 5.0`

每个原子 QA 的分数由两层组成：
- `qa_score` (0-1)：图片对该 QA 问题的回答是否正确
- `tax_score` (0-1)：检测到的物体是否符合 Taxonomy 诊断检查点
- `atom_score = qa_score × tax_score`（无 Taxonomy 信息时 tax_score = 1.0）

**Authenticity Score** (0-5)：由 Reflector 基于视觉观察和专家证据综合判断，衡量伪影严重程度和图片整体质量。

## 环境准备

### 1. 环境变量（必须）

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
export MACA_PATH=/opt/maca   # MetaX GPU 服务器必须设置
```

### 2. 激活 Python 环境

```bash
source /mnt/afs/zhengmingkai/miniconda3/envs/themis/bin/activate
# 或
conda activate themis
```

### 3. 进入项目目录

```bash
cd /path/to/THEMIS
```

## 数据准备

### 1. 准备 GenEval2 数据文件

将 GenEval2 的 prompt 数据文件放到 THEMIS 项目根目录下，命名为 `geneval2_data.jsonl`。

每行是一个 JSON 记录，格式如下：

```json
{
  "prompt": "four brown monkeys and a metal bicycle",
  "vqa_list": [
    ["How many monkeys are in the image?", "four"],
    ["Are the monkeys brown?", "yes"],
    ["Is there a bicycle?", "yes"],
    ["Is the bicycle metal?", "yes"]
  ],
  "skills": ["count", "attribute", "count", "attribute"]
}
```

字段说明：
- `prompt`：文本生成 prompt
- `vqa_list`：原子 QA 对列表，每对是 `[question, answer]`
- `skills`（可选）：与 `vqa_list` 平行的技能标签

### 2. 准备待测评图片

用 GenEval2 的 prompt 生成图片后，将图片放到一个文件夹中（如 `test_image/`）。**图片文件名必须以 prompt_id 命名**，即 `geneval2_data.jsonl` 中的行号（从 0 开始）：

```
test_image/
├── 0.png      # 对应 geneval2_data.jsonl 第 0 行的 prompt
├── 1.png      # 对应第 1 行
├── 2.png
├── ...
└── 100.png
```

支持的图片格式：`.png`、`.jpg`、`.jpeg`

## 统一入口

所有评估任务通过 `t2i_harness/run.py` 启动：

```bash
python t2i_harness/run.py --mode <模式> --step <步骤> [选项...]
```

## 两种执行模式

| 模式 | 适用场景 | 特点 |
|------|----------|------|
| `sync` | 调试/单图测试 | 串行执行，日志最清晰 |
| `async` | 日常批量评估 | API 并发 + GPU 流水线重叠，**默认模式** |

## 快速上手

### 测评 test_image 文件夹中的图片

假设你已用 GenEval2 的 prompt 生成了图片，放在 `test_image/` 目录下，`geneval2_data.jsonl` 已放在 THEMIS 根目录。

```bash
# 完整流水线（Atomize + Router + Judge + Expert + Reflector），异步模式，10 张图
python t2i_harness/run.py --mode async --step 1234 --image-dir test_image --limit 10 --api-concurrency 5

# 串行模式（调试用，日志最清晰）
python t2i_harness/run.py --mode sync --step 1234 --image-dir test_image --limit 5

# 测评单张图片（如 prompt_id=0 对应的 0.png）
python t2i_harness/run.py --mode sync --step 1234 --image-dir test_image --image-id 0

# 仅跑 Step 1+2（Atomize + Router + Judge，不跑 GPU 专家和 Reflector）
python t2i_harness/run.py --mode async --step 12 --image-dir test_image --limit 100 --api-concurrency 5

# 仅跑 Step 3（已有 approved plans，只跑 GPU 专家推理）
python t2i_harness/run.py --mode async --step 3 --gpu-groups 1

# 仅跑 Step 4（已有 expert results，只跑 Reflector）
python t2i_harness/run.py --mode async --step 4 --image-dir test_image --api-concurrency 5
```

### GPU 服务器上的批量评估

```bash
# 2 卡环境，10 路 API 并发，全流程
python t2i_harness/run.py --mode async --step 1234 --image-dir test_image --limit 500 \
  --api-concurrency 10 --gpu-preset 2x_c500 --api-retry 2

# 8 卡环境，20 路 API 并发，全流程
python t2i_harness/run.py --mode async --step 1234 --image-dir test_image --limit 1000 \
  --api-concurrency 20 --gpu-preset 8x_c500_fast --api-retry 2

# 自定义 GPU 配置文件
python t2i_harness/run.py --mode async --step 1234 --gpu-config my_config.json
```

## 完整参数列表

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `async` | 执行模式：`sync` / `async` |
| `--step` | `123` | 执行步骤：`1` / `2` / `12` / `3` / `4` / `123` / `1234` |
| `--limit` | `0` (全部) | 最大处理图片数 |
| `--image-id` | - | 只处理指定 prompt_id 的单张图 |
| `--max-iterations` | `2` | Router-Judge 最大迭代轮数 |
| `--image-dir` | `test_images/` | 输入图片目录 |
| `--output-dir` | `output` | 输出目录名（相对 `t2i_harness/`）或绝对路径 |
| `--geneval2-jsonl` | `geneval2_data.jsonl` | GenEval2 数据文件路径 |
| `--save-feedback` | `false` | 保存 Judge 反馈详情 |

### GPU 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gpu-groups` | `1` | 并行 GPU 组数 |
| `--gpu-config` | - | 自定义 GPU 分配 JSON 文件路径 |
| `--gpu-preset` | - | GPU 预设名（如 `2x_c500`, `8x_c500`） |

### Async 模式参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api-concurrency` | `5` | 最大 API 并发数 |
| `--api-retry` | `0` | API 调用失败重试次数 |
| `--temp-router` | `0.0` | Router API 温度参数 |
| `--temp-judge` | `0.0` | Judge API 温度参数 |
| `--temp-reflector` | `0.5` | Reflector API 温度参数 |

## 输出文件结构

```
t2i_harness/output/
├── atomized/           # Step 0 原子化后的 prompt 结构
├── plans/              # Step 1 Router 生成的初始 plan
├── approved_plans/     # Step 2 Judge 审批后的 plan
├── judge_feedback/     # Judge 反馈详情（需 --save-feedback）
├── expert_results/     # Step 3 专家推理结果
├── final_reports/      # Step 4 Reflector 最终评分报告
└── ...
```

### 最终评分报告格式

每张图片一个 JSON 文件 `final_reports/final_evaluation_report_{prompt_id}.json`：

```json
{
  "atom_reviews": [
    {
      "atom_index": 0,
      "question": "How many monkeys are in the image?",
      "expected": "four",
      "qa_score": 1.0,
      "tax_score": 0.9,
      "atom_score": 0.9,
      "expert_evidence": "...",
      "reasoning": "..."
    }
  ],
  "artifact_review": "...",
  "alignment_score": 4.32,
  "authenticity_score": 3.85,
  "per_atom_scores": [0.9, 0.85, 1.0, 0.8],
  "key_defects": ["..."],
  "metadata": {
    "original_image": "...",
    "prompt_id": "0",
    "prompt_text": "four brown monkeys and a metal bicycle",
    "reflector_cost_seconds": 3.21
  }
}
```

## 代码结构

```
t2i_harness/
├── run.py               # 统一入口，CLI 参数解析
├── common.py            # 共享工具函数、路径常量、GenEval2 数据加载
├── step0_atomize.py     # Step 0: Prompt 原子化与 Taxonomy 关联
├── step1_router.py      # Step 1: Router Agent（VLM 规划 + 原子 QA 判定）
├── step2_judge.py       # Step 2: Judge Agent（VLM 审核）
├── step4_reflector.py   # Step 4: Reflector Agent（VLM 综合评分）
├── dispatch_sync.py     # 串行模式调度
├── dispatch_async.py    # 异步流水线模式调度
└── README.md            # 本文件
```

**注意**：Step 3（Expert 执行）复用 `c2i_harness/step3_execute.py` 中的 `ExpertManager` 和 `execute_plan`，无需重复实现。

## 模型角色分配

| 角色 | 模型 | 温度 | 职责 |
|------|------|------|------|
| Router | Qwen3.6-Plus | 0.0 | 原子 QA 判定、Taxonomy 检查点验证、伪影检测、专家选择 |
| Judge | Qwen3.6-Plus | 0.0 | 审核 Router 的评估计划，确保逻辑严谨性和完整性 |
| Reflector | Qwen3.7-Plus | 0.5 | 综合专家证据和 Router 评估，输出最终 alignment_score 和 authenticity_score |

Router 和 Judge 使用相同模型但不同系统消息，Reflector 使用不同模型以减少评分偏差。

## 评估流水线详解

### Step 0: Prompt 原子化（自动执行）

将 GenEval2 的 VQA 格式 prompt 分解为结构化的原子 QA 对，并为每个物体关联 ImageNet Taxonomy 信息：

1. 将 `[question, answer]` 对标准化为包含 `atom_index`、`answer_type`、`skill`、`target_object` 的原子
2. 从 QA 问题中提取物体名称（如 "monkeys" → "monkey"）
3. 将物体映射到 ImageNet class_id（内置映射表 + 可选 JSON 覆盖）
4. 加载该 class_id 的 Taxonomy 描述和诊断检查点

### Step 1: Router（VLM 规划）

Router 接收图片 + prompt + 原子 QA + Taxonomy 信息，完成：
- **原子 QA 判定**：对每个原子预测答案并判断是否正确
- **Taxonomy 检查点验证**：验证图片中物体是否符合 Taxonomy 诊断特征
- **伪影检测**：扫描全图寻找 AI 生成伪影
- **专家选择**：根据可见实体和伪影风险选择 3-8 个专家

### Step 2: Judge（VLM 审核）

Judge 审查 Router 的评估计划，检查：
- 原子 QA 覆盖率和判定合理性
- Taxonomy 检查点判断是否合理
- 伪影观测是否一致
- 专家选择是否适当

如拒绝，Router 根据反馈修订计划（最多 `--max-iterations` 轮）。

### Step 3: Expert（本地 GPU 推理）

复用 C2I 系统的专家模型执行评估计划，包括：
- 细粒度分类器、开放词汇检测器、动物姿态检测
- 深度估计、拓扑边界分析、感知质量审计、文字审计

### Step 4: Reflector（VLM 综合评分）

Reflector 综合所有信息输出最终评分：
- 对每个原子 QA 评估 `qa_score` 和 `tax_score`，计算 `atom_score = qa_score × tax_score`
- `alignment_score = mean(per_atom_scores) × 5.0`（代码硬计算，非 LLM 自报）
- `authenticity_score` 由 Reflector 基于视觉观察和专家证据综合判断

## 常见问题

### Q: 图片文件名如何命名？

图片文件名必须以 `geneval2_data.jsonl` 中的行号（0-based）为文件名。例如第 0 行的 prompt 对应 `0.png`，第 1 行对应 `1.png`。支持 `.png`、`.jpg`、`.jpeg` 格式。

### Q: 如何只跑 API 部分（不跑 GPU 专家）？

```bash
python t2i_harness/run.py --mode async --step 12 --image-dir test_image --limit 100
```

### Q: 如何只重跑 Reflector？

```bash
python t2i_harness/run.py --mode async --step 4 --image-dir test_image
```

### Q: `geneval2_data.jsonl` 应该放在哪里？

默认放在 THEMIS 项目根目录下。也可通过 `--geneval2-jsonl` 参数指定自定义路径。

### Q: 物体没有对应的 ImageNet 类怎么办？

系统内置了常见物体的 ImageNet class_id 映射表。未映射的物体仍可正常评估（原子 QA 照常工作），但不会有关联的 Taxonomy 诊断检查点（`tax_score` 默认为 1.0）。可在 `t2i_harness/object_to_classid.json` 中添加自定义映射来扩展。

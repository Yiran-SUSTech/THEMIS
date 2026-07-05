# THEMIS C2I 评估系统 — 启动指南

## 概述

THEMIS C2I 是一个 Agentic 图像质量评估系统，对 AI 生成的图片输出两个分数：
- **Alignment Score (0-5)** — 图片是否符合目标类别
- **Artifact Score (0-5)** — 画面质量（伪影、融化、结构崩坏等）

评估流水线分 4 步：

```
Step 1 Router  →  Step 2 Judge  →  Step 3 Expert  →  Step 4 Reflector
 (VLM规划)        (VLM审核)       (本地GPU推理)      (VLM综合评分)
```

## 环境准备

### 1. 环境变量（必须）

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
export MACA_PATH=/opt/maca   # MetaX GPU 服务器必须设置
```

建议写入 `~/.bashrc` 永久生效。

### 2. 激活 Python 环境

```bash
source /mnt/afs/zhengmingkai/miniconda3/envs/themis/bin/activate
# 或
conda activate themis
```

### 3. 进入项目目录

```bash
cd /mnt/afs/zhengmingkai/hhy/themis/THEMIS
```

## 统一入口

所有评估任务通过 `c2i_faster/run.py` 启动：

```bash
python c2i_faster/run.py --mode <模式> --step <步骤> [选项...]
```

## 三种执行模式

| 模式 | 适用场景 | 特点 |
|------|----------|------|
| `sync` | 调试/单图测试 | 串行执行，日志最清晰 |
| `async` | 日常批量评估 | API 并发 + GPU 流水线重叠，**默认模式** |
| `batch` | 大规模评估 (1000+图) | 批量 API 提交，成本约 50% 折扣，延迟分钟~小时级 |

## 快速上手

### 测试单张图（验证环境）

```bash
python c2i_faster/run.py --mode sync --step 123 --image-id 000000
```

### 跑 10 张图（异步模式，5 路 API 并发）

```bash
python c2i_faster/run.py --mode async --step 123 --limit 10 --api-concurrency 5

# 2 卡环境 10路并发 无 Session 模式
python c2i_faster/run.py --mode async --step 1234 --limit 40 --api-concurrency 10 --gpu-preset 2x_c500

# 2 卡环境 10路并发 Session 模式（共享对话上下文）
python c2i_faster/run.py --mode async --step 1234 --limit 40 --api-concurrency 10 --gpu-preset 2x_c500 --session

# 8 卡环境 20路并发 Session 模式（共享对话上下文） 
python c2i_faster/run.py --mode async --step 1234 --limit 40 --api-concurrency 20 --gpu-preset 8x_c500_fast --session

# 8 卡环境 20路并发 Session 模式（共享对话上下文） 使用参考图片 生成checklist API访问失败再重试2次
# 不启用 pose 硬封顶，Reflector 根据自身视觉判断打分
python c2i_faster/run.py --mode async --step 1234 --limit 40 --api-concurrency 20 --gpu-preset 8x_c500_fast --session --enable-checklist --ref-enable --api-retry 2 --image-dir /mnt/afs/zhengmingkai/zyr/THEMIS/IMF_XL2-sample_class_num-100-per_class_img-5

# 消融实验：无专家模式（仅 Router 直接打分，无 Judge/Expert/Reflector）
python c2i_faster/run.py --mode async --without-expert --limit 40 --api-concurrency 20 --api-retry 2

# 消融实验：启用硬封顶
python c2i_faster/run.py --mode async --step 1234 --limit 40 \
  --api-concurrency 20 --gpu-preset 8x_c500_fast \
  --session --enable-checklist --ref-enable --api-retry 2 \
  --pose-hard-cap

# 自定义配置文件
python c2i_faster/run.py --mode async --step 1234 --gpu-config my_config.json
```

### 全流程含 Reflector（Step 1-4）

```bash
python c2i_faster/run.py --mode sync --step 1234 --limit 5 --session
```

### 只跑 GPU 推理（已有 approved plans）

```bash
python c2i_faster/run.py --mode async --step 3 --gpu-groups 1
```

### 大规模 Batch 模式

```bash
python c2i_faster/run.py --mode batch --step 12 --limit 1000 --poll-interval 30
# batch 完成后，跑 GPU 层
python c2i_faster/run.py --mode async --step 3 --gpu-groups 2
```

## 完整参数列表

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `async` | 执行模式：`sync` / `async` / `batch` |
| `--step` | `123` | 执行步骤：`1` / `2` / `12` / `3` / `4` / `123` / `1234` |
| `--limit` | `0` (全部) | 最大处理图片数 |
| `--image-id` | - | 只处理指定 ID 的单张图 |
| `--max-iterations` | `2` | Router-Judge 最大迭代轮数 |
| `--image-dir` | `test_images/` | 输入图片目录 |
| `--class-ids` | `test_images/class_ids.txt` | 图片-类别映射文件 |
| `--save-feedback` | `false` | 保存 Judge 反馈详情 |
| `--session` | `false` | Step 4 使用 conversation session |
| `--save-pose-viz` | `false` | 保存骨骼可视化图 |
| `--without-expert` | `false` | 消融模式：仅 Router 直接打分，跳过 Judge/Expert/Reflector |

### GPU 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gpu-groups` | `1` | 并行 GPU 组数（2 = 同时跑两张图的专家） |
| `--gpu-config` | - | 自定义 GPU 分配 JSON 文件路径 |

### Async 模式参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api-concurrency` | `5` | 最大 API 并发数 |

### Batch 模式参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-dir` | `c2i_faster/output/batch/` | Batch JSONL 文件存储目录 |
| `--poll-interval` | `30` | 轮询 batch 状态间隔（秒） |

## 输出文件结构

```
c2i_faster/output/
├── plans/              # Step 1 Router 生成的初始 plan
├── approved_plans/     # Step 2 Judge 审批后的 plan
├── judge_feedback/     # Judge 反馈详情（需 --save-feedback）
├── expert_results/     # Step 3 专家推理结果
├── final_reports/      # Step 4 Reflector 最终评分报告
├── without_expert_reports/  # --without-expert 模式下 Router 直接打分报告
├── batch/              # Batch 模式的 JSONL 文件
├── depth_maps/         # 深度图输出
├── sam_masks/          # 分割掩码输出
└── pose_visualizations/ # 骨骼可视化（需 --save-pose-viz）
```

## 输入数据格式

### test_images/ 目录

```
test_images/
├── 000000.png
├── 000001.png
├── ...
└── class_ids.txt
```

### class_ids.txt 格式

```
000000 0
000001 0
000002 1
...
```

每行：`<图片ID> <ImageNet类别ID>`

## 性能参考

| 配置 | 单图延迟 | 吞吐量 |
|------|----------|--------|
| sync, 全流程 | ~90s | 1 img/90s |
| async, 5 并发, 1 GPU 组 | ~43s/img | 10 img/7min |
| async, 10 并发, 2 GPU 组 | ~20s/img（预估） | - |
| batch, Step 1+2 | 分钟级（异步） | 不受 QPS 限制 |

## 代码结构

```
c2i_faster/
├── run.py               # 统一入口
├── common.py            # 共享工具函数、常量
├── dispatch_sync.py     # 串行模式
├── dispatch_async.py    # 异步流水线模式
├── dispatch_batch.py    # Batch API 模式
├── step1_router.py      # Router Agent（VLM 规划）
├── step2_judge.py       # Judge Agent（VLM 审核）
├── step3_execute.py     # Expert Manager（本地 GPU 推理）
├── step4_reflector.py   # Reflector Agent（VLM 综合评分）
└── conversation_session.py  # Session 管理
```

## 消融实验：无专家模式（--without-expert）

为验证系统中加入专家证据能使测评结果更贴近人类打分，提供 `--without-expert` 消融模式。

### 工作原理

在该模式下，系统**跳过** Judge、Expert、Reflector 三个环节，仅保留 Router：
- Router 接收待测评图片 + 类别 + taxonomy info checklist
- Router 直接完成 Step 1（checkpoint 验证）+ Step 2（artifact 检测）+ Step 3（直接打分）
- 输出 `alignment_score` 和 `artifact_score`（0-5 连续分数）

Router 的 prompt 与正常模式保持高度一致（Step 1 和 Step 2 完全相同），仅将 Step 3 从"选择专家"替换为"直接打分"，确保消融对比的公平性。

### 使用示例

```bash
# 40 张图，20 路 API 并发，仅 Router 直接打分
python c2i_faster/run.py --mode async --without-expert --limit 40 --api-concurrency 20 --api-retry 2

# Session 模式（共享对话上下文）
python c2i_faster/run.py --mode async --without-expert --limit 40 --api-concurrency 20 --session --api-retry 2

# 串行模式（调试用）
python c2i_faster/run.py --mode sync --without-expert --limit 5
```

### 输出

报告保存在 `c2i_faster/output/without_expert_reports/` 目录下，每张图一个 JSON 文件：

```json
{
  "image_description": "...",
  "image_class": "golden retriever",
  "checkpoint_verdicts": [...],
  "artifact_observations": [...],
  "alignment_score": 4.32,
  "artifact_score": 3.85,
  "alignment_reasoning": "...",
  "artifact_reasoning": "...",
  "metadata": {
    "original_image": "...",
    "class_id": 207,
    "class_label": "golden retriever",
    "router_cost_seconds": 3.21,
    "mode": "without_expert"
  }
}
```

### 与正常模式对比

| 维度 | 正常模式 | 无专家模式 |
|------|----------|------------|
| Router | 生成 plan | 直接打分 |
| Judge | 审核 plan | 跳过 |
| Expert | 本地 GPU 推理 | 跳过 |
| Reflector | 综合评分 | 跳过 |
| 输出目录 | `final_reports/` | `without_expert_reports/` |
| GPU 需求 | 需要 | 不需要 |
| 单图成本 | 高（多次 API + GPU） | 低（单次 API） |

## 常见问题

### Q: `TypeError: expected str, bytes or os.PathLike object, not NoneType`

设置 `export MACA_PATH=/opt/maca`。这是 MetaX triton backend 需要的环境变量。

### Q: DINO 加载很慢（100s+）

首次加载需要下载 bert-base-uncased tokenizer（约 440MB）。后续运行会使用缓存。

### Q: 如何只重跑 Reflector？

```bash
python c2i_faster/run.py --mode sync --step 4 --session
```

### Q: 如何查看单图的各专家耗时？

```bash
cat c2i_faster/output/expert_results/expert_results_000000.json | python -m json.tool | grep execution_time_ms
```

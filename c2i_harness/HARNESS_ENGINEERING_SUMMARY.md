# C2I Harness Engineering 修改总结

## 概述

本文件夹 `c2i_harness` 基于原 `c2i_faster` 代码进行重构，核心目标是将 THEMRIS C2I 评测系统从松散的 LLM 调用模式转变为符合 **Harness Engineering** 原则的可靠评测框架。修改涵盖角色分离、上下文隔离、会话模式移除、温度控制等多个维度。

---

## 一、修改清单

### 1. 模型角色分离（缓解"既当运动员又当裁判"问题）

| 角色 | 模型 | 文件 | 常量 |
|------|------|------|------|
| Router | `qwen3.6-plus` | `step1_router.py` | `ROUTER_MODEL` |
| Judge | `qwen3.6-plus` | `step2_judge.py` | `JUDGE_MODEL` |
| Reflector | `qwen3.7-plus` | `step4_reflector.py` | `REFLECTOR_MODEL` |

- Router 负责生成评测计划，Judge 负责审核计划，两者使用同一模型 `qwen3.6-plus`，构成"计划-审核"闭环。
- Reflector 作为最终裁决者，使用更强的 `qwen3.7-plus` 模型，与 Router/Judge 形成模型层面的隔离。
- 三个角色均通过同一个 API Key（`DASHSCOPE_API_KEY`）调用，但模型不同，确保裁判角色具备独立判断能力。

### 2. 移除 Session 模式

**原问题**：Session 模式将三个角色的指令捆绑在单个 system message 中，每次 API 调用都会重发完整对话历史，导致 token 累积浪费，且 Session 与非 Session 模式下 Reflector 行为不一致。

**修改内容**：

- **删除 `conversation_session.py` 的所有引用**：`dispatch_async.py`、`dispatch_sync.py` 不再导入 `ConversationSession`、`build_combined_system_content`、`build_reflector_only_system_content`。
- **移除 `--use-session` 命令行参数**：`run.py` 中不再提供 session 相关选项。
- **移除所有 `session` 参数传递**：`generate_plan()`、`revise_plan()`、`review_plan()`、`run_reflector()` 等函数签名中不再包含 `session` 参数。
- **`conversation_session.py` 文件保留但不再被任何活跃模块导入**，仅作为历史参考。

### 3. 独立 System Message（指令解耦）

**原问题**：Session 模式下，三个角色的指令被拼接成一个超长 system message，每次调用都携带全部角色指令，造成冗余。

**修改内容**：

每个角色现在拥有独立的、精简的 system message，只包含该角色所需的信息：

- **Router** system message：
  - 角色定位："You are a highly logical Router Agent for image auditing."
  - 通用路由指令（`_COMMON_ROUTER_INSTRUCTIONS`）
  - 专家注册表摘要（包含全部 7 个专家的 `expert_id`、`description`、`best_for` 字段，由 Router LLM 在 Step 3 中根据图片内容自行选择 3-8 个相关专家）

- **Judge** system message：
  - 角色定位："You are a meta-cognitive Judge Agent."
  - 专家注册表摘要

- **Reflector** system message：
  - 独立的 Reflector 系统模板（`_REFLECTOR_SYSTEM_TEMPLATE` 或 `_REFLECTOR_CHECKLIST_SYSTEM_TEMPLATE`）
  - 不包含 Router/Judge 的指令

每次 API 调用只传递 `[system_message, user_message]` 两条消息，不携带历史对话。

### 4. 角色级温度控制

**原问题**：Session 模式下温度参数由 session 统一管理，无法对不同角色设置不同温度；非 Session 模式下温度硬编码在代码中。

**修改内容**：

在 `run.py` 中新增三个命令行参数：

```bash
--temp-router      # Router 温度，默认 0.0（确定性输出）
--temp-judge       # Judge 温度，默认 0.0（确定性输出）
--temp-reflector   # Reflector 温度，默认 0.5（允许一定创造性）
```

参数传递链路：

```
run.py (CLI args)
  → dispatch_async.py / dispatch_sync.py (temp_router, temp_judge, temp_reflector)
    → step1_router.py: generate_plan(temperature=temp_router)
                       revise_plan(temperature=temp_router)
                       generate_direct_score(temperature=temp_router)
    → step2_judge.py: review_plan(temperature=temp_judge)
    → step4_reflector.py: run_reflector(temperature=temp_reflector)
```

涉及的函数签名修改：

| 函数 | 文件 | 新增参数 |
|------|------|----------|
| `generate_plan()` | `step1_router.py` | `temperature: float = 0.0` |
| `revise_plan()` | `step1_router.py` | `temperature: float = 0.0` |
| `generate_direct_score()` | `step1_router.py` | `temperature: float = 0.0` |
| `_call_router_api()` | `step1_router.py` | `temperature: float = 0.0` |
| `review_plan()` | `step2_judge.py` | `temperature: float = 0.0` |
| `run_reflector()` | `step4_reflector.py` | `temperature: float = 0.5` |
| `_sync_router_judge()` | `dispatch_async.py` | `temp_router`, `temp_judge` |
| `_api_worker()` | `dispatch_async.py` | `temp_router`, `temp_judge` |
| `_sync_reflector()` | `dispatch_async.py` | `temperature` |
| `_reflector_worker()` | `dispatch_async.py` | `temp_reflector` |
| `_run_full_pipeline()` | `dispatch_async.py` | `temp_router`, `temp_judge`, `temp_reflector` |
| `_run_step12_only()` | `dispatch_async.py` | `temp_router`, `temp_judge` |
| `_run_step4_only()` | `dispatch_async.py` | `temp_reflector` |
| `run_async_pipeline()` | `dispatch_async.py` | `temp_router`, `temp_judge`, `temp_reflector` |
| `_run_single_image()` | `dispatch_sync.py` | `temp_router`, `temp_judge` |
| `run_sync_pipeline()` | `dispatch_sync.py` | `temp_router`, `temp_judge`, `temp_reflector` |

### 5. 上下文过滤（仅传必要信息）

**原问题**：Session 模式下，每次 API 调用都携带完整对话历史（包括之前角色的交互记录、历史图片 URL 等），造成 token 浪费。

**修改内容**：

- **Router 调用**：仅传递 `[system_message, {image + prompt}]`，不携带 Judge 的反馈历史（revision 时仅在 user prompt 中包含 feedback 文本）。
- **Judge 调用**：仅传递 `[system_message, {image + prompt}]`，不携带 Router 的 system 指令或历史 Judge 判决。
- **Reflector 调用**：仅传递 `[system_message, {image + expert_results + prompt}]`，不携带 Router/Judge 的对话历史。
- **图片历史修剪**：每次调用只传递当前图片的 base64 编码，不携带历史图片。

### 6. 批处理模式（dispatch_batch.py）

`dispatch_batch.py` 使用 OpenAI Batch API 进行大规模离线处理，本身不使用 session 模式。已确认其模型常量正确：

```python
ROUTER_MODEL = "qwen3.6-plus"
JUDGE_MODEL = "qwen3.6-plus"
```

批处理模式仅处理 Step 1+2（Router + Judge），Step 3（GPU 执行）复用 async dispatcher，不涉及 Reflector，因此无需额外的模型分离。温度在批处理 JSONL 中硬编码为 0（批处理场景下确定性优先）。

---

## 二、Harness Engineering 具体应用情况

Harness Engineering 的核心思想是将 LLM 评测系统视为一个工程化的"测试框架"（harness），通过结构化设计确保评测的可靠性、可复现性和公平性。以下是本项目中 harness engineering 原则的具体应用：

### 原则 1：角色隔离（Role Isolation）

**应用**：三个角色（Router、Judge、Reflector）使用独立的 system message，不共享上下文。每个角色的 API 调用是独立的无状态请求，不携带其他角色的指令或历史交互。

**收益**：
- 消除了角色间的指令干扰（一个角色不会因为看到其他角色的指令而产生偏向）
- 每次调用的 token 消耗可预测，不会因对话历史累积而增长
- 角色行为可独立调试和复现

### 原则 2：模型分离（Model Separation）

**应用**：Router/Judge 使用 `qwen3.6-plus`，Reflector 使用 `qwen3.7-plus`。最终裁决者使用更强且不同的模型，避免同一模型"既当运动员又当裁判"。

**收益**：
- 减少模型自评偏差（同模型倾向于给自己的输出更高评分）
- Reflector 作为独立裁决者，具备更强的推理能力
- 同一 API Key 调用不同模型，保持部署简洁

### 原则 3：确定性可控（Determinism Control）

**应用**：通过 `--temp-router`、`--temp-judge`、`--temp-reflector` 命令行参数，为每个角色独立设置温度。

**收益**：
- Router/Judge 默认 temperature=0.0，确保计划生成和审核的确定性，提高可复现性
- Reflector 默认 temperature=0.5，允许最终裁决具备一定的推理多样性
- 实验者可通过命令行参数灵活调整，无需修改代码

### 原则 4：上下文最小化（Context Minimization）

**应用**：每次 API 调用只传递该角色完成任务所需的最小信息：
- Router：图片 + 分类标签 + 专家注册表摘要 + 路由指令
- Judge：图片 + 分类标签 + 待审计划 + 专家注册表摘要
- Reflector：图片 + 分类标签 + 专家执行结果 + 路由计划（可选）

**收益**：
- Token 消耗降低（无对话历史累积）
- 减少上下文噪声对模型判断的干扰
- API 调用延迟更稳定

### 原则 5：无状态调用（Stateless Invocation）

**应用**：移除 session 模式后，所有 API 调用都是无状态的。每个角色的每次调用都是独立的 `[system, user]` 消息对，不依赖之前的调用历史。

**收益**：
- 消除了 session 模式下 token 累积导致的成本增长
- 消除了 session 与非 session 模式下 Reflector 行为不一致的问题
- 简化了错误恢复（单次调用失败只需重试该调用，不需回滚 session 状态）

### 原则 6：可复现性（Reproducibility）

**应用**：
- 固定模型常量（`ROUTER_MODEL`、`JUDGE_MODEL`、`REFLECTOR_MODEL`）
- 命令行参数控制温度
- 无状态调用确保相同输入产生相同输出（temperature=0 时）
- 计划、审核反馈、专家结果、最终报告均持久化到磁盘

**收益**：
- 实验结果可复现
- 可通过命令行参数精确记录实验配置
- 中间结果可追溯

---

## 三、文件修改对照表

| 文件 | 修改类型 | 关键变化 |
|------|----------|----------|
| `run.py` | 新增参数 | `--temp-router`, `--temp-judge`, `--temp-reflector`；移除 `--use-session` |
| `step1_router.py` | 签名修改 | `generate_plan()`, `revise_plan()`, `generate_direct_score()`, `_call_router_api()` 新增 `temperature` 参数；移除 `session` 参数 |
| `step2_judge.py` | 签名修改 | `review_plan()` 新增 `temperature` 参数；移除 `session` 参数；模型改为 `qwen3.6-plus` |
| `step4_reflector.py` | 签名修改+模型变更 | `run_reflector()` 新增 `temperature` 参数；移除 `session` 参数；模型改为 `qwen3.7-plus` |
| `dispatch_async.py` | 结构重构 | 移除 `conversation_session` 导入；所有 worker 函数新增温度参数；移除 session 创建逻辑；移除 5-tuple/4-tuple 解包逻辑 |
| `dispatch_sync.py` | 结构重构 | 移除 `conversation_session` 导入；`_run_single_image()` 和 `run_sync_pipeline()` 新增温度参数；移除 session 参数 |
| `dispatch_batch.py` | 模型更新 | `ROUTER_MODEL` 和 `JUDGE_MODEL` 确认为 `qwen3.6-plus` |
| `conversation_session.py` | 保留不导入 | 文件保留但不再被任何活跃模块导入 |

---

## 四、使用示例

### 单图测评（sync 模式）

```bash
python run.py --mode sync --step 1234 --image-id IMG_001 \
  --temp-router 0 --temp-judge 0 --temp-reflector 0.5
```

### 批量异步测评（async 模式）

```bash
python run.py --mode async --step 1234 --limit 100 \
  --api-concurrency 5 --gpu-preset 2x_c500 \
  --temp-router 0 --temp-judge 0 --temp-reflector 0.3
```

### 仅运行 Reflector（从磁盘加载中间结果）

```bash
python run.py --mode async --step 4 --limit 100 \
  --temp-reflector 0.5
```

### 大规模批处理（batch 模式，仅 Step 1+2）

```bash
python run.py --mode batch --step 12 --limit 1000
```

---

## 五、验证结果

- **语法检查**：所有 11 个 Python 文件通过 `py_compile` 编译检查。
- **模型常量验证**：Router/Judge = `qwen3.6-plus`，Reflector = `qwen3.7-plus`。
- **Session 引用验证**：无任何活跃模块导入 `conversation_session`，无 `use_session` 或 `session=None` 残留。
- **温度参数验证**：`run.py` CLI 参数正确传递到所有 dispatcher 和 step 函数。
- **函数签名验证**：所有核心函数的 `temperature` 参数已正确添加。

---

## 六、工作流改进（2026-08 更新）

基于 `C2I_WORKFLOW_REVISION.md` 方案，对 Router、Judge、Reflector 三个角色进行了证据驱动和自审机制改进。详见 `C2I_WORKFLOW_REVISION.md`。

### 1. Router — 证据驱动专家验证计划

**改进前**：Router Step 3 基于图中可见实体和硬编码规则选择专家，仅输出 `selected_experts`（专家名 + target_subject）。

**改进后**：Router 先对 Taxonomy checkpoint 做初步判断（Step 1，不变），再基于判断结果制定**证据驱动的专家验证计划**（Step 3 重写）：

- 对每个 checkpoint 判断哪些可以用专家模型求证
- 对 `is_present=false` 的 checkpoint 强烈建议 `fine_grained_classifier` 验证物种身份
- 对伪影观察分配 `perceptual_quality_auditor` 或 `topology_boundary_auditor`
- 无法用专家验证的点列入 `unverifiable_points`

**新输出结构**：

```json
{
  "expert_verification_plan": [
    {
      "expert_name": "fine_grained_classifier",
      "target_subject": "monkey",
      "verification_goals": ["body_structure checkpoint", "facial_features checkpoint"],
      "reason": "Verify species identity for is_present=false checkpoints",
      "weight": 1.0
    }
  ],
  "unverifiable_points": ["texture/covering quality cannot be verified by expert models"]
}
```

向后兼容：自动生成 `selected_experts` 供 Step 3 Expert 执行使用。

### 2. Judge — 验证覆盖度审查

**新增审查维度**：检查 Router 的 `expert_verification_plan` 完备性：

- 所有关键 taxonomy checkpoint 是否都有专家覆盖（或合理标注为 unverifiable）
- `verification_goals` 是否与 checkpoint 一一对应
- `unverifiable_points` 中的点是否确实无法用专家验证

**新输出字段**：

```json
{
  "coverage_assessment": {
    "checkpoints_covered": 8,
    "checkpoints_total": 10,
    "artifacts_covered": 2,
    "artifacts_total": 3,
    "unverifiable_count": 2
  }
}
```

### 3. Reflector — Self-Reflection 机制（两轮 API 调用）

**改进前**：Reflector 单次 API 调用直接输出最终评分。

**改进后**：采用方案 A（两轮 API 调用）实现 Self-Reflection：

1. **Round 1（初步评分）**：使用原始 Reflector 系统模板，输出 `alignment_score`、`authenticity_score` + reasoning
2. **Round 2（自审修订）**：使用 `_REFLECTOR_SELF_REFLECTION_TEMPLATE`，审查自身 Round 1 输出：
   - 分数与 reasoning 是否一致
   - 专家证据是否充分利用
   - 参考校准是否合理
   - 是否过于宽松/严格
   - 每个 checkpoint 的 is_present 判断是否独立验证
3. **合并**：Round 2 分数优先，Round 1 分数保存在 `preliminary_scores` 供审计

**新输出字段**：

```json
{
  "alignment_score": 3.20,
  "authenticity_score": 3.80,
  "preliminary_scores": {
    "alignment_score": 3.50,
    "authenticity_score": 4.00
  },
  "self_reflection_notes": "Lowered alignment_score from 3.50 to 3.20 because...",
  "score_changes": {
    "alignment_score": "3.50 → 3.20",
    "authenticity_score": "4.00 → 3.80"
  }
}
```

**执行顺序**：Round 1（LLM 初步评分）→ Round 2（LLM Self-Reflection）→ `_calibrate_scores`（代码级硬规则）。

### 4. `_calibrate_scores` 开关控制

**改进前**：`_calibrate_scores` 中的分类器封顶和姿态封顶均硬编码启用，无法关闭。

**改进后**：新增 `enable_classifier_cap` 参数（默认 `True`），可通过 CLI 控制：

| 操作 | 参数 | 默认 | 说明 |
|------|------|------|------|
| 分类器封顶 | `enable_classifier_cap` | `True` | Top-1 不匹配 → alignment ≤ 2.0；Top-3 不含目标 → alignment ≤ 1.0 |
| 姿态封顶 | `pose_hard_cap` | `False` | 低置信度关节比例过高时封顶 authenticity_score（默认关闭，因 domain-shift 风险） |

使用 `--no-classifier-cap` 可完全信任 Reflector 的判断，不做代码级封顶。

### 5. 新增 CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-self-reflection` | `True` | 启用 Reflector Self-Reflection（两轮 API 调用，默认开启） |
| `--no-self-reflection` | - | 禁用 Self-Reflection（单轮模式，节省 API 成本） |
| `--no-classifier-cap` | - | 禁用分类器封顶，完全信任 Reflector 判断 |

### 6. API 成本影响

| 模式 | API 调用次数/图 | 适用场景 |
|------|-----------------|----------|
| 默认（Self-Reflection ON） | 2× Reflector | 高质量评估，默认推荐 |
| `--no-self-reflection` | 1× Reflector | 快速调试/低成本场景 |

### 7. 更新后的使用示例

```bash
# 完整流水线 + Self-Reflection（默认）
python run.py --mode async --step 1234 --limit 100 \
  --api-concurrency 5 --temp-reflector 0.5

# 禁用 Self-Reflection（快速模式）
python run.py --mode async --step 1234 --limit 100 \
  --no-self-reflection

# 禁用分类器封顶（完全信任 LLM 判断）
python run.py --mode async --step 1234 --limit 100 \
  --no-classifier-cap

# 同时禁用 Self-Reflection 和分类器封顶
python run.py --mode sync --step 1234 --image-id IMG_001 \
  --no-self-reflection --no-classifier-cap
```

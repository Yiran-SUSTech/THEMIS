# T2I Agentic 测评系统方案

## 一、背景与动机

### 1.1 现有系统对比

| 维度 | GenEval2 (T2I) | THEMIS C2I |
|------|----------------|------------|
| 测评对象 | Text-to-Image（复杂 prompt → 图片） | Class-to-Image（ImageNet 类别 → 图片） |
| 测评方法 | 单一 VQA 模型（Qwen3-VL-8B）回答原子问题 | Agentic 多角色协作（Router→Judge→Expert→Reflector） |
| 先验知识 | 无 taxonomy，仅依赖 prompt 分解的原子 QA | ImageNet taxonomy enriched_description + diagnostic_checkpoints |
| 评测维度 | 原子级 QA 准确率（count/attribute/object/position/verb） | alignment_score（类别一致性）+ authenticity_score（图片真实性/质量） |
| 专家模型 | 无 | 7 个专家（分类器、检测器、深度估计、姿态估计等） |
| 打分方式 | Soft-TIFA（VQA 模型对正确答案的概率） | 0-5 连续评分（Reflector 综合专家证据） |
| 组合性分析 | 按 atom_count 和 skill 分组统计 | 无（c2i 每张图只有一个类别） |

### 1.2 核心问题

GenEval2 的 Soft-TIFA 方法存在以下局限：

1. **单一模型依赖**：全部评测依赖一个 VQA 模型，无交叉验证，容易产生系统性偏差
2. **无 taxonomy 先验**：对于 "golden retriever" 这类具体物种，只问 "Are there any dogs?"，不检查物种级诊断特征（毛色、体型、耳形等）
3. **无 artifact 检测**：只评估 prompt 对齐性，不评估图片本身的生成质量（伪影、结构崩塌等）
4. **无 Agentic 推理**：VQA 问题是固定的，不根据图片内容动态调整评测策略
5. **缺少专家工具链**：没有深度估计、姿态检测、分割等专业工具的支撑

### 1.3 设计目标

构建一个 **T2I Agentic 测评系统**，融合 GenEval2 的原子 QA 分解和 THEMIS 的 taxonomy 先验知识 + Agentic 框架，实现：

- **Prompt 级对齐评测**：评估生成图片与 text prompt 的语义对齐度（继承 GenEval2 的原子分解思路）
- **Object 级 taxonomy 评测**：对 prompt 中出现的每个物体，利用 ImageNet taxonomy 的 diagnostic_checkpoints 做物种级特征验证（继承 THEMIS 的 taxonomy 先验）
- **Artifact 级质量评测**：检测 AI 生成伪影，评估图片真实性（继承 THEMIS 的专家工具链）
- **组合性分析**：按 atom_count 和 skill 分组统计（继承 GenEval2 的分析维度）

---

## 二、系统架构

### 2.1 整体流程

```
Text Prompt (e.g., "four brown monkeys and a metal bicycle")
    │
    ▼
┌─────────────────────────────────────────────┐
│  Step 0: Prompt Atomization & Taxonomy Link │
│  - 分解 prompt 为原子 QA (GenEval2 风格)      │
│  - 识别 prompt 中的物体，链接到 ImageNet 类别  │
│  - 为每个物体加载 diagnostic_checkpoints      │
│  - 输出: enriched_atoms + taxonomy_context    │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Step 1: Router                              │
│  - 分析图片：物体检测、属性识别、关系判断       │
│  - 对照原子 QA 清单，初步回答每个原子问题       │
│  - 对照 taxonomy checkpoints，做物体级诊断     │
│  - 检测 artifact（伪影）                      │
│  - 选择专家并分配权重                          │
│  - 输出: evaluation_plan (JSON)              │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Step 2: Judge                               │
│  - 审核 Router 的计划合理性                    │
│  - 检查原子 QA 是否有遗漏                      │
│  - 检查专家选择是否合理                        │
│  - 输出: approve / reject + feedback          │
└─────────────────────────────────────────────┘
    │ (reject → 回到 Step 1 修订)
    ▼
┌─────────────────────────────────────────────┐
│  Step 3: Expert Execution                    │
│  - fine_grained_classifier: 验证每个物体类别   │
│  - open_vocabulary_detector: 检测物体位置/数量 │
│  - animal_pose_auditor: 姿态结构审计           │
│  - geometric_depth_auditor: 深度/空间一致性    │
│  - topology_boundary_auditor: 边界完整性       │
│  - perceptual_quality_auditor: 感知质量        │
│  - image_text_auditor: 文字检测（如有）         │
│  - 输出: expert_testimonies (JSON)            │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Step 4: Reflector                           │
│  - 综合 Router 观察 + 专家证据                 │
│  - 对每个原子 QA 给出最终判决（含 taxonomy      │
│    checklist 验证）                            │
│  - 产出最终评分:                               │
│    - alignment_score (0-5)                   │
│    - authenticity_score (0-5)                │
│  - 输出: final_report (JSON)                  │
└─────────────────────────────────────────────┘
```

### 2.2 与 C2I 系统的关键差异

| 维度 | C2I 系统 | T2I 系统（本方案） |
|------|---------|-------------------|
| 输入 | class_id (int) → 单一类别 | text prompt (str) → 复杂描述 |
| 物体数量 | 1 个（class subject） | 1-N 个（prompt 中提及的所有物体） |
| Taxonomy 链接 | class_id 直接索引 | 需要从 prompt 中提取物体名，映射到 ImageNet 类别 |
| 对齐评测维度 | 类别一致性 (alignment_score) | 原子 QA 对齐 + 物体级 taxonomy 对齐 |
| 评分体系 | alignment_score + artifact_score (各 0-5) | alignment_score + authenticity_score (各 0-5) |
| 组合性 | 无 | 按 atom_count 和 skill 分组分析 |

---

## 三、核心模块详细设计

### 3.1 Step 0: Prompt Atomization & Taxonomy Link

这是 T2I 系统独有的新模块，负责将 text prompt 分解为可评测的原子，并链接到 taxonomy 先验知识。

#### 3.1.1 Prompt 原子分解

采用 GenEval2 的分解方式，将 prompt 拆解为原子 QA 对：

```
Prompt: "four brown monkeys and a metal bicycle"

Atoms:
  1. {"question": "How many monkeys are in the image?", "answer": "four", "skill": "count", "target_object": "monkey"}
  2. {"question": "Are the monkeys brown?", "answer": "Yes", "skill": "attribute", "target_object": "monkey"}
  3. {"question": "Are there any monkeys in the image?", "answer": "Yes", "skill": "object", "target_object": "monkey"}
  4. {"question": "How many bicycles are in the image?", "answer": "one", "skill": "count", "target_object": "bicycle"}
  5. {"question": "Is the bicycle metal?", "answer": "Yes", "skill": "attribute", "target_object": "bicycle"}
  6. {"question": "Are there any bicycles in the image?", "answer": "Yes", "skill": "object", "target_object": "bicycle"}
```

**实现方式**：两种选项

- **选项 A（静态预生成）**：复用 GenEval2 已有的 `geneval2_data.jsonl`，其中 800 条 prompt 已预分解好原子 QA。直接加载使用。
- **选项 B（动态生成）**：用 LLM 实时分解任意 prompt。适合自定义 prompt 集。推荐用 Qwen3.6-plus 做分解，输出 JSON 格式的原子列表。

**推荐方案**：选项 A 为主（复用 GenEval2 数据），选项 B 为辅（支持自定义 prompt）。

#### 3.1.2 Taxonomy 链接

对 prompt 中识别出的每个物体，映射到 ImageNet 类别，加载对应的 diagnostic_checkpoints：

```
物体提取: "monkey" → ImageNet 搜索 → class_id=376 ("guenon, guenon monkey")
                           → enriched_description: "Medium-sized Old World monkey..."
                           → diagnostic_checkpoints:
                             Head_and_Face: ["Large, forward-facing eyes...", ...]
                             Body_and_Plumage: ["Fur is short and dense...", ...]
                             Limbs_and_Anatomy: ["Long limbs adapted for...", ...]

物体提取: "bicycle" → ImageNet 搜索 → class_id=444 ("bicycle, bike")
                          → enriched_description: "Human-powered vehicle..."
                          → diagnostic_checkpoints:
                            (如有; 否则标记为 "no taxonomy prior available")
```

**实现方式**：

1. 构建 `object_to_classid` 映射表（从 ImageNet 1000 类的 class_name 和 class_display_name 中提取同义词）
2. 对 prompt 中的每个物体名词，查找最匹配的 ImageNet 类别
3. 加载该类别的 `taxonomy_enriched_Batch_X.json` 和 `taxonomy_enriched_Batch_X_structured.json`
4. 如果找不到精确匹配，使用 superclass 级别的先验知识（如 "monkey" → primate superclass）
5. 如果完全无法匹配（如 "croissant" = class_id=926），仍尝试加载 taxonomy
6. 如果物体不在 ImageNet 中（如 "sparkling"），跳过 taxonomy 链接，仅做原子 QA 评测

#### 3.1.3 输出格式

```json
{
  "prompt": "four brown monkeys and a metal bicycle",
  "atom_count": 6,
  "atoms": [
    {
      "question": "How many monkeys are in the image?",
      "answer": "four",
      "answer_type": "count",
      "skill": "count",
      "target_object": "monkey",
      "weight": 1.0
    },
    ...
  ],
  "objects": [
    {
      "object_name": "monkey",
      "class_id": 376,
      "class_name": "guenon",
      "taxonomy_description": "Medium-sized Old World monkey...",
      "diagnostic_checkpoints": {
        "Head_and_Face": [...],
        "Body_and_Plumage": [...],
        "Limbs_and_Anatomy": [...]
      },
      "attributes_from_prompt": ["brown"],
      "count_from_prompt": 4
    },
    {
      "object_name": "bicycle",
      "class_id": 444,
      "class_name": "bicycle",
      "taxonomy_description": "...",
      "diagnostic_checkpoints": {...},
      "attributes_from_prompt": ["metal"],
      "count_from_prompt": 1
    }
  ]
}
```

### 3.2 Step 1: Router

Router 的职责是分析图片，对照原子 QA 和 taxonomy checkpoints 做初步评测，并选择专家。

#### 3.2.1 Router System Message

```
You are a Router Agent for Text-to-Image (T2I) evaluation. You analyze a generated
image against a text prompt that has been decomposed into atomic QA pairs and
enriched with taxonomy prior knowledge for each object.

Your tasks:
1. **Object Detection & Counting**: For each object in the prompt, identify its
   presence and count in the image.
2. **Attribute Verification**: For each attribute mentioned in the prompt (color,
   material, pattern), verify it matches the generated image.
3. **Relation Verification**: For spatial/positional relations (in front of, behind,
   on top of, to the left of), verify the spatial arrangement.
4. **Action Verification**: For verbs (chasing, playing with, jumping over),
   verify the action is depicted.
5. **Taxonomy Checkpoint Verification**: For each object with taxonomy prior,
   verify the diagnostic checkpoints (same as C2I Router).
6. **Artifact Detection**: Scan for AI-generation artifacts (same as C2I Router).
7. **Expert Selection**: Select experts based on visible entities and artifact risks.

[Expert Registry Summary - all 7 experts]
```

#### 3.2.2 Router User Prompt

```
Analyze the provided image against the text prompt and its atomic decomposition.

**[Text Prompt]**
four brown monkeys and a metal bicycle

**[Atomic QA Pairs]**
1. How many monkeys are in the image? → Expected: four (skill: count)
2. Are the monkeys brown? → Expected: Yes (skill: attribute)
3. Are there any monkeys in the image? → Expected: Yes (skill: object)
4. How many bicycles are in the image? → Expected: one (skill: count)
5. Is the bicycle metal? → Expected: Yes (skill: attribute)
6. Are there any bicycles in the image? → Expected: Yes (skill: object)

**[Taxonomy Prior Knowledge]**
Object 1: monkey (class: guenon)
  Taxonomy Description: Medium-sized Old World monkey...
  Diagnostic Checkpoints:
    Head_and_Face: [...]
    Body_and_Plumage: [...]
    Limbs_and_Anatomy: [...]

Object 2: bicycle (class: bicycle)
  Taxonomy Description: ...
  Diagnostic Checkpoints: ...

**[Output JSON Schema]**
{
  "image_description": "...",
  "atom_verdicts": [
    {"atom_index": 0, "question": "...", "expected": "...", "predicted": "...",
     "is_correct": true, "confidence": 0.9, "reasoning": "..."}
  ],
  "checkpoint_verdicts": [
    {"object": "monkey", "checkpoint": "...", "category": "...",
     "is_testable": true, "is_present": true, "reasoning": "..."}
  ],
  "artifact_observations": [...],
  "selected_experts": [...],
  "focus_areas": [...]
}
```

#### 3.2.3 关键设计决策

- **原子 QA 预判**：Router 对每个原子问题给出初步回答（predicted answer），而非等到 Reflector 阶段才用 VQA 模型回答。这样可以与专家证据交叉验证。
- **Taxonomy checkpoint 仅用于有先验知识的物体**：对于无法映射到 ImageNet 的物体（如 "sparkling"），跳过 taxonomy 评测。
- **专家选择支持多物体**：`fine_grained_classifier` 可以对多个物体分别运行；`animal_pose_auditor` 仅对有四肢的动物运行。

### 3.3 Step 2: Judge

Judge 的职责与 C2I 系统类似，但审核维度增加了原子 QA 覆盖性检查。

#### 3.3.1 Judge 审核维度

1. **原子 QA 覆盖性**：Router 是否回答了所有原子问题？是否有遗漏？
2. **Taxonomy Checkpoint 合理性**：Router 的 is_testable/is_present 判断是否合理？
3. **Artifact 观察**：严重度评级是否一致？是否有遗漏的伪影？
4. **专家选择**：每个专家的 target_subject 是否兼容？关键专家是否缺失？
5. **权重分配**：主物体的专家权重是否高于辅助物体？

### 3.4 Step 3: Expert Execution

专家模型复用 C2I 系统的 7 个专家，但需要适配多物体场景：

| 专家 | C2I 用法 | T2I 适配 |
|------|---------|---------|
| `fine_grained_classifier` | 对 1 个 class subject 运行 | 对 prompt 中的每个物体分别运行，返回 Top-5 预测 |
| `open_vocabulary_detector` | 检测 class subject | 检测 prompt 中所有物体，验证数量和位置 |
| `animal_pose_auditor` | 对 class subject 运行 | 仅对 prompt 中的动物物体运行 |
| `geometric_depth_auditor` | 全图深度估计 | 不变（全图分析） |
| `topology_boundary_auditor` | 对 class subject 分割 | 对 prompt 中的每个物体分别分割 |
| `perceptual_quality_auditor` | 全图质量评估 | 不变（全图分析） |
| `image_text_auditor` | 检测文字 | 不变（仅 prompt 要求文字时激活） |

**新增专家（可选）**：

- `counting_verifier`：基于检测器的 BBox 数量，精确验证 count 类原子 QA。当前可以通过 `open_vocabulary_detector` 间接实现。

### 3.5 Step 4: Reflector

Reflector 的职责是综合所有证据，产出最终评分。与 C2I 系统一致，每张图给出**两个分数**：alignment_score 和 authenticity_score。

#### 3.5.1 评分体系

| 评分维度 | 范围 | 计算方式 | 说明 |
|----------|------|----------|------|
| `alignment_score` | 0-5 | `mean(per_atom_scores) × 5` | 图片与 text prompt 的对齐度（含原子 QA 准确性 + taxonomy checklist 符合度） |
| `authenticity_score` | 0-5 | Reflector 综合判断 | 图片真实性/质量（伪影越少分越高，与 C2I 的 artifact 评估方式一致） |
| `per_atom_scores` | 0-1 (每个原子) | Reflector 对每个原子 QA 给出 | 原子级对齐分数（兼容 GenEval2 的 Soft-TIFA），同时融合 taxonomy checklist 验证 |

#### 3.5.2 per_atom_score 如何打出：原子 QA + Taxonomy Checklist 融合

每个原子的分数**不是单纯的 QA 对错**，而是融合了两层验证：

**第一层：原子 QA 准确性**
- Reflector 对照原子的 question 和 expected answer，结合 Router 预判和专家证据，判断答案是否正确
- 例如 atom 0: "How many monkeys?" → expected: "four"，专家检测器检测到 4 只 → QA 正确

**第二层：Taxonomy Checklist 验证**（仅对有 taxonomy 先验的物体的原子）
- 对于涉及物体的原子（如 object、count、attribute 类），Reflector 同时验证该物体是否符合 taxonomy diagnostic_checkpoints
- 例如 atom 2: "Are there any monkeys?" → 不仅检查"是否有猴子"，还要验证检测到的物体**确实是猴子**（而非猩猩或其他灵长类），依据是 taxonomy 中 monkey 的 diagnostic_checkpoints（如 "Large forward-facing eyes with distinctive brow ridges"、"Long limbs with grasping hands and feet" 等）以及 fine_grained_classifier 的 Top-K 预测

**融合逻辑**：

```
对于每个 atom:
  1. QA 正确性判断: expected answer 与实际是否一致 → qa_score (0-1)
  2. Taxonomy 验证（如有先验）: 物体是否符合 diagnostic_checkpoints → tax_score (0-1)
  3. 最终 atom_score = qa_score × tax_score（如果 taxonomy 不适用，则 atom_score = qa_score）
```

**具体例子**：

```
atom 0: "How many monkeys?" → expected: "four"
  - QA 层: 专家检测器检测到 4 只 → qa_score = 1.0
  - Taxonomy 层: fine_grained_classifier Top-1 = "guenon" (prob=0.72)，
    Router/Reflector 通过 checklist 验证猴子的面部特征、毛发、四肢均符合 → tax_score = 0.9
  - atom_score = 1.0 × 0.9 = 0.9

atom 3: "How many bicycles?" → expected: "one"
  - QA 层: 专家检测器检测到 2 辆（多生成了一辆）→ qa_score = 0.0
  - Taxonomy 层: 检测到的物体确实是自行车 → tax_score = 1.0
  - atom_score = 0.0 × 1.0 = 0.0（QA 错误直接归零）

atom 1: "Are the monkeys brown?" → expected: "Yes"
  - QA 层: Router 观察到猴子是棕色的 → qa_score = 1.0
  - Taxonomy 层: 棕色符合 guenon 的 "Fur is short and dense, typically brown or grey" → tax_score = 1.0
  - atom_score = 1.0 × 1.0 = 1.0

atom 5: "Are there any bicycles?" → expected: "Yes"
  - QA 层: 检测器确认有自行车 → qa_score = 1.0
  - Taxonomy 层: 无 taxonomy 先验（bicycle 的 checkpoint 不涉及颜色/材质）→ tax_score 不适用
  - atom_score = 1.0（仅看 QA）
```

#### 3.5.3 alignment_score 计算

```python
alignment_score = mean(per_atom_scores) * 5.0
```

示例：6 个原子的 per_atom_scores = [0.9, 1.0, 1.0, 0.0, 0.8, 1.0]
→ mean = 0.783 → alignment_score = 0.783 × 5.0 = **3.92**

#### 3.5.4 authenticity_score 计算

与 C2I 的 artifact 评估方式**完全一致**，由 Reflector 综合以下信息判断（分数越高表示图片质量越好、伪影越少）：

- Router 的 artifact_observations（伪影类型、位置、严重度）
- 专家证据（pose auditor 的低置信关键点、depth auditor 的深度不一致、topology auditor 的边界融合等）
- Reflector 自己的视觉观察

打分逻辑：5.0 = 无伪影，严重结构崩塌 → ≤1.0，多个小问题叠加扣分。

#### 3.5.5 Reflector System Message

```
You are the Reflector of a T2I image evaluation system. Review the Router's
assessment, expert evidence, and atomic QA verdicts, then produce the final
evaluation. Output JSON only.

**Core Principles:**
1. For each atom, evaluate TWO layers:
   a) QA correctness: Does the image answer the atom's question correctly?
   b) Taxonomy conformance (if applicable): Does the detected object match the
      taxonomy diagnostic_checkpoints for its class?
   The final atom_score = qa_score × tax_score. If taxonomy is not applicable,
   atom_score = qa_score.
2. Expert detector/classifier hard data is more reliable than Router's visual
   impression for object presence and counting.
3. For attribute verification: Router's visual observation is primary; experts
   are supplementary.
4. For authenticity: Router's direct visual observation is primary; experts are
   supplementary. Expert silence does NOT override Router's findings.
5. Be critical — do NOT rubber-stamp the Router's assessment.

**Scoring:**
- per_atom_scores: For each atomic QA, assign 0.0-1.0 based on:
  - qa_score (0-1): Is the answer correct?
  - tax_score (0-1): Does the object conform to taxonomy checkpoints?
  - atom_score = qa_score × tax_score (or qa_score if no taxonomy)
- alignment_score: mean(per_atom_scores) × 5.0
- authenticity_score (0-5): How authentic is the image? Higher = fewer artifacts,
  better quality. A flawless image scores 5.0. Structural defects severely
  reduce this score.

**Output JSON:**
{
  "atom_reviews": [
    {"atom_index": 0, "question": "...", "expected": "...",
     "qa_score": 1.0, "tax_score": 0.9, "atom_score": 0.9,
     "expert_evidence": "...", "reasoning": "..."}
  ],
  "artifact_review": "...",
  "alignment_score": 0.0,
  "authenticity_score": 0.0,
  "per_atom_scores": [0.9, 1.0, 1.0, 0.0, 0.8, 1.0],
  "key_defects": ["..."]
}
```

#### 3.5.6 兼容 GenEval2 分析

由于 `per_atom_scores` 与 GenEval2 的 Soft-TIFA 分数一一对应，可以直接复用 GenEval2 的分析脚本：

- **Per-skill analysis**：按 skill 分组统计 per_atom_scores 的平均值
- **Per-atomicity analysis**：按 atom_count 分组统计 gmean(per_atom_scores)
- **Overall AM/GM**：全数据集的 mean/gmean

---

## 四、数据流与文件结构

### 4.1 输入数据

```
D:\THEMIS\
├── taxonomy_info/                    # 已有，复用
│   └── taxonomy_enriched_Batch_*.json
├── taxonomy_info_structural/         # 已有，复用
│   └── taxonomy_enriched_Batch_*_structured.json
├── expert_registry.json              # 已有，复用
├── c2i_harness/                      # 已有 C2I 系统
└── t2i_harness/                      # 新建 T2I 系统
    ├── run.py
    ├── common.py                     # 复用 C2I 的通用工具
    ├── step0_atomize.py              # 新增：Prompt 分解 + Taxonomy 链接
    ├── step1_router.py               # 修改：适配多物体 + 原子 QA
    ├── step2_judge.py                # 修改：增加原子 QA 覆盖性检查
    ├── step3_execute.py              # 修改：适配多物体专家执行
    ├── step4_reflector.py            # 修改：增加 per_atom_scores 评分
    ├── dispatch_async.py             # 修改：适配新流程
    ├── dispatch_sync.py              # 修改：适配新流程
    ├── object_to_classid.json        # 新增：物体名→ImageNet class_id 映射
    └── geneval2_data.jsonl           # 复制自 GenEval2
```

### 4.2 物体→类别映射构建

从 ImageNet 1000 类中提取所有 class_name 和 class_display_name 的同义词，构建映射表：

```python
# object_to_classid.json 示例
{
  "monkey": 376,          # guenon
  "baboon": 377,
  "gorilla": 366,
  "chimpanzee": 367,
  "dog": [151, 152, ...],  # 多个犬种，取最通用的
  "bicycle": 444,
  "car": 468,              # cab → car? 需要同义词扩展
  "elephant": 386,
  "bird": 7,               # cock → bird? 使用 superclass
  "fish": 0,               # tench → fish? 使用 superclass
  ...
}
```

对于无法精确匹配的物体，按以下优先级回退：
1. 精确匹配 class_name
2. 匹配 class_display_name 中的同义词
3. 匹配 superclass（如 "monkey" → primate superclass）
4. 标记为 "no taxonomy prior"（跳过 taxonomy 评测，仅做原子 QA）

### 4.3 输出数据

```
output_t2i_<timestamp>/
├── atomized_prompts/         # Step 0 输出
│   └── atoms_<prompt_id>.json
├── plans/                    # Step 1 输出
│   └── plan_<prompt_id>.json
├── judge_feedback/           # Step 2 输出
│   └── feedback_<prompt_id>_iter<N>.json
├── expert_results/           # Step 3 输出
│   └── testimony_<prompt_id>.json
├── final_reports/            # Step 4 输出
│   └── report_<prompt_id>.json
└── summary.json              # 全局统计
    ├── overall_alignment: 3.92
    ├── overall_authenticity: 3.50
    ├── soft_tifa_am: 78.3
    ├── soft_tifa_gm: 72.1
    ├── per_skill: {"count": 0.78, "attribute": 0.65, ...}
    └── per_atomicity: {"3": 0.82, "5": 0.71, "7": 0.63, "10": 0.51}
```

---

## 五、Harness Engineering 应用

T2I 系统继承 C2I harness 的全部工程原则，并针对 T2I 场景做适配：

| 原则 | C2I 应用 | T2I 适配 |
|------|---------|---------|
| 角色隔离 | 独立 system message | 不变 |
| 模型分离 | Router/Judge=qwen3.6-plus, Reflector=qwen3.7-plus | 不变 |
| 确定性可控 | --temp-router/judge/reflector | 不变 |
| 上下文最小化 | 每次调用只传必要信息 | 不变，但 user prompt 增加 atoms + taxonomy context |
| 无状态调用 | 移除 session 模式 | 不变 |
| 可复现性 | 固定模型 + 温度 + 持久化 | 不变，增加 atomized_prompts 持久化 |

### 新增 Harness 原则

| 原则 | 说明 |
|------|------|
| **原子可追溯性** | 每个 per_atom_score 可追溯到 qa_score、tax_score、Router 预判、专家证据、Reflector 判决 |
| **Taxonomy 透明性** | 每个 atom 的 tax_score 可追溯到具体的 diagnostic_checkpoints 验证结果 |
| **双轨评分兼容** | per_atom_scores 兼容 GenEval2 的 Soft-TIFA，可直接对比 |

---

## 六、实施计划

### Phase 1: 基础框架搭建

1. 创建 `t2i_harness/` 文件夹，复制 `c2i_harness/` 的核心文件
2. 构建 `object_to_classid.json` 映射表
3. 实现 `step0_atomize.py`：加载 GenEval2 数据 + taxonomy 链接
4. 修改 `run.py`：增加 `--benchmark-data` 参数指向 GenEval2 数据

### Phase 2: 核心模块改造

5. 修改 `step1_router.py`：适配多物体 + 原子 QA 预判
6. 修改 `step2_judge.py`：增加原子 QA 覆盖性审核
7. 修改 `step3_execute.py`：多物体专家执行（分类器多目标、检测器多类）
8. 修改 `step4_reflector.py`：实现 per_atom_scores（qa_score × tax_score 融合）+ alignment_score（mean × 5）+ authenticity_score

### Phase 3: 调度与集成

9. 修改 `dispatch_async.py` 和 `dispatch_sync.py`：适配 Step 0 → 1 → 2 → 3 → 4 流程
10. 实现兼容 GenEval2 的分析脚本（per-skill, per-atomicity 统计）

### Phase 4: 测试与验证

11. 用 GenEval2 的低组合性 prompt（atom_count=3）做单图测试
12. 用高组合性 prompt（atom_count=10）做对比测试
13. 与 GenEval2 原始 Soft-TIFA 分数做相关性分析
14. 撰写实验报告

---

## 七、具体示例

### 完整流程示例

**输入**:
- Prompt: `"four brown monkeys and a metal bicycle"`
- 图片: AI 生成的图片

**Step 0 输出**:
```json
{
  "prompt": "four brown monkeys and a metal bicycle",
  "atom_count": 6,
  "atoms": [
    {"question": "How many monkeys are in the image?", "answer": "four", "skill": "count", "target_object": "monkey"},
    {"question": "Are the monkeys brown?", "answer": "Yes", "skill": "attribute", "target_object": "monkey"},
    {"question": "Are there any monkeys in the image?", "answer": "Yes", "skill": "object", "target_object": "monkey"},
    {"question": "How many bicycles are in the image?", "answer": "one", "skill": "count", "target_object": "bicycle"},
    {"question": "Is the bicycle metal?", "answer": "Yes", "skill": "attribute", "target_object": "bicycle"},
    {"question": "Are there any bicycles in the image?", "answer": "Yes", "skill": "object", "target_object": "bicycle"}
  ],
  "objects": [
    {
      "object_name": "monkey",
      "class_id": 376,
      "class_name": "guenon",
      "taxonomy_description": "Medium-sized Old World monkey with distinctive facial markings...",
      "diagnostic_checkpoints": {
        "Head_and_Face": ["Large forward-facing eyes with distinctive brow ridges.", ...],
        "Body_and_Plumage": ["Fur is short and dense, typically brown or grey.", ...],
        "Limbs_and_Anatomy": ["Long limbs with grasping hands and feet.", ...]
      },
      "attributes_from_prompt": ["brown"],
      "count_from_prompt": 4
    },
    {
      "object_name": "bicycle",
      "class_id": 444,
      "class_name": "bicycle",
      "taxonomy_description": "Human-powered vehicle with two wheels...",
      "diagnostic_checkpoints": {...},
      "attributes_from_prompt": ["metal"],
      "count_from_prompt": 1
    }
  ]
}
```

**Step 1 (Router) 输出摘要**:
```json
{
  "image_description": "Four brown monkeys sitting near a metal bicycle in a grassy field.",
  "atom_verdicts": [
    {"atom_index": 0, "predicted": "four", "is_correct": true, "confidence": 0.85},
    {"atom_index": 1, "predicted": "Yes", "is_correct": true, "confidence": 0.90},
    {"atom_index": 2, "predicted": "Yes", "is_correct": true, "confidence": 0.95},
    {"atom_index": 3, "predicted": "one", "is_correct": true, "confidence": 0.90},
    {"atom_index": 4, "predicted": "Yes", "is_correct": true, "confidence": 0.80},
    {"atom_index": 5, "predicted": "Yes", "is_correct": true, "confidence": 0.95}
  ],
  "checkpoint_verdicts": [
    {"object": "monkey", "checkpoint": "Large forward-facing eyes...", "is_present": true, ...},
    {"object": "monkey", "checkpoint": "Fur is short and dense...", "is_present": true, ...},
    ...
  ],
  "artifact_observations": [
    {"artifact_type": "melting", "location": "monkey #3 right hand", "severity": 2.0}
  ],
  "selected_experts": [
    {"expert_name": "fine_grained_classifier", "target_subject": "monkey", "weight": 0.30},
    {"expert_name": "fine_grained_classifier", "target_subject": "bicycle", "weight": 0.20},
    {"expert_name": "open_vocabulary_detector", "target_subject": "all", "weight": 0.20},
    {"expert_name": "animal_pose_auditor", "target_subject": "monkey", "weight": 0.15},
    {"expert_name": "perceptual_quality_auditor", "target_subject": "scene", "weight": 0.15}
  ]
}
```

**Step 3 (Expert) 输出摘要**:
```json
{
  "fine_grained_classifier": {
    "monkey": {"top_5": ["guenon", "baboon", "gorilla", "macaque", "chimp"], "top_1_prob": 0.72},
    "bicycle": {"top_5": ["bicycle", "mountain bike", "moped", "unicycle", "tricycle"], "top_1_prob": 0.91}
  },
  "open_vocabulary_detector": {
    "monkey": {"count": 4, "boxes": [...]},
    "bicycle": {"count": 1, "boxes": [...]}
  },
  "animal_pose_auditor": {
    "monkey": {"low_confidence_ratio": 0.28, "artifact_risk_level": "MEDIUM", ...}
  }
}
```

**Step 4 (Reflector) 最终输出**:
```json
{
  "atom_reviews": [
    {"atom_index": 0, "question": "How many monkeys?", "expected": "four",
     "qa_score": 1.0, "tax_score": 0.9, "atom_score": 0.9,
     "reasoning": "Detector confirms 4 monkeys. Classifier Top-1=guenon (0.72). Checkpoints: forward-facing eyes ✓, brown fur ✓, long limbs ✓."},
    {"atom_index": 1, "question": "Are the monkeys brown?", "expected": "Yes",
     "qa_score": 1.0, "tax_score": 1.0, "atom_score": 1.0,
     "reasoning": "Monkeys appear brown. Consistent with guenon taxonomy: 'Fur typically brown or grey'."},
    {"atom_index": 2, "question": "Are there any monkeys?", "expected": "Yes",
     "qa_score": 1.0, "tax_score": 0.9, "atom_score": 0.9,
     "reasoning": "Monkeys clearly present. Classifier confirms species-level match."},
    {"atom_index": 3, "question": "How many bicycles?", "expected": "one",
     "qa_score": 0.0, "tax_score": 1.0, "atom_score": 0.0,
     "reasoning": "Detector found 2 bicycles (expected 1). Objects are correctly bicycles, but count is wrong."},
    {"atom_index": 4, "question": "Is the bicycle metal?", "expected": "Yes",
     "qa_score": 0.8, "tax_score": 1.0, "atom_score": 0.8,
     "reasoning": "Bicycle appears metallic but finish is slightly off. No taxonomy applicable for material."},
    {"atom_index": 5, "question": "Are there any bicycles?", "expected": "Yes",
     "qa_score": 1.0, "tax_score": 1.0, "atom_score": 1.0,
     "reasoning": "Bicycle clearly present. No taxonomy verification needed."}
  ],
  "alignment_score": 3.67,
  "authenticity_score": 3.5,
  "per_atom_scores": [0.9, 1.0, 0.9, 0.0, 0.8, 1.0],
  "soft_tifa_am": 76.7,
  "soft_tifa_gm": 0.0,
  "key_defects": ["Slight melting on monkey #3 right hand", "Extra bicycle generated (expected 1, found 2)"]
}
```

> 注：`soft_tifa_gm` 为 0.0 是因为 atom 3 的 score 为 0.0，几何平均对零值敏感。这是与 GenEval2 原始 Soft-TIFA 一致的行为。

---

## 八、与 GenEval2 的兼容性

### 8.1 数据兼容

T2I 系统直接加载 `geneval2_data.jsonl`，复用其 prompt 和 vqa_list。每个 prompt 的 atoms 通过 `normalize_atoms()` 转换为统一的 atom 格式。

### 8.2 评分兼容

| GenEval2 指标 | T2I 系统对应 | 计算方式 |
|---------------|-------------|----------|
| Soft-TIFA AM | `mean(per_atom_scores) * 100` | 算术平均 |
| Soft-TIFA GM | `gmean(per_atom_scores) * 100` | 几何平均 |
| Per-skill accuracy | `per_skill_scores` | 按 skill 分组平均 |
| Per-atomicity accuracy | `per_atomicity_scores` | 按 atom_count 分组 GM |

### 8.3 增量价值

T2I 系统在 GenEval2 基础上增加的评测维度：

| 新增维度 | 来源 | 价值 |
|----------|------|------|
| `alignment_score` (0-5) | mean(per_atom_scores) × 5，per_atom_scores 融合 QA + taxonomy | 整体对齐度，比 GenEval2 的纯 QA 平均更全面（含物种级特征验证） |
| `authenticity_score` (0-5) | Router + 专家 + Reflector | 生成质量/真实性，GenEval2 完全不评估 |
| `per_atom_scores` 融合 taxonomy | qa_score × tax_score | 每个原子的分数不仅看 QA 对错，还验证物体是否符合 taxonomy 特征 |
| `per_atom_reasoning` | Reflector 推理链 | 可解释性，GenEval2 只有概率分数 |
| `expert_evidence` | 7 个专家模型 | 交叉验证，GenEval2 只用一个 VQA 模型 |

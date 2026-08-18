# T2I 评估系统流程改进方案

## 一、改进后的完整流水线

```
Step 0  Atomize + Taxonomy Preparation
  ├── 0a: 将 GenEval2 VQA 分解为结构化原子
  ├── 0b: 从原子中提取物体名
  ├── 0c: 为每个物体关联 Taxonomy
  │   ├── 具体类（有 class_id 映射）→ 加载已有 taxonomy_info + diagnostic_checkpoints
  │   └── 泛类（无 class_id 映射）→ 调用 LLM 生成通用诊断特征
  └── 0d: 输出完整的 atomized_data（所有物体都有 taxonomy checklist）

Step 1  Router（初步判定 + 证据驱动专家计划）
  ├── 1a: 原子 QA 初步判定（predicted, is_correct, confidence）
  ├── 1b: Taxonomy checkpoint 验证（is_testable, is_present）
  ├── 1c: 伪影检测（type, location, severity）
  └── 1d: 专家验证计划（哪些 checkpoint/atom 可以用专家求证 → 指定专家 + 验证目标）

Step 2  Judge（审核专家计划的完备性）
  ├── 审查：所有 checkpoint 和 atom 是否都有对应验证手段
  ├── 审查：专家分配是否合理（检测器用于计数，分类器用于物种，姿态用于四肢）
  └── 审查：权重分配是否合理

Step 3  Expert（执行专家计划，提供硬数据证据）

Step 4  Reflector（综合评分 + Self-Reflection）
  ├── Round 1：初步评分
  │   ├── 输入：Router 初步判定 + 专家硬数据 + 人类打分参考
  │   ├── 每个 atom：qa_score（专家验证后）× tax_score（专家验证后）
  │   ├── alignment_score = mean(per_atom_scores) × 5.0
  │   ├── authenticity_score（参考人类打分校准）
  │   └── 输出初步评分 + reasoning
  └── Round 2：Self-Reflection（自审修订）
      ├── 输入：Round 1 输出 + 原始上下文（图片 + 专家证据 + 参考图）
      ├── 审查：分数与 reasoning 是否一致、专家证据是否充分利用、
      │         参考校准是否合理、是否过于宽松/严格
      └── 输出修订后的最终评分 + self_reflection_notes
```

### 与当前流程的关键区别

| 维度 | 当前流程 | 改进流程 |
|------|---------|---------|
| Taxonomy 生成时机 | Router 内部处理（或不处理） | Step 0 统一准备，Router 之前完成 |
| 泛类处理 | 强行映射到某个 ImageNet 类或不加载 | LLM 生成通用诊断特征 |
| 专家选择依据 | 图中可见实体 + 硬编码规则 | Taxonomy checkpoint + 原子 QA 的验证需求 |
| Reflector 校准 | 无参考，自评 | 引入人类打分参考锚点 |
| Reflector Self-Reflection | 无（单次 API 调用） | 两轮 API 调用：初步评分 → 自审修订 |
| Router-Judge-Expert 信息流 | Router 选专家 → Judge 审 → Expert 跑 | Router 做初步判定 + 指定验证目标 → Judge 审验证覆盖 → Expert 针对性验证 |

---

## 二、逐文件修改方案

### 2.1 `step0_atomize.py` — 新增泛类 Taxonomy 生成

#### 新增函数：`generate_generic_taxonomy`

```python
def generate_generic_taxonomy(
    client: OpenAI,
    object_name: str,
    prompt_text: str,
    api_retry: int = 0,
    temperature: float = 0.0,
) -> dict:
    """为泛类物体生成通用诊断特征。

    返回格式与已有 diagnostic_checkpoints 结构一致：
    {
        "object_name": "monkey",
        "is_generic": True,
        "class_id": None,
        "class_name": "monkey (generic)",
        "taxonomy_description": "灵长类动物，面部扁平双眼朝前，有对生拇指...",
        "diagnostic_checkpoints": {
            "body_structure": "灵长类体型，四肢比例协调，躯干覆盖毛发",
            "facial_features": "双眼朝前，面部扁平，吻部不突出",
            "limbs": "四肢，有对生拇指可抓握",
            "covering": "体表毛发覆盖，无羽毛或鳞片"
        }
    }
    """
```

LLM Prompt 设计：
```
你是生物分类学专家。给定物体名称 "{object_name}"（来自 prompt: "{prompt_text}"），
生成简洁但能区分该大类与其他大类的诊断特征。

要求：
1. 3-5 个 checkpoint，每个一句话
2. 每个 checkpoint 必须是图片中可视觉验证的特征
3. 不要涉及物种级特征（如特定花纹、颊囊等）
4. 重点：能将该大类与其他易混淆大类区分开的特征
5. 输出 JSON：{"taxonomy_description": "一句话总述", "diagnostic_checkpoints": {"部位": "特征描述"}}
```

#### 新增函数：`enrich_with_generic_taxonomy`

```python
def enrich_with_generic_taxonomy(
    atomized_data: dict,
    client: OpenAI,
    api_retry: int = 0,
    temperature: float = 0.0,
) -> dict:
    """为 atomized_data 中没有 taxonomy 的泛类物体生成通用诊断特征。

    原地修改 atomized_data["objects"]，为每个 is_generic=True 的物体
    调用 generate_generic_taxonomy 补全 taxonomy 信息。
    """
    objects = atomized_data.get("objects", [])
    for obj in objects:
        if obj.get("is_generic", False) and not obj.get("diagnostic_checkpoints"):
            generated = generate_generic_taxonomy(
                client, obj["object_name"],
                atomized_data.get("prompt", ""),
                api_retry=api_retry,
                temperature=temperature,
            )
            obj.update(generated)
    return atomized_data
```

#### 修改 `atomize_prompt` 中的物体处理逻辑

当前逻辑（`link_taxonomy` 返回 None 时）：
```python
# 当前：创建空 taxonomy 的 fallback
obj_info = {
    "object_name": obj_name,
    "class_id": None,
    "class_name": "",
    "taxonomy_description": "",
    "diagnostic_checkpoints": {},
}
```

改为：
```python
# 改进：标记为泛类，等待 Step 0d 生成
obj_info = {
    "object_name": obj_name,
    "class_id": None,
    "is_generic": True,  # 新增标记
    "class_name": "",
    "taxonomy_description": "",
    "diagnostic_checkpoints": {},
}
```

对于有 class_id 的物体，添加：
```python
obj_info["is_generic"] = False
```

---

### 2.2 `step1_router.py` — 证据驱动的专家计划

#### 修改 `_COMMON_ROUTER_INSTRUCTIONS`

**Step 1-3（初步判定）保持不变**：原子 QA 判定、Taxonomy 验证、伪影检测。

**Step 4 重写为"专家验证计划"**：

```text
**Step 4 — Expert Verification Plan**
Based on your preliminary judgments in Steps 1-3, specify which points can be
verified by expert models. For each verification need:

a) For each Taxonomy checkpoint that you marked is_present=true or is_present=false:
   - Can an expert model provide hard evidence to confirm/deny your judgment?
   - If yes, assign the appropriate expert (see capability mapping below).

b) For each count-type atom (e.g., "How many monkeys?"):
   - Assign "open_vocabulary_detector" with the target_subject to verify count.

c) For each object-presence atom (e.g., "Are there monkeys?"):
   - Assign "open_vocabulary_detector" for detection evidence.
   - Assign "fine_grained_classifier" for classification evidence.

d) For each attribute atom (e.g., "Are the monkeys brown?"):
   - If the attribute is a color/material: "fine_grained_classifier" may help
     but your visual observation is primary. Only assign if classification
     labels contain color/material cues.
   - If no expert can verify: leave unassigned (VLM-only judgment).

e) For taxonomy checkpoints with no expert available:
   - Leave unassigned. Your visual observation is the primary evidence.

Expert Capability Mapping:
  - open_vocabulary_detector: Object detection + counting (bounding boxes)
  - fine_grained_classifier: ImageNet species classification (top-3 labels)
  - animal_pose_auditor: Limb/keypoint verification (limbed subjects only)
  - topology_boundary_auditor: Shape/contour verification (segmentation)
  - geometric_depth_auditor: Spatial relationship verification (depth map)
  - perceptual_quality_auditor: Artifact/distortion verification
  - image_text_auditor: Text verification (OCR)

Rules:
- Select 3-8 experts. Same expert_id may appear with different target_subjects.
- Each expert entry MUST specify verification_goals (list of checkpoint/atom IDs).
- "fine_grained_classifier" is recommended for each main object.
- "animal_pose_auditor" ONLY for limbed subjects (people, dogs, cats, etc.).
- "image_text_auditor" ONLY if text is visible.

**Step 5 — Weights** (same as before)
```

#### 修改输出 JSON Schema

```json
{
  "image_description": "str",
  "atom_verdicts": [
    {"atom_index": int, "question": "str", "expected": "str",
     "predicted": "str", "is_correct": bool, "confidence": float,
     "reasoning": "str"}
  ],
  "checkpoint_verdicts": [
    {"object": "str", "checkpoint": "str", "category": "str",
     "is_testable": bool, "is_present": bool, "reasoning": "str"}
  ],
  "artifact_observations": [
    {"artifact_type": "str", "location": "str", "severity": float,
     "reasoning": "str"}
  ],
  "expert_verification_plan": [
    {"expert_name": "str",
     "target_subject": "str",
     "verification_goals": ["str"],
     "reason": "str",
     "weight": float}
  ],
  "unverifiable_points": ["str"],
  "focus_areas": ["str"]
}
```

**新增字段说明**：
- `expert_verification_plan`：替换原来的 `selected_experts`，增加 `verification_goals` 字段，明确每个专家要验证什么
- `unverifiable_points`：列出无法用专家验证的 checkpoint/atom，标注为 VLM-only 判定

#### 修改 `validate_plan`

```python
# 验证 expert_verification_plan
if "expert_verification_plan" not in plan:
    print("  [WARN] Plan missing 'expert_verification_plan'")
    return False

for ev in plan["expert_verification_plan"]:
    if "expert_name" not in ev:
        return False
    if "verification_goals" not in ev or not ev["verification_goals"]:
        print(f"  [WARN] Expert '{ev.get('expert_name')}' has no verification_goals")
        return False
    if "target_subject" not in ev:
        return False
```

#### 兼容 `selected_experts`

在 `generate_plan` 返回前，将 `expert_verification_plan` 同时映射为 `selected_experts`（保留 `expert_name`、`target_subject`、`weight`），确保 Step 3 的 `execute_plan` 可以正常解析：

```python
plan["selected_experts"] = [
    {
        "expert_name": ev["expert_name"],
        "target_subject": ev["target_subject"],
        "reason": ev.get("reason", ""),
        "weight": ev.get("weight", 0.0),
    }
    for ev in plan.get("expert_verification_plan", [])
]
```

---

### 2.3 `step2_judge.py` — 审查验证覆盖度

#### 修改 Judge 指令

新增审查维度：

```text
**审查维度 4 — 专家验证覆盖度**
检查 Router 的 expert_verification_plan：
1. 所有 count 类型的原子是否都分配了 open_vocabulary_detector？
2. 所有 object-presence 原子是否有检测器或分类器覆盖？
3. 所有 is_present=false 的 taxonomy checkpoint，是否有专家可以提供反证？
4. verification_goals 是否与 checkpoint/atom 一一对应？
5. unverifiable_points 中的点是否确实无法用专家验证？

如果覆盖不完整，在 suggestions 中指出缺失的验证目标。
```

#### 修改输出 Schema

```json
{
  "is_approved": bool,
  "reasons_for_rejection": "str",
  "suggestions": [
    {"type": "missing_verification", "point": "str", "recommended_expert": "str"}
  ],
  "coverage_assessment": {
    "atoms_covered": int,
    "atoms_total": int,
    "checkpoints_covered": int,
    "checkpoints_total": int
  }
}
```

---

### 2.4 `step4_reflector.py` — 引入人类打分参考

#### 新增：参考标注加载

```python
T2I_REF_ANNOTATIONS_JSON = T2I_DIR / "t2i_ref_annotations.json"

@lru_cache(maxsize=1)
def _load_t2i_ref_annotations() -> dict:
    """加载 T2I 人类打分参考数据。

    格式：{image_name: {"prompt": str, "alignment_score": float, "authenticity_score": float}}
    """
    if not T2I_REF_ANNOTATIONS_JSON.exists():
        return {}
    with open(T2I_REF_ANNOTATIONS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)
```

#### 新增：参考图片选择

```python
def select_t2i_reference_images(
    image_dir: Path,
    prompt_text: str,
    num_refs: int = 3,
) -> list[dict]:
    """选择与当前 prompt 最相似的参考图片。

    选择策略：
    1. 从参考标注中筛选与 prompt 共享物体关键词的图片
    2. 按 alignment_score 分为低/中/高三段
    3. 每段选一张代表图，确保分数跨度覆盖 0-5
    4. 返回 [{image_name, image_path, alignment_score, authenticity_score, prompt}]
    """
```

#### 修改 `_REFLECTOR_SYSTEM_TEMPLATE`

新增人类参考校准指令：

```text
6. HUMAN REFERENCE CALIBRATION:
   - You will be provided with human-scored reference images.
   - Use these as calibration anchors: if the target image has similar quality
     to a reference with alignment_score=4.0, assign a similar alignment_score.
   - If the target image has similar artifact severity to a reference with
     authenticity_score=2.0, assign a similar authenticity_score.
   - References are anchors, not templates. Adjust based on the target's
     specific features and expert evidence.
```

#### 修改 `build_reflector_prompt`

新增参考图片上下文：

```python
def build_reflector_prompt(
    prompt_text: str,
    atomized_data: dict,
    expert_results_str: str,
    router_plan: dict | None = None,
    ref_images: list[dict] | None = None,  # 新增
) -> str:
    # ... 原有逻辑 ...

    # 新增：参考图片文本
    ref_text = ""
    if ref_images:
        ref_lines = []
        for ref in ref_images:
            ref_lines.append(
                f"  - Image: {ref['image_name']} | "
                f"Prompt: {ref.get('prompt', 'N/A')} | "
                f"Human Alignment: {ref['alignment_score']} | "
                f"Human Authenticity: {ref['authenticity_score']}"
            )
        ref_text = (
            "\n**[Human-Annotated Reference Images]**\n"
            "Compare the target image against these references. "
            "If the target has similar quality, assign a similar score.\n"
            + "\n".join(ref_lines)
        )

    # 在 prompt 末尾添加 ref_text
```

#### 修改 `run_reflector`

```python
def run_reflector(
    client: OpenAI,
    image_path: str,
    prompt_id: str,
    prompt_text: str,
    atomized_data: dict,
    expert_results: dict,
    experts_registry_str: str,
    router_plan: dict | None = None,
    ref_image_dir: Path | None = None,  # 新增
    api_retry: int = 0,
    temperature: float = 0.5,
) -> dict | None:
    # ... 原有逻辑 ...

    # 新增：选择参考图片
    ref_images = []
    if ref_image_dir:
        ref_images = select_t2i_reference_images(ref_image_dir, prompt_text)

    prompt = build_reflector_prompt(
        prompt_text, atomized_data, expert_results_str, router_plan,
        ref_images=ref_images,
    )

    # 新增：将参考图片追加到 user_content
    for ref in ref_images:
        try:
            ref_b64 = encode_image(ref["image_path"])
            user_content.append({
                "type": "text",
                "text": f"[Reference: {ref['image_name']} | "
                        f"Alignment={ref['alignment_score']} | "
                        f"Authenticity={ref['authenticity_score']}]",
            })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{ref_b64}"},
            })
        except Exception as e:
            print(f"  [WARN] Failed to load reference image {ref['image_path']}: {e}")

    # ... API 调用 ...
```

#### 新增：`t2i_ref_annotations.json` 格式

```json
{
  "ref_0.png": {
    "prompt": "four brown monkeys and a metal bicycle",
    "alignment_score": 4.5,
    "authenticity_score": 3.8
  },
  "ref_1.png": {
    "prompt": "a green backpack and a pig",
    "alignment_score": 2.0,
    "authenticity_score": 4.2
  }
}
```

---

### 2.4.1 `step4_reflector.py` — Self-Reflection 机制（方案 A：两轮 API 调用）

#### 新增：Self-Reflection 系统模板

```python
_REFLECTOR_SELF_REFLECTION_TEMPLATE = """You are the Reflector performing self-reflection on your initial assessment. You have just completed a preliminary evaluation of an AI-generated image. Now critically review your own assessment and produce the final, revised evaluation.

**Self-Reflection Checklist:**
1. Score-Reasoning Consistency: Do your scores align with your reasoning?
   - If your reasoning describes serious issues but the score is high → lower it.
   - If your reasoning is positive but the score is low → raise it.
   - Look for contradictions between per_atom_scores and the reasoning text.

2. Expert Evidence Utilization: Did you properly consider ALL expert testimony?
   - Were there classifier results (top-3 labels) you ignored or underweighted?
   - Were there detector counts that contradict your atom verdicts?
   - Were there auxiliary images (depth maps, segmentation masks) you didn't reference?

3. Reference Calibration: If human-scored reference images were provided:
   - Are your scores consistent with the reference anchors?
   - If a reference with similar quality has alignment=4.0, is your score in a similar range?

4. Leniency Bias: Are you rubber-stamping the Router's assessment too readily?
   - The Router's checkpoint verdicts are preliminary — did you independently verify them?
   - Did you accept is_present=true without checking expert evidence?

5. Harshness Bias: Are you over-penalizing minor issues?
   - A minor texture anomaly (severity 1) should not drop authenticity_score by more than 0.5.
   - Multiple minor issues compound, but one minor issue should not dominate.

6. Atom Score Review: For each atom:
   - Is qa_score justified by the expert evidence (or VLM observation if no expert)?
   - Is tax_score appropriate? If no taxonomy info, tax_score should be 1.0.

**Output the SAME JSON schema as your initial assessment, with revised scores.**
Add a "self_reflection_notes" field documenting:
  - What you changed and why
  - Which checklist items triggered adjustments
  - Whether your final scores are higher, lower, or same as initial (with reasoning)
"""
```

#### 新增：Self-Reflection 执行函数

```python
def _run_self_reflection_round(
    client: OpenAI,
    system_message: dict,
    user_content: list[dict],
    round1_result: dict,
    api_retry: int = 0,
    temperature: float = 0.5,
) -> dict | None:
    """执行 Round 2 Self-Reflection API 调用。

    利用对话历史：[system, round1_user, round1_assistant, round2_user]
    模型可以看到自己的初步评分并据此修订。
    """
    round2_prompt = (
        "Review your assessment above using the Self-Reflection Checklist. "
        "Output revised JSON with the same schema, plus a 'self_reflection_notes' field."
    )

    messages = [
        system_message,
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": json.dumps(round1_result, indent=2, ensure_ascii=False)},
        {"role": "user", "content": round2_prompt},
    ]

    try:
        completion = api_call_with_retry(
            client.chat.completions.create,
            model=REFLECTOR_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_retries=api_retry,
            label="Reflector-SelfReflection",
            extra_body={"enable_thinking": False},
        )
        raw_content = completion.choices[0].message.content
        if raw_content is None or raw_content.strip() == "":
            reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
            if reasoning:
                raw_content = reasoning
            else:
                return None
        result = parse_json_safely(raw_content)
        return result
    except Exception as e:
        print(f"  [WARN] Self-reflection round failed: {type(e).__name__}: {e}")
        return None
```

#### 新增：合并函数

```python
def _merge_self_reflection(round1: dict, round2: dict) -> dict:
    """合并两轮结果。Round 2 的分数优先，但保留 Round 1 的数据用于审计。"""
    # 保存 Round 1 分数用于对比
    round1_scores = {
        "alignment_score": round1.get("alignment_score"),
        "authenticity_score": round1.get("authenticity_score"),
        "per_atom_scores": round1.get("per_atom_scores", []),
    }

    # 使用 Round 2 的分数作为最终结果
    merged = round2.copy()
    merged["preliminary_scores"] = round1_scores
    merged["self_reflection_notes"] = round2.get("self_reflection_notes", "")

    # 记录分数变化
    r1_align = round1.get("alignment_score", 0)
    r2_align = round2.get("alignment_score", 0)
    if abs(r1_align - r2_align) > 0.01:
        merged["score_changes"] = {
            "alignment_score": f"{r1_align:.2f} → {r2_align:.2f}",
            "authenticity_score": f"{round1.get('authenticity_score', 0):.2f} → {round2.get('authenticity_score', 0):.2f}",
        }

    return merged
```

#### 修改 `run_reflector` — 插入 Round 2

```python
def run_reflector(...) -> dict | None:
    # ... Round 1: 原有 API 调用（不变）...
    result = parse_json_safely(raw_content)
    if result is None:
        print(f"  [ERROR] Reflector Round 1 returned unparseable JSON")
        return None

    # ── Round 2: Self-Reflection（新增）──
    round2_result = _run_self_reflection_round(
        client, system_message, user_content, result,
        api_retry=api_retry, temperature=temperature,
    )
    if round2_result is not None:
        result = _merge_self_reflection(result, round2_result)
        print(f"  [INFO] Self-reflection completed. "
              f"Alignment: {result.get('alignment_score', 'N/A')}, "
              f"Authenticity: {result.get('authenticity_score', 'N/A')}")
    else:
        print(f"  [WARN] Self-reflection round failed, using Round 1 scores")

    # ... 原有后续逻辑：metadata + _calibrate_scores ...
    result["metadata"] = {
        "original_image": image_path,
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "reflector_cost_seconds": round(cost_time, 2),
        "self_reflection_enabled": True,
        "self_reflection_succeeded": round2_result is not None,
    }
    result = _calibrate_scores(result)
    return result
```

#### Self-Reflection 输出格式

最终报告中新增字段：

```json
{
  "atom_reviews": [...],
  "alignment_score": 3.85,
  "authenticity_score": 3.92,
  "per_atom_scores": [0.9, 0.85, ...],
  "preliminary_scores": {
    "alignment_score": 4.20,
    "authenticity_score": 4.10,
    "per_atom_scores": [0.95, 0.90, ...]
  },
  "self_reflection_notes": "Lowered alignment from 4.20 to 3.85 because atom 2 reasoning described a count mismatch but qa_score was 0.95. Lowered authenticity because the texture anomaly was more severe than initially assessed.",
  "score_changes": {
    "alignment_score": "4.20 → 3.85",
    "authenticity_score": "4.10 → 3.92"
  },
  "metadata": {
    "self_reflection_enabled": true,
    "self_reflection_succeeded": true,
    ...
  }
}
```

---

### 2.5 `dispatch_sync.py` 和 `dispatch_async.py`

#### 在 Step 0 之后、Step 1 之前调用泛类 Taxonomy 生成

```python
# Step 0: Atomize
atomized_data = atomize_prompt(prompt_data)
save_atomized_prompt(atomized_data, ATOMIZED_DIR, img_id)

# Step 0d: 为泛类生成 Taxonomy（新增）
from step0_atomize import enrich_with_generic_taxonomy
atomized_data = enrich_with_generic_taxonomy(
    atomized_data, client,
    api_retry=api_retry, temperature=0.0,  # 泛类生成用 temperature=0
)
# 保存更新后的 atomized_data（包含生成的 taxonomy）
save_atomized_prompt(atomized_data, ATOMIZED_DIR, img_id)

# Step 1: Router
plan = _run_single_image(
    client, str(img_path), img_id, prompt_text, atomized_data, ...
)
```

#### 在 Step 4 传递参考图片目录

```python
# Step 4: Reflector
ref_image_dir = Path(args.ref_image_dir) if args.ref_image_dir else None

report = run_reflector(
    client=client,
    image_path=image_path,
    prompt_id=image_id,
    prompt_text=prompt_text,
    atomized_data=atomized_data,
    expert_results=bundle,
    experts_registry_str=experts_registry_str,
    router_plan=plan,
    ref_image_dir=ref_image_dir,  # 新增
    api_retry=api_retry,
    temperature=temp_reflector,
)
```

---

### 2.6 `run.py` — 新增 CLI 参数

```python
parser.add_argument("--ref-image-dir", type=str, default="",
                    help="Directory containing human-scored reference images "
                         "for Reflector calibration (default: not used)")
parser.add_argument("--ref-annotations", type=str,
                    default="t2i_harness/t2i_ref_annotations.json",
                    help="Path to human reference annotations JSON")
```

---

## 三、新数据结构汇总

### 3.1 增强后的 `atomized_data`

```json
{
  "prompt_id": "0",
  "prompt": "four brown monkeys and a metal bicycle",
  "atom_count": 7,
  "atoms": [
    {
      "atom_index": 0,
      "question": "How many monkeys are in the image?",
      "expected": "four",
      "answer_type": "count",
      "skill": "count",
      "target_object": "monkeys",
      "weight": 1.0
    }
  ],
  "objects": [
    {
      "object_name": "monkey",
      "class_id": null,
      "is_generic": true,
      "class_name": "monkey (generic)",
      "taxonomy_description": "灵长类动物，面部扁平双眼朝前...",
      "diagnostic_checkpoints": {
        "body_structure": "灵长类体型，四肢比例协调，躯干覆盖毛发",
        "facial_features": "双眼朝前，面部扁平，吻部不突出",
        "limbs": "四肢，有对生拇指可抓握",
        "covering": "体表毛发覆盖，无羽毛或鳞片"
      },
      "attributes_from_prompt": ["brown"],
      "count_from_prompt": 4
    },
    {
      "object_name": "bicycle",
      "class_id": 444,
      "is_generic": false,
      "class_name": "bicycle",
      "taxonomy_description": "...(从 taxonomy_info 加载)",
      "diagnostic_checkpoints": {"...(从 structured_taxonomy_info 加载)"},
      "attributes_from_prompt": ["metal"],
      "count_from_prompt": 1
    }
  ]
}
```

### 3.2 Router 输出（新增 expert_verification_plan）

```json
{
  "image_description": "Four brown monkeys next to a metal bicycle...",
  "atom_verdicts": [
    {"atom_index": 0, "question": "How many monkeys?", "expected": "four",
     "predicted": "four", "is_correct": true, "confidence": 0.95,
     "reasoning": "I count four monkey-like figures in the image."}
  ],
  "checkpoint_verdicts": [
    {"object": "monkey", "checkpoint": "body_structure", "category": "body",
     "is_testable": true, "is_present": true,
     "reasoning": "Subjects have primate body structure with four limbs."}
  ],
  "artifact_observations": [
    {"artifact_type": "texture_anomaly", "location": "monkey fur",
     "severity": 1.5, "reasoning": "Slight texture inconsistency on second monkey's fur."}
  ],
  "expert_verification_plan": [
    {
      "expert_name": "open_vocabulary_detector",
      "target_subject": "monkey",
      "verification_goals": ["atom_0_count", "checkpoint_body_structure"],
      "reason": "Verify monkey count and provide bounding boxes for body shape analysis",
      "weight": 0.3
    },
    {
      "expert_name": "fine_grained_classifier",
      "target_subject": "monkey",
      "verification_goals": ["checkpoint_facial_features", "checkpoint_covering"],
      "reason": "Classify the primate species and verify facial features + fur coverage",
      "weight": 0.25
    },
    {
      "expert_name": "animal_pose_auditor",
      "target_subject": "monkey",
      "verification_goals": ["checkpoint_limbs"],
      "reason": "Verify limb structure via keypoint detection",
      "weight": 0.2
    },
    {
      "expert_name": "perceptual_quality_auditor",
      "target_subject": "whole_image",
      "verification_goals": ["artifact_texture_anomaly"],
      "reason": "Verify texture anomaly severity via distortion analysis",
      "weight": 0.15
    },
    {
      "expert_name": "open_vocabulary_detector",
      "target_subject": "bicycle",
      "verification_goals": ["atom_3_count", "atom_4_attribute"],
      "reason": "Detect bicycle and verify count + metallic appearance",
      "weight": 0.1
    }
  ],
  "unverifiable_points": [
    "atom_1_attribute_brown: color verification relies on VLM visual observation",
    "atom_5_position: spatial relationship relies on VLM judgment"
  ],
  "focus_areas": ["monkey count verification", "primate body structure", "bicycle detection"]
}
```

### 3.3 `t2i_ref_annotations.json`（新文件）

```json
{
  "ref_0.png": {
    "prompt": "four brown monkeys and a metal bicycle",
    "alignment_score": 4.5,
    "authenticity_score": 3.8
  },
  "ref_1.png": {
    "prompt": "a green backpack and a pig",
    "alignment_score": 2.0,
    "authenticity_score": 4.2
  },
  "ref_2.png": {
    "prompt": "a monkey behind a penguin",
    "alignment_score": 1.5,
    "authenticity_score": 2.5
  }
}
```

---

## 四、实现顺序

| 阶段 | 文件 | 改动 | 状态 |
|------|------|------|------|
| 0 | `step4_reflector.py` | `_calibrate_scores` 删除 per_atom_scores fallback 补全逻辑（Reflector prompt 已要求输出 per_atom_scores） | **已完成** |
| 1 | `step0_atomize.py` | 新增 `generate_generic_taxonomy` + `enrich_with_generic_taxonomy`；修改 `atomize_prompt` 标记 `is_generic` | **已完成** |
| 2 | `step1_router.py` | 重写 Step 4 指令为"专家验证计划"；修改输出 schema；新增 `verification_goals`；修改 `validate_plan` | **已完成** |
| 3 | `step2_judge.py` | 新增验证覆盖度审查维度；修改输出 schema | **已完成** |
| 4 | `step4_reflector.py` | 新增参考标注加载 + 参考图片选择；修改 system template + prompt；修改 `run_reflector` 签名 | **已完成** |
| 5 | `step4_reflector.py` | 新增 Self-Reflection 机制（`_REFLECTOR_SELF_REFLECTION_TEMPLATE` + `_run_self_reflection_round` + `_merge_self_reflection`）；修改 `run_reflector` 插入 Round 2 | **已完成** |
| 6 | `dispatch_sync.py` | Step 0 后调用 `enrich_with_generic_taxonomy`；Step 4 传 `ref_image_dir` + `enable_self_reflection` | **已完成** |
| 7 | `dispatch_async.py` | 同步 dispatch_sync 的改动 | **已完成** |
| 8 | `run.py` | 新增 `--ref-image-dir`、`--ref-annotations`、`--enable-self-reflection` / `--no-self-reflection` CLI 参数 | **已完成** |
| 9 | 全文件语法检查 | `py_compile` + import 测试 | **已完成** |
| 10 | 更新 `README.md` | 更新流水线描述和参数说明 | ✅ 已完成 |

---

## 五、关键设计决策

### 5.1 泛类 Taxonomy 生成放在 Step 0 还是 Router 内？

**放在 Step 0**（dispatch 在 Step 0 后调用 `enrich_with_generic_taxonomy`）。

理由：
- 用户明确要求"先有 taxonomy info，然后让 router 根据 taxonomy 进行判定"
- Router 之前完成 Taxonomy 生成，确保 Router 和 Reflector 使用同一套 taxonomy
- 避免在 Router 的单次 API 调用中混合"生成"和"判定"两种任务
- Step 0 是纯预处理阶段，适合做 Taxonomy 准备

### 5.2 专家验证计划如何与 Step 3 的 `execute_plan` 兼容？

Router 输出 `expert_verification_plan`，但在 `generate_plan` 返回前同时生成 `selected_experts`（兼容字段）：

```python
plan["selected_experts"] = [
    {"expert_name": ev["expert_name"], "target_subject": ev["target_subject"],
     "reason": ev.get("reason", ""), "weight": ev.get("weight", 0.0)}
    for ev in plan.get("expert_verification_plan", [])
]
```

Step 3 的 `execute_plan` 只读取 `selected_experts`，不需要修改。

### 5.3 人类参考如何选择？

采用 C2I 的策略，适配 T2I：
1. 从 `t2i_ref_annotations.json` 加载参考标注
2. 按 prompt 中的物体关键词匹配（共享物体越多越相关）
3. 按 alignment_score 分低/中/高三段，每段选一张
4. 参考图片和分数一起传给 Reflector API（图片 + 文本标注）

### 5.4 `unverifiable_points` 的作用

Router 输出 `unverifiable_points` 列表，标注哪些 checkpoint/atom 只能靠 VLM 判断（如颜色属性、空间关系）。这样：
- Judge 可以审查"是否所有可验证的点都分配了专家"
- Reflector 知道哪些判定有硬证据支撑、哪些只有 VLM 软判断
- 评分时可以对有专家证据的判定给更高置信度

### 5.5 `_calibrate_scores` 的 per_atom_scores fallback 已删除

Reflector 的 prompt（`_REFLECTOR_SYSTEM_TEMPLATE`）明确要求输出 `per_atom_scores: [float]`，因此 Reflector 会按 prompt 输出该字段。`_calibrate_scores` 中"从 `atom_reviews` 补全 `per_atom_scores`"的 fallback 逻辑已删除。

当前 T2I `_calibrate_scores` 只保留两项操作：
1. `alignment_score = mean(per_atom_scores) × 5.0`（确定性计算，覆盖 Reflector 自报值）
2. `authenticity_score` 钳位到 [0, 5]

与 C2I 的对比：

| 维度 | C2I `_calibrate_scores` | T2I `_calibrate_scores` |
|------|------------------------|------------------------|
| 分类器封顶 | 有，可由 `enable_classifier_cap` 控制 | 无（T2I 通过 atom scores 处理） |
| 姿态封顶 | 有，可由 `pose_hard_cap` 控制 | 无 |
| alignment 计算 | 不覆盖 Reflector 的值（仅封顶） | 强制覆盖 = `mean(per_atom_scores) × 5.0` |
| 钳位 [0, 5] | 有 | 有 |
| 可控参数 | `enable_classifier_cap` + `pose_hard_cap` | 无（全部操作始终执行） |

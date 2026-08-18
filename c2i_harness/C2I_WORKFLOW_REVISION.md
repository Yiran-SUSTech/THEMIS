# C2I 评估系统流程改进方案

## 一、改进后的完整流水线

```
Step 1  Router（初步判定 + 证据驱动专家计划）
  ├── 1a: Taxonomy Checkpoint 验证（已有，保持不变）
  │   └── 对每个 diagnostic_checkpoint 判断 is_testable, is_present
  ├── 1b: 伪影检测（已有，保持不变）
  │   └── type, location, severity, reasoning
  └── 1c: 证据驱动专家计划（重写 Step 3）
      ├── 对每个 checkpoint：哪些可以用专家模型求证？
      ├── 对每个 is_present=false 的 checkpoint：是否有专家可提供反证？
      ├── 输出 expert_verification_plan（含 verification_goals）
      └── 输出 unverifiable_points（只能 VLM 判断的点）

Step 2  Judge（审核专家计划的完备性）
  ├── 原有审查维度保持不变
  └── 新增：专家验证覆盖度审查
      ├── 所有 checkpoint 是否都有对应验证手段（专家或 VLM）
      ├── verification_goals 是否与 checkpoint 一一对应
      └── unverifiable_points 中的点是否确实无法用专家验证

Step 3  Expert（执行专家计划，提供硬数据证据）

Step 4  Reflector（综合评分 + Self-Reflection）
  ├── Round 1：初步评分
  │   ├── 输入：Router 初步判定 + 专家硬数据 + 人类打分参考（已有）
  │   ├── 审查 Router 的 checkpoint verdicts 是否过于宽松
  │   ├── 审查伪影严重度是否被低估
  │   ├── 输出 alignment_score + artifact_score + reasoning
  │   └── _calibrate_scores 代码级后处理（已有，新增 enable_classifier_cap 开关）
  └── Round 2：Self-Reflection（自审修订，新增）
      ├── 输入：Round 1 输出 + 原始上下文（图片 + 专家证据 + 参考图）
      ├── 审查：分数与 reasoning 是否一致、专家证据是否充分利用、
      │         参考校准是否合理、是否过于宽松/严格
      └── 输出修订后的最终评分 + self_reflection_notes
```

### 与当前流程的关键区别

| 维度 | 当前流程 | 改进流程 |
|------|---------|---------|
| Router Step 3（专家选择） | 基于可见实体 + 硬编码规则 | 证据驱动：基于 checkpoint 的专家验证需求 |
| 专家计划结构 | `selected_experts`（仅指定专家 + target_subject） | `expert_verification_plan`（含 `verification_goals` + `unverifiable_points`） |
| Judge 审查维度 | 审查计划合理性 | 新增验证覆盖度审查 |
| Reflector 评估方式 | 单次 API 调用 | 两轮 API 调用：初步评分 → Self-Reflection 修订 |
| Reflector 输出 | alignment_score + artifact_score | + preliminary_scores + self_reflection_notes + score_changes |

---

## 二、逐文件修改方案

### 2.1 `step1_router.py` — 证据驱动专家计划

#### 当前状态

C2I Router 已有 4 步指令：
- Step 1 — Checkpoint Verification (STRICT)：已有，**保持不变**
- Step 2 — Artifact Detection (THOROUGH)：已有，**保持不变**
- Step 3 — Expert Selection：**需重写为"专家验证计划"**
- Step 4 — Weights：已有，**保持不变**

#### 重写 Step 3 指令

```text
**Step 3 — Expert Verification Plan**
Based on your preliminary judgments in Steps 1-2, specify which points can be
verified by expert models. For each verification need:

a) For each Taxonomy checkpoint that you marked is_present=true or is_present=false:
   - Can an expert model provide hard evidence to confirm/deny your judgment?
   - If yes, assign the appropriate expert with specific verification_goals.

b) For checkpoints you marked is_present=false (potential mismatch):
   - Strongly consider assigning "fine_grained_classifier" to verify species identity.
   - Consider "open_vocabulary_detector" if object presence is in question.

c) For artifact observations you found:
   - Assign "perceptual_quality_auditor" to verify distortion/artifact severity.
   - Consider "topology_boundary_auditor" for structural/shape issues.

d) For checkpoints with no expert available:
   - Leave unassigned. Your visual observation is the primary evidence.
   - List them in "unverifiable_points".

Expert Capability Mapping:
  - open_vocabulary_detector: Object detection + bounding boxes
  - fine_grained_classifier: ImageNet species classification (top-3 labels)
  - animal_pose_auditor: Limb/keypoint verification (limbed subjects only)
  - topology_boundary_auditor: Shape/contour verification (segmentation)
  - geometric_depth_auditor: Spatial relationship verification (depth map)
  - perceptual_quality_auditor: Artifact/distortion verification
  - image_text_auditor: Text verification (OCR)

Rules:
- Select 3-8 experts. Same expert_id may appear with different target_subjects.
- Each expert entry MUST specify verification_goals (list of checkpoint descriptions).
- "fine_grained_classifier" is recommended for the class subject.
- "animal_pose_auditor" ONLY for limbed subjects (people, dogs, cats, etc.).
- "image_text_auditor" ONLY if text is visible.
```

#### 修改输出 JSON Schema

```json
{
  "image_description": "Brief description of all visible entities and their roles",
  "image_class": "ImageNet class label",
  "checkpoint_verdicts": [
    {"checkpoint": "str", "category": "str", "is_testable": bool, "is_present": bool, "reasoning": "str"}
  ],
  "artifact_observations": [
    {"artifact_type": "str", "location": "str", "severity": float, "reasoning": "str"}
  ],
  "expert_verification_plan": [
    {"expert_name": "str",
     "target_subject": "str",
     "verification_goals": ["str"],
     "reason": "str",
     "weight": float}
  ],
  "unverifiable_points": ["str"],
  "focus_areas": ["str"],
  "custom_prompts_for_reflector": "str"
}
```

**字段变化**：
- `selected_experts` → `expert_verification_plan`：新增 `verification_goals` 字段
- 新增 `unverifiable_points`：列出只能靠 VLM 判断的 checkpoint

#### 兼容 `selected_experts`

在 `generate_plan` 返回前，将 `expert_verification_plan` 映射为 `selected_experts`，确保 Step 3 的 `execute_plan` 可以正常解析：

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

#### Router 输出示例

```json
{
  "image_description": "A guenon monkey sitting on a branch...",
  "image_class": "guenon",
  "checkpoint_verdicts": [
    {"checkpoint": "Round face with forward-facing eyes", "category": "head",
     "is_testable": true, "is_present": true,
     "reasoning": "Subject has a round face with clearly forward-facing eyes."},
    {"checkpoint": "Fur covering the body in brown/tan tones", "category": "body",
     "is_testable": true, "is_present": false,
     "reasoning": "Fur color is more grey than the expected brown/tan."}
  ],
  "artifact_observations": [
    {"artifact_type": "texture_anomaly", "location": "tail region",
     "severity": 2.0, "reasoning": "Tail fur shows melting artifacts."}
  ],
  "expert_verification_plan": [
    {
      "expert_name": "fine_grained_classifier",
      "target_subject": "guenon",
      "verification_goals": [
        "Round face with forward-facing eyes",
        "Fur covering the body in brown/tan tones"
      ],
      "reason": "Verify species identity and fur color via classification labels",
      "weight": 0.35
    },
    {
      "expert_name": "animal_pose_auditor",
      "target_subject": "guenon",
      "verification_goals": ["Four limbs with opposable thumbs"],
      "reason": "Verify limb structure via keypoint detection",
      "weight": 0.25
    },
    {
      "expert_name": "perceptual_quality_auditor",
      "target_subject": "whole_image",
      "verification_goals": ["texture_anomaly:tail_region"],
      "reason": "Verify tail melting artifact severity via distortion analysis",
      "weight": 0.20
    },
    {
      "expert_name": "topology_boundary_auditor",
      "target_subject": "guenon",
      "verification_goals": ["Body shape and contour"],
      "reason": "Verify body contour via segmentation",
      "weight": 0.20
    }
  ],
  "unverifiable_points": [
    "Tail length proportion: requires precise measurement not available from experts"
  ],
  "focus_areas": ["fur color mismatch", "tail melting artifact"],
  "custom_prompts_for_reflector": "Pay special attention to fur color — Router flagged grey vs expected brown/tan."
}
```

---

### 2.2 `step2_judge.py` — 新增验证覆盖度审查

#### 新增审查维度

```text
**审查维度 — 专家验证覆盖度**
检查 Router 的 expert_verification_plan：
1. 所有关键 taxonomy checkpoint 是否都有专家覆盖（或合理标注为 unverifiable）？
2. 所有 is_present=false 的 checkpoint，是否有 fine_grained_classifier 或
   open_vocabulary_detector 提供反证？
3. 所有 artifact_observations 是否都有 perceptual_quality_auditor 覆盖？
4. verification_goals 是否与 checkpoint_verdicts 中的描述一一对应？
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
    "checkpoints_covered": int,
    "checkpoints_total": int,
    "artifacts_covered": int,
    "artifacts_total": int,
    "unverifiable_count": int
  }
}
```

---

### 2.3 `step4_reflector.py` — Self-Reflection 机制（方案 A：两轮 API 调用）

#### 当前状态

C2I Reflector 已有：
- 两个系统模板（`_REFLECTOR_SYSTEM_TEMPLATE` + `_REFLECTOR_CHECKLIST_SYSTEM_TEMPLATE`）
- 参考图片校准（`select_reference_images` + `_build_ref_images_text` + `_append_ref_images_to_content`）
- 代码级后处理（`_calibrate_scores`：分类器不匹配封顶[可由 `enable_classifier_cap` 控制] + 姿态低置信封顶[可由 `pose_hard_cap` 控制]）
- 单次 API 调用

**需新增**：在 Round 1 之后插入 Round 2 Self-Reflection。

#### 新增：Self-Reflection 系统模板

```python
_REFLECTOR_SELF_REFLECTION_TEMPLATE = """You are the Reflector performing self-reflection on your initial assessment. You have just completed a preliminary evaluation of an AI-generated image. Now critically review your own assessment and produce the final, revised evaluation.

**Self-Reflection Checklist:**
1. Score-Reasoning Consistency: Do your scores align with your reasoning?
   - If your alignment_reasoning describes checkpoint mismatches but alignment_score is high → lower it.
   - If your artifact_reasoning describes severe issues but artifact_score is high → lower it.
   - Look for contradictions between the reasoning text and the numerical scores.

2. Expert Evidence Utilization: Did you properly consider ALL expert testimony?
   - Were there classifier results (top-3 labels) you ignored or underweighted?
   - Did the classifier Top-1 match the target class? If not, did you adequately cap alignment?
   - Were there auxiliary images (depth maps, segmentation masks) you didn't reference?
   - Did the pose auditor's keypoint analysis reveal structural issues you overlooked?

3. Reference Calibration: If human-scored reference images were provided:
   - Are your scores consistent with the reference anchors?
   - If a reference with similar quality has alignment=4.0, is your score in a similar range?
   - Reference anchors should prevent both inflated and deflated scores.

4. Leniency Bias: Are you rubber-stamping the Router's assessment too readily?
   - The Router's checkpoint verdicts are preliminary — did you independently verify them?
   - Did you accept is_present=true without checking expert evidence?
   - The Router may miss subtle artifacts — did you look for additional issues?

5. Harshness Bias: Are you over-penalizing minor issues?
   - A minor texture anomaly (severity 1) should not drop artifact_score by more than 0.5.
   - Multiple minor issues compound, but one minor issue should not dominate.
   - Pose low-confidence keypoints alone (without visual confirmation) are a weak signal.

6. Checkpoint Review: For each checkpoint:
   - Did you agree/disagree with the Router's is_present verdict?
   - If you disagreed, did you explain why?
   - If the Router was too lenient, did you flag it?

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
    round1_scores = {
        "alignment_score": round1.get("alignment_score"),
        "artifact_score": round1.get("artifact_score"),
    }

    merged = round2.copy()
    merged["preliminary_scores"] = round1_scores
    merged["self_reflection_notes"] = round2.get("self_reflection_notes", "")

    r1_align = round1.get("alignment_score", 0)
    r2_align = round2.get("alignment_score", 0)
    r1_artifact = round1.get("artifact_score", 0)
    r2_artifact = round2.get("artifact_score", 0)

    if abs(r1_align - r2_align) > 0.01 or abs(r1_artifact - r2_artifact) > 0.01:
        merged["score_changes"] = {
            "alignment_score": f"{r1_align:.2f} → {r2_align:.2f}",
            "artifact_score": f"{r1_artifact:.2f} → {r2_artifact:.2f}",
        }

    return merged
```

#### 修改 `run_reflector` — 插入 Round 2

```python
def run_reflector(
    client: OpenAI,
    image_path: str,
    class_id: int,
    class_label: str,
    expert_results: dict,
    experts_registry_str: str,
    router_plan: dict | None = None,
    ref_images: list[dict] | None = None,
    enable_checklist: bool = False,
    enable_self_reflection: bool = True,  # 新增
    api_retry: int = 0,
    temperature: float = 0.5,
    pose_hard_cap: bool = False,
    enable_classifier_cap: bool = True,  # 新增：控制分类器封顶
) -> dict | None:
    # ... Round 1: 原有 API 调用（不变）...
    result = parse_json_safely(raw_content)
    if result is None:
        print(f"  [ERROR] Reflector Round 1 returned unparseable JSON: {raw_content[:300]}")
        return None

    # ── Round 2: Self-Reflection（新增）──
    if enable_self_reflection:
        round2_result = _run_self_reflection_round(
            client, system_message, user_content, result,
            api_retry=api_retry, temperature=temperature,
        )
        if round2_result is not None:
            result = _merge_self_reflection(result, round2_result)
            print(f"  [INFO] Self-reflection completed. "
                  f"Alignment: {result.get('alignment_score', 'N/A')}, "
                  f"Artifact: {result.get('artifact_score', 'N/A')}")
        else:
            print(f"  [WARN] Self-reflection round failed, using Round 1 scores")

    # ... 原有后续逻辑：metadata + _calibrate_scores + checklist normalization ...
    result["metadata"] = {
        "original_image": image_path,
        "class_id": class_id,
        "class_label": class_label,
        "taxonomy_available": taxonomy_info is not None,
        "auxiliary_images_included": ...,
        "ref_images_enabled": ref_images is not None and len(ref_images) > 0,
        "checklist_enabled": enable_checklist,
        "self_reflection_enabled": enable_self_reflection,  # 新增
        "self_reflection_succeeded": round2_result is not None if enable_self_reflection else None,  # 新增
        "reflector_cost_seconds": round(cost_time, 2),
    }
    result = _calibrate_scores(result, expert_results, router_plan,
                               pose_hard_cap=pose_hard_cap,
                               enable_classifier_cap=enable_classifier_cap)
    if enable_checklist:
        result = _normalize_checklist_output(result, structured_taxonomy_info)
    return result
```

#### Self-Reflection 输出格式

最终报告中新增字段：

```json
{
  "checkpoint_review": "...",
  "artifact_review": "...",
  "alignment_score": 3.42,
  "artifact_score": 3.15,
  "alignment_reasoning": "...",
  "artifact_reasoning": "...",
  "key_defects": ["..."],
  "preliminary_scores": {
    "alignment_score": 4.10,
    "artifact_score": 3.80
  },
  "self_reflection_notes": "Lowered alignment from 4.10 to 3.42 because the classifier Top-1 was 'baboon' not 'guenon', which I underweighted in Round 1. Lowered artifact because the tail melting was more severe upon re-examination.",
  "score_changes": {
    "alignment_score": "4.10 → 3.42",
    "artifact_score": "3.80 → 3.15"
  },
  "metadata": {
    "self_reflection_enabled": true,
    "self_reflection_succeeded": true,
    ...
  }
}
```

#### 与 `_calibrate_scores` 的关系

Self-Reflection（Round 2）发生在 `_calibrate_scores` **之前**：
1. Round 1 → 初步评分
2. Round 2 → Self-Reflection 修订
3. `_calibrate_scores` → 代码级硬规则

`_calibrate_scores` 的各项操作现在均可由参数控制：

| 操作 | 控制参数 | 默认值 | CLI 开关 |
|------|---------|--------|---------|
| 分类器封顶（Top-1 不匹配→2.0，不在 Top-3→1.0） | `enable_classifier_cap` | True | `--no-classifier-cap` 关闭 |
| 姿态封顶（低置信比封顶 artifact） | `pose_hard_cap` | False | `--pose-hard-cap` 启用 |
| 钳位 [0, 5] | 无（始终执行） | — | — |

**默认行为**：分类器封顶启用，姿态封顶关闭。`_calibrate_scores` 对分类器不匹配仍有最终否决权。

**完全信任 LLM**：使用 `--no-classifier-cap` 关闭分类器封顶后，Reflector（含 Self-Reflection）的判断为最终结果，代码不做 alignment 封顶。

---

### 2.4 `dispatch_sync.py` 和 `dispatch_async.py`

#### 传递 `enable_self_reflection` 参数

```python
# Step 4: Reflector
report = run_reflector(
    client=client,
    image_path=image_path,
    class_id=class_id,
    class_label=class_label,
    expert_results=bundle,
    experts_registry_str=experts_registry_str,
    router_plan=plan,
    ref_images=ref_images,
    enable_checklist=enable_checklist,
    enable_self_reflection=enable_self_reflection,  # 新增
    api_retry=api_retry,
    temperature=temp_reflector,
    pose_hard_cap=pose_hard_cap,
    enable_classifier_cap=enable_classifier_cap,  # 新增
)
```

---

### 2.5 `run.py` — 新增 CLI 参数

```python
parser.add_argument("--enable-self-reflection", action="store_true", default=True,
                    help="Enable Reflector self-reflection (two-round API calls). "
                         "Default: enabled. Use --no-self-reflection to disable.")
parser.add_argument("--no-self-reflection", action="store_false", dest="enable_self_reflection",
                    help="Disable Reflector self-reflection (single-round mode for faster processing)")
parser.add_argument("--no-classifier-cap", action="store_false", dest="enable_classifier_cap", default=True,
                    help="Disable classifier-based alignment capping (Top-1/Top-3 mismatch caps). "
                         "By default, alignment is capped to 2.0 if classifier Top-1 mismatches target, "
                         "and 1.0 if target not in Top-3. Use --no-classifier-cap to trust the "
                         "Reflector's judgment entirely without code-level capping.")
```

---

## 三、实现顺序

| 阶段 | 文件 | 改动 | 状态 |
|------|------|------|------|
| 0a | `step4_reflector.py` | `_calibrate_scores` 新增 `enable_classifier_cap` 参数；`run_reflector` 新增参数并传递 | **已完成** |
| 0b | `dispatch_sync.py` + `dispatch_async.py` | 全链路传递 `enable_classifier_cap` 参数 | **已完成** |
| 0c | `run.py` | 新增 `--no-classifier-cap` CLI 参数；3 处 pipeline 调用传递参数 | **已完成** |
| 1 | `step1_router.py` | 重写 Step 3 指令为"专家验证计划"；修改输出 schema；新增 `verification_goals` + `unverifiable_points`；修改 `validate_plan`；兼容 `selected_experts` | **已完成** |
| 2 | `step2_judge.py` | 新增验证覆盖度审查维度；修改输出 schema | **已完成** |
| 3 | `step4_reflector.py` | 新增 Self-Reflection 机制（`_REFLECTOR_SELF_REFLECTION_TEMPLATE` + `_run_self_reflection_round` + `_merge_self_reflection`）；修改 `run_reflector` 插入 Round 2 + `enable_self_reflection` 参数 | **已完成** |
| 4 | `dispatch_sync.py` | 传递 `enable_self_reflection` 参数 | **已完成** |
| 5 | `dispatch_async.py` | 同步 dispatch_sync 的改动 | **已完成** |
| 6 | `run.py` | 新增 `--enable-self-reflection` / `--no-self-reflection` CLI 参数 | **已完成** |
| 7 | 全文件语法检查 | `py_compile` + import 测试 | **已完成** |
| 8 | 更新 `HARNESS_ENGINEERING_SUMMARY.md` | 更新流水线描述和参数说明 | ✅ 已完成 |

---

## 四、关键设计决策

### 4.1 C2I Router 的 Step 1（Checkpoint Verification）是否需要修改？

**不需要。** C2I Router 已有的 Step 1 指令（"Checkpoint Verification STRICT"）已经很好地完成了初步判定。改进仅针对 Step 3（Expert Selection），将其从"基于可见实体"改为"基于 checkpoint 验证需求"。

### 4.2 Self-Reflection 与 `_calibrate_scores` 的执行顺序

```
Round 1 (LLM 初步评分) → Round 2 (LLM Self-Reflection) → _calibrate_scores (代码级硬规则，可开关)
```

- Round 2 的 Self-Reflection 是 LLM 层面的自审：检查 reasoning 与分数的一致性、专家证据的利用度
- `_calibrate_scores` 是代码层面的硬约束：分类器不匹配 → 封顶（`enable_classifier_cap`，默认启用），姿态低置信 → 封顶（`pose_hard_cap`，默认关闭）
- 两者互补：Self-Reflection 处理"软性偏差"（过于宽松/严格），`_calibrate_scores` 处理"硬性约束"（分类器与目标类不匹配）
- 使用 `--no-classifier-cap` 可完全关闭分类器封顶，让 Reflector（含 Self-Reflection）的判断为最终结果

### 4.3 Self-Reflection 的 API 成本

| 模式 | API 调用次数 | 适用场景 |
|------|------------|---------|
| `--enable-self-reflection`（默认） | 2× | 正式评估，需要最高评分质量 |
| `--no-self-reflection` | 1× | 快速测试/调试，节省 API 成本 |

### 4.4 `enable_checklist` 与 `enable_self_reflection` 的组合

| `enable_checklist` | `enable_self_reflection` | 行为 |
|----|----|------|
| False | False | Round 1 标准模板，无 Self-Reflection |
| False | True | Round 1 标准模板 + Round 2 Self-Reflection |
| True | False | Round 1 checklist 模板，无 Self-Reflection |
| True | True | Round 1 checklist 模板 + Round 2 Self-Reflection |

Self-Reflection 模板与 checklist/非 checklist 模板兼容：Round 2 要求输出"与 Round 1 相同的 schema"，因此无论 Round 1 用哪个模板，Round 2 都会输出兼容的 JSON。

### 4.5 C2I vs T2I Self-Reflection 的差异

| 维度 | C2I Self-Reflection | T2I Self-Reflection |
|------|---------------------|---------------------|
| 分数名称 | alignment_score + artifact_score | alignment_score + authenticity_score |
| 检查重点 | checkpoint verdicts + 分类器匹配 + 姿态证据 | per_atom_scores (qa_score × tax_score) + 原子 QA 正确性 |
| `_calibrate_scores` 行为 | 分类器封顶[可开关] + 姿态封顶[可开关] + 钳位 | alignment_score 强制覆盖 = mean(per_atom_scores) × 5.0 + 钳位 |
| 参考校准 | 已有（`select_reference_images`） | 改进方案中新增 |
| 实现方式 | 完全相同（对话历史 + Self-Reflection 模板） | 完全相同 |

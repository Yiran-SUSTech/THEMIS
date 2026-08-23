# T2I 测评数据采样提示词

为 THEMIS T2I 测评系统（`t2i_harness`）采样待测评 prompt 的 LLM 提示词。输出直接就是 `geneval2_data.jsonl` 格式。

## 使用方法

1. 将下方提示词完整复制给长输出 LLM（Qwen3.6-Plus / GPT 等）
2. 按需修改「采样配额」中的 N 和分布比例
3. 输出保存为 `geneval2_data.jsonl`，放到 THEMIS 根目录（或运行时用 `--geneval2-jsonl` 指定路径）
4. 用其中的 prompt 逐条生成图片，**图片文件名 = 行号**（第 0 行 → `0.png`）
5. 运行：`python t2i_harness/run.py --mode async --step 1234 --image-dir <图片目录>`

## 提示词正文（复制以下全部内容）

```text
你是一个文生图（T2I）模型测评数据集的采样引擎。你的任务：生成 N 条高质量的英文测评 prompt，并为每条 prompt 标注原子 QA 对，输出严格符合下方格式规范的 JSONL。

## 输出格式
- 只输出 JSONL：每行一个合法 JSON 对象，共 N=100 行
- 不要输出任何解释、markdown 代码块标记或额外文字
- 每行三个字段：
  {"prompt": "<英文短语式文生图提示词>", "vqa_list": [["<question>", "<answer>"], ...], "skills": ["<skill>", ...]}
- vqa_list 与 skills 必须等长且一一对应

## vqa_list 构建规则（最重要）
1. 完整覆盖：prompt 中的每个物体、每个数量、每个属性、每组空间关系都必须有至少一个 QA 对应；QA 不得引入 prompt 未提及的信息
2. 问题必须严格使用以下四种模板之一（下游解析器按模板提取物体名，偏离模板会导致 taxonomy 关联失败）：
   - 数量："How many {物体复数} are in the image?"
   - 存在："Are there any {物体复数} in the image?"
   - 属性（复数物体）："Are the {物体复数} {属性}?"
   - 属性（单数物体）："Is the {物体} {属性}?"
3. 答案格式：
   - 数量题：英文数字单词，取值 one / two / three / four
   - 存在题、是否题：yes 或 no
   - 属性题优先改写为是否存在形式（"Are the monkeys brown?" → "yes"），避免开放式属性值
4. 一个 QA 只验证一个原子事实：禁止 "Are there two brown dogs?"（复合了数量+颜色），应拆成两个 QA
5. 所有 QA 必须可以通过看图客观判定；禁止主观词（beautiful、cute、realistic、high quality）

## skills 标签（闭合集合，每题选一个最贴切的）
- count：数量、物体存在
- attribute：颜色、材质、形状等属性
- position：空间位置关系（left / right / above / below / on / under / between / next to / in front of / behind）

## 采样配额（本次执行参数）
- 总条数 N=100
- 原子数分布：2 个原子约 25%、3 个约 40%、4 个约 25%、5 个约 10%
- 物体语义类别 ≥ 15 类（动物、交通工具、家具、食物、乐器、容器、工具、体育器材、植物、昆虫等）
- 物体必须是单个英文常用名词，禁止复合名词（如 teddy bear、fire truck）
- 约 70% 的物体从以下 ImageNet 可映射池中选取：
  dog, cat, horse, elephant, zebra, bear, sheep, cow, pig, rabbit, bird, owl, duck, penguin, fish, frog, butterfly, spider, snake, bicycle, car, truck, motorcycle, bus, train, airplane, boat, umbrella, clock, trumpet, banana, apple, orange, pizza, donut, croissant, bagel, flower, rose, bottle, cup, book, chair, table, guitar, piano
- 约 30% 使用映射外物体（如 monkey, giraffe, kangaroo, llama, hamster, flamingo, peacock, camel, turtle, lizard），用于测试泛类 taxonomy 生成路径
- 同一条 prompt 内物体种类 1-3 个，数量 one 到 four
- 空间关系与属性类型轮换使用，避免全部集中在 left/right 和颜色
- 任意两条 prompt 语义不得重复

## 禁止事项
- 禁止模糊量词（several、many、a few、some）
- 禁止需要世界知识或常识推理才能回答的 QA（如 "Is this the Eiffel Tower?"）
- 禁止 prompt 中出现文字渲染要求（如 "a sign that says HELLO"）
- 禁止抽象概念、名人、地标、品牌
- 禁止输出 JSONL 以外的任何内容

## 参考示例（格式与粒度标准）
{"prompt": "four brown monkeys and a metal bicycle", "vqa_list": [["How many monkeys are in the image?", "four"], ["Are the monkeys brown?", "yes"], ["Are there any bicycles in the image?", "yes"], ["Is the bicycle metal?", "yes"]], "skills": ["count", "attribute", "count", "attribute"]}

现在开始输出 N=100 行 JSONL。
```

## 变体用法

**从已有数据选子集**：把「采样配额」一节替换为：

```text
- 从我提供的 prompt 列表中选择 N=100 条，使 skill 覆盖、原子数分布、物体类别分布尽量均衡
- 保持每条原有的 vqa_list 和 skills 不变，只做选择，不改写
- prompt 列表如下：<粘贴>
```

## 采样后的校验清单

- [ ] 行数 = N，每行 JSON 可解析（`python -c "import json;[json.loads(l) for l in open('geneval2_data.jsonl',encoding='utf-8')]"`）
- [ ] 每行的 vqa_list 与 skills 等长
- [ ] 数量题答案是 one/two/three/four 之一
- [ ] 问题全部匹配四种模板（可抽样人工检查 target_object 提取）
- [ ] 图片文件名与行号一一对应，无缺失（缺失的图片会被静默跳过）
- [ ] 多模型对比时：同一份 jsonl，每个模型一个图片目录，`--image-dir` + `--output-dir` 区分

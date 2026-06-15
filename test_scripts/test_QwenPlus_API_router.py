import os
import base64
import json
from openai import OpenAI

# 1. 工具函数：图像转 Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 2. 加载专家注册表
def load_experts_registry(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.dumps(json.load(f), indent=2)

# 3. 新增：加载并检索 Taxonomy 先验知识
def get_taxonomy_description(file_path, target_label):
    """
    根据 class_label 模糊匹配 taxonomy 中的 enriched_description。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            taxonomy_data = json.load(f)
            # 在列表中寻找匹配项（支持名称部分包含匹配）
            for item in taxonomy_data:
                if target_label.lower() in item['class_name'].lower():
                    return item['enriched_description']
        return "No specific biological prior knowledge found for this class."
    except Exception as e:
        return f"Error loading taxonomy: {e}"

# 4. 初始化客户端
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# --- 配置区 ---
image_path = "test_images_GPT-XL-c2i_XL/000052.png"
experts_json_path = "expert_registry.json"
taxonomy_json_path = "taxonomy_enriched.json" # 确保路径正确
class_label = "bald eagle"
# 新增：保存路径配置
output_plan_dir = "evaluation_plans_output" 
os.makedirs(output_plan_dir, exist_ok=True) # 如果文件夹不存在则创建
# --------------

# 准备数据
base64_image = encode_image(image_path)
experts_content = load_experts_registry(experts_json_path)
# 检索该类别的先验知识
prior_knowledge = get_taxonomy_description(taxonomy_json_path, class_label)

# 5. 构建增强型 Planner Prompt（整合先验知识）
planner_prompt = f"""You are the Lead Strategic Planner for an advanced AI image evaluation system. 
Your task is to analyze the provided image and its specific class category to formulate a rigorous "Evaluation Plan" using the Expert Registry.

**[Input Data]**
- **Class Label:** {class_label}
- **Taxonomy Knowledge (Ground Truth):** {prior_knowledge}
- **Expert Registry (Available Tools):** {experts_content}

**[Strategic Instruction]**
1. **Identify Category Archetype:** Determine if the class "{class_label}" is an **Organism** (animal/plant), a **Rigid Object** (architecture/tool/vehicle), or a **Natural Scene** (landscape/texture).
2. **Feature Mapping:** Based on the Taxonomy Knowledge, extract 2-3 "Non-negotiable" features (e.g., specific symmetry for buildings, anatomical counts for animals, or textural coherence for landscapes).
3. **Visual Risk Assessment:** Scrutinize the image for category-specific flaws:
   - *Organisms:* Look for "Melting" limbs, missing parts, or anatomical hallucinations.
   - *Rigid Objects:* Look for warped lines, perspective distortion, or "fusing" into the background.
   - *Scenes:* Look for repetitive patterns (mode collapse) or illogical spatial bleeding.
4. **Tool Selection:** Map the identified risks to the specific `expert_id` in the Registry.

**[Output Requirements]**
- You must prioritize **fine_grained_classifier** for identity verification.
- You must use **open_vocabulary_detector** if the class requires locating specific parts (e.g., eyes of an eagle, wheels of a car).
- Justify weights based on the "Structural Criticality" of the feature.

Return JSON ONLY in this exact schema:
{{
  "category_archetype": "Organism | Rigid Object | Natural Scene",
  "semantic_baseline": "A concise summary of the key diagnostic features for this class.",
  "initial_observation": "Major visual anomalies or successes found during preliminary scan.",
  "evaluation_plan": [
    {{
      "stage": 1,
      "expert_id": "string (must match the registry)",
      "rationale": "Logical necessity based on category archetype and taxonomy details.",
      "expected_evidence": "The specific metric or visual proof this expert must provide.",
      "weight": 0.0-1.0
    }}
  ],
  "conflict_resolution_strategy": "The prioritized 'Source of Truth' when experts provide conflicting scores for this specific archetype."
}}"""

# 6. 调用 API 制定计划
try:
    completion = client.chat.completions.create(
        model="qwen3.6-plus",
        messages=[
            {
                "role": "system",
                "content": "You are a highly logical Router Agent for image auditing. You must prioritize the provided Taxonomy Knowledge as the source of truth."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": planner_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    
    # 获取返回的内容并解析为字典
    plan_dict = json.loads(completion.choices[0].message.content)
    
    # 6. 【核心保存逻辑】
    # 提取文件名（不含后缀）作为保存的文件名，例如 000052
    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_plan_dir, f"plan_{image_filename}.json")
    
    # 将额外信息（原图路径、类别、生成时间）也存进去，方便后续追踪
    plan_dict["metadata"] = {
        "original_image": image_path,
        "class_label": class_label,
    }
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(plan_dict, f, indent=4, ensure_ascii=False)
    
    print(f"Plan saved successfully to: {save_path}")

except Exception as e:
    print(f"Error during API call: {e}")
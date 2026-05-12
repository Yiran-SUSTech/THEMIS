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
client = OpenAI(
    api_key="sk-9165cc69015b4a12ab542fb5edc20612",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# --- 配置区 ---
image_path = "test_images_GPT-XL-c2i_XL/000052.png"
experts_json_path = "expert_registry.json"
taxonomy_json_path = "taxonomy_enriched.json" # 确保路径正确
class_label = "bald eagle"
# --------------

# 准备数据
base64_image = encode_image(image_path)
experts_content = load_experts_registry(experts_json_path)
# 检索该类别的先验知识
prior_knowledge = get_taxonomy_description(taxonomy_json_path, class_label)

# 5. 构建增强型 Planner Prompt（整合先验知识）
planner_prompt = f"""You are the Lead Strategic Planner for an advanced AI image evaluation system. 
Your task is to analyze the provided image and class label to formulate a precise "Evaluation Plan" using available expert models.

**Input Data:**
- Class Label: {class_label}
- **Prior Knowledge (Taxonomy):** {prior_knowledge}

- Expert Registry (Available Tools): 
{experts_content}

**Instruction:**
1. **Analyze the Class:** Use the provided "Prior Knowledge" to identify the absolute "Golden Standards" for {class_label}. Focus on unique anatomical features (e.g., beak shape, feather patterns, limb structure) mentioned in the taxonomy.
2. **Visual Inspection:** Scan the attached image for initial "red flags" that violate the Prior Knowledge or basic physical laws.
3. **Strategize:** Select relevant expert models from the Registry to conduct a deep-dive audit.

**Requirements:**
- High priority should be given to **fine_grained_classifier (EVA-02)** to verify the species identity against the taxonomy description.
- Use **animal_pose_auditor** or **topology_boundary_auditor** if the taxonomy mentions complex limbs or silhouettes.
- Justify the weights based on how critical a feature is in the taxonomy (e.g., if the taxonomy emphasizes "bald head," assign higher weight to a detail-checking expert).

Return JSON ONLY in this exact schema:
{{
  "semantic_baseline": "string (Summary of key features from Taxonomy)",
  "initial_observation": "string (Anomalies found when comparing image to Taxonomy)",
  "evaluation_plan": [
    {{
      "stage": 1,
      "expert_id": "string (from registry)",
      "rationale": "string (referencing why this is needed based on Taxonomy/Image)",
      "expected_evidence": "string",
      "weight": 0.0-1.0
    }}
  ],
  "conflict_resolution_strategy": "string"
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
    
    plan_json = completion.choices[0].message.content
    print("Successfully generated evaluation plan with Taxonomy Priors:")
    print(plan_json)

except Exception as e:
    print(f"Error during API call: {e}")
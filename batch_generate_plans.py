import os
import base64
import json
import re
from openai import OpenAI

# ==================== 1. 配置区 ====================
IMAGE_DIR = "/mnt/afs/zhengmingkai/zyr/THEMIS/test_images_GPT-XL-c2i_XL"
CLASS_IDS_TXT = "/mnt/afs/zhengmingkai/zyr/THEMIS/test_images_GPT-XL-c2i_XL/class_ids.txt"
TAXONOMY_JSON = "/mnt/afs/zhengmingkai/zyr/THEMIS/taxonomy_enriched_all.json"
EXPERTS_REGISTRY_JSON = "expert_registry.json"

OUTPUT_PLAN_DIR = "evaluation_plans_output"
os.makedirs(OUTPUT_PLAN_DIR, exist_ok=True)

# 初始化阿里云 DashScope 兼容客户端
client = OpenAI(
    api_key="sk-9165cc69015b4a12ab542fb5edc20612",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# ===================================================


# ==================== 2. 数据辅助加载函数 ====================
def encode_image(image_path):
    """图像转 Base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_experts_registry(file_path):
    """载入专家模型库定义"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.dumps(json.load(f), indent=2)


def parse_class_ids(txt_path):
    """
    解析 class_ids.txt 文件。
    返回字典结构: { "000002": 29, "000010": 74, ... }
    """
    img_to_class_map = {}
    if not os.path.exists(txt_path):
        print(f"[-] Error: Class map file not found at {txt_path}")
        return img_to_class_map
        
    print(f"--> Parsing class index mapping from {txt_path}...")
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 使用正则或 split 劈开“图片名编号”与“类别index”
            parts = re.split(r'\s+', line)
            if len(parts) >= 2:
                img_id = parts[0]
                try:
                    class_id = int(parts[1])
                    img_to_class_map[img_id] = class_id
                except ValueError:
                    continue
    return img_to_class_map


def load_taxonomy_map(json_path):
    """
    将 Taxonomy 数组加载为以 class_id 为 Key 的字典，实现 O(1) 的精准极速检索。
    """
    taxonomy_map = {}
    if not os.path.exists(json_path):
        print(f"[-] Error: Taxonomy knowledge file not found at {json_path}")
        return taxonomy_map
        
    print(f"--> Loading ground-truth taxonomy database from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            c_id = item.get("class_id")
            if c_id is not None:
                taxonomy_map[int(c_id)] = {
                    "class_name": item.get("class_name", "Unknown"),
                    "enriched_description": item.get("enriched_description", "No description available.")
                }
    return taxonomy_map


# ==================== 3. 核心批量执行主函数 ====================
def main():
    # Step 1: 预加载基础数据与映射路由
    img_to_class_map = parse_class_ids(CLASS_IDS_TXT)
    taxonomy_db = load_taxonomy_map(TAXONOMY_JSON)
    experts_content = load_experts_registry(EXPERTS_REGISTRY_JSON)
    
    # Step 2: 扫描并筛选出目录下的前 20 张合法图片
    all_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_files.sort()  # 严格排序，确保取到的是前 20 张
    
    target_files = all_files[:20]
    print(f"\n[Ready] Found {len(all_files)} total images. Selected first {len(target_files)} samples for Router Planning.")
    print("="*60)
    
    # Step 3: 循环请求 Router
    for idx, img_name in enumerate(target_files, start=1):
        img_id = os.path.splitext(img_name)[0]  # 提取例如 "000002"
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        print(f"\n[{idx}/20] Processing Image Target: {img_name}")
        
        # 精准查找匹配的 Class ID 和 Taxonomy
        if img_id not in img_to_class_map:
            print(f"    [-] Skipped: Image ID '{img_id}' not indexed in class_ids.txt.")
            continue
            
        target_class_id = img_to_class_map[img_id]
        
        if target_class_id not in taxonomy_db:
            print(f"    [-] Skipped: Class ID '{target_class_id}' absent from taxonomy matrix.")
            continue
            
        # 捕获精准先验知识
        class_label = taxonomy_db[target_class_id]["class_name"]
        prior_knowledge = taxonomy_db[target_class_id]["enriched_description"]
        
        print(f"    ├─ Mapped Class ID: {target_class_id} --> Class Label: '{class_label}'")
        
        # 图像 Base64 编码
        try:
            base64_image = encode_image(img_path)
        except Exception as e:
            print(f"    [-] Error encoding image: {e}")
            continue

        # 动态组装精准的增强型 Prompt
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

        # 调用阿里大模型 API 制定计划
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
            
            # 解析并注入完备的溯源元数据 (Metadata)
            plan_dict = json.loads(completion.choices[0].message.content)
            plan_dict["metadata"] = {
                "original_image_path": img_path,
                "image_filename": img_name,
                "class_id": target_class_id,
                "class_label": class_label,
            }
            
            # 保存输出
            save_path = os.path.join(OUTPUT_PLAN_DIR, f"plan_{img_id}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(plan_dict, f, indent=4, ensure_ascii=False)
                
            print(f"    └─ [Success] Evaluation plan safely stored at: {save_path}")
            
        except Exception as e:
            print(f"    └─ [CRITICAL ERROR] Failed routing token or parsing payload for {img_name}: {e}")

    print("\n" + "="*20 + " All 20 Tasks Batched Completed " + "="*20)


if __name__ == "__main__":
    main()
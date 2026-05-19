import os
import base64
import json
import re
import time  # 🚀 引入时间模块
from openai import OpenAI

# ==================== 1. 配置区 ====================
IMAGE_DIR = "/mnt/afs/zhengmingkai/zyr/THEMIS/test_images_GPT-XL-c2i_XL"
CLASS_IDS_TXT = "/mnt/afs/zhengmingkai/zyr/THEMIS/test_images_GPT-XL-c2i_XL/class_ids.txt"
TAXONOMY_JSON = "/mnt/afs/zhengmingkai/zyr/THEMIS/taxonomy_enriched_all.json"
EXPERTS_REGISTRY_JSON = "expert_registry.json"

OUTPUT_PLAN_DIR = "evaluation_plans_output"
os.makedirs(OUTPUT_PLAN_DIR, exist_ok=True)

# 初始化客户端
client = OpenAI(
    api_key="sk-9165cc69015b4a12ab542fb5edc20612",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# ===================================================

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_experts_registry(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.dumps(json.load(f), indent=2)

def parse_class_ids(txt_path):
    img_to_class_map = {}
    if not os.path.exists(txt_path):
        print(f"[-] Error: Class map file not found at {txt_path}")
        return img_to_class_map
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = re.split(r'\s+', line)
            if len(parts) >= 2:
                try:
                    img_to_class_map[parts[0]] = int(parts[1])
                except ValueError: continue
    return img_to_class_map

def load_taxonomy_map(json_path):
    taxonomy_map = {}
    if not os.path.exists(json_path):
        return taxonomy_map
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

# ==================== 2. 主执行函数 ====================
def main():
    img_to_class_map = parse_class_ids(CLASS_IDS_TXT)
    taxonomy_db = load_taxonomy_map(TAXONOMY_JSON)
    experts_content = load_experts_registry(EXPERTS_REGISTRY_JSON)
    
    all_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_files.sort()
    target_files = all_files[:20]
    
    print(f"\n[Ready] Starting Router Planning for first {len(target_files)} samples.")
    print("="*60)
    
    # 用于存储每张图耗时数据的字典
    time_records = []
    
    for idx, img_name in enumerate(target_files, start=1):
        img_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        if img_id not in img_to_class_map: continue
        target_class_id = img_to_class_map[img_id]
        if target_class_id not in taxonomy_db: continue
            
        class_label = taxonomy_db[target_class_id]["class_name"]
        prior_knowledge = taxonomy_db[target_class_id]["enriched_description"]
        
        try:
            base64_image = encode_image(img_path)
        except Exception: continue

        planner_prompt = f"""You are the Lead Strategic Planner for an advanced AI image evaluation system. 
Your task is to analyze the provided image and its specific class category to formulate a rigorous "Evaluation Plan" using the Expert Registry.

**[Input Data]**
- **Class Label:** {class_label}
- **Taxonomy Knowledge (Ground Truth):** {prior_knowledge}
- **Expert Registry (Available Tools):** {experts_content}

**[Strategic Instruction]**
1. **Identify Category Archetype:** Determine if the class "{class_label}" is an **Organism** (animal/plant), a **Rigid Object** (architecture/tool/vehicle), or a **Natural Scene** (landscape/texture).
2. **Feature Mapping:** Based on the Taxonomy Knowledge, extract 2-3 "Non-negotiable" features.
3. **Visual Risk Assessment:** Scrutinize the image for category-specific flaws.
4. **Tool Selection:** Map the identified risks to the specific `expert_id` in the Registry.

Return JSON ONLY in this exact schema:
{{
  "category_archetype": "Organism | Rigid Object | Natural Scene",
  "semantic_baseline": "A concise summary of the key diagnostic features.",
  "initial_observation": "Major visual anomalies found.",
  "evaluation_plan": [
    {{
      "stage": 1,
      "expert_id": "string",
      "rationale": "string",
      "expected_evidence": "string",
      "weight": 0.0-1.0
    }}
  ],
  "conflict_resolution_strategy": "string"
}}"""

        # 🚀 记录请求发起的时间戳
        start_time = time.time()
        
        try:
            print(f"[{idx}/20] Sending request for {img_name} ({class_label})...")
            completion = client.chat.completions.create(
                model="qwen3.6-plus",
                messages=[
                    {"role": "system", "content": "You are a highly logical Router Agent."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": planner_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            # 🚀 计算得到响应的耗时（秒）
            cost_time = time.time() - start_time
            time_records.append({"img_name": img_name, "cost": cost_time})
            print(f"    ├─ Response received. Cost: {cost_time:.2f} seconds.")
            
            plan_dict = json.loads(completion.choices[0].message.content)
            plan_dict["metadata"] = {
                "original_image_path": img_path,
                "image_filename": img_name,
                "class_id": target_class_id,
                "class_label": class_label,
                "router_cost_seconds": round(cost_time, 2)  # 顺便把耗时也记录在 plan 文件里
            }
            
            save_path = os.path.join(OUTPUT_PLAN_DIR, f"plan_{img_id}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(plan_dict, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            print(f"    └─ [ERROR] Failed for {img_name}: {e}")

    # ==================== 3. 耗时统计报告打印 ====================
    print("\n" + "="*20 + " Router Performance Report " + "="*20)
    if time_records:
        total_time = 0
        for record in time_records:
            print(f" -> {record['img_name']} | Router Cost: {record['cost']:.2f}s")
            total_time += record['cost']
        
        avg_time = total_time / len(time_records)
        print("-" * 67)
        print(f" Successfully Planned: {len(time_records)} / 20 images.")
        print(f" Total Elapsed Time  : {total_time:.2f} seconds.")
        print(f" Average Plan Time   : {avg_time:.2f} seconds per image. 🚀")
    else:
        print("No successful time records collected.")
    print("="*67)


if __name__ == "__main__":
    main()
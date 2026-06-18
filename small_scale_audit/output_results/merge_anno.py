import json
from collections import Counter

def majority_vote(statuses):
    counter = Counter(statuses)
    if len(counter) == 3:
        return "⚪ N/A"
    most_common = counter.most_common(1)[0]
    if most_common[1] >= 2:
        return most_common[0]
    if len(counter) == 2:
        sorted_items = counter.most_common(2)
        if sorted_items[0][1] == sorted_items[1][1]:
            for s in statuses:
                if s == "⚪ N/A":
                    return "⚪ N/A"
            return "🔴 Missing"
        return sorted_items[0][0]
    return most_common[0]

def merge_artifact_scores(scores):
    mean_val = sum(scores) / len(scores)
    distances = [(abs(s - mean_val), s) for s in scores]
    distances.sort(key=lambda x: x[0], reverse=True)
    remaining = [s for _, s in distances[1:]]
    final = sum(remaining) / len(remaining)
    return final

def main():
    base_dir = r"./small_scale_audit/output_results"
    
    with open(f"{base_dir}\\User_1_final_annotations.json", "r", encoding="utf-8") as f:
        user1 = json.load(f)
    with open(f"{base_dir}\\User_2_final_annotations.json", "r", encoding="utf-8") as f:
        user2 = json.load(f)
    with open(f"{base_dir}\\User_3_final_annotations.json", "r", encoding="utf-8") as f:
        user3 = json.load(f)

    all_images = set(user1.keys()) | set(user2.keys()) | set(user3.keys())
    
    merged = {}

    for img_name in sorted(all_images):
        d1 = user1.get(img_name)
        d2 = user2.get(img_name)
        d3 = user3.get(img_name)
        
        ref = d1 or d2 or d3
        
        merged_fine_grained = {}
        all_categories = set()
        if d1: all_categories.update(d1["fine_grained_details"].keys())
        if d2: all_categories.update(d2["fine_grained_details"].keys())
        if d3: all_categories.update(d3["fine_grained_details"].keys())
        
        for category in sorted(all_categories):
            merged_fine_grained[category] = {}
            
            checkpoints = set()
            if d1 and category in d1["fine_grained_details"]:
                checkpoints.update(d1["fine_grained_details"][category].keys())
            if d2 and category in d2["fine_grained_details"]:
                checkpoints.update(d2["fine_grained_details"][category].keys())
            if d3 and category in d3["fine_grained_details"]:
                checkpoints.update(d3["fine_grained_details"][category].keys())
            
            for cp in sorted(checkpoints):
                statuses = []
                if d1 and category in d1["fine_grained_details"] and cp in d1["fine_grained_details"][category]:
                    statuses.append(d1["fine_grained_details"][category][cp])
                if d2 and category in d2["fine_grained_details"] and cp in d2["fine_grained_details"][category]:
                    statuses.append(d2["fine_grained_details"][category][cp])
                if d3 and category in d3["fine_grained_details"] and cp in d3["fine_grained_details"][category]:
                    statuses.append(d3["fine_grained_details"][category][cp])
                
                merged_fine_grained[category][cp] = majority_vote(statuses)
        
        checked_count = 0
        missing_count = 0
        for category in merged_fine_grained:
            for cp in merged_fine_grained[category]:
                if merged_fine_grained[category][cp] == "🟢 Checked":
                    checked_count += 1
                elif merged_fine_grained[category][cp] == "🔴 Missing":
                    missing_count += 1
        
        if checked_count + missing_count > 0:
            alignment_score = round(checked_count / (checked_count + missing_count) * 100, 2) / 100
        else:
            alignment_score = 0.0
        
        artifact_scores = []
        if d1: artifact_scores.append(d1["scores"]["artifact_score"])
        if d2: artifact_scores.append(d2["scores"]["artifact_score"])
        if d3: artifact_scores.append(d3["scores"]["artifact_score"])
        
        merged_artifact_scores = merge_artifact_scores(artifact_scores)
        
        veto_activated = False
        if d1: veto_activated = veto_activated or d1.get("veto_activated", False)
        if d2: veto_activated = veto_activated or d2.get("veto_activated", False)
        if d3: veto_activated = veto_activated or d3.get("veto_activated", False)
        
        merged[img_name] = {
            "image_name": img_name,
            "class_id": ref["class_id"],
            "class_name": ref["class_name"],
            "veto_activated": veto_activated,
            "scores": {
                "alignment_score": alignment_score,
                "artifact_score": merged_artifact_scores,
                "total_score": round(alignment_score * merged_artifact_scores, 2)
            },
            "fine_grained_details": merged_fine_grained
        }
    
    output_path = f"{base_dir}\\merged_annotations.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)
    
    print(f"Merged {len(merged)} images, saved to {output_path}")

if __name__ == "__main__":
    main()
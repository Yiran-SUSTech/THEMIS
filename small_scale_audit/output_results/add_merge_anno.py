"""
将 merged_annotations.json 中每张图片的 alignment_score 和 artifact_score
合并到 aggregated_scores.xlsx 中，作为单独的两列。
"""

import json
import os
import pandas as pd
import numpy as np

MERGED_JSON = os.path.join(os.path.dirname(__file__), "merged_annotations.json")
EXCEL_PATH = os.path.join(os.path.dirname(__file__), "aggregated_scores.xlsx")

# 读取 Excel
df = pd.read_excel(EXCEL_PATH)

# 读取 JSON
with open(MERGED_JSON, "r", encoding="utf-8") as f:
    merged_data = json.load(f)

# 初始化新列
merged_alignment = []
merged_artifact = []
missing_ids = []

for idx, row in df.iterrows():
    image_name = row["image_name"]  # e.g. "000000.png"

    if image_name in merged_data:
        scores = merged_data[image_name].get("scores", {})
        merged_alignment.append(scores.get("alignment_score", np.nan)*5)
        merged_artifact.append(scores.get("artifact_score", np.nan))
    else:
        merged_alignment.append(np.nan)
        merged_artifact.append(np.nan)
        missing_ids.append(image_name)

# 添加新列
df["Merged_alignment_score"] = merged_alignment
df["Merged_artifact_score"] = merged_artifact

# 保存回 Excel
df.to_excel(EXCEL_PATH, index=False)

print(f"Done. Wrote {len(df)} rows to {EXCEL_PATH}")
print(f"Merged_alignment_score non-NaN: {df['Merged_alignment_score'].notna().sum()}")
print(f"Merged_artifact_score non-NaN: {df['Merged_artifact_score'].notna().sum()}")
if missing_ids:
    print(f"\nMissing entries for {len(missing_ids)} images:")
    print(missing_ids)
else:
    print("\nAll 1000 images have merged annotations.")
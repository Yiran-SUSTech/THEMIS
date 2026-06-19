"""
将 c2i_faster/output/final_reports 中的 AI 测评结果
(alignment_score, artifact_score) 合并到
small_scale_audit/output_results/aggregated_scores.xlsx 中。
"""

import json
import os
import pandas as pd
import numpy as np

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "c2i_faster", "output", "final_reports")
EXCEL_PATH = os.path.join(os.path.dirname(__file__), "small_scale_audit_recorrect", "output_results", "aggregated_scores.xlsx")

# 读取 Excel
df = pd.read_excel(EXCEL_PATH)

# 初始化新列
ai_alignment = []
ai_artifact = []
missing_ids = []

for idx, row in df.iterrows():
    image_name = row["image_name"]  # e.g. "000000.png"
    image_id = os.path.splitext(image_name)[0]  # e.g. "000000"
    report_path = os.path.join(REPORTS_DIR, f"final_evaluation_report_{image_id}.json")

    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        ai_alignment.append(report.get("alignment_score", np.nan))
        ai_artifact.append(report.get("artifact_score", np.nan))
    else:
        ai_alignment.append(np.nan)
        ai_artifact.append(np.nan)
        missing_ids.append(image_id)

# 添加新列
df["AI_alignment_score"] = ai_alignment
df["AI_artifact_score"] = ai_artifact

# 保存回 Excel
df.to_excel(EXCEL_PATH, index=False)

print(f"Done. Wrote {len(df)} rows to {EXCEL_PATH}")
print(f"AI_alignment_score non-NaN: {df['AI_alignment_score'].notna().sum()}")
print(f"AI_artifact_score non-NaN: {df['AI_artifact_score'].notna().sum()}")
if missing_ids:
    print(f"\nMissing reports for {len(missing_ids)} images:")
    print(missing_ids)
else:
    print("\nAll 1000 images have reports.")

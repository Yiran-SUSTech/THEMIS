"""
找出3个标注员分歧较大的图片，输出到CSV文件供重新标注参考。
"""

import os
import pandas as pd

EXCEL_PATH = os.path.join(os.path.dirname(__file__), "output_results", "aggregated_scores.xlsx")
OUTPUT_DIR = os.path.dirname(__file__)


def main():
    df = pd.read_excel(EXCEL_PATH)

    thresholds = [0.5, 1.0, 1.5, 2.0]

    for score_type in ["alignment", "artifact"]:
        max_min_col = f"{score_type}_max-min"
        mean_col = f"{score_type}_mean"

        print(f"\n{'='*60}")
        print(f"  {score_type.upper()} 分歧分析")
        print(f"{'='*60}")

        for thresh in thresholds:
            mask = df[max_min_col] >= thresh
            count = mask.sum()
            print(f"\n  max-min >= {thresh}: {count} 张图片")

            if count > 0 and count <= 30:
                subset = df.loc[mask, ["image_name", "class_name",
                                        f"User_1_{score_type}", f"User_2_{score_type}", f"User_3_{score_type}",
                                        mean_col, max_min_col]]
                print(subset.to_string(index=False))

        # 输出详细CSV：所有max-min >= 2.0的图片
        thresh = 2.0
        mask = df[max_min_col] >= thresh
        subset = df.loc[mask, ["image_name", "class_id", "class_name",
                                f"User_1_{score_type}", f"User_2_{score_type}", f"User_3_{score_type}",
                                mean_col, max_min_col]].copy()
        subset = subset.sort_values(max_min_col, ascending=False)

        out_path = os.path.join(OUTPUT_DIR, f"disagreement_{score_type}_maxmin_ge_{int(thresh)}.csv")
        subset.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n  已保存 {len(subset)} 条记录到: {out_path}")

    # 综合分歧：alignment和artifact都分歧大的图片
    align_mask = df["alignment_max-min"] >= 2.0
    artifact_mask = df["artifact_max-min"] >= 2.0
    both_mask = align_mask & artifact_mask

    print(f"\n{'='*60}")
    print(f"  alignment AND artifact 都分歧大 (max-min >= 2.0)")
    print(f"{'='*60}")
    print(f"\n  共 {both_mask.sum()} 张图片")

    if both_mask.sum() > 0:
        subset = df.loc[both_mask, ["image_name", "class_id", "class_name",
                                     "User_1_alignment", "User_2_alignment", "User_3_alignment",
                                     "User_1_artifact", "User_2_artifact", "User_3_artifact",
                                     "alignment_max-min", "artifact_max-min"]].copy()
        subset = subset.sort_values("alignment_max-min", ascending=False)
        out_path = os.path.join(OUTPUT_DIR, "disagreement_both_maxmin_ge_2.csv")
        subset.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  已保存到: {out_path}")


if __name__ == "__main__":
    main()
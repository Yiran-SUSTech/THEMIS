"""
汇总人类标注员的标注结果，计算逐图均值/方差，并分析标注一致性。

用法:
    python small_scale_audit_recorrect/analyze_annotations.py
    python small_scale_audit_recorrect/analyze_annotations.py --annotator 1 2 3
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path


OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(OUTPUT_DIR, "output_results")


def load_user_annotations(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    scores = {}
    for img_name, entry in data.items():
        s = entry.get("scores", {})
        scores[img_name] = {
            "alignment_score": s.get("alignment_score"),
            "artifact_score": s.get("artifact_score"),
        }
    return scores


def compute_icc_2_1(data: np.ndarray) -> float:
    n, k = data.shape
    grand_mean = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ms_rows = ss_rows / (n - 1)
    ms_cols_val = ss_cols / (k - 1)
    ms_error = (ss_total - ss_rows - ss_cols + n * grand_mean ** 2 * k - n * grand_mean ** 2 * k) / ((n - 1) * (k - 1))
    ss_interaction = ss_total - ss_rows - ss_cols + n * k * grand_mean ** 2
    ms_error = ss_interaction / ((n - 1) * (k - 1))
    icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols_val - ms_error) / n)
    return icc


def compute_fleiss_kappa(data: np.ndarray, bins: list) -> float:
    n = data.shape[0]
    k = len(bins) - 1
    digitized = np.digitize(data, bins) - 1
    digitized = np.clip(digitized, 0, k - 1)
    ratings = np.zeros((n, k))
    for i in range(n):
        for j in range(data.shape[1]):
            ratings[i, digitized[i, j]] += 1
    P_i = (np.sum(ratings ** 2, axis=1) - data.shape[1]) / (data.shape[1] * (data.shape[1] - 1))
    P_bar = P_i.mean()
    p_j = ratings.sum(axis=0) / (n * data.shape[1])
    P_e = np.sum(p_j ** 2)
    if P_e == 1.0:
        return 1.0
    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa


def main():
    parser = argparse.ArgumentParser(description="汇总人类标注员的标注结果，分析标注一致性")
    parser.add_argument(
        "--annotator",
        nargs="+",
        type=int,
        default=None,
        help="指定标注员编号，如 --annotator 1 2 3。不指定则使用所有标注员",
    )
    args = parser.parse_args()

    user_files = sorted(Path(RESULTS_DIR).glob("User_*_final_annotations.json"))
    if not user_files:
        print("错误: 未找到标注文件")
        return

    all_user_names = [f.stem.replace("_final_annotations", "") for f in user_files]

    if args.annotator is not None:
        selected = [f"User_{a}" for a in args.annotator]
        user_names = [u for u in all_user_names if u in selected]
        if not user_names:
            print(f"错误: 指定的标注员 {args.annotator} 未找到，可用: {all_user_names}")
            return
        print(f"使用指定的标注员: {user_names}")
    else:
        user_names = all_user_names
        print(f"使用所有标注员: {user_names}")

    all_data = {}
    for uname in user_names:
        fpath = os.path.join(RESULTS_DIR, f"{uname}_final_annotations.json")
        all_data[uname] = load_user_annotations(fpath)

    common_imgs = None
    for uname in user_names:
        s = set(all_data[uname].keys())
        common_imgs = s if common_imgs is None else common_imgs & s
    common_imgs = sorted(common_imgs)
    print(f"共同标注图片数: {len(common_imgs)}")

    rows = []
    for img in common_imgs:
        row = {"image_name": img}
        for uname in user_names:
            entry = all_data[uname][img]
            row[f"{uname}_alignment"] = entry["alignment_score"]
            row[f"{uname}_artifact"] = entry["artifact_score"]
        rows.append(row)
    df = pd.DataFrame(rows)

    for metric in ["alignment", "artifact"]:
        cols = [f"{u}_{metric}" for u in user_names]
        vals = df[cols].values
        df[f"{metric}_mean"] = vals.mean(axis=1)
        df[f"{metric}_var"] = vals.var(axis=1, ddof=0)
        df[f"{metric}_std"] = vals.std(axis=1, ddof=0)
        df[f"{metric}_range"] = vals.max(axis=1) - vals.min(axis=1)

    csv_path = os.path.join(OUTPUT_DIR, "annotations_summary.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n汇总表已保存到 {csv_path}")

    print("\n" + "=" * 60)
    print("人类标注一致性分析报告")
    print("=" * 60)

    for metric in ["alignment", "artifact"]:
        cols = [f"{u}_{metric}" for u in user_names]
        print(f"\n--- {metric}_score ---")

        print(f"\n  各标注员整体均值:")
        for u in user_names:
            print(f"    {u}: {df[f'{u}_{metric}'].mean():.4f}")

        print(f"\n  逐图统计:")
        print(f"    均值(mean of means): {df[f'{metric}_mean'].mean():.4f}")
        print(f"    方差均值(mean of var): {df[f'{metric}_var'].mean():.4f}")
        print(f"    标准差均值(mean of std): {df[f'{metric}_std'].mean():.4f}")
        print(f"    极差均值(mean of range): {df[f'{metric}_range'].mean():.4f}")
        print(f"    极差中位数(median of range): {df[f'{metric}_range'].median():.4f}")
        print(f"    极差最大值: {df[f'{metric}_range'].max():.4f}")

        print(f"\n  逐图方差分布:")
        var_bins = [0, 0.1, 0.25, 0.5, 1.0, 2.0, 10.0]
        var_labels = ["0-0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-2", "2+"]
        df[f"{metric}_var_bin"] = pd.cut(df[f"{metric}_var"], bins=var_bins, labels=var_labels, include_lowest=True)
        for vl in var_labels:
            cnt = (df[f"{metric}_var_bin"] == vl).sum()
            pct = cnt / len(df) * 100
            print(f"    {vl:>10s}: {cnt:>4d} ({pct:>5.1f}%)")

        print(f"\n  标注员间 Pearson 相关系数:")
        corr_matrix = df[cols].corr()
        for i in range(len(user_names)):
            for j in range(i + 1, len(user_names)):
                r = corr_matrix.iloc[i, j]
                print(f"    {user_names[i]} vs {user_names[j]}: r = {r:.4f}")

        print(f"\n  配对 t 检验:")
        for i in range(len(user_names)):
            for j in range(i + 1, len(user_names)):
                col_i = f"{user_names[i]}_{metric}"
                col_j = f"{user_names[j]}_{metric}"
                t_stat, p_val = stats.ttest_rel(df[col_i], df[col_j])
                mean_diff = df[col_i].mean() - df[col_j].mean()
                print(f"    {user_names[i]} vs {user_names[j]}: 均值差={mean_diff:+.4f}, t={t_stat:.3f}, p={p_val:.4f} {'*' if p_val < 0.05 else ''}")
        print(f"    (* 表示 p<0.05)")

        print(f"\n  ICC(2,1) 组内相关系数:")
        icc = compute_icc_2_1(df[cols].values)
        print(f"    ICC = {icc:.4f}")

        print(f"\n  Fleiss' Kappa (5档: 0-1,1-2,2-3,3-4,4-5):")
        kappa = compute_fleiss_kappa(df[cols].values, [0, 1, 2, 3, 4, 5])
        print(f"    Kappa = {kappa:.4f}")

    print("\n--- 分歧最大的 Top 20 图片 ---")
    for metric in ["alignment", "artifact"]:
        top20 = df.nlargest(20, f"{metric}_range")
        print(f"\n  {metric}_score:")
        print(f"  {'image':>12s}", end="")
        for u in user_names:
            print(f"  {u:>12s}", end="")
        print(f"  {'mean':>6s}  {'range':>6s}")
        for _, row in top20.iterrows():
            print(f"  {row['image_name']:>12s}", end="")
            for u in user_names:
                print(f"  {row[f'{u}_{metric}']:>12.2f}", end="")
            print(f"  {row[f'{metric}_mean']:>6.2f}  {row[f'{metric}_range']:>6.2f}")

    print("\n--- 可视化 ---")
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for mi, metric in enumerate(["alignment", "artifact"]):
        cols = [f"{u}_{metric}" for u in user_names]

        ax0 = axes[mi][0]
        ax0.hist(df[f"{metric}_range"].values, bins=20, color="#4C72B0", edgecolor="white")
        ax0.set_xlabel("Range (max - min)")
        ax0.set_ylabel("Count")
        ax0.set_title(f"{metric}_score\nRange distribution across images")
        ax0.axvline(df[f"{metric}_range"].mean(), color="red", linestyle="--", label=f"mean={df[f'{metric}_range'].mean():.2f}")
        ax0.legend()

        ax1 = axes[mi][1]
        bp_data = [df[f"{u}_{metric}"].values for u in user_names]
        bp = ax1.boxplot(bp_data, labels=user_names, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors[:len(user_names)]):
            patch.set_facecolor(c)
        ax1.set_ylabel("Score")
        ax1.set_title(f"{metric}_score\nAnnotator score distribution")
        ax1.grid(axis="y", alpha=0.3)

        ax2 = axes[mi][2]
        score_bins = [0, 1, 2, 3, 4, 5]
        score_labels = ["0-1", "1-2", "2-3", "3-4", "4-5"]
        width = 0.2
        x = np.arange(len(score_labels))
        for ui, u in enumerate(user_names):
            counts = pd.cut(df[f"{u}_{metric}"], bins=score_bins, labels=score_labels, include_lowest=True).value_counts().reindex(score_labels, fill_value=0)
            ax2.bar(x + ui * width, counts.values, width, label=u, color=colors[ui % len(colors)])
        ax2.set_xticks(x + width * (len(user_names) - 1) / 2)
        ax2.set_xticklabels(score_labels)
        ax2.set_xlabel("Score bin")
        ax2.set_ylabel("Count")
        ax2.set_title(f"{metric}_score\nScore bin distribution per annotator")
        ax2.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "annotations_consistency_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"可视化图表已保存到 {plot_path}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
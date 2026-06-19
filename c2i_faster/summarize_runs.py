"""
汇总多次测评运行的结果，比较 alignment_score 和 artifact_score 的稳定性。

用法:
    python c2i_faster/summarize_runs.py c2i_faster/output_DiT_val_1 c2i_faster/output_DiT_val_2 c2i_faster/output_DiT_val_3
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


def load_scores_from_run(run_dir: str) -> dict:
    """从一次运行的 final_reports 目录中读取所有图片的 alignment_score 和 artifact_score。"""
    final_reports_dir = os.path.join(run_dir, "final_reports")
    if not os.path.isdir(final_reports_dir):
        print(f"警告: {final_reports_dir} 不存在，跳过")
        return {}

    scores = {}
    for fname in sorted(os.listdir(final_reports_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(final_reports_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 从文件名提取图片编号，如 final_evaluation_report_000000.json -> 000000
            img_id = fname.replace("final_evaluation_report_", "").replace(".json", "")
            scores[img_id] = {
                "alignment_score": data.get("alignment_score"),
                "artifact_score": data.get("artifact_score"),
            }
        except Exception as e:
            print(f"警告: 读取 {fpath} 失败: {e}")
    return scores


def main():
    parser = argparse.ArgumentParser(description="汇总多次测评运行的结果")
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="要汇总的运行输出文件夹路径（可传入多个）",
    )
    args = parser.parse_args()

    # 为每个运行目录取一个简短标签
    run_labels = []
    for d in args.run_dirs:
        label = Path(d).name  # e.g. output_DiT_val_1
        run_labels.append(label)

    # 读取所有运行的分数
    all_runs = {}
    for label, run_dir in zip(run_labels, args.run_dirs):
        all_runs[label] = load_scores_from_run(run_dir)

    # 收集所有图片ID（取交集，只保留每次运行都有的图片）
    all_img_ids = None
    for label in run_labels:
        ids = set(all_runs[label].keys())
        if all_img_ids is None:
            all_img_ids = ids
        else:
            all_img_ids &= ids
    if all_img_ids is None:
        print("错误: 没有找到任何数据")
        return
    all_img_ids = sorted(all_img_ids)

    # 构建DataFrame
    rows = []
    for img_id in all_img_ids:
        row = {"image_id": img_id}
        for label in run_labels:
            scores = all_runs[label][img_id]
            row[f"{label}_alignment_score"] = scores["alignment_score"]
            row[f"{label}_artifact_score"] = scores["artifact_score"]
        rows.append(row)

    df = pd.DataFrame(rows)

    # === 逐图稳定性统计 ===
    for metric in ["alignment_score", "artifact_score"]:
        cols = [f"{label}_{metric}" for label in run_labels]
        values = df[cols].values
        df[f"{metric}_mean"] = values.mean(axis=1)
        df[f"{metric}_std"] = values.std(axis=1, ddof=0)
        df[f"{metric}_range"] = values.max(axis=1) - values.min(axis=1)
        df[f"{metric}_min"] = values.min(axis=1)
        df[f"{metric}_max"] = values.max(axis=1)

    # 保存为CSV
    output_path = os.path.join("c2i_faster", "runs_summary.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"汇总完成，共 {len(df)} 张图片，结果已保存到 {output_path}")

    # === 整体稳定性报告 ===
    print("\n" + "=" * 60)
    print("稳定性分析报告")
    print("=" * 60)

    # === 各次运行整体均值对比 ===
    print("\n--- 各次运行整体均值对比 ---")
    for metric in ["alignment_score", "artifact_score"]:
        print(f"\n  {metric}:")
        means = []
        for label in run_labels:
            col = f"{label}_{metric}"
            m = df[col].mean()
            means.append(m)
            print(f"    {label}: {m:.4f}")
        print(f"    均值范围: [{min(means):.4f}, {max(means):.4f}]")
        print(f"    均值极差: {max(means) - min(means):.4f}")
        
        print(f"\n  配对 t 检验 (检验两次运行均值是否显著不同):")
        for i in range(len(run_labels)):
            for j in range(i + 1, len(run_labels)):
                col_i = f"{run_labels[i]}_{metric}"
                col_j = f"{run_labels[j]}_{metric}"
                t_stat, p_value = stats.ttest_rel(df[col_i], df[col_j])
                mean_diff = df[col_i].mean() - df[col_j].mean()
                print(f"    {run_labels[i]} vs {run_labels[j]}: 均值差={mean_diff:+.4f}, t={t_stat:.3f}, p={p_value:.4f} {'*' if p_value < 0.05 else ''}")
        print(f"    (* 表示 p<0.05，即差异显著)")

    # === 按极差区间可视化 min/max 分布 ===
    plot_range_bins = [(0, 1, "0-1"), (1, 2, "1-2"), (2, 3, "2-3"), (3, 5.01, "3+")]
    metrics = ["alignment_score", "artifact_score"]
    fig, axes = plt.subplots(len(metrics), len(plot_range_bins), figsize=(5 * len(plot_range_bins), 5 * len(metrics)), squeeze=False)

    for mi, metric in enumerate(metrics):
        for bi, (lo, hi, label) in enumerate(plot_range_bins):
            ax = axes[mi][bi]
            mask = (df[f"{metric}_range"] >= lo) & (df[f"{metric}_range"] < hi)
            subset = df[mask]
            if len(subset) == 0:
                ax.set_title(f"{metric}\nrange {label}\n(n=0)")
                ax.set_xlim(0, 5)
                ax.set_ylim(0, 5)
                continue

            min_vals = subset[f"{metric}_min"].values
            max_vals = subset[f"{metric}_max"].values

            positions = [0, 1]
            bp = ax.boxplot([min_vals, max_vals], positions=positions, widths=0.6,
                            patch_artist=True, showfliers=True)
            bp["boxes"][0].set_facecolor("#4C72B0")
            bp["boxes"][1].set_facecolor("#DD8452")

            ax.set_xticks(positions)
            ax.set_xticklabels(["min", "max"])
            ax.set_ylabel("Score")
            ax.set_ylim(-0.2, 5.2)
            ax.set_title(f"{metric}\nrange {label}\n(n={len(subset)})")
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
            ax.axhline(y=1.5, color="gray", linestyle="--", alpha=0.3)
            ax.axhline(y=2.5, color="gray", linestyle="--", alpha=0.3)
            ax.axhline(y=3.5, color="gray", linestyle="--", alpha=0.3)
            ax.axhline(y=4.5, color="gray", linestyle="--", alpha=0.3)
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join("c2i_faster", "runs_stability_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n可视化图表已保存到 {plot_path}")
    print("稳定性分析报告")
    print("=" * 60)

    for metric in ["alignment_score", "artifact_score"]:
        print(f"\n--- {metric} ---")
        print(f"  逐图标准差均值:  {df[f'{metric}_std'].mean():.4f}")
        print(f"  逐图标准差中位数: {df[f'{metric}_std'].median():.4f}")
        print(f"  逐图极差均值:    {df[f'{metric}_range'].mean():.4f}")
        print(f"  逐图极差中位数:  {df[f'{metric}_range'].median():.4f}")
        print(f"  逐图极差最大值:  {df[f'{metric}_range'].max():.4f}")
        print(f"  极差>1.0的图片数: {(df[f'{metric}_range'] > 1.0).sum()}")
        print(f"  极差>0.5的图片数: {(df[f'{metric}_range'] > 0.5).sum()}")

    # === 运行间 Pearson 相关系数 ===
    print("\n--- 运行间 Pearson 相关系数 ---")
    for metric in ["alignment_score", "artifact_score"]:
        cols = [f"{label}_{metric}" for label in run_labels]
        corr_matrix = df[cols].corr()
        print(f"\n  {metric}:")
        for i in range(len(run_labels)):
            for j in range(i + 1, len(run_labels)):
                r = corr_matrix.iloc[i, j]
                print(f"    {run_labels[i]} vs {run_labels[j]}: r = {r:.4f}")

    # === ICC (Intraclass Correlation Coefficient) ===
    print("\n--- ICC(2,1) 组内相关系数 ---")
    for metric in ["alignment_score", "artifact_score"]:
        cols = [f"{label}_{metric}" for label in run_labels]
        icc = compute_icc_2_1(df[cols].values)
        print(f"  {metric}: ICC(2,1) = {icc:.4f}")

    # === 按分数区间分析分歧 ===
    print("\n--- 按分数区间分析分歧 ---")
    bins = [0, 1, 2, 3, 4, 5]
    bin_labels = ["0-1", "1-2", "2-3", "3-4", "4-5"]
    for metric in ["alignment_score", "artifact_score"]:
        df[f"{metric}_bin"] = pd.cut(df[f"{metric}_mean"], bins=bins, labels=bin_labels, include_lowest=True)
        print(f"\n  {metric} (按3次运行均值分段):")
        print(f"  {'区间':>6s}  {'图片数':>6s}  {'平均极差':>8s}  {'平均std':>8s}  {'极差>1数':>8s}")
        print(f"  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")
        for bl in bin_labels:
            subset = df[df[f"{metric}_bin"] == bl]
            if len(subset) == 0:
                print(f"  {bl:>6s}  {0:>6d}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>8s}")
                continue
            avg_range = subset[f"{metric}_range"].mean()
            avg_std = subset[f"{metric}_std"].mean()
            n_large = (subset[f"{metric}_range"] > 1.0).sum()
            print(f"  {bl:>6s}  {len(subset):>6d}  {avg_range:>8.4f}  {avg_std:>8.4f}  {n_large:>8d}")

    # === 按极差区间统计图片数量 ===
    print("\n--- 按极差区间统计图片数量 ---")
    range_bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]
    range_bin_labels = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-2.5", "2.5-3", "3-5"]
    for metric in ["alignment_score", "artifact_score"]:
        df[f"{metric}_range_bin"] = pd.cut(df[f"{metric}_range"], bins=range_bins, labels=range_bin_labels, include_lowest=True)
        print(f"\n  {metric} (极差分布):")
        print(f"  {'极差区间':>8s}  {'图片数':>6s}  {'占比':>8s}")
        print(f"  {'-'*8}  {'-'*6}  {'-'*8}")
        for bl in range_bin_labels:
            count = (df[f"{metric}_range_bin"] == bl).sum()
            pct = count / len(df) * 100
            print(f"  {bl:>8s}  {count:>6d}  {pct:>7.1f}%")

    # === 分歧最大的 Top 20 图片 ===
    print("\n--- 分歧最大的 Top 20 图片 ---")
    for metric in ["alignment_score", "artifact_score"]:
        top20 = df.nlargest(20, f"{metric}_range")
        print(f"\n  {metric}:")
        print(f"  {'image_id':>10s}", end="")
        for label in run_labels:
            print(f"  {label:>20s}", end="")
        print(f"  {'mean':>6s}  {'range':>6s}")
        for _, row in top20.iterrows():
            print(f"  {row['image_id']:>10s}", end="")
            for label in run_labels:
                print(f"  {row[f'{label}_{metric}']:>20.4f}", end="")
            print(f"  {row[f'{metric}_mean']:>6.2f}  {row[f'{metric}_range']:>6.2f}")

    print("\n" + "=" * 60)


def compute_icc_2_1(data: np.ndarray) -> float:
    """
    计算ICC(2,1)：每个被试由同一组评分者评分，衡量绝对一致性。
    data: shape (n_subjects, n_raters)
    """
    n, k = data.shape
    grand_mean = data.mean()

    ss_between = n * np.sum((data.mean(axis=0) - grand_mean) ** 2)
    ss_within = np.sum((data.mean(axis=1) - grand_mean) ** 2) * k - ss_between
    ss_total = np.sum((data - grand_mean) ** 2)

    ms_between = ss_between / (k - 1)
    ms_within_subj = (ss_total - ss_between - (np.sum((data.mean(axis=1) - grand_mean) ** 2) * k - ss_between)) / ((n - 1) * (k - 1))

    ms_subjects = np.sum((data.mean(axis=1) - grand_mean) ** 2) * k / (n - 1)
    ms_error = (ss_total - np.sum((data.mean(axis=1) - grand_mean) ** 2) * k - ss_between) / ((n - 1) * (k - 1))

    icc = (ms_subjects - ms_error) / (ms_subjects + (k - 1) * ms_error + k * (ms_between - ms_error) / n)
    return icc


if __name__ == "__main__":
    main()

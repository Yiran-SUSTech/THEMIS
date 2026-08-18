"""
analyze_cls_sampling.py
========================
分析 DiT 1000 类 vs 随机采样 100 类 vs 聚类 100 类的指标分布差异。

输入: D:\THEMIS\DiT_1Kcls_rand100cls_cluster100cls.xlsx (Sheet1)
输出:
  1. bar_metrics_with_ci.png   - 各指标柱状图 (带 95% CI 误差棒)
  2. deviation_from_baseline.png - 各 100 类样本相对 1000 类基线的偏差
  3. boxplot_rand_vs_cluster.png  - 随机 vs 聚类 的分布箱线图
  4. summary_stats.csv         - 汇总统计表
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = r"d:\THEMIS"
XLSX_PATH = os.path.join(BASE_DIR, "DiT_1Kcls_rand100cls_cluster100cls.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "cls_sampling_analysis_output")

# 4 个核心指标
METRICS = ["alignment_norm", "authenticity_norm", "composite_product", "composite_harmonic"]
METRIC_LABELS = {
    "alignment_norm": "Alignment Norm",
    "authenticity_norm": "Authenticity Norm",
    "composite_product": "Composite Product",
    "composite_harmonic": "Composite Harmonic",
}

# 分组
BASELINE = "Sys_DiT_ref_class1000"
RAND_PREFIX = "Sys_DiT_ref_rand100cls"
CLUSTER_PREFIX = "Sys_DiT_ref_cluster100cls"


def classify_source(name):
    if name == BASELINE:
        return "Baseline (1000 cls)"
    if name.startswith(RAND_PREFIX):
        return "Random 100 cls"
    if name.startswith(CLUSTER_PREFIX):
        return "Cluster 100 cls"
    return "Other"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_excel(XLSX_PATH, sheet_name="Sheet1")
    df["group"] = df["source"].apply(classify_source)
    print(f"加载数据: {len(df)} 行")
    print(df[["source", "group"] + METRICS].to_string(index=False))

    # 颜色
    GROUP_COLORS = {
        "Baseline (1000 cls)": "#2ca02c",
        "Random 100 cls": "#1f77b4",
        "Cluster 100 cls": "#d62728",
    }

    # ========== 图 1: 各指标柱状图 (带 95% CI 误差棒) ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    x = np.arange(len(df))
    bar_width = 0.6

    for ax_idx, metric in enumerate(METRICS):
        ax = axes[ax_idx]
        ci_low = df[f"{metric}_ci_low"].values
        ci_high = df[f"{metric}_ci_high"].values
        yerr = np.vstack([df[metric].values - ci_low, ci_high - df[metric].values])
        colors = [GROUP_COLORS[g] for g in df["group"]]

        bars = ax.bar(x, df[metric].values, width=bar_width, color=colors,
                      edgecolor="black", linewidth=0.8, zorder=3)
        ax.errorbar(x, df[metric].values, yerr=yerr, fmt="none",
                    ecolor="black", capsize=5, capthick=1.5, linewidth=1.5, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(df["source"], rotation=35, ha="right", fontsize=10)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=14, fontweight="bold")
        ax.set_title(METRIC_LABELS[metric], fontsize=16, fontweight="bold")
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)

        # 在柱顶标注数值
        for bar, val in zip(bars, df[metric].values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 图例
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, edgecolor="black", label=g)
                      for g, c in GROUP_COLORS.items()]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3,
               fontsize=13, framealpha=0.9, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("DiT: 1000-class Baseline vs Random/Cluster 100-class Samples",
                 fontsize=18, fontweight="bold", y=1.04)
    plt.tight_layout()
    out1 = os.path.join(OUTPUT_DIR, "bar_metrics_with_ci.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[1] 保存: {out1}")

    # ========== 图 2: 偏差图 (相对 1000 类基线) ==========
    baseline_vals = df[df["source"] == BASELINE].set_index("source")
    baseline_row = baseline_vals.iloc[0]
    sub_df = df[df["source"] != BASELINE].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax_idx, metric in enumerate(METRICS):
        ax = axes[ax_idx]
        deviations = sub_df[metric].values - baseline_row[metric]
        colors = [GROUP_COLORS[g] for g in sub_df["group"]]
        bars = ax.barh(np.arange(len(sub_df)), deviations, color=colors,
                       edgecolor="black", linewidth=0.8, zorder=3)
        ax.set_yticks(np.arange(len(sub_df)))
        ax.set_yticklabels(sub_df["source"], fontsize=14)
        ax.set_xlabel(f"Deviation from Baseline", fontsize=16)
        ax.set_title(METRIC_LABELS[metric], fontsize=17, fontweight="bold")
        ax.axvline(0, color="black", linewidth=1.5, linestyle="-", zorder=4)
        ax.tick_params(axis="x", labelsize=14)
        ax.grid(axis="x", alpha=0.3, linestyle="--", zorder=0)

        # 标注偏差值在条带内部中心
        for bar, val in zip(bars, deviations):
            x_pos = val / 2  # 条带中心
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.4f}", va="center", ha="center",
                    fontsize=12, fontweight="bold", color="Black")

        # 设置 x 轴范围, 留出 padding 防止超出
        x_min = min(deviations.min(), 0)
        x_max = max(deviations.max(), 0)
        padding = (x_max - x_min) * 0.15
        ax.set_xlim(x_min - padding, x_max + padding)

    fig.legend(handles=legend_handles, loc="upper center", ncol=3,
               fontsize=14, framealpha=0.9, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Deviation from 1000-class Baseline (Random & Cluster 100-class)",
                 fontsize=19, fontweight="bold", y=1.03)
    plt.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, "deviation_from_baseline.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[2] 保存: {out2}")

    # ========== 图 3: 箱线图 (随机 vs 聚类, 3 次运行) ==========
    fig, axes = plt.subplots(1, 4, figsize=(20, 7))
    for ax_idx, metric in enumerate(METRICS):
        ax = axes[ax_idx]
        rand_vals = df[df["group"] == "Random 100 cls"][metric].values
        cluster_vals = df[df["group"] == "Cluster 100 cls"][metric].values
        baseline_val = [baseline_row[metric]]

        positions = [0, 1, 2]
        bp = ax.boxplot([baseline_val, rand_vals, cluster_vals],
                        positions=positions, widths=0.5, patch_artist=True,
                        showmeans=True, meanprops=dict(marker="D", markerfacecolor="white",
                                                       markeredgecolor="black", markersize=8))
        for patch, color in zip(bp["boxes"],
                                [GROUP_COLORS["Baseline (1000 cls)"],
                                 GROUP_COLORS["Random 100 cls"],
                                 GROUP_COLORS["Cluster 100 cls"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 散点叠加
        for pos, vals, color in [(0, baseline_val, GROUP_COLORS["Baseline (1000 cls)"]),
                                  (1, rand_vals, GROUP_COLORS["Random 100 cls"]),
                                  (2, cluster_vals, GROUP_COLORS["Cluster 100 cls"])]:
            jitter = np.random.RandomState(42).normal(0, 0.04, size=len(vals))
            ax.scatter([pos + j for j in jitter], vals, color=color,
                       edgecolor="black", linewidth=0.5, s=70, zorder=5)

        ax.set_xticks(positions)
        ax.set_xticklabels(["Baseline\n(1000 cls)", "Random\n(100 cls)", "Cluster\n(100 cls)"],
                           fontsize=14)
        ax.set_title(METRIC_LABELS[metric], fontsize=17, fontweight="bold")
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)

    fig.suptitle("Distribution Comparison: Baseline vs Random vs Cluster (3 runs each)",
                 fontsize=19, fontweight="bold", y=1.02)
    plt.tight_layout()
    out3 = os.path.join(OUTPUT_DIR, "boxplot_rand_vs_cluster.png")
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[3] 保存: {out3}")

    # ========== 汇总统计表 ==========
    summary_rows = []
    for metric in METRICS:
        bval = baseline_row[metric]
        rand_vals = df[df["group"] == "Random 100 cls"][metric].values
        cluster_vals = df[df["group"] == "Cluster 100 cls"][metric].values

        for name, vals in [("Random 100 cls", rand_vals), ("Cluster 100 cls", cluster_vals)]:
            deviations = vals - bval
            summary_rows.append({
                "Metric": metric,
                "Group": name,
                "Baseline (1000 cls)": round(bval, 4),
                "Mean": round(np.mean(vals), 4),
                "Std": round(np.std(vals, ddof=1), 4),
                "Min": round(np.min(vals), 4),
                "Max": round(np.max(vals), 4),
                "Mean Deviation": round(np.mean(deviations), 4),
                "Max |Deviation|": round(np.max(np.abs(deviations)), 4),
                "Relative Deviation %": round(np.mean(np.abs(deviations)) / bval * 100, 2),
            })

    summary_df = pd.DataFrame(summary_rows)
    out4 = os.path.join(OUTPUT_DIR, "summary_stats.csv")
    summary_df.to_csv(out4, index=False, encoding="utf-8-sig")
    print(f"[4] 保存: {out4}")
    print("\n=== 汇总统计 ===")
    print(summary_df.to_string(index=False))

    print(f"\n所有输出已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

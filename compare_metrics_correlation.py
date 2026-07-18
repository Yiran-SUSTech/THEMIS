"""
compare_metrics_correlation.py

分析 FID / Human_composite / Sys_composite 三个指标之间的相关性与误差。
- 任务一: 基于 composite_product
- 任务二: 基于 composite_harmonic

以 Human_composite 作为 ground-truth 基准，计算:
  - Spearman 等级相关系数 ρ
  - Pearson 相关系数 r
  - Normalized RMSE

注意:
  - FID 越低越好 (ranking: ascending, rank 1 = best = 最低 FID)
  - Composite 越高越好 (ranking: descending, rank 1 = best = 最高 composite)
  - 因此 FID 与 composite 之间预期为负相关 (这是正常的)
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


# ==================== 工具函数 ====================

def rank_series(values, ascending=True):
    """对序列排名, rank=1 表示最佳。
    ascending=True  -> 最小值 rank=1 (FID 这种, 越低越好)
    ascending=False -> 最大值 rank=1 (composite 这种, 越高越好)
    method='average' 与 Spearman 默认行为一致 (处理 tie)
    """
    return values.rank(method="average", ascending=ascending)


def minmax_normalize_align_higher_better(values):
    """Min-Max 归一化到 [0,1], 使"越大越好"。
    适用于 composite (原本就越高越好)。
    """
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - vmin) / (vmax - vmin)


def minmax_normalize_fid_align_higher_better(values):
    """Min-Max 归一化 FID 到 [0,1], 并反转方向使"越大越好"。
    FID 越低越好, 所以反转: norm = (max - fid) / (max - min)
    这样 FID 最低的模型 norm=1 (最佳)。
    """
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (vmax - values) / (vmax - vmin)


def compute_metrics(x_vals, y_vals):
    """计算 x 相对于 y (ground-truth) 的 Spearman / Pearson / normalized RMSE。

    x, y 均为对齐到"越高越好"方向的 [0,1] 归一化值。
    """
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)

    # Spearman (用原始值的 rank, scipy 内部会 rank)
    rho, p_rho = spearmanr(x_vals, y_vals)
    # Pearson (用归一化后的值)
    r, p_r = pearsonr(x_arr, y_arr)
    # Normalized RMSE (在 [0,1] 归一化空间)
    rmse = float(np.sqrt(np.mean((x_arr - y_arr) ** 2)))

    return {
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "pearson_r": float(r),
        "pearson_p": float(p_r),
        "normalized_rmse": rmse,
    }


def run_task(df, model_col, fid_col, human_col, sys_col, task_name, composite_type):
    """执行一个分析任务"""
    print("\n" + "=" * 78)
    print(f"  {task_name}")
    print(f"  Ground-truth 基准: {human_col}")
    print(f"  对比指标: {fid_col}  /  {sys_col}")
    print("=" * 78)

    # ---- 1. 独立排序 ----
    # FID: 越低越好 -> ascending=True (rank 1 = 最低 FID = 最佳)
    # Human / Sys composite: 越高越好 -> ascending=False (rank 1 = 最高 = 最佳)
    df_eval = df[[model_col, fid_col, human_col, sys_col]].copy()

    df_eval["FID_rank"] = rank_series(df_eval[fid_col], ascending=True)
    df_eval["Human_rank"] = rank_series(df_eval[human_col], ascending=False)
    df_eval["Sys_rank"] = rank_series(df_eval[sys_col], ascending=False)

    # 排序展示表 (按 Human_rank 升序, 即按人类评价最佳 -> 最差)
    display_df = df_eval.sort_values("Human_rank").reset_index(drop=True)
    display_df = display_df[[model_col, fid_col, "FID_rank",
                             human_col, "Human_rank",
                             sys_col, "Sys_rank"]]
    # 保留 4 位小数
    for c in [fid_col, human_col, sys_col]:
        display_df[c] = display_df[c].round(4)
    for c in ["FID_rank", "Human_rank", "Sys_rank"]:
        display_df[c] = display_df[c].round(4)

    print(f"\n[1] 各指标独立排序 (按 Human {composite_type} 升序展示, rank=1 为最佳):")
    print(display_df.to_string(index=False))

    # ---- 2-5. 相关性与误差 ----
    # 归一化到 [0,1] 且方向对齐为"越大越好"
    fid_norm = minmax_normalize_fid_align_higher_better(df_eval[fid_col])
    human_norm = minmax_normalize_align_higher_better(df_eval[human_col])
    sys_norm = minmax_normalize_align_higher_better(df_eval[sys_col])

    # FID vs Human
    m_fid = compute_metrics(fid_norm, human_norm)
    # Sys vs Human
    m_sys = compute_metrics(sys_norm, human_norm)

    # 汇总表
    summary_rows = [
        {
            "对比对 (Comparison)": f"FID  vs  Human_{composite_type}",
            "Spearman ρ": round(m_fid["spearman_rho"], 4),
            "Spearman p": round(m_fid["spearman_p"], 4),
            "Pearson r": round(m_fid["pearson_r"], 4),
            "Pearson p": round(m_fid["pearson_p"], 4),
            "Normalized RMSE": round(m_fid["normalized_rmse"], 4),
        },
        {
            "对比对 (Comparison)": f"Sys_{composite_type}  vs  Human_{composite_type}",
            "Spearman ρ": round(m_sys["spearman_rho"], 4),
            "Spearman p": round(m_sys["spearman_p"], 4),
            "Pearson r": round(m_sys["pearson_r"], 4),
            "Pearson p": round(m_sys["pearson_p"], 4),
            "Normalized RMSE": round(m_sys["normalized_rmse"], 4),
        },
    ]
    summary_df = pd.DataFrame(summary_rows)

    print(f"\n[2-5] 相关性与误差汇总 (以 Human_{composite_type} 为 ground-truth):")
    print(summary_df.to_string(index=False))

    # 说明
    print(f"\n说明:")
    print(f"  - Spearman ρ: 基于排名的等级相关, |ρ|→1 表示排序一致性越高")
    print(f"  - Pearson r: 基于归一化值的线性相关, |r|→1 表示线性关系越强")
    print(f"  - Normalized RMSE: 在 [0,1] 归一化空间下的 RMSE, 越小表示越接近 ground-truth")
    print(f"  - FID 越低越好, composite 越高越好, 二者负相关为正常预期")

    return display_df, summary_df


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(
        description="Compare FID / Human_composite / Sys_composite: Spearman, Pearson, normalized RMSE"
    )
    parser.add_argument("--input", default="metrics_compare.xlsx",
                        help="输入 Excel 文件路径 (default: metrics_compare.xlsx)")
    parser.add_argument("--output-dir", default="metrics_compare_results",
                        help="输出目录 (default: metrics_compare_results)")
    parser.add_argument("--model-col", default=None,
                        help="模型名列 (default: 自动检测第一列)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 读取数据
    df = pd.read_excel(args.input)
    # 第一列通常是模型名
    if args.model_col:
        model_col = args.model_col
    else:
        model_col = df.columns[0]
    df = df.rename(columns={model_col: "model"})
    model_col = "model"

    print(f"读取数据: {args.input}")
    print(f"模型列: {model_col}")
    print(f"数据预览:")
    print(df.to_string(index=False))

    required = ["FID", "Human_composite_product", "Human_composite_harmonic",
                "Sys_composite_product", "Sys_composite_harmonic"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}, 当前列: {list(df.columns)}")

    # ---- 任务一: composite_product ----
    disp1, summ1 = run_task(
        df, model_col,
        fid_col="FID",
        human_col="Human_composite_product",
        sys_col="Sys_composite_product",
        task_name="任务一: 基于 composite_product 的相关性分析",
        composite_type="composite_product",
    )
    disp1.to_csv(os.path.join(args.output_dir, "task1_ranking_product.csv"),
                 index=False, encoding="utf-8-sig")
    summ1.to_csv(os.path.join(args.output_dir, "task1_correlation_product.csv"),
                 index=False, encoding="utf-8-sig")

    # ---- 任务二: composite_harmonic ----
    disp2, summ2 = run_task(
        df, model_col,
        fid_col="FID",
        human_col="Human_composite_harmonic",
        sys_col="Sys_composite_harmonic",
        task_name="任务二: 基于 composite_harmonic 的相关性分析",
        composite_type="composite_harmonic",
    )
    disp2.to_csv(os.path.join(args.output_dir, "task2_ranking_harmonic.csv"),
                 index=False, encoding="utf-8-sig")
    summ2.to_csv(os.path.join(args.output_dir, "task2_correlation_harmonic.csv"),
                 index=False, encoding="utf-8-sig")

    # ---- 综合对比表 ----
    combined = pd.concat([summ1, summ2], ignore_index=True)
    combined.to_csv(os.path.join(args.output_dir, "combined_correlation_summary.csv"),
                    index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("  综合对比 (Task 1 + Task 2)")
    print("=" * 78)
    print(combined.to_string(index=False))

    print(f"\n输出文件保存在: {args.output_dir}/")
    print(f"  - task1_ranking_product.csv")
    print(f"  - task1_correlation_product.csv")
    print(f"  - task2_ranking_harmonic.csv")
    print(f"  - task2_correlation_harmonic.csv")
    print(f"  - combined_correlation_summary.csv")


if __name__ == "__main__":
    main()

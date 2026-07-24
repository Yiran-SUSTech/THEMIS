"""
contour_and_correlation_analysis.py
====================================
等高线数据提取与可视化 + 相关性与一致性分析

功能:
  1. 等高线可视化: 从多个 analysis_output_* 目录读取 joint_probability_surfaces.csv,
     在同一张图上绘制不同来源的等高线 (支持多个联合概率水平, 每个概率生成一张图)。
     线型约定: Sys 来源用实线, Human 来源用虚线; 同模型 Sys/Human 共享颜色。
  2. 相关性分析: 计算 10 个 composite 指标的 (Sys vs Human) 和 ((-FID) vs Human)
     的 Spearman / Pearson / Normalized RMSE。

前置条件:
  - 需先运行 extract_and_analyze_scores.py (带 --vendi-ratio-csv) 生成
    joint_probability_surfaces.csv 和 cross_class_macro_avg.csv

使用方式:
    # 多概率等高线 + 相关性分析
    python contour_and_correlation_analysis.py \
        --source Sys_DiT_ref Human_DiT Sys_VAR_ref_500 Human_VAR_500 Sys_IMF_ref Human_IMF \
        --joint_prob 25 50 75

    # 仅画等高线 (单概率)
    python contour_and_correlation_analysis.py --source Sys_DiT_ref Human_DiT --joint_prob 50 --no-correlation

    # 仅做相关性分析
    python contour_and_correlation_analysis.py --source Sys_DiT_ref Human_DiT Sys_VAR_ref_500 Human_VAR_500 --no-contour

输出:
    <output_dir>/contour_comparison_p<level>.png   - 多来源等高线对比图 (每个概率一张)
    <output_dir>/correlation_summary.csv           - 一致性汇总表
    <output_dir>/model_scores.csv                  - 各模型的指标值明细
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from pathlib import Path

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = r"d:\THEMIS"

# ==================== 模型 FID 映射 ====================
# (model_key -> (full_name, fid_value))
# model_key 从 source 名称中提取, 如 "Sys_VAR_ref_500" -> "VAR"
MODEL_FID = {
    "DiT":        ("DiT-XL",     2.27),
    "VAR":        ("VAR-d24",    2.09),
    "IMF":        ("iMF-XL",     1.54),
    "IMFfdloss":  ("iMF-FDloss", 0.72),
    "JiTfdloss":  ("JiT-FDloss", 0.72),
}

# 需要分析的 10 个指标
METRICS_TO_ANALYZE = [
    "composite_product",
    "composite_harmonic",
    "composite_product_x_vendi_ratio",
    "composite_product_x_vendi_ratio_train",
    "composite_harmonic_x_vendi_ratio",
    "composite_harmonic_x_vendi_ratio_train",
    "composite_product_x_MEAN_VENDI_RATIO",
    "composite_harmonic_x_MEAN_VENDI_RATIO",
    "composite_product_x_MEAN_VENDI_RATIO_TRAIN",
    "composite_harmonic_x_MEAN_VENDI_RATIO_TRAIN",
]


# ==================== 工具函数 ====================

def extract_model_key(source_name):
    """从 source 名称提取模型关键字。

    'Sys_VAR_ref_500'   -> 'VAR'
    'Human_DiT'         -> 'DiT'
    'Sys_IMFfdloss_ref' -> 'IMFfdloss'
    'Human_JiTfdloss'   -> 'JiTfdloss'
    """
    name = source_name
    # 去除前缀
    for prefix in ["Sys_", "Human_", "User_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # 去除后缀 (按长度降序匹配, 避免短后缀先匹配)
    suffixes = [
        "_ref_500", "_ref_600", "_ref_800", "_ref_400", "_ref_200",
        "_ref", "_500", "_600", "_800", "_400", "_200",
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def is_system_source(source_name):
    """判断是否为系统打分来源"""
    return source_name.startswith("Sys_")


def is_human_source(source_name):
    """判断是否为人类打分来源"""
    return source_name.startswith("Human_") or source_name.startswith("User_")


def find_analysis_dirs():
    """搜索 BASE_DIR 下所有 analysis_output_* 目录, 返回路径列表。"""
    result = []
    if not os.path.isdir(BASE_DIR):
        return result
    for name in os.listdir(BASE_DIR):
        full = os.path.join(BASE_DIR, name)
        if os.path.isdir(full) and name.startswith("analysis_output_"):
            result.append(full)
    return result


def find_source_dir(source_name, analysis_dirs=None):
    """在 analysis_output_* 目录中查找包含指定 source 的目录。

    查找策略 (按优先级):
      1. 精确匹配 AVG_PERCLASS 行 (如 "Sys_DiT_ref_AVG_PERCLASS") — 避免前缀歧义
      2. 前缀匹配 source 列 (如 "Sys_DiT_ref_1" 匹配 source_name="Sys_DiT_ref")

    Returns:
        dir_path or None
    """
    if analysis_dirs is None:
        analysis_dirs = find_analysis_dirs()

    # 第一轮: 精确匹配 {source_name}_AVG_PERCLASS 行
    target_perclass = f"{source_name}_AVG_PERCLASS"
    for d in analysis_dirs:
        csv_path = os.path.join(d, "cross_class_macro_avg.csv")
        if not os.path.isfile(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig", usecols=["source"])
            sources_in_csv = set(df["source"].dropna().tolist())
            if target_perclass in sources_in_csv:
                return d
        except Exception:
            continue

    # 第二轮: 前缀匹配 (兜底)
    for d in analysis_dirs:
        csv_path = os.path.join(d, "cross_class_macro_avg.csv")
        if not os.path.isfile(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig", usecols=["source"])
            sources_in_csv = set(df["source"].dropna().tolist())
            for s in sources_in_csv:
                if s == source_name or s.startswith(source_name + "_"):
                    return d
        except Exception:
            continue
    return None


def load_contour_surface(surface_csv, group_name):
    """从 joint_probability_surfaces.csv 读取指定 group 的概率曲面。

    Returns:
        thresholds: 1D array of threshold values
        prob_grid: 2D array [n_thresh, n_thresh] of probability
        或 (None, None) 如果读取失败
    """
    if not os.path.isfile(surface_csv):
        return None, None
    try:
        df = pd.read_csv(surface_csv, encoding="utf-8-sig")
    except Exception as e:
        print(f"[WARN] 读取曲面数据失败: {surface_csv}: {e}")
        return None, None

    sub = df[df["group"] == group_name]
    if sub.empty:
        return None, None

    # 构建 2D 网格
    a_thresh = sorted(sub["alignment_threshold"].unique())
    r_thresh = sorted(sub["artifact_threshold"].unique())
    n = len(a_thresh)
    prob = np.zeros((n, n))
    # 用 pivot 构建 2D 矩阵
    pivot = sub.pivot_table(
        index="alignment_threshold", columns="artifact_threshold", values="probability"
    )
    prob = pivot.values
    thresholds = np.array(a_thresh)
    return thresholds, prob


def extract_contour_line(thresholds, prob_grid, level):
    """从概率曲面提取指定 level 的等高线数据点。

    Returns:
        list of np.array, 每个是 (N, 2) 的 [x, y] 坐标
    """
    if thresholds is None or prob_grid is None:
        return []
    # 用 matplotlib contour 提取等高线
    fig_tmp, ax_tmp = plt.subplots()
    try:
        cs = ax_tmp.contour(
            thresholds, thresholds, prob_grid.T,
            levels=[level], colors=["blue"], linewidths=1,
        )
        segments = []
        # 兼容不同 matplotlib 版本
        if hasattr(cs, "allsegs") and cs.allsegs:
            for seg in cs.allsegs[0]:
                if len(seg) > 0:
                    segments.append(np.array(seg))
        plt.close(fig_tmp)
        return segments
    except Exception as e:
        plt.close(fig_tmp)
        print(f"[WARN] 等高线提取失败 (level={level}): {e}")
        return []


def minmax_normalize_higher_better(values):
    """Min-Max 归一化到 [0,1], 方向: 越大越好。"""
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - vmin) / (vmax - vmin)


def minmax_normalize_fid_higher_better(values):
    """Min-Max 归一化 FID 到 [0,1] 并反转方向 (FID 越低越好 -> 归一化后越大越好)。
    使用 -FID 进行归一化。
    """
    neg_values = -values
    vmin, vmax = neg_values.min(), neg_values.max()
    if vmax - vmin < 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (neg_values - vmin) / (vmax - vmin)


def compute_correlation_metrics(x_vals, y_vals):
    """计算 x 相对于 y (ground-truth) 的 Spearman / Pearson / Normalized RMSE。

    x, y 均为对齐到"越高越好"方向的 [0,1] 归一化值。
    """
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    n = len(x_arr)
    if n < 2:
        return {"spearman_rho": float("nan"), "spearman_p": float("nan"),
                "pearson_r": float("nan"), "pearson_p": float("nan"),
                "normalized_rmse": float("nan")}

    rho, p_rho = spearmanr(x_vals, y_vals)
    r, p_r = pearsonr(x_arr, y_arr)
    rmse = float(np.sqrt(np.mean((x_arr - y_arr) ** 2)))
    return {
        "spearman_rho": float(rho) if not np.isnan(rho) else float("nan"),
        "spearman_p": float(p_rho) if not np.isnan(p_rho) else float("nan"),
        "pearson_r": float(r) if not np.isnan(r) else float("nan"),
        "pearson_p": float(p_r) if not np.isnan(p_r) else float("nan"),
        "normalized_rmse": rmse,
    }


# ==================== 等高线可视化 ====================

# 各概率水平对应的坐标轴范围: {probability: (xlim, ylim)}
CONTOUR_AXIS_RANGES = {
    75: ((0.0, 0.8), (0.0, 0.5)),
    50: ((0.0, 1.0), (0.0, 0.75)),
    25: ((0.0, 1.0), (0.0, 1.0)),
}


def plot_contour_comparison(source_groups, joint_probs, output_dir):
    """绘制多个来源的等高线对比图 (每个概率水平生成一张图)。

    线型约定:
      - Sys 来源: 实线 (颜色按模型区分)
      - Human 来源: 虚线 (颜色按模型区分, 与同模型 Sys 共享颜色)

    Args:
        source_groups: list of source group names (如 ["Sys_DiT_ref", "Human_DiT", ...])
        joint_probs: 联合概率值列表 (如 [25, 50, 75])
        output_dir: 输出目录
    """
    # 搜索 analysis_output_* 目录
    analysis_dirs = find_analysis_dirs()
    if not analysis_dirs:
        print("[ERROR] 未找到 analysis_output_* 目录")
        return

    # 为每个 source 加载曲面数据 (只加载一次, 复用于所有概率水平)
    source_surfaces = {}  # {source_name: (thresholds, prob_grid)}
    for src in source_groups:
        found_dir = find_source_dir(src, analysis_dirs)
        if found_dir is None:
            print(f"  [WARN] 未找到 source '{src}' 的分析目录, 跳过")
            continue
        surface_csv = os.path.join(found_dir, "joint_probability_surfaces.csv")
        thresholds, prob = load_contour_surface(surface_csv, src)
        if thresholds is None:
            print(f"  [WARN] source '{src}' 在 {found_dir} 中无曲面数据, 跳过")
            continue
        source_surfaces[src] = (thresholds, prob)
        print(f"  [OK] 曲面数据加载: {src} (from {os.path.basename(found_dir)})")

    if not source_surfaces:
        print("[ERROR] 无可用的等高线数据, 跳过可视化")
        return

    # 按模型分配颜色 (同模型的 Sys 和 Human 共享颜色, 用线型区分)
    model_keys = sorted(set(extract_model_key(s) for s in source_surfaces.keys()))
    n_models = len(model_keys)
    cmap_colors = plt.cm.tab10(np.linspace(0, 0.95, max(n_models, 1)))
    model_color = {mk: cmap_colors[i] for i, mk in enumerate(model_keys)}

    # 对每个概率水平生成一张图
    for jp in joint_probs:
        level = jp / 100.0
        print(f"\n--> 等高线可视化: {len(source_surfaces)} 个来源, 联合概率 = {jp}% (level={level:.2f})")

        # 提取等高线数据
        contour_data = {}  # {source_name: segments}
        for src, (thresholds, prob) in source_surfaces.items():
            segments = extract_contour_line(thresholds, prob, level)
            if not segments:
                print(f"  [WARN] source '{src}' 在 level={level:.2f} 无等高线数据, 跳过")
                continue
            contour_data[src] = segments

        if not contour_data:
            print(f"  [ERROR] 无可用的等高线数据 (level={level:.2f}), 跳过此概率")
            continue

        # 绘图
        n_sources = len(contour_data)
        fig_w = max(10, min(20, 6 + n_sources * 1.2))
        fig_h = max(8, min(18, 6 + n_sources * 0.8))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        for src, segments in contour_data.items():
            mk = extract_model_key(src)
            color = model_color.get(mk, "gray")
            # Sys 用实线, Human 用虚线
            ls = "--" if is_human_source(src) else "-"
            for seg_idx, seg in enumerate(segments):
                label = src if seg_idx == 0 else ""
                ax.plot(seg[:, 0], seg[:, 1], color=color, linestyle=ls,
                        linewidth=2.5, alpha=0.85, label=label)

        ax.set_xlabel("Alignment threshold (normalized)", fontsize=12)
        ax.set_ylabel("Artifact threshold (normalized)", fontsize=12)
        ax.set_title(f"Joint Probability Contour Comparison\n"
                     f"P(alignment >= s, artifact >= t) = {jp}%",
                     fontsize=14, fontweight="bold")

        # 根据概率水平设置坐标轴范围
        xlim, ylim = CONTOUR_AXIS_RANGES.get(int(jp), ((0.0, 1.0), (0.0, 1.0)))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("auto")
        ax.grid(alpha=0.3, linestyle="--")

        # 图例
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9,
                  title="Source", title_fontsize=10)

        plt.tight_layout()
        jp_label = f"{jp:g}"
        out_path = os.path.join(output_dir, f"contour_comparison_p{jp_label}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> 等高线对比图保存至: {out_path}")


# ==================== 相关性分析 ====================

def load_model_scores(source_groups):
    """从 analysis_output_* 目录加载各模型的指标值。

    对于每个 source, 从 cross_class_macro_avg.csv 中提取 AVG_PERCLASS 行的指标值。

    Returns:
        DataFrame: 每行一个 source, 列包括 source, model_key, role(Sys/Human),
                   以及 10 个指标的值
    """
    analysis_dirs = find_analysis_dirs()
    if not analysis_dirs:
        print("[ERROR] 未找到 analysis_output_* 目录")
        return pd.DataFrame()

    rows = []
    for src in source_groups:
        found_dir = find_source_dir(src, analysis_dirs)
        if found_dir is None:
            print(f"  [WARN] 未找到 source '{src}' 的分析目录, 跳过")
            continue

        csv_path = os.path.join(found_dir, "cross_class_macro_avg.csv")
        if not os.path.isfile(csv_path):
            print(f"  [WARN] {found_dir} 中无 cross_class_macro_avg.csv, 跳过 '{src}'")
            continue

        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception as e:
            print(f"  [WARN] 读取 {csv_path} 失败: {e}")
            continue

        # 查找 AVG_PERCLASS 行
        target_row_name = f"{src}_AVG_PERCLASS"
        row = df[df["source"] == target_row_name]
        if row.empty:
            # 也尝试直接匹配 source 名称
            row = df[df["source"] == src]
        if row.empty:
            print(f"  [WARN] 在 {os.path.basename(found_dir)} 中未找到 '{target_row_name}' 行, 跳过")
            continue

        row_data = row.iloc[0].to_dict()
        model_key = extract_model_key(src)
        role = "Sys" if is_system_source(src) else "Human"

        entry = {
            "source": src,
            "model_key": model_key,
            "role": role,
        }
        # 提取 10 个指标的值
        for metric in METRICS_TO_ANALYZE:
            entry[metric] = float(row_data.get(metric, float("nan")))

        rows.append(entry)
        print(f"  [OK] {src} ({role}, model={model_key})")

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def run_correlation_analysis(source_groups, output_dir):
    """执行相关性分析, 生成汇总表。

    对于每个指标, 计算:
      - Sys vs Human: Spearman / Pearson / NRMSE
      - (-FID) vs Human: Spearman / Pearson / NRMSE

    需要将 Sys 和 Human 按 model_key 配对。
    """
    print(f"\n--> 相关性分析: 加载模型指标数据...")

    scores_df = load_model_scores(source_groups)
    if scores_df.empty:
        print("[ERROR] 无可用数据, 跳过相关性分析")
        return pd.DataFrame(), pd.DataFrame()

    # 按 model_key 配对 Sys 和 Human
    sys_df = scores_df[scores_df["role"] == "Sys"].set_index("model_key")
    human_df = scores_df[scores_df["role"] == "Human"].set_index("model_key")

    # 找到同时有 Sys 和 Human 数据的模型
    common_models = sorted(set(sys_df.index) & set(human_df.index))
    if not common_models:
        print("[ERROR] 未找到同时有 Sys 和 Human 数据的模型")
        print(f"  Sys models: {sorted(sys_df.index.tolist())}")
        print(f"  Human models: {sorted(human_df.index.tolist())}")
        return pd.DataFrame(), pd.DataFrame()

    print(f"  匹配到的模型: {common_models} ({len(common_models)} 个)")

    # 构建 model-level 数据表
    model_rows = []
    for mk in common_models:
        model_full_name, fid = MODEL_FID.get(mk, (mk, float("nan")))
        row = {
            "model_key": mk,
            "model": model_full_name,
            "FID": fid,
            "neg_FID": -fid if not np.isnan(fid) else float("nan"),
        }
        for metric in METRICS_TO_ANALYZE:
            row[f"Sys_{metric}"] = float(sys_df.loc[mk, metric]) if mk in sys_df.index else float("nan")
            row[f"Human_{metric}"] = float(human_df.loc[mk, metric]) if mk in human_df.index else float("nan")
        model_rows.append(row)

    model_scores = pd.DataFrame(model_rows)
    model_scores_csv = os.path.join(output_dir, "model_scores.csv")
    model_scores.to_csv(model_scores_csv, index=False, encoding="utf-8-sig")
    print(f"\n  各模型指标值明细:")
    print(model_scores.to_string(index=False))
    print(f"\n  -> 保存至: {model_scores_csv}")

    # 计算相关性
    summary_rows = []
    for metric in METRICS_TO_ANALYZE:
        sys_col = f"Sys_{metric}"
        human_col = f"Human_{metric}"

        # 提取非 NaN 的配对数据
        valid = model_scores[[sys_col, human_col]].dropna()
        n_valid = len(valid)
        if n_valid < 2:
            print(f"  [WARN] 指标 '{metric}' 有效数据不足 ({n_valid} < 2), 跳过")
            continue

        # 归一化到 [0,1] (越大越好)
        sys_norm = minmax_normalize_higher_better(valid[sys_col])
        human_norm = minmax_normalize_higher_better(valid[human_col])

        # Sys vs Human
        m_sys = compute_correlation_metrics(sys_norm, human_norm)
        summary_rows.append({
            "对比对 (Comparison)": f"Sys_{metric}  vs  Human_{metric}",
            "n": n_valid,
            "Spearman ρ": round(m_sys["spearman_rho"], 4),
            "Spearman p": round(m_sys["spearman_p"], 4),
            "Pearson r": round(m_sys["pearson_r"], 4),
            "Pearson p": round(m_sys["pearson_p"], 4),
            "Normalized RMSE": round(m_sys["normalized_rmse"], 4),
        })

        # (-FID) vs Human
        fid_valid = model_scores[["FID", human_col]].dropna()
        n_fid = len(fid_valid)
        if n_fid >= 2:
            fid_norm = minmax_normalize_fid_higher_better(fid_valid["FID"])
            human_norm_fid = minmax_normalize_higher_better(fid_valid[human_col])
            m_fid = compute_correlation_metrics(fid_norm, human_norm_fid)
            summary_rows.append({
                "对比对 (Comparison)": f"(-FID)  vs  Human_{metric}",
                "n": n_fid,
                "Spearman ρ": round(m_fid["spearman_rho"], 4),
                "Spearman p": round(m_fid["spearman_p"], 4),
                "Pearson r": round(m_fid["pearson_r"], 4),
                "Pearson p": round(m_fid["pearson_p"], 4),
                "Normalized RMSE": round(m_fid["normalized_rmse"], 4),
            })

    summary_df = pd.DataFrame(summary_rows)

    print(f"\n{'=' * 78}")
    print(f"  相关性与一致性汇总")
    print(f"{'=' * 78}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("  [无有效结果]")

    # 说明
    print(f"\n说明:")
    print(f"  - Spearman ρ: 基于排名的等级相关, |ρ|→1 表示排序一致性越高")
    print(f"  - Pearson r: 基于归一化值的线性相关, |r|→1 表示线性关系越强")
    print(f"  - Normalized RMSE: 在 [0,1] 归一化空间下的 RMSE, 越小越接近 ground-truth")
    print(f"  - FID 越低越好, 使用 -FID 进行归一化使方向对齐 (越大越好)")

    summary_csv = os.path.join(output_dir, "correlation_summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"\n  -> 汇总表保存至: {summary_csv}")

    return model_scores, summary_df


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(
        description="等高线数据可视化 + 相关性一致性分析"
    )
    parser.add_argument(
        "--source",
        nargs="+",
        required=True,
        help="数据来源列表 (如: Sys_DiT_ref Human_DiT Sys_VAR_ref_500 Human_VAR_500)"
    )
    parser.add_argument(
        "--joint_prob",
        nargs="+",
        type=float,
        default=[75],
        help="联合概率值列表, 每个概率生成一张等高线图 (如: 25 50 75, 默认: 75)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录 (默认: contour_correlation_output)"
    )
    parser.add_argument(
        "--no-contour",
        action="store_true",
        help="跳过等高线可视化"
    )
    parser.add_argument(
        "--no-correlation",
        action="store_true",
        help="跳过相关性分析"
    )
    args = parser.parse_args()

    output_dir = args.output or os.path.join(BASE_DIR, "contour_correlation_output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"数据来源: {args.source}")
    print(f"联合概率: {args.joint_prob}")
    print(f"输出目录: {output_dir}")

    # 1. 等高线可视化
    if not args.no_contour:
        plot_contour_comparison(args.source, args.joint_prob, output_dir)

    # 2. 相关性分析
    if not args.no_correlation:
        run_correlation_analysis(args.source, output_dir)

    print(f"\n所有输出保存在: {output_dir}/")


if __name__ == "__main__":
    main()

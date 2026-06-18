"""
评估 AI 测评分数与人类标注均值的一致性。
指标包括：Pearson、Spearman、ICC(2,1)、CCC、MAE、RMSE、Bland-Altman 图。
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXCEL_PATH = os.path.join(os.path.dirname(__file__), "output_results", "aggregated_scores.xlsx")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "agreement_analysis_merged")


def pearson_r(x, y):
    r, p = stats.pearsonr(x, y)
    return r, p


def spearman_r(x, y):
    r, p = stats.spearmanr(x, y)
    return r, p


def icc_2_1(x, y):
    """ICC(2,1): two-way random, absolute agreement, single measures."""
    n = len(x)
    ms_between = np.var((x + y) / 2, ddof=1) * 2
    ms_within = (np.var(x - (x + y) / 2, ddof=1) + np.var(y - (x + y) / 2, ddof=1))
    ms_subjects = np.var((x + y) / 2, ddof=1) * 2
    ms_error = ms_within
    ms_raters = np.var([np.mean(x), np.mean(y)], ddof=1) * n if n > 1 else 0

    ss_total = np.sum((np.concatenate([x, y]) - np.mean(np.concatenate([x, y]))) ** 2)
    ss_between = np.sum(((x + y) / 2 - np.mean(np.concatenate([x, y]))) ** 2) * 2
    ss_within = ss_total - ss_between

    grand_mean = np.mean(np.concatenate([x, y]))
    ss_raters = n * ((np.mean(x) - grand_mean) ** 2 + (np.mean(y) - grand_mean) ** 2)
    ss_error = ss_within - ss_raters

    df_between = n - 1
    df_raters = 1
    df_error = df_between * df_raters

    ms_between_val = ss_between / df_between if df_between > 0 else 0
    ms_raters_val = ss_raters / df_raters if df_raters > 0 else 0
    ms_error_val = ss_error / df_error if df_error > 0 else 0

    icc_val = (ms_between_val - ms_error_val) / (
        ms_between_val + (df_raters + 1) * (ms_raters_val - ms_error_val) / (df_raters + 1)
        + (df_raters + 1) * ms_error_val / n
    ) if ms_error_val > 0 else np.nan

    # Simplified ICC(2,1) formula
    icc_val = (ms_between_val - ms_error_val) / (
        ms_between_val + (2 - 1) * (ms_raters_val - ms_error_val) / 2
        + 2 * ms_error_val / n
    ) if (ms_between_val + ms_error_val) > 0 else np.nan

    return icc_val


def icc_simple(x, y):
    """Simplified ICC(2,1) using correlation and bias correction."""
    r, _ = stats.pearsonr(x, y)
    mean_diff = np.mean(x) - np.mean(y)
    var_pooled = (np.var(x, ddof=1) + np.var(y, ddof=1)) / 2
    bias_correction = 1 - (mean_diff ** 2) / (2 * var_pooled) if var_pooled > 0 else 0
    return r * bias_correction


def concordance_cc(x, y):
    """Lin's Concordance Correlation Coefficient."""
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    r, _ = stats.pearsonr(x, y)
    c_b = (2 * var_x * var_y) / (
        var_x + var_y + (mean_x - mean_y) ** 2
    ) if (var_x + var_y + (mean_x - mean_y) ** 2) > 0 else 0
    return r * c_b


def mae(x, y):
    return np.mean(np.abs(x - y))


def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def bland_altman_plot(x, y, label, save_path):
    mean_vals = (x + y) / 2
    diff_vals = x - y
    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals, ddof=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mean_vals, diff_vals, alpha=0.3, s=10, c="steelblue")
    ax.axhline(mean_diff, color="red", linestyle="-", linewidth=1.5,
               label=f"Mean diff = {mean_diff:.3f}")
    ax.axhline(mean_diff + 1.96 * std_diff, color="gray", linestyle="--",
               label=f"+1.96 SD = {mean_diff + 1.96 * std_diff:.3f}")
    ax.axhline(mean_diff - 1.96 * std_diff, color="gray", linestyle="--",
               label=f"-1.96 SD = {mean_diff - 1.96 * std_diff:.3f}")
    ax.set_xlabel(f"Mean of {label}", fontsize=12)
    ax.set_ylabel(f"AI - Human ({label})", fontsize=12)
    ax.set_title(f"Bland-Altman Plot: {label}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def scatter_plot(x, y, label, save_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, alpha=0.3, s=10, c="steelblue")
    lims = [0, 5]
    ax.plot(lims, lims, "r--", linewidth=1, label="y = x (perfect agreement)")
    ax.set_xlabel(f"Human {label} (mean)", fontsize=12)
    ax.set_ylabel(f"AI {label}", fontsize=12)
    ax.set_title(f"Scatter: AI vs Human ({label})", fontsize=14)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_excel(EXCEL_PATH)

    alignment_gt = "Merged_alignment_score"
    artifact_gt = "Merged_artifact_score"

    pairs = {
        "alignment": (alignment_gt, "AI_alignment_score"),
        "artifact": (artifact_gt, "AI_artifact_score"),
    }

    results = {}

    for name, (human_col, ai_col) in pairs.items():
        mask = df[human_col].notna() & df[ai_col].notna()
        human = df.loc[mask, human_col].values.astype(float)
        ai = df.loc[mask, ai_col].values.astype(float)
        n = len(human)

        pr, pr_p = pearson_r(human, ai)
        sr, sr_p = spearman_r(human, ai)
        ccc = concordance_cc(human, ai)
        icc = icc_simple(human, ai)
        mae_val = mae(human, ai)
        rmse_val = rmse(human, ai)
        mean_ai = np.mean(ai)
        mean_human = np.mean(human)
        bias = mean_ai - mean_human

        results[name] = {
            "N": n,
            "Human_mean": round(mean_human, 4),
            "AI_mean": round(mean_ai, 4),
            "Bias(AI-Human)": round(bias, 4),
            "Pearson_r": round(pr, 4),
            "Pearson_p": f"{pr_p:.2e}",
            "Spearman_r": round(sr, 4),
            "Spearman_p": f"{sr_p:.2e}",
            "CCC": round(ccc, 4),
            "ICC(2,1)": round(icc, 4),
            "MAE": round(mae_val, 4),
            "RMSE": round(rmse_val, 4),
        }

        bland_altman_plot(ai, human, name,
                          os.path.join(OUTPUT_DIR, f"bland_altman_{name}.png"))
        scatter_plot(human, ai, name,
                     os.path.join(OUTPUT_DIR, f"scatter_{name}.png"))

    print("=" * 70)
    print("AI vs Human Agreement Analysis")
    print("=" * 70)
    for name, r in results.items():
        print(f"\n--- {name.upper()} ---")
        for k, v in r.items():
            print(f"  {k:20s}: {v}")

    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(OUTPUT_DIR, "agreement_metrics.csv"))

    print(f"\nPlots and metrics saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
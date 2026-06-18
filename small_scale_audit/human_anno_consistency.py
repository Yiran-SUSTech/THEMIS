"""
计算3个标注员之间的一致性（ICC等指标），作为人类上界参照。
包含 ICC(2,1)、ICC(3,1)、ICC(2,k)、ICC(3,k)。
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

EXCEL_PATH = os.path.join(os.path.dirname(__file__), "output_results", "aggregated_scores.xlsx")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "human_agreement_analysis")


def concordance_cc(x, y):
    """Lin's Concordance Correlation Coefficient."""
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    r, _ = stats.pearsonr(x, y)
    c_b = (2 * var_x * var_y) / (
        var_x + var_y + (mean_x - mean_y) ** 2
    ) if (var_x + var_y + (mean_x - mean_y) ** 2) > 0 else 0
    return r * c_b


def compute_pairwise_metrics(a, b, label_a, label_b):
    mask = np.isfinite(a) & np.isfinite(b)
    a_clean = a[mask]
    b_clean = b[mask]
    n = len(a_clean)

    if n < 3:
        return None

    r_p, p_p = stats.pearsonr(a_clean, b_clean)
    r_s, p_s = stats.spearmanr(a_clean, b_clean)
    ccc = concordance_cc(a_clean, b_clean)

    # ICC(2,1) pairwise
    mean_diff = np.mean(a_clean) - np.mean(b_clean)
    var_pooled = (np.var(a_clean, ddof=1) + np.var(b_clean, ddof=1)) / 2
    bias_correction = 1 - (mean_diff ** 2) / (2 * var_pooled) if var_pooled > 0 else 0
    icc_2_1 = r_p * bias_correction

    mae = np.mean(np.abs(a_clean - b_clean))
    rmse = np.sqrt(np.mean((a_clean - b_clean) ** 2))
    bias = np.mean(a_clean) - np.mean(b_clean)

    return {
        "N": n,
        "Bias": round(bias, 4),
        "Pearson_r": round(r_p, 4),
        "Spearman_r": round(r_s, 4),
        "CCC": round(ccc, 4),
        "ICC(2,1)": round(icc_2_1, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
    }


def icc_anova(a, b, c):
    """
    3-way ICC using ANOVA approach.
    Two-way model, both random (2) and mixed (3) effects.
    """
    mask = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    a, b, c = a[mask], b[mask], c[mask]
    n = len(a)
    k = 3  # number of raters

    data = np.column_stack([a, b, c])
    grand_mean = np.mean(data)

    ss_total = np.sum((data - grand_mean) ** 2)

    subject_means = np.mean(data, axis=1)
    ss_subjects = k * np.sum((subject_means - grand_mean) ** 2)

    rater_means = np.mean(data, axis=0)
    ss_raters = n * np.sum((rater_means - grand_mean) ** 2)

    ss_error = ss_total - ss_subjects - ss_raters

    df_subjects = n - 1
    df_raters = k - 1
    df_error = df_subjects * df_raters

    ms_subjects = ss_subjects / df_subjects
    ms_raters = ss_raters / df_raters
    ms_error = ss_error / df_error

    # ICC(2,1): two-way random, absolute agreement, single measure
    icc_2_1 = (ms_subjects - ms_error) / (
        ms_subjects + (k - 1) * ms_error + k * (ms_raters - ms_error) / n
    )

    # ICC(3,1): two-way mixed, absolute agreement, single measure
    # 标注员是固定的，不推广到其他标注员
    icc_3_1 = (ms_subjects - ms_error) / (
        ms_subjects + (k - 1) * ms_error
    )

    # ICC(2,k): two-way random, average measures
    icc_2_k = (ms_subjects - ms_error) / ms_subjects

    # ICC(3,k): two-way mixed, average measures
    icc_3_k = (ms_subjects - ms_error) / (
        ms_subjects + (ms_error - ms_raters) / n
    )

    return {
        "N": n,
        "ICC(2,1)_single_random": round(icc_2_1, 4),
        "ICC(3,1)_single_mixed": round(icc_3_1, 4),
        "ICC(2,k)_avg_random": round(icc_2_k, 4),
        "ICC(3,k)_avg_mixed": round(icc_3_k, 4),
        "Rater_means": {
            "Rater1": round(np.mean(a), 4),
            "Rater2": round(np.mean(b), 4),
            "Rater3": round(np.mean(c), 4),
        },
        "Rater_stds": {
            "Rater1": round(np.std(a, ddof=1), 4),
            "Rater2": round(np.std(b, ddof=1), 4),
            "Rater3": round(np.std(c, ddof=1), 4),
        },
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_excel(EXCEL_PATH)

    user_cols_align = ["User_1_alignment", "User_2_alignment", "User_3_alignment"]
    user_cols_artifact = ["User_1_artifact", "User_2_artifact", "User_3_artifact"]

    pairs = [
        ("User_1", "User_2"),
        ("User_1", "User_3"),
        ("User_2", "User_3"),
    ]

    all_results = {}

    print("=" * 70)
    print("Human Annotator Agreement Analysis")
    print("=" * 70)

    for score_type, cols in [("alignment", user_cols_align), ("artifact", user_cols_artifact)]:
        print(f"\n{'='*50}")
        print(f"  {score_type.upper()}")
        print(f"{'='*50}")

        a = df[cols[0]].values.astype(float)
        b = df[cols[1]].values.astype(float)
        c = df[cols[2]].values.astype(float)

        result_3way = icc_anova(a, b, c)
        all_results[f"{score_type}_3way"] = result_3way
        print(f"\n  3-way ICC (all raters):")
        print(f"    ICC(2,1) single random:  {result_3way['ICC(2,1)_single_random']}")
        print(f"    ICC(3,1) single mixed:   {result_3way['ICC(3,1)_single_mixed']}")
        print(f"    ICC(2,k) avg random:     {result_3way['ICC(2,k)_avg_random']}")
        print(f"    ICC(3,k) avg mixed:      {result_3way['ICC(3,k)_avg_mixed']}")
        print(f"    Rater means: {result_3way['Rater_means']}")
        print(f"    Rater stds:  {result_3way['Rater_stds']}")

        print(f"\n  Pairwise comparisons:")
        all_results[f"{score_type}_pairwise"] = {}
        for r1, r2 in pairs:
            c1 = f"{r1}_{score_type}"
            c2 = f"{r2}_{score_type}"
            metrics = compute_pairwise_metrics(
                df[c1].values.astype(float),
                df[c2].values.astype(float),
                r1, r2,
            )
            if metrics:
                all_results[f"{score_type}_pairwise"][f"{r1}_vs_{r2}"] = metrics
                print(f"\n    {r1} vs {r2}:")
                for k, v in metrics.items():
                    print(f"      {k:15s}: {v}")

    # Save results to CSV
    rows_3way = []
    for key in ["alignment_3way", "artifact_3way"]:
        r = all_results[key]
        rows_3way.append({
            "score_type": key.replace("_3way", ""),
            "N": r["N"],
            "ICC(2,1)_single_random": r["ICC(2,1)_single_random"],
            "ICC(3,1)_single_mixed": r["ICC(3,1)_single_mixed"],
            "ICC(2,k)_avg_random": r["ICC(2,k)_avg_random"],
            "ICC(3,k)_avg_mixed": r["ICC(3,k)_avg_mixed"],
            "Rater1_mean": r["Rater_means"]["Rater1"],
            "Rater2_mean": r["Rater_means"]["Rater2"],
            "Rater3_mean": r["Rater_means"]["Rater3"],
            "Rater1_std": r["Rater_stds"]["Rater1"],
            "Rater2_std": r["Rater_stds"]["Rater2"],
            "Rater3_std": r["Rater_stds"]["Rater3"],
        })
    df_3way = pd.DataFrame(rows_3way)
    df_3way.to_csv(os.path.join(OUTPUT_DIR, "human_3way_icc.csv"), index=False)

    rows_pairwise = []
    for key in ["alignment_pairwise", "artifact_pairwise"]:
        score_type = key.replace("_pairwise", "")
        for pair_name, metrics in all_results[key].items():
            rows_pairwise.append({
                "score_type": score_type,
                "pair": pair_name,
                **metrics,
            })
    df_pairwise = pd.DataFrame(rows_pairwise)
    df_pairwise.to_csv(os.path.join(OUTPUT_DIR, "human_pairwise_metrics.csv"), index=False)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - human_3way_icc.csv")
    print(f"  - human_pairwise_metrics.csv")


if __name__ == "__main__":
    main()
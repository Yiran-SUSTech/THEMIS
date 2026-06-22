import json
import os
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from collections import defaultdict

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = r"d:\THEMIS"

SOURCE_GROUPS = {
    "DiT_val (no-ref)": ["DiT_val_1", "DiT_val_2", "DiT_val_3"],
    "DiT_val_ref": ["DiT_val_ref_1", "DiT_val_ref_2", "DiT_val_ref_3"],
    "Checklist_ref": ["Checklist_ref_1", "Checklist_ref_2", "Checklist_ref_3"],
    "DiT_val_temmp5": ["DiT_val_temmp5_1", "Checklist_temmp5_1"],
    "Without_expert": ["Without_expert_1"],
    "Human": ["User_1", "User_2", "User_3"],
}

SOURCES = {
    "DiT_val_1": {
        "type": "final_report",
        "path": os.path.join(BASE_DIR, "c2i_faster", "output_DiT_val_1", "final_reports"),
        "prefix": "final_evaluation_report_",
    },
    "DiT_val_2": {
        "type": "final_report",
        "path": os.path.join(BASE_DIR, "c2i_faster", "output_DiT_val_2", "final_reports"),
        "prefix": "final_evaluation_report_",
    },
    "DiT_val_3": {
        "type": "final_report",
        "path": os.path.join(BASE_DIR, "c2i_faster", "output_DiT_val_3", "final_reports"),
        "prefix": "final_evaluation_report_",
    },
    "DiT_val_ref_1": {
        "type": "final_report",
        "path": os.path.join(BASE_DIR, "c2i_faster", "output_DiT_val_ref_1", "final_reports"),
        "prefix": "final_evaluation_report_",
    },
    "DiT_val_ref_2": {
        "type": "final_report",
        "path": os.path.join(BASE_DIR, "c2i_faster", "output_DiT_val_ref_2", "final_reports"),
        "prefix": "final_evaluation_report_",
    },
    "DiT_val_ref_3": {
        "type": "final_report",
        "path": os.path.join(BASE_DIR, "c2i_faster", "output_DiT_val_ref_3", "final_reports"),
        "prefix": "final_evaluation_report_",
    },
    "Checklist_ref_1": {
        "type": "checklist",
        "path": os.path.join(BASE_DIR, "c2i_faster", "output_DiT_val_ref_1", "checklist_annotations"),
        "prefix": "checklist_",
    },
    "Checklist_ref_2": {
        "type": "checklist",
        "path": os.path.join(BASE_DIR, "c2i_faster", "output_DiT_val_ref_2", "checklist_annotations"),
        "prefix": "checklist_",
    },
    "Checklist_ref_3": {
        "type": "checklist",
        "path": os.path.join(BASE_DIR, "c2i_faster", "output_DiT_val_ref_3", "checklist_annotations"),
        "prefix": "checklist_",
    },
    "User_1": {
        "type": "human",
        "path": os.path.join(BASE_DIR, "small_scale_audit_recorrect", "output_results", "User_1_final_annotations.json"),
    },
    "User_2": {
        "type": "human",
        "path": os.path.join(BASE_DIR, "small_scale_audit_recorrect", "output_results", "User_2_final_annotations.json"),
    },
    "User_3": {
        "type": "human",
        "path": os.path.join(BASE_DIR, "small_scale_audit_recorrect", "output_results", "User_3_final_annotations.json"),
    },
    "User_4": {
        "type": "human",
        "path": os.path.join(BASE_DIR, "small_scale_audit_recorrect", "output_results", "User_4_final_annotations.json"),
    },
    "User_5": {
        "type": "human",
        "path": os.path.join(BASE_DIR, "small_scale_audit_recorrect", "output_results", "User_5_final_annotations.json"),
    },
    "User_55": {
        "type": "human",
        "path": os.path.join(BASE_DIR, "small_scale_audit_recorrect", "output_results", "User_55_final_annotations.json"),
    },
    "User_6": {
        "type": "human",
        "path": os.path.join(BASE_DIR, "small_scale_audit_recorrect", "output_results", "User_6_final_annotations.json"),
    },
    "User_7": {
        "type": "human",
        "path": os.path.join(BASE_DIR, "small_scale_audit_recorrect", "output_results", "User_7_final_annotations.json"),
    },
    "User_8": {
        "type": "human",
        "path": os.path.join(BASE_DIR, "small_scale_audit_recorrect", "output_results", "User_8_final_annotations.json"),
    },
    "DiT_val_temmp5_1": {
        "type": "final_report",
        "path": os.path.join(BASE_DIR, "c2i_faster", "output_DiT_val_temmp5_1", "final_reports"),
        "prefix": "final_evaluation_report_",
    },
    "Checklist_temmp5_1": {
        "type": "checklist",
        "path": os.path.join(BASE_DIR, "c2i_faster", "output_DiT_val_temmp5_1", "checklist_annotations"),
        "prefix": "checklist_",
    },
    "Without_expert_1": {
        "type": "final_report",
        "path": os.path.join(BASE_DIR, "c2i_faster", "output_DiT_val_without_expert_1", "without_expert_reports"),
        "prefix": "direct_score_",
    },
}


def extract_final_report_scores(dir_path, prefix):
    results = {}
    if not os.path.isdir(dir_path):
        print(f"[WARN] Directory not found: {dir_path}")
        return results
    for fname in sorted(os.listdir(dir_path)):
        if not fname.startswith(prefix) or not fname.endswith(".json"):
            continue
        idx_str = fname.replace(prefix, "").replace(".json", "")
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        fpath = os.path.join(dir_path, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            align = data.get("alignment_score")
            artifact = data.get("artifact_score")
            if align is not None and artifact is not None:
                results[idx] = {"alignment_score": float(align), "artifact_score": float(artifact)}
        except Exception as e:
            print(f"[WARN] Failed to read {fpath}: {e}")
    return results


def extract_checklist_scores(dir_path, prefix):
    results = {}
    if not os.path.isdir(dir_path):
        print(f"[WARN] Directory not found: {dir_path}")
        return results
    for fname in sorted(os.listdir(dir_path)):
        if not fname.startswith(prefix) or not fname.endswith(".json"):
            continue
        idx_str = fname.replace(prefix, "").replace(".json", "")
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        fpath = os.path.join(dir_path, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            scores = data.get("scores", {})
            align = scores.get("alignment_score")
            artifact = scores.get("artifact_score")
            if align is not None and artifact is not None:
                results[idx] = {"alignment_score": float(align), "artifact_score": float(artifact)}
        except Exception as e:
            print(f"[WARN] Failed to read {fpath}: {e}")
    return results


def extract_human_scores(json_path):
    results = {}
    if not os.path.isfile(json_path):
        print(f"[WARN] File not found: {json_path}")
        return results
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {json_path}: {e}")
        return results
    for key, val in data.items():
        image_name = val.get("image_name", key) if isinstance(val, dict) else key
        idx_str = image_name.replace(".png", "").replace(".jpg", "")
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        scores = val.get("scores", {})
        align = scores.get("alignment_score")
        artifact = scores.get("artifact_score")
        if align is not None and artifact is not None:
            results[idx] = {"alignment_score": float(align), "artifact_score": float(artifact)}
    return results


def collect_all_scores():
    all_data = {}
    for source_name, cfg in SOURCES.items():
        print(f"Extracting: {source_name} ...")
        if cfg["type"] == "final_report":
            scores = extract_final_report_scores(cfg["path"], cfg["prefix"])
        elif cfg["type"] == "checklist":
            scores = extract_checklist_scores(cfg["path"], cfg["prefix"])
        elif cfg["type"] == "human":
            scores = extract_human_scores(cfg["path"])
        else:
            scores = {}
        for idx, sc in scores.items():
            if idx not in all_data:
                all_data[idx] = {}
            all_data[idx][source_name] = sc
        print(f"  -> {len(scores)} images extracted")
    return all_data


def build_dataframe(all_data):
    rows = []
    for idx in sorted(all_data.keys()):
        for source_name, sc in all_data[idx].items():
            rows.append({
                "image_id": idx,
                "source": source_name,
                "alignment_score": sc["alignment_score"],
                "artifact_score": sc["artifact_score"],
            })
    df = pd.DataFrame(rows)
    return df


def save_csv(df, output_path):
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"CSV saved to: {output_path}")


def plot_distributions(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sources = sorted(df["source"].unique())

    for score_col, score_label in [("alignment_score", "Alignment Score"), ("artifact_score", "Artifact Score")]:
        n_src = len(sources)
        n_cols = 4
        n_rows = (n_src + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        fig.suptitle(f"{score_label} Distribution by Source", fontsize=18, fontweight="bold")
        axes_flat = axes.flatten()
        for i, src in enumerate(sources):
            ax = axes_flat[i]
            vals = df[df["source"] == src][score_col].dropna().values
            if len(vals) == 0:
                ax.set_title(f"{src}\n(no data)")
                continue
            ax.hist(vals, bins=20, edgecolor="black", alpha=0.7, color="steelblue")
            ax.set_title(f"{src}\nn={len(vals)}, mean={np.mean(vals):.2f}, std={np.std(vals):.2f}", fontsize=9)
            ax.set_xlabel(score_label)
            ax.set_ylabel("Count")
        for j in range(len(sources), len(axes_flat)):
            axes_flat[j].set_visible(False)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(output_dir, f"distribution_{score_col}.png"), dpi=150)
        plt.close(fig)
        print(f"Saved: distribution_{score_col}.png")

    for score_col, score_label in [("alignment_score", "Alignment Score"), ("artifact_score", "Artifact Score")]:
        fig, ax = plt.subplots(figsize=(14, 6))
        data_for_box = []
        labels_for_box = []
        for src in sources:
            vals = df[df["source"] == src][score_col].dropna().values
            if len(vals) > 0:
                data_for_box.append(vals)
                labels_for_box.append(src)
        if data_for_box:
            bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
            colors = plt.cm.Set3(np.linspace(0, 1, len(data_for_box)))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
        ax.set_title(f"{score_label} Box Plot by Source", fontsize=14, fontweight="bold")
        ax.set_ylabel(score_label)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"boxplot_{score_col}.png"), dpi=150)
        plt.close(fig)
        print(f"Saved: boxplot_{score_col}.png")


def compute_icc(x, y, icc_type="2,1"):
    """
    计算两个评分者之间的ICC（k=2）。
    icc_type:
      "2,1" -> ICC(2,1): Two-way random, absolute agreement, single measures
      "3,1" -> ICC(3,1): Two-way mixed, consistency, single measures
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    k = 2
    if n < 3:
        return np.nan

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    grand_mean = (mean_x + mean_y) / 2.0

    ss_raters = n * (mean_x - grand_mean) ** 2 + n * (mean_y - grand_mean) ** 2
    ss_subjects = np.sum(((x + y) / 2.0 - grand_mean) ** 2) * k
    ss_total = np.sum((np.concatenate([x, y]) - grand_mean) ** 2)
    ss_error = ss_total - ss_subjects - ss_raters

    ms_subjects = ss_subjects / (n - 1)
    ms_raters = ss_raters / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    if icc_type == "2,1":
        denom = ms_subjects + (k - 1) * ms_error + k * (ms_raters - ms_error) / n
        if denom <= 0:
            return np.nan
        icc = (ms_subjects - ms_error) / denom
    elif icc_type == "3,1":
        denom = ms_subjects + (k - 1) * ms_error
        if denom <= 0:
            return np.nan
        icc = (ms_subjects - ms_error) / denom
    else:
        return np.nan

    return icc


def compute_icc_k(data_matrix, icc_type="2,1"):
    """
    计算k个评分者（多次运行）的ICC。
    data_matrix: shape (n_subjects, k_raters)
    icc_type:
      "2,1" -> ICC(2,1): Two-way random, absolute agreement, single measures
      "3,1" -> ICC(3,1): Two-way mixed, consistency, single measures
    """
    data_matrix = np.asarray(data_matrix, dtype=float)
    n, k = data_matrix.shape
    if n < 3 or k < 2:
        return np.nan

    grand_mean = data_matrix.mean()
    row_means = data_matrix.mean(axis=1)
    col_means = data_matrix.mean(axis=0)

    ss_total = np.sum((data_matrix - grand_mean) ** 2)
    ss_subjects = k * np.sum((row_means - grand_mean) ** 2)
    ss_raters = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_subjects - ss_raters

    ms_subjects = ss_subjects / (n - 1)
    ms_raters = ss_raters / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    if icc_type == "2,1":
        denom = ms_subjects + (k - 1) * ms_error + k * (ms_raters - ms_error) / n
        if denom <= 0:
            return np.nan
        icc = (ms_subjects - ms_error) / denom
    elif icc_type == "3,1":
        denom = ms_subjects + (k - 1) * ms_error
        if denom <= 0:
            return np.nan
        icc = (ms_subjects - ms_error) / denom
    else:
        return np.nan

    return icc


def compute_correlations(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sources = sorted(df["source"].unique())
    pivot_align = df.pivot_table(index="image_id", columns="source", values="alignment_score")
    pivot_artifact = df.pivot_table(index="image_id", columns="source", values="artifact_score")

    results = []
    for score_name, pivot in [("alignment_score", pivot_align), ("artifact_score", pivot_artifact)]:
        for i, s1 in enumerate(sources):
            for j, s2 in enumerate(sources):
                if i >= j:
                    continue
                pair = pivot[[s1, s2]].dropna()
                if len(pair) < 5:
                    continue
                r_p, p_p = stats.pearsonr(pair[s1], pair[s2])
                r_s, p_s = stats.spearmanr(pair[s1], pair[s2])
                icc_21 = compute_icc(pair[s1].values, pair[s2].values, "2,1")
                icc_31 = compute_icc(pair[s1].values, pair[s2].values, "3,1")
                results.append({
                    "score_type": score_name,
                    "source_1": s1,
                    "source_2": s2,
                    "n_common_images": len(pair),
                    "pearson_r": round(r_p, 4),
                    "pearson_p": round(p_p, 6),
                    "spearman_r": round(r_s, 4),
                    "spearman_p": round(p_s, 6),
                    "icc_2_1_absolute_agreement": round(icc_21, 4) if not np.isnan(icc_21) else "N/A",
                    "icc_3_1_consistency": round(icc_31, 4) if not np.isnan(icc_31) else "N/A",
                })

    corr_df = pd.DataFrame(results)
    corr_csv_path = os.path.join(output_dir, "correlations.csv")
    corr_df.to_csv(corr_csv_path, index=False, encoding="utf-8-sig")
    print(f"Correlations saved to: {corr_csv_path}")

    for score_name in ["alignment_score", "artifact_score"]:
        sub = corr_df[corr_df["score_type"] == score_name]
        if sub.empty:
            continue
        sources_in_sub = sorted(set(sub["source_1"].tolist() + sub["source_2"].tolist()))
        n_src = len(sources_in_sub)
        idx_map = {s: i for i, s in enumerate(sources_in_sub)}

        mat_dict = {
            "Pearson": np.full((n_src, n_src), np.nan),
            "Spearman": np.full((n_src, n_src), np.nan),
            "ICC_2_1": np.full((n_src, n_src), np.nan),
            "ICC_3_1": np.full((n_src, n_src), np.nan),
        }
        col_map = {
            "Pearson": "pearson_r",
            "Spearman": "spearman_r",
            "ICC_2_1": "icc_2_1_absolute_agreement",
            "ICC_3_1": "icc_3_1_consistency",
        }
        for _, row in sub.iterrows():
            i, j = idx_map[row["source_1"]], idx_map[row["source_2"]]
            for mat_name, col_name in col_map.items():
                val = row[col_name]
                if val != "N/A":
                    mat_dict[mat_name][i, j] = val
                    mat_dict[mat_name][j, i] = val
        for mat in mat_dict.values():
            np.fill_diagonal(mat, 1.0)

        for mat_name, mat in mat_dict.items():
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(mat, cmap="RdYlBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(n_src))
            ax.set_yticks(range(n_src))
            ax.set_xticklabels(sources_in_sub, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(sources_in_sub, fontsize=8)
            for ii in range(n_src):
                for jj in range(n_src):
                    val = mat[ii, jj]
                    if not np.isnan(val):
                        ax.text(jj, ii, f"{val:.2f}", ha="center", va="center", fontsize=7,
                                color="white" if abs(val) > 0.6 else "black")
            label = {"Pearson": "Pearson r", "Spearman": "Spearman r",
                     "ICC_2_1": "ICC(2,1)", "ICC_3_1": "ICC(3,1)"}[mat_name]
            plt.colorbar(im, ax=ax, label=label)
            ax.set_title(f"{label} - {score_name}", fontsize=14, fontweight="bold")
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f"corr_{mat_name}_{score_name}.png"), dpi=150)
            plt.close(fig)
            print(f"Saved: corr_{mat_name}_{score_name}.png")

    return corr_df


def compute_stability(df, output_dir, groups=None):
    os.makedirs(output_dir, exist_ok=True)
    if groups is None:
        groups = SOURCE_GROUPS

    results = []
    for group_name, source_list in groups.items():
        for score_col in ["alignment_score", "artifact_score"]:
            sub = df[df["source"].isin(source_list)]
            pivot = sub.pivot_table(index="image_id", columns="source", values=score_col)
            common = pivot.dropna()
            n_images = len(common)
            if n_images < 2:
                print(f"[WARN] Not enough common images for {group_name} {score_col}: {n_images}")
                continue

            per_image_std = common.std(axis=1)
            per_image_mean = common.mean(axis=1)
            per_image_cv = per_image_std / per_image_mean.replace(0, np.nan)

            mean_score = common.values.mean()
            std_across_runs = per_image_std.mean()
            cv_across_runs = per_image_cv.mean()

            icc_21 = compute_icc_k(common.values, "2,1")
            icc_31 = compute_icc_k(common.values, "3,1")

            pairwise_corrs = []
            for i in range(len(source_list)):
                for j in range(i + 1, len(source_list)):
                    s1, s2 = source_list[i], source_list[j]
                    if s1 in common.columns and s2 in common.columns:
                        pair = common[[s1, s2]].dropna()
                        if len(pair) >= 5:
                            r, _ = stats.pearsonr(pair[s1], pair[s2])
                            pairwise_corrs.append(r)

            avg_pairwise_corr = np.mean(pairwise_corrs) if pairwise_corrs else np.nan

            results.append({
                "group": group_name,
                "score_type": score_col,
                "n_common_images": n_images,
                "mean_score": round(mean_score, 4),
                "std_across_runs (avg per-image std)": round(std_across_runs, 4),
                "cv_across_runs (avg per-image CV)": round(cv_across_runs, 4),
                "icc_2_1 (absolute agreement)": round(icc_21, 4) if not np.isnan(icc_21) else "N/A",
                "icc_3_1 (consistency)": round(icc_31, 4) if not np.isnan(icc_31) else "N/A",
                "avg_pairwise_pearson_r": round(avg_pairwise_corr, 4) if not np.isnan(avg_pairwise_corr) else "N/A",
            })

    stability_df = pd.DataFrame(results)
    stability_csv = os.path.join(output_dir, "stability.csv")
    stability_df.to_csv(stability_csv, index=False, encoding="utf-8-sig")
    print(f"Stability results saved to: {stability_csv}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax_idx, score_col in enumerate(["alignment_score", "artifact_score"]):
        ax = axes[ax_idx]
        sub_results = [r for r in results if r["score_type"] == score_col]
        group_names = [r["group"] for r in sub_results]
        std_vals = [r["std_across_runs (avg per-image std)"] for r in sub_results]
        icc_21_vals = [r["icc_2_1 (absolute agreement)"] if r["icc_2_1 (absolute agreement)"] != "N/A" else 0 for r in sub_results]
        icc_31_vals = [r["icc_3_1 (consistency)"] if r["icc_3_1 (consistency)"] != "N/A" else 0 for r in sub_results]

        x = np.arange(len(group_names))
        width = 0.25
        bars1 = ax.bar(x - width, std_vals, width, label="Avg per-image Std", color="steelblue", alpha=0.8)
        ax2 = ax.twinx()
        bars2 = ax2.bar(x, icc_21_vals, width, label="ICC(2,1) absolute", color="coral", alpha=0.8)
        bars3 = ax2.bar(x + width, icc_31_vals, width, label="ICC(3,1) consistency", color="seagreen", alpha=0.8)

        ax.set_xlabel("Group")
        ax.set_ylabel("Avg per-image Std", color="steelblue")
        ax2.set_ylabel("ICC", color="coral")
        ax2.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(group_names, rotation=15, ha="right")
        ax.set_title(f"{score_col} - Stability Metrics", fontweight="bold")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "stability_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: stability_comparison.png")

    return stability_df


def analyze_system_vs_human(df, output_dir, groups=None):
    os.makedirs(output_dir, exist_ok=True)
    if groups is None:
        groups = SOURCE_GROUPS

    human_group_names = [g for g in groups if "Human" in g or "User" in g]
    system_group_names = [g for g in groups if g not in human_group_names]

    human_sources = []
    for g in human_group_names:
        human_sources.extend(groups[g])
    system_sources = []
    for g in system_group_names:
        system_sources.extend(groups[g])

    if not human_sources or not system_sources:
        print("[INFO] No human+system pair found, skipping system_vs_human analysis.")
        bias_df = pd.DataFrame()
        dist_df = pd.DataFrame()
        return bias_df, dist_df

    sub_sys = df[df["source"].isin(system_sources)]
    sub_human = df[df["source"].isin(human_sources)]

    sys_avg = sub_sys.pivot_table(index="image_id", columns="source", values=["alignment_score", "artifact_score"])
    human_avg = sub_human.pivot_table(index="image_id", columns="source", values=["alignment_score", "artifact_score"])

    sys_groups = {g: groups[g] for g in system_group_names}

    results = []
    for score_col in ["alignment_score", "artifact_score"]:
        human_per_image = human_avg[score_col]
        human_mean_per_image = human_per_image.mean(axis=1)
        human_grand_mean = human_mean_per_image.mean()

        for group_name, src_list in sys_groups.items():
            sys_per_image = df[df["source"].isin(src_list)].pivot_table(
                index="image_id", columns="source", values=score_col)
            sys_mean_per_image = sys_per_image.mean(axis=1)
            sys_grand_mean = sys_mean_per_image.mean()

            common_idx = sys_mean_per_image.index.intersection(human_mean_per_image.index)
            sys_vals = sys_mean_per_image.loc[common_idx]
            human_vals = human_mean_per_image.loc[common_idx]

            diff = sys_vals - human_vals
            mean_diff = diff.mean()
            median_diff = diff.median()
            pct_sys_higher = (diff > 0).mean() * 100
            pct_sys_lower = (diff < 0).mean() * 100

            r_pearson, p_pearson = stats.pearsonr(sys_vals, human_vals)
            r_spearman, p_spearman = stats.spearmanr(sys_vals, human_vals)
            icc_21 = compute_icc(sys_vals.values, human_vals.values, "2,1")
            icc_31 = compute_icc(sys_vals.values, human_vals.values, "3,1")

            results.append({
                "score_type": score_col,
                "system_group": group_name,
                "system_mean": round(sys_grand_mean, 4),
                "human_mean": round(human_grand_mean, 4),
                "mean_diff (sys-human)": round(mean_diff, 4),
                "median_diff (sys-human)": round(median_diff, 4),
                "pct_sys_higher (%)": round(pct_sys_higher, 2),
                "pct_sys_lower (%)": round(pct_sys_lower, 2),
                "pearson_r (avg-vs-avg)": round(r_pearson, 4),
                "spearman_r (avg-vs-avg)": round(r_spearman, 4),
                "icc_2_1 (avg-vs-avg)": round(icc_21, 4) if not np.isnan(icc_21) else "N/A",
                "icc_3_1 (avg-vs-avg)": round(icc_31, 4) if not np.isnan(icc_31) else "N/A",
            })

    bias_df = pd.DataFrame(results)
    bias_csv = os.path.join(output_dir, "system_vs_human_bias.csv")
    bias_df.to_csv(bias_csv, index=False, encoding="utf-8-sig")
    print(f"System vs Human bias saved to: {bias_csv}")

    for score_col in ["alignment_score", "artifact_score"]:
        n_sys = len(sys_groups)
        fig, axes = plt.subplots(1, n_sys, figsize=(7 * n_sys, 6))
        if n_sys == 1:
            axes = [axes]
        fig.suptitle(f"{score_col}: System Avg vs Human Avg", fontsize=16, fontweight="bold")

        for ax_idx, (group_name, src_list) in enumerate(sys_groups.items()):
            ax = axes[ax_idx]
            sys_per_image = df[df["source"].isin(src_list)].pivot_table(
                index="image_id", columns="source", values=score_col)
            sys_mean_per_image = sys_per_image.mean(axis=1)
            human_per_image = human_avg[score_col]
            human_mean_per_image = human_per_image.mean(axis=1)

            common_idx = sys_mean_per_image.index.intersection(human_mean_per_image.index)
            x = human_mean_per_image.loc[common_idx]
            y = sys_mean_per_image.loc[common_idx]

            ax.scatter(x, y, alpha=0.15, s=8, c="steelblue")
            lim_min = min(x.min(), y.min()) - 0.2
            lim_max = max(x.max(), y.max()) + 0.2
            ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", alpha=0.7, label="y=x (perfect agreement)")
            ax.set_xlim(lim_min, lim_max)
            ax.set_ylim(lim_min, lim_max)
            ax.set_xlabel("Human Avg Score")
            ax.set_ylabel("System Avg Score")

            r_s, _ = stats.spearmanr(x, y)
            r_p, _ = stats.pearsonr(x, y)
            mean_d = (y - x).mean()
            ax.set_title(f"{group_name}\nSpearman r={r_s:.3f}, Pearson r={r_p:.3f}\nMean diff={mean_d:+.3f}", fontsize=10)
            ax.legend(fontsize=7, loc="upper left")
            ax.set_aspect("equal")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(os.path.join(output_dir, f"scatter_system_vs_human_{score_col}.png"), dpi=150)
        plt.close(fig)
        print(f"Saved: scatter_system_vs_human_{score_col}.png")

    for score_col in ["alignment_score", "artifact_score"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        human_per_image = human_avg[score_col]
        human_mean_per_image = human_per_image.mean(axis=1)

        all_means = {"Human avg": human_mean_per_image}
        for group_name, src_list in sys_groups.items():
            sys_per_image = df[df["source"].isin(src_list)].pivot_table(
                index="image_id", columns="source", values=score_col)
            all_means[group_name + " avg"] = sys_per_image.mean(axis=1)

        data_for_box = []
        labels_for_box = []
        for name, series in all_means.items():
            data_for_box.append(series.dropna().values)
            labels_for_box.append(name)

        bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(data_for_box)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(f"{score_col}: Score Distribution Comparison (Per-Image Avg)", fontweight="bold")
        ax.set_ylabel(score_col)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"boxplot_system_vs_human_{score_col}.png"), dpi=150)
        plt.close(fig)
        print(f"Saved: boxplot_system_vs_human_{score_col}.png")

    dist_stats_rows = []
    for score_col in ["alignment_score", "artifact_score"]:
        fig, ax = plt.subplots(figsize=(12, 6))

        human_per_image = human_avg[score_col]
        human_mean_per_image = human_per_image.mean(axis=1)

        all_means = {"Human (3人平均)": human_mean_per_image}
        for group_name, src_list in sys_groups.items():
            sys_per_image = df[df["source"].isin(src_list)].pivot_table(
                index="image_id", columns="source", values=score_col)
            all_means[group_name + " (3次平均)"] = sys_per_image.mean(axis=1)

        colors_line = plt.cm.Set1(np.linspace(0, 0.8, len(all_means)))
        for idx, (name, series) in enumerate(all_means.items()):
            vals = series.dropna().values
            ax.hist(vals, bins=30, density=True, alpha=0.35, color=colors_line[idx], label=name)
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(vals)
            x_grid = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 300)
            ax.plot(x_grid, kde(x_grid), color=colors_line[idx], linewidth=2.5)

            dist_stats_rows.append({
                "score_type": score_col,
                "source": name,
                "n": len(vals),
                "mean": round(np.mean(vals), 4),
                "median": round(np.median(vals), 4),
                "std": round(np.std(vals), 4),
                "min": round(np.min(vals), 4),
                "max": round(np.max(vals), 4),
                "pct_<=1": round((vals <= 1).mean() * 100, 2),
                "pct_1-2": round(((vals > 1) & (vals <= 2)).mean() * 100, 2),
                "pct_2-3": round(((vals > 2) & (vals <= 3)).mean() * 100, 2),
                "pct_3-4": round(((vals > 3) & (vals <= 4)).mean() * 100, 2),
                "pct_4-5": round((vals > 4).mean() * 100, 2),
                "skewness": round(stats.skew(vals), 4),
                "kurtosis": round(stats.kurtosis(vals), 4),
            })

        ax.set_xlabel(score_col)
        ax.set_ylabel("Density")
        ax.set_title(f"{score_col}: Score Distribution Overlay (Per-Image Avg)", fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"overlay_dist_{score_col}.png"), dpi=150)
        plt.close(fig)
        print(f"Saved: overlay_dist_{score_col}.png")

    dist_df = pd.DataFrame(dist_stats_rows)
    dist_csv = os.path.join(output_dir, "distribution_stats.csv")
    dist_df.to_csv(dist_csv, index=False, encoding="utf-8-sig")
    print(f"Distribution stats saved to: {dist_csv}")

    return bias_df, dist_df


def main():
    parser = argparse.ArgumentParser(description="Extract and analyze evaluation scores")
    parser.add_argument(
        "--source",
        nargs="+",
        default=list(SOURCE_GROUPS.keys()),
        choices=list(SOURCE_GROUPS.keys()),
        help=f"Which source groups to include. Available: {list(SOURCE_GROUPS.keys())}. "
             f"Default: all groups."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory. Default: analysis_output or analysis_output_<sources>"
    )
    args = parser.parse_args()

    selected_groups = {k: SOURCE_GROUPS[k] for k in args.source if k in SOURCE_GROUPS}
    selected_source_names = set()
    for src_list in selected_groups.values():
        selected_source_names.update(src_list)

    if args.output:
        output_dir = args.output
    else:
        tag = "_".join(args.source).replace(" ", "")
        output_dir = os.path.join(BASE_DIR, f"analysis_output_{tag}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Selected groups: {list(selected_groups.keys())}")
    print(f"Selected sources: {sorted(selected_source_names)}")
    print(f"Output dir: {output_dir}")

    all_data = collect_all_scores()
    df_full = build_dataframe(all_data)

    df = df_full[df_full["source"].isin(selected_source_names)].copy()
    df.reset_index(drop=True, inplace=True)

    csv_path = os.path.join(output_dir, "all_scores.csv")
    save_csv(df, csv_path)

    print(f"\nTotal records: {len(df)}")
    print(f"Unique images: {df['image_id'].nunique()}")
    print(f"Sources: {sorted(df['source'].unique())}")
    print()

    plot_distributions(df, output_dir)
    corr_df = compute_correlations(df, output_dir)
    stability_df = compute_stability(df, output_dir, groups=selected_groups)

    human_group_names = [g for g in selected_groups if "Human" in g or "User" in g]
    system_group_names = [g for g in selected_groups if g not in human_group_names]

    if human_group_names and system_group_names:
        bias_df, dist_df = analyze_system_vs_human(df, output_dir, groups=selected_groups)
    else:
        print("[INFO] Skipping system_vs_human analysis (need both human and system groups).")
        bias_df = pd.DataFrame()
        dist_df = pd.DataFrame()

    print("\n=== Summary ===")
    print(f"\nCSV file: {csv_path}")
    print(f"\nCorrelations:\n{corr_df.to_string(index=False)}")
    print(f"\nStability:\n{stability_df.to_string(index=False)}")
    if not bias_df.empty:
        print(f"\nSystem vs Human Bias:\n{bias_df.to_string(index=False)}")
    if not dist_df.empty:
        print(f"\nDistribution Stats:\n{dist_df.to_string(index=False)}")
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
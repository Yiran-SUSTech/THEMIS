"""ROC / AUROC analysis: THEMIS system scores (final_reports) vs checklist judgments.

Definitions
-----------
- y_true (binary, "human-style" label): derived from checklist_annotations —
  an image is POSITIVE if its checklist score >= threshold
  (alignment: 5 x weighted-correct / total; authenticity: 0-5 integer scale).
- y_score (continuous, system): the corresponding score from final_reports
  (the Reflector's final output for the same image).

Produces, per run and pooled across runs:
  - roc_alignment.png, roc_authenticity.png
  - auroc_summary.json  (per-run AUROC, pooled AUROC with bootstrap 95% CI,
    across-run mean/std, Spearman correlation between system and checklist)
  - per_image_scores.csv (all paired scores for downstream analysis)

Usage (from repo root or t2i_harness/):
  python t2i_harness/plot_roc_auroc.py \
      --runs output_flux2_dev_800_1 output_flux2_dev_800_2 output_flux2_dev_800_3 \
      --align-threshold 4.0 --auth-threshold 4.0 \
      --out-dir t2i_harness/roc_analysis
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

T2I_DIR = Path(__file__).resolve().parent

METRICS = {
    "alignment": {
        "title": "Alignment",
        "sys_key": "alignment_score",
        "cl_key": "alignment_score",
    },
    "authenticity": {
        "title": "Authenticity",
        "sys_key": "authenticity_score",
        "cl_key": "authenticity_score",
    },
}

RUN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
POOLED_COLOR = "#000000"
GRID_BG = "#e8f1fa"
GRID_COLOR = "#a8c4e0"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Data loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _img_id_sort_key(img_id: str):
    return int(img_id) if img_id.isdigit() else 10 ** 9


def load_run(run_dir: Path) -> tuple[list[dict], dict]:
    """Load paired (system, checklist) records for one run directory.

    Returns (records, info) where each record is
    {img_id, sys_alignment, sys_authenticity, cl_alignment, cl_authenticity, cl_total}.
    """
    fr_dir = run_dir / "final_reports"
    cl_dir = run_dir / "checklist_annotations"

    fr_ids = {p.stem.replace("final_evaluation_report_", ""): p
              for p in fr_dir.glob("final_evaluation_report_*.json")}
    cl_ids = {p.stem.replace("checklist_", ""): p
              for p in cl_dir.glob("checklist_*.json")}
    all_ids = sorted(set(fr_ids) & set(cl_ids), key=_img_id_sort_key)

    records, bad = [], 0
    for img_id in all_ids:
        try:
            fr = json.loads(fr_ids[img_id].read_text(encoding="utf-8"))
            cl = json.loads(cl_ids[img_id].read_text(encoding="utf-8"))
            cl_scores = cl.get("scores", {})
            rec = {
                "img_id": img_id,
                "sys_alignment": float(fr["alignment_score"]),
                "sys_authenticity": float(fr["authenticity_score"]),
                "cl_alignment": float(cl_scores["alignment_score"]),
                "cl_authenticity": float(cl_scores["authenticity_score"]),
                "cl_total": float(cl_scores.get("total_score", np.nan)),
            }
            records.append(rec)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            bad += 1
            print(f"  [WARN] {run_dir.name} img {img_id}: skipped ({type(e).__name__}: {e})")

    info = {
        "n_pairs": len(records),
        "n_final_reports": len(fr_ids),
        "n_checklists": len(cl_ids),
        "n_unpaired_or_bad": len(set(fr_ids) ^ set(cl_ids)) + bad,
    }
    return records, info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Statistics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_metric_stats(records: list[dict], metric: str, threshold: float,
                          n_boot: int, seed: int) -> dict:
    sys_key, cl_key = f"sys_{metric}", f"cl_{metric}"
    y_score = np.array([r[sys_key] for r in records], dtype=float)
    y_true = np.array([r[cl_key] for r in records], dtype=float) >= threshold

    n_pos, n_neg = int(y_true.sum()), int((~y_true).sum())
    out = {
        "threshold": threshold,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "positive_rate": round(n_pos / len(records), 4) if records else None,
    }

    if n_pos == 0 or n_neg == 0:
        out["auroc"] = None
        out["note"] = "only one class present after thresholding — AUROC undefined"
        return out

    out["auroc"] = round(float(roc_auc_score(y_true, y_score)), 4)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if 0 < yt.sum() < len(yt):
            aucs.append(roc_auc_score(yt, ys))
    if aucs:
        out["auroc_ci95"] = [round(float(np.percentile(aucs, 2.5)), 4),
                              round(float(np.percentile(aucs, 97.5)), 4)]

    if HAS_SCIPY:
        rho, _ = spearmanr(y_score, np.array([r[cl_key] for r in records], dtype=float))
        out["spearman_sys_vs_checklist"] = round(float(rho), 4)

    return out


def plot_roc_figure(fig, ax, curves: list[dict], title: str, subtitle: str):
    """curves: [{label, fpr, tpr, auroc, color, lw}]"""
    ax.set_facecolor(GRID_BG)
    ax.grid(True, color=GRID_COLOR, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)

    ax.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1.6,
            zorder=1, label="Chance (y = x)")

    for c in curves:
        label = c["label"] if c["auroc"] is None else \
            f"{c['label']}  (AUROC = {c['auroc']:.3f})"
        ax.plot(c["fpr"], c["tpr"], color=c["color"], linewidth=c["lw"],
                zorder=2, label=label)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(title, fontsize=14, pad=10)
    ax.tick_params(labelsize=11.5)
    leg = ax.legend(loc="lower right", fontsize=11, framealpha=0.95,
                    handlelength=1.4, labelspacing=0.9, borderpad=0.8)
    for t in leg.get_texts():
        t.set_fontsize(11)
    if subtitle:
        fig.text(0.5, 0.005, subtitle, ha="center", fontsize=10, color="#444444")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Run directory name(s) under t2i_harness/ or absolute path(s)")
    parser.add_argument("--align-threshold", type=float, default=4.0,
                        help="checklist alignment >= this -> positive (default 4.0)")
    parser.add_argument("--auth-threshold", type=float, default=4.0,
                        help="checklist authenticity >= this -> positive (default 4.0)")
    parser.add_argument("--out-dir", default=str(T2I_DIR / "roc_analysis"))
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="bootstrap resamples for pooled AUROC CI (default 1000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    thresholds = {"alignment": args.align_threshold, "authenticity": args.auth_threshold}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = []
    for r in args.runs:
        p = Path(r)
        if not p.is_absolute():
            p = T2I_DIR / r
        if not p.exists():
            sys.exit(f"[ERROR] run directory not found: {p}")
        run_dirs.append(p)

    # ---- load all runs ----
    all_records = {}  # run_name -> records
    print("=" * 62)
    print("Loading runs")
    print("=" * 62)
    for rd in run_dirs:
        records, info = load_run(rd)
        all_records[rd.name] = records
        print(f"  {rd.name}: {info['n_pairs']} paired images "
              f"(final_reports={info['n_final_reports']}, "
              f"checklists={info['n_checklists']}, "
              f"unpaired/bad={info['n_unpaired_or_bad']})")

    # ---- per-metric analysis ----
    summary = {
        "config": {
            "runs": [rd.name for rd in run_dirs],
            "thresholds": thresholds,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
        },
        "runs": {},
        "pooled": {},
        "across_runs": {},
    }
    csv_rows = []

    for run_name, records in all_records.items():
        for r in records:
            csv_rows.append({"run": run_name, **r})

    for metric, mcfg in METRICS.items():
        print()
        print("=" * 62)
        print(f"{mcfg['title']} — y_true: checklist >= {thresholds[metric]}, "
              f"y_score: final_report")
        print("=" * 62)

        curves = []
        per_run_aurocs = []
        for i, (run_name, records) in enumerate(all_records.items()):
            stats = compute_metric_stats(records, metric, thresholds[metric],
                                          args.bootstrap, args.seed + i)
            summary["runs"].setdefault(run_name, {})[metric] = stats

            auc = stats["auroc"]
            if auc is not None:
                per_run_aurocs.append(auc)

            if stats["positive_rate"] is None:
                print(f"  {run_name}: no paired images (final_reports/checklists "
                      f"empty or missing) — run skipped, no curve drawn")
                continue

            y_true = np.array([r[f"cl_{metric}"] for r in records], dtype=float) \
                >= thresholds[metric]
            y_score = np.array([r[f"sys_{metric}"] for r in records], dtype=float)
            if stats["n_pos"] and stats["n_neg"]:
                fpr, tpr, _ = roc_curve(y_true, y_score)
            else:
                fpr, tpr = [], []

            print(f"  {run_name}: AUROC = {auc if auc is None else f'{auc:.4f}'}"
                  f"  (pos {stats['n_pos']} / neg {stats['n_neg']}"
                  f"  [{stats['positive_rate'] * 100:.1f}% positive])")

            curves.append({
                "label": run_name,
                "fpr": fpr, "tpr": tpr, "auroc": auc,
                "color": RUN_COLORS[i % len(RUN_COLORS)], "lw": 1.8,
            })

        # pooled across runs
        pooled_records = [r for recs in all_records.values() for r in recs]
        pooled_stats = compute_metric_stats(pooled_records, metric,
                                            thresholds[metric],
                                            args.bootstrap, args.seed)
        summary["pooled"][metric] = pooled_stats
        if pooled_stats["auroc"] is not None:
            y_true = np.array([r[f"cl_{metric}"] for r in pooled_records], dtype=float) \
                >= thresholds[metric]
            y_score = np.array([r[f"sys_{metric}"] for r in pooled_records], dtype=float)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            # curves.append({
            #     "label": f"Pooled (n={len(pooled_records)})",
            #     "fpr": fpr, "tpr": tpr,
            #     "auroc": pooled_stats["auroc"],
            #     "color": POOLED_COLOR, "lw": 2.4,
            # })
            ci = pooled_stats.get("auroc_ci95")
            ci_str = f", 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
            print(f"  POOLED : AUROC = {pooled_stats['auroc']:.4f}{ci_str} "
                  f"(n={len(pooled_records)})")

        if per_run_aurocs:
            summary["across_runs"][metric] = {
                "auroc_mean": round(float(np.mean(per_run_aurocs)), 4),
                "auroc_std": round(float(np.std(per_run_aurocs, ddof=1)), 4)
                if len(per_run_aurocs) > 1 else 0.0,
                "n_runs": len(per_run_aurocs),
            }
            ar = summary["across_runs"][metric]
            print(f"  Across runs: mean = {ar['auroc_mean']:.4f} "
                  f"+/- {ar['auroc_std']:.4f} (n={ar['n_runs']})")

        # ---- figure ----
        fig, ax = plt.subplots(figsize=(6.4, 6.2))
        sub = (f"positive: human anno. {mcfg['title'].lower()} >= "
               f"{thresholds[metric]:g};  score: system {mcfg['title'].lower()}")
        plot_roc_figure(fig, ax, curves, f"ROC — {mcfg['title']}", sub)
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        fig_path = out_dir / f"roc_{metric}.png"
        fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  [SAVED] {fig_path}")

    # ---- outputs ----
    json_path = out_dir / "auroc_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  [SAVED] {json_path}")

    csv_path = out_dir / "per_image_scores.csv"
    if csv_rows:
        keys = ["run", "img_id", "sys_alignment", "sys_authenticity",
                "cl_alignment", "cl_authenticity", "cl_total"]
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for row in sorted(csv_rows, key=lambda r: (r["run"], _img_id_sort_key(r["img_id"]))):
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
        print(f"  [SAVED] {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

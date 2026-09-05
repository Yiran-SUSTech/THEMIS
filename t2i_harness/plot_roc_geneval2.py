"""ROC / AUROC: GenEval2 Soft-TIFA scores vs THEMIS checklist labels.

Protocol follows GenEval2 paper Sec 5.3 (Table 3/7):
  - y_true: binary prompt-level label — positive iff ALL atoms are correct
    ("a prompt is incorrect if any component atom is incorrect").
    Temporary human proxy: THEMIS checklist_annotations fine_grained_details.
  - y_score: continuous per-image metric score (Soft-TIFA_AM / Soft-TIFA_GM)
    from the GenEval2 evaluation JSONs.

Two modes:
  1. ONE GenEval2 score file + multiple --checklist-run dirs:
     one ROC curve per checklist labeling (same score, different labels).
  2. Multiple GenEval2 score files + ONE --checklist-run dir:
     one ROC curve per score file (same labels, different scores).

Usage:
  python t2i_harness/plot_roc_geneval2.py \
      --geneval2-scores "D:/NewMetric/GenEval2/flux2_dev_scores.json" \
      --checklist-run output_flux2_dev_800_1 output_flux2_dev_800_2 output_flux2_dev_800_3 \
      --out-dir t2i_harness/roc_analysis_geneval2 --no-themis

  --majority-vote            add a consensus curve (labels = majority vote over the checklist runs)
  --no-themis                do not overlay the THEMIS alignment curve
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

RUN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
POOLED_COLOR = "#000000"
THEMIS_COLOR = "#7b2d8b"
GRID_BG = "#e8f1fa"
GRID_COLOR = "#a8c4e0"

SORT_KEY = lambda x: int(x) if x.isdigit() else 10 ** 9


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Label loading (y_true): THEMIS checklist as temporary human labels
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_checklist_labels(run_dir: Path) -> dict[str, dict]:
    """Return {img_id: {all_correct, n_atoms, n_correct, alignment}}.

    run_dir may be the run root (contains checklist_annotations/) or the
    checklist_annotations folder itself.

    all_correct follows the GenEval2 prompt-level rule: positive iff every
    atom in fine_grained_details is correct (weights ignored — conjunction).
    """
    cl_dir = run_dir / "checklist_annotations" if \
        (run_dir / "checklist_annotations").is_dir() else run_dir
    labels = {}
    for p in sorted(cl_dir.glob("checklist_*.json"), key=lambda q: SORT_KEY(q.stem.split("_")[1])):
        img_id = p.stem.replace("checklist_", "")
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  [WARN] unparseable checklist {p.name}, skipped")
            continue
        entries = [rec
                   for cat in d.get("fine_grained_details", {}).values()
                   for rec in cat.values()]
        if not entries:
            continue
        n_correct = sum(1 for r in entries if r.get("correct"))
        labels[img_id] = {
            "all_correct": all(r.get("correct") for r in entries),
            "n_atoms": len(entries),
            "n_correct": n_correct,
            "alignment": float(d.get("scores", {}).get("alignment_score", np.nan)),
        }
    return labels


def majority_vote_labels(label_dicts: list[dict]) -> dict[str, dict]:
    """Majority vote of all_correct across runs; ties fall back to first run."""
    img_ids = set()
    for ld in label_dicts:
        img_ids |= set(ld.keys())
    voted = {}
    for img_id in img_ids:
        votes = [ld[img_id]["all_correct"] for ld in label_dicts if img_id in ld]
        if not votes:
            continue
        pos = sum(votes) > len(votes) / 2
        base = next(ld[img_id] for ld in label_dicts if img_id in ld)
        voted[img_id] = {**base, "all_correct": pos,
                         "n_votes": len(votes), "n_yes": sum(votes)}
    return voted


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Score loading (y_score): GenEval2 evaluation JSONs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_geneval2_scores(path: Path) -> dict[str, dict]:
    """Return {img_id: {am, gm}} from a GenEval2 scores JSON."""
    d = json.loads(path.read_text(encoding="utf-8"))
    scores = {}
    for r in d.get("results", []):
        sid = r.get("sample_id", "")
        if not sid.startswith("t2i_"):
            continue
        img_id = str(int(sid.split("_")[-1]))
        scores[img_id] = {"am": float(r["sample_am"]), "gm": float(r["sample_gm"])}
    return scores


def load_themis_scores(run_dir: Path) -> dict[str, float]:
    """Return {img_id: alignment_score} from final_reports."""
    fr_dir = run_dir / "final_reports" if \
        (run_dir / "final_reports").is_dir() else run_dir
    scores = {}
    for p in fr_dir.glob("final_evaluation_report_*.json"):
        img_id = p.stem.replace("final_evaluation_report_", "")
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            scores[img_id] = float(d["alignment_score"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return scores


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Statistics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def auroc_with_ci(y_true: np.ndarray, y_score: np.ndarray,
                  n_boot: int, seed: int) -> tuple[float, list | None]:
    auc = roc_auc_score(y_true, y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if 0 < yt.sum() < len(yt):
            aucs.append(roc_auc_score(yt, ys))
    ci = ([round(float(np.percentile(aucs, 2.5)), 4),
           round(float(np.percentile(aucs, 97.5)), 4)]
          if aucs else None)
    return float(auc), ci


RUN_MARKERS = ["o", "s", "^", "D", "v", "P"]


def plot_roc(fig, ax, curves: list[dict], title: str, subtitle: str):
    ax.set_facecolor(GRID_BG)
    ax.grid(True, color=GRID_COLOR, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1.6,
            zorder=1, label="Chance (y = x)")
    for j, c in enumerate(curves):
        label = c["label"] if c["auroc"] is None else \
            f"{c['label']}  (AUROC = {c['auroc']:.3f})"
        ax.plot(c["fpr"], c["tpr"], color=c["color"], linewidth=c["lw"],
                zorder=2, label=label, linestyle=c.get("ls", "-"),
                marker=RUN_MARKERS[j % len(RUN_MARKERS)], markevery=0.07,
                markersize=4.5, markerfacecolor=c["color"],
                markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(title, fontsize=14, pad=10)
    ax.tick_params(labelsize=11.5)
    ax.legend(loc="lower right", fontsize=10.5, framealpha=0.95,
              handlelength=1.5, labelspacing=0.9, borderpad=0.8)
    if subtitle:
        fig.text(0.5, 0.005, subtitle, ha="center", fontsize=10, color="#444444")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geneval2-scores", nargs="+", required=True,
                    help="GenEval2 scores JSON path(s). With multiple --checklist-run "
                         "dirs exactly ONE file is allowed (one curve per labeling); "
                         "with a single checklist run, multiple files are overlaid")
    ap.add_argument("--checklist-run", nargs="+", default=["output_flux2_dev_800_1"],
                    help="THEMIS run dir(s) for checklist labels (under t2i_harness/, "
                         "or absolute). Multiple dirs -> one ROC curve per labeling")
    ap.add_argument("--majority-vote", action="store_true",
                    help="add a consensus curve: labels = majority vote over the "
                         "checklist runs (multi-labeling mode only)")
    ap.add_argument("--themis-run", default=None,
                    help="THEMIS run dir for the comparison alignment curve "
                         "(default: first --checklist-run)")
    ap.add_argument("--no-themis", dest="themis", action="store_false",
                    help="do not overlay the THEMIS alignment curve")
    ap.add_argument("--out-dir", default=str(T2I_DIR / "roc_analysis_geneval2"))
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else T2I_DIR / p

    ge2_files = [Path(p) for p in args.geneval2_scores]
    checklist_dirs = [_resolve(r) for r in args.checklist_run]
    multi_label = len(checklist_dirs) > 1
    if multi_label and len(ge2_files) > 1:
        sys.exit("ERROR: multiple --geneval2-scores together with multiple "
                 "--checklist-run is ambiguous. Either pass ONE score file with "
                 "several checklist runs (one curve per labeling), or several "
                 "score files with ONE checklist run.")

    # ---- labels ----
    print("=" * 62)
    print("Labels (y_true): THEMIS checklist, GenEval2 prompt-level rule")
    print("=" * 62)
    label_sets = []
    for rd in checklist_dirs:
        labels = load_checklist_labels(rd)
        n_pos = sum(1 for v in labels.values() if v["all_correct"])
        print(f"  {rd.name}: {len(labels)} labeled, {n_pos} positive "
              f"({n_pos / max(len(labels), 1) * 100:.1f}%)")
        label_sets.append((rd, labels))

    voted = majority_vote_labels([ld for _, ld in label_sets]) if multi_label else None
    if voted is not None:
        n_pos = sum(1 for v in voted.values() if v["all_correct"])
        print(f"  majority-vote consensus: {len(voted)} images, {n_pos} positive "
              f"({n_pos / max(len(voted), 1) * 100:.1f}%)")

    # ---- scores ----
    ge2_scores_list = [load_geneval2_scores(p) for p in ge2_files]
    for p, s in zip(ge2_files, ge2_scores_list):
        print(f"  GenEval2 scores: {p.name} ({len(s)} samples)")

    themis_scores = None
    if args.themis:
        trd = _resolve(args.themis_run or checklist_dirs[0])
        themis_scores = load_themis_scores(trd)
        print(f"  THEMIS alignment (comparison): {trd.name} "
              f"({len(themis_scores)} reports)")

    summary = {"config": {
        "mode": ("one GenEval2 score file vs multiple checklist labelings"
                 if multi_label else "multiple GenEval2 score files vs one labeling"),
        "geneval2_files": [p.name for p in ge2_files],
        "checklist_runs": [rd.name for rd in checklist_dirs],
        "majority_vote_curve": bool(args.majority_vote and multi_label),
        "y_true_rule": "positive iff ALL checklist atoms correct (GenEval2 prompt-level)",
        "bootstrap": args.bootstrap,
    }, "checklist_runs": {}, "across_labelings": {}, "majority_vote": {},
       "runs": {}, "pooled": {}, "themis_comparison": {}}
    csv_rows = []

    for metric, fname, mtitle in [("gm", "roc_geneval2_gm.png", "Soft-TIFA_GM"),
                                  ("am", "roc_geneval2_am.png", "Soft-TIFA_AM")]:
        print()
        print("=" * 62)
        print(f"{mtitle} — y_true: all-atoms-correct, y_score: {mtitle}")
        print("=" * 62)

        curves = []

        if multi_label:
            # mode 1: same score file, one curve per checklist labeling
            ge2 = ge2_scores_list[0]
            per_label_auc = []
            for i, (rd, labels) in enumerate(label_sets):
                ids = sorted(set(labels) & set(ge2), key=SORT_KEY)
                yt = np.array([1.0 if labels[x]["all_correct"] else 0.0 for x in ids])
                ys = np.array([ge2[x][metric] for x in ids])
                auc, ci = auroc_with_ci(yt, ys, args.bootstrap, args.seed + i)
                per_label_auc.append(auc)
                fpr, tpr, _ = roc_curve(yt, ys)
                curves.append({"label": rd.name, "fpr": fpr, "tpr": tpr,
                               "auroc": auc, "color": RUN_COLORS[i % len(RUN_COLORS)],
                               "lw": 1.9})
                entry = {"auroc": round(auc, 4), "auroc_ci95": ci,
                         "n": len(ids), "n_positive": int(yt.sum())}
                if HAS_SCIPY:
                    rho, _ = spearmanr(ys, yt)
                    entry["spearman"] = round(float(rho), 4)
                summary["checklist_runs"].setdefault(rd.name, {})[metric] = entry
                print(f"  {rd.name}: n={len(ids)}, AUROC = {auc:.4f}"
                      + (f"  CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""))

                if metric == "gm":
                    for x in ids:
                        csv_rows.append({"img_id": x, f"all_correct_{i + 1}":
                                         int(labels[x]["all_correct"]),
                                         f"n_correct_{i + 1}": labels[x]["n_correct"],
                                         f"n_atoms_{i + 1}": labels[x]["n_atoms"],
                                         "ge2_gm": ge2[x]["gm"], "ge2_am": ge2[x]["am"]})

            if len(per_label_auc) > 1:
                summary["across_labelings"][metric] = {
                    "mean": round(float(np.mean(per_label_auc)), 4),
                    "std": round(float(np.std(per_label_auc, ddof=1)), 4)}
                print(f"  across labelings: mean = {np.mean(per_label_auc):.4f} "
                      f"+/- {np.std(per_label_auc, ddof=1):.4f}")

            if args.majority_vote and voted is not None:
                ids = sorted(set(voted) & set(ge2), key=SORT_KEY)
                yt = np.array([1.0 if voted[x]["all_correct"] else 0.0 for x in ids])
                ys = np.array([ge2[x][metric] for x in ids])
                auc, ci = auroc_with_ci(yt, ys, args.bootstrap, args.seed)
                fpr, tpr, _ = roc_curve(yt, ys)
                curves.append({"label": f"Majority vote (n={len(ids)})", "fpr": fpr,
                               "tpr": tpr, "auroc": auc, "color": POOLED_COLOR, "lw": 2.4})
                summary["majority_vote"][metric] = {
                    "auroc": round(auc, 4), "auroc_ci95": ci, "n": len(ids)}
                print(f"  MAJORITY-VOTE labels: n={len(ids)}, AUROC = {auc:.4f}"
                      + (f"  CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""))

            if args.themis and themis_scores is not None and metric == "gm":
                ids = sorted(set(voted) & set(ge2) & set(themis_scores), key=SORT_KEY)
                yt = np.array([1.0 if voted[x]["all_correct"] else 0.0 for x in ids])
                y_th = np.array([themis_scores[x] for x in ids])
                auc_t, ci_t = auroc_with_ci(yt, y_th, args.bootstrap, args.seed)
                summary["themis_comparison"]["alignment"] = {
                    "auroc": round(auc_t, 4), "auroc_ci95": ci_t,
                    "labels": "majority vote over checklist runs",
                    "note": "same Reflector responses as labels — same-source advantage"}
                fpr, tpr, _ = roc_curve(yt, y_th)
                curves.append({"label": "THEMIS alignment* (consensus labels)",
                               "fpr": fpr, "tpr": tpr, "auroc": auc_t,
                               "color": THEMIS_COLOR, "lw": 2.0, "ls": "-."})
                print(f"  THEMIS alignment (consensus labels*): AUROC = {auc_t:.4f}"
                      + (f"  CI [{ci_t[0]:.3f}, {ci_t[1]:.3f}]" if ci_t else ""))

        else:
            # mode 2: same labeling, one curve per GenEval2 score file
            labels = label_sets[0][1]
            img_ids = sorted(set(labels) & set.intersection(
                *[set(s) for s in ge2_scores_list]), key=SORT_KEY)
            if args.themis and themis_scores is not None:
                img_ids = [i for i in img_ids if i in themis_scores]
            y_true = np.array([1.0 if labels[i]["all_correct"] else 0.0
                               for i in img_ids])
            print(f"  paired images: {len(img_ids)} "
                  f"(pos {int(y_true.sum())} / neg {int(len(y_true) - y_true.sum())})")

            per_run = []
            y_scores_runs = []
            for i, s in enumerate(ge2_scores_list):
                y_score = np.array([s[i2][metric] for i2 in img_ids])
                y_scores_runs.append(y_score)
                auc, ci = auroc_with_ci(y_true, y_score, args.bootstrap, args.seed + i)
                summary["runs"].setdefault(f"run{i + 1}", {})[metric] = {
                    "auroc": round(auc, 4), "auroc_ci95": ci}
                per_run.append(auc)
                fpr, tpr, _ = roc_curve(y_true, y_score)
                curves.append({"label": f"Run {i + 1}", "fpr": fpr, "tpr": tpr,
                               "auroc": auc, "color": RUN_COLORS[i % len(RUN_COLORS)],
                               "lw": 1.8})
                print(f"  run {i + 1}: AUROC = {auc:.4f}"
                      + (f"  CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""))
                if HAS_SCIPY:
                    rho, _ = spearmanr(y_score, y_true)
                    summary["runs"][f"run{i + 1}"][metric]["spearman"] = \
                        round(float(rho), 4)

                if metric == "gm":
                    for k, i2 in enumerate(img_ids):
                        csv_rows.append({"img_id": i2,
                                         "all_correct": int(labels[i2]["all_correct"]),
                                         "n_atoms": labels[i2]["n_atoms"],
                                         "n_correct": labels[i2]["n_correct"],
                                         f"ge2_gm_run{i + 1}": y_score[k]})

            # collapse deterministic (byte-identical) runs into one curve
            runs_identical = all(
                np.array_equal(y_scores_runs[0], ys) for ys in y_scores_runs[1:])
            if runs_identical and len(y_scores_runs) > 1:
                print(f"  (all {len(y_scores_runs)} runs are identical — deterministic "
                      f"decoding; curves collapsed)")
                for c in curves[1:]:
                    curves.remove(c)
                curves[0]["label"] = f"{mtitle} ({len(y_scores_runs)} identical runs)"

            # pooled
            y_score_pool = np.mean(y_scores_runs, axis=0)
            auc, ci = auroc_with_ci(y_true, y_score_pool, args.bootstrap, args.seed)
            summary["pooled"][metric] = {"auroc": round(auc, 4), "auroc_ci95": ci,
                                         "n": len(img_ids),
                                         "runs_identical": bool(runs_identical)}
            print(f"  POOLED (mean score): AUROC = {auc:.4f}"
                  + (f"  CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""))
            if not runs_identical:
                fpr, tpr, _ = roc_curve(y_true, y_score_pool)
                curves.append({"label": f"Pooled mean (n={len(img_ids)})", "fpr": fpr,
                               "tpr": tpr, "auroc": auc, "color": POOLED_COLOR, "lw": 2.4})
            if len(per_run) > 1:
                print(f"  across runs: mean = {np.mean(per_run):.4f} "
                      f"+/- {np.std(per_run, ddof=1):.4f}")

            # THEMIS reference curve
            if args.themis and themis_scores is not None and metric == "gm":
                y_themis = np.array([themis_scores[i] for i in img_ids])
                auc_t, ci_t = auroc_with_ci(y_true, y_themis, args.bootstrap, args.seed)
                summary["themis_comparison"]["alignment"] = {
                    "auroc": round(auc_t, 4), "auroc_ci95": ci_t,
                    "note": "same Reflector response as labels — same-source advantage"}
                fpr, tpr, _ = roc_curve(y_true, y_themis)
                curves.append({"label": "THEMIS alignment*", "fpr": fpr, "tpr": tpr,
                               "auroc": auc_t, "color": THEMIS_COLOR, "lw": 2.0,
                               "ls": "-."})
                print(f"  THEMIS alignment (reference*): AUROC = {auc_t:.4f}"
                      + (f"  CI [{ci_t[0]:.3f}, {ci_t[1]:.3f}]" if ci_t else ""))

        fig, ax = plt.subplots(figsize=(6.4, 6.2))
        sub = (f"positive: all checklist atoms correct (GenEval2 prompt-level); "
               f"score: {mtitle}"
               + ("; *labels and alignment share the same Reflector responses"
                  if any(c["label"].startswith("THEMIS") for c in curves) else ""))
        plot_roc(fig, ax, curves, f"ROC — Human annotators vs {mtitle}", sub)
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        fig_path = out_dir / fname
        fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  [SAVED] {fig_path}")

    # ---- outputs ----
    json_path = out_dir / "auroc_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"\n  [SAVED] {json_path}")

    if csv_rows:
        merged = {}
        for row in csv_rows:
            key = row["img_id"]
            merged.setdefault(key, {"img_id": key}).update(row)
        keys = ["img_id"] + [k for k in merged[next(iter(merged))].keys()
                             if k != "img_id"]
        csv_path = out_dir / "per_image_scores.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for img_id in sorted(merged, key=SORT_KEY):
                f.write(",".join(str(merged[img_id].get(k, "")) for k in keys) + "\n")
        print(f"  [SAVED] {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

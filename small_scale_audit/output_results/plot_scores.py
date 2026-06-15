import csv
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent / "aggregated_human_scores.csv"

with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

x = [r["image_name"].replace(".png", "") for r in rows]

for metric in ["alignment", "artifact"]:
    fig, ax = plt.subplots(figsize=(len(x) * 0.16 + 1, 4))

    lines = {
        f"User_1": (f"User_1_{metric}", "C0", "-"),
        f"User_2": (f"User_2_{metric}", "C1", "-"),
        f"User_3": (f"User_3_{metric}", "C2", "-"),
        "z-score": (f"{metric}_zscore", "C3", "--"),
        "additive": (f"{metric}_additive", "C4", "--"),
    }

    for label, (col, color, ls) in lines.items():
        vals = [float(r[col]) if r[col] else None for r in rows]
        ax.plot(x, vals, color=color, linestyle=ls, marker="o", markersize=3, label=label)

    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_xlabel("Image")
    ax.set_ylabel(f"{metric.capitalize()} Score")
    ax.set_title(f"{metric.capitalize()} Scores per Image")
    ax.legend()
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    # plt.subplots_adjust(left=0.06, right=0.98)

    out_path = Path(__file__).resolve().parent / f"{metric}_scores_plot.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

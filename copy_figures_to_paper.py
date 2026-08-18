"""
Copy relevant figures from analysis_output_* folders to AAAI_AuthorKit27/Figures.

Each model gets its own subfolder under Figures/, preserving original filenames.

Usage:
    python copy_figures_to_paper.py
    python copy_figures_to_paper.py --dry-run          # preview without copying
    python copy_figures_to_paper.py --clean            # clear target before copying
"""

import argparse
import shutil
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

# ── Model → source folder mapping ────────────────────────────────────────
MODELS = {
    "DiT":       ROOT / "analysis_output_Sys_DiT_ref_500_Human_DiT_500",
    "VAR":       ROOT / "analysis_output_Sys_VAR_ref_500_Human_VAR_500",
    "IMF":       ROOT / "analysis_output_Sys_IMF_ref_500_Human_IMF_500",
    "IMFfdloss": ROOT / "analysis_output_Sys_IMFfdloss_ref_500_Human_IMFfdloss_500",
    "JiTfdloss": ROOT / "analysis_output_Sys_JiTfdloss_ref_500_Human_JiTfdloss_500",
}

# ── Figures to copy (original filenames, kept as-is) ─────────────────────
FIGURES = [
    "roc_curve_group_avg_alignment_score.png",
    "roc_curve_group_avg_authenticity_score.png",
    "roc_curve_individual_alignment_score.png",
    "roc_curve_individual_authenticity_score.png",
    "corr_ICC_2_1_alignment_score.png",
    "corr_ICC_2_1_authenticity_score.png",
    "distribution_alignment_score.png",
    "distribution_authenticity_score.png",
    "joint_probability_contour.png",
]

DEST_BASE = ROOT / "AAAI_AuthorKit27" / "Figures"


def copy_figures(dry_run: bool = False, clean: bool = False) -> None:
    if clean and DEST_BASE.exists():
        if dry_run:
            print(f"[DRY-RUN] Would remove: {DEST_BASE}")
        else:
            shutil.rmtree(DEST_BASE)
            print(f"Cleaned: {DEST_BASE}")

    DEST_BASE.mkdir(parents=True, exist_ok=True)

    total = 0
    missing: list[str] = []

    for model, src_folder in MODELS.items():
        dest_folder = DEST_BASE / model
        if not dest_folder.exists():
            if dry_run:
                print(f"[DRY-RUN] Would create: {dest_folder}")
            else:
                dest_folder.mkdir(parents=True, exist_ok=True)

        for fig in FIGURES:
            src = src_folder / fig
            dst = dest_folder / fig
            if not src.exists():
                missing.append(str(src))
                continue
            if dry_run:
                print(f"[DRY-RUN] {src}  →  {dst}")
            else:
                shutil.copy2(src, dst)
            total += 1

        n = len(list(dest_folder.iterdir())) if dest_folder.exists() else 0
        action = "Would copy" if dry_run else "Copied"
        print(f"{action} to {model}/: {len(FIGURES)} files")

    action = "Would copy" if dry_run else "Copied"
    print(f"\n{action} total: {total} files across {len(MODELS)} models")

    if missing:
        print(f"\n[WARNING] {len(missing)} missing files:")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy figures to AAAI paper Figures folder.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without copying.")
    parser.add_argument("--clean", action="store_true", help="Clear target folder before copying.")
    args = parser.parse_args()
    copy_figures(dry_run=args.dry_run, clean=args.clean)

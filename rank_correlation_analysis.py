#!/usr/bin/env python
"""
Rank correlation and error analysis for model quality evaluation.

Compares FID, Human scores, and THEMIS (Ours) scores across 5 c2i models.
Evaluates whether THEMIS aligns better with human judgment than FID does.

Metrics computed:
  1. Rankings under each metric
  2. Spearman's rho  — rank correlation with Human
  3. Pearson's r     — linear correlation with Human (spacing fidelity)
  4. Normalized RMSE — absolute error after min-max scaling to [0, 1]

Note: FID is "lower is better"; Human and Ours are "higher is better".
      We negate FID (-FID) so all metrics share the same direction
      (higher = better) before computing correlations and NRMSE.
"""

import json
import numpy as np
from scipy import stats


# ============================================================
# Experimental Results
# ============================================================
models = [
    "DiT-XL/2-G (cfg=1.50)",
    "VAR-d24",
    "iMF-XL/2",
    "iMF-XL-FD-loss-post-trained",
    "JiT-H-FD-loss-post-trained",
]

# FID: lower is better
fid   = np.array([2.27, 2.09, 1.54, 0.72, 0.72])
# Human & Ours: composite = mean(Alignment) * mean(Artifact), higher is better
human = np.array([11.89, 11.99, 11.34, 12.62, 12.60])
ours  = np.array([10.72, 10.77, 10.10, 11.49, 11.36])

# Negate FID so higher = better (consistent with Human/Ours)
fid_neg = -fid


def minmax_normalize(x):
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


# ============================================================
# 1. Rankings
# ============================================================
print("=" * 80)
print("1. MODEL RANKINGS (rank 1 = best)")
print("=" * 80)

# FID: lower = better  -> rankdata ascending (smallest = rank 1)
fid_rank = stats.rankdata(fid, method='average')
# Human & Ours: higher = better -> rankdata of negated values (largest = rank 1)
human_rank = stats.rankdata(-human, method='average')
ours_rank = stats.rankdata(-ours, method='average')

header = f"{'Model':<45} {'FID':>6} {'Human':>6} {'Ours':>6}"
print(header)
print("-" * len(header))
for i, m in enumerate(models):
    print(f"{m:<45} {fid_rank[i]:>6.1f} {human_rank[i]:>6.1f} {ours_rank[i]:>6.1f}")
print("-" * len(header))
print("Note: FID rank 1 = lowest FID (best); Human/Ours rank 1 = highest score (best)")
print()


# ============================================================
# 2. Spearman's rho (rank correlation with Human)
# ============================================================
print("=" * 80)
print("2. SPEARMAN'S rho  (rank correlation with Human)")
print("=" * 80)

rho_fid, p_fid = stats.spearmanr(fid_neg, human)
rho_ours, p_ours = stats.spearmanr(ours, human)

print(f"  FID  vs Human:  rho = {rho_fid:+.4f}  (p = {p_fid:.4f})")
print(f"  Ours vs Human:  rho = {rho_ours:+.4f}  (p = {p_ours:.4f})")
print()
print(f"  Ours-Human Spearman {'>>' if rho_ours > rho_fid else 'is NOT >'} FID-Human Spearman")
print(f"  Difference: {rho_ours - rho_fid:+.4f}")
print()


# ============================================================
# 3. Pearson's r (linear correlation with Human)
# ============================================================
print("=" * 80)
print("3. PEARSON'S r  (linear correlation with Human — spacing fidelity)")
print("=" * 80)

r_fid, p_r_fid = stats.pearsonr(fid_neg, human)
r_ours, p_r_ours = stats.pearsonr(ours, human)

print(f"  FID  vs Human:  r = {r_fid:+.4f}  (p = {p_r_fid:.4f})")
print(f"  Ours vs Human:  r = {r_ours:+.4f}  (p = {p_r_ours:.4f})")
print()
print("  Interpretation: higher r => metric not only ranks correctly")
print("  but also preserves the 'how much better/worse' spacing.")
print(f"  Ours {'preserves' if r_ours > r_fid else 'does NOT preserve'} spacing better than FID.")
print()


# ============================================================
# 4. Normalized RMSE (min-max scaled to [0, 1])
# ============================================================
print("=" * 80)
print("4. NORMALIZED RMSE  (min-max scaled to [0, 1], then RMSE vs Human)")
print("=" * 80)

fid_norm   = minmax_normalize(fid_neg)
human_norm = minmax_normalize(human)
ours_norm  = minmax_normalize(ours)

nrmse_fid  = rmse(fid_norm, human_norm)
nrmse_ours = rmse(ours_norm, human_norm)

print(f"  {'Metric':<15} {'NRMSE':>10}")
print(f"  {'-' * 25}")
print(f"  {'FID (neg)':<15} {nrmse_fid:>10.4f}")
print(f"  {'Ours':<15} {nrmse_ours:>10.4f}")
print()
print(f"  Ours NRMSE is {'lower' if nrmse_ours < nrmse_fid else 'HIGHER'} "
      f"-> {'better' if nrmse_ours < nrmse_fid else 'WORSE'} fit to Human")
print()

# Show normalized values
print("  Normalized values (0 = worst, 1 = best):")
nhdr = f"  {'Model':<45} {'FID':>6} {'Human':>6} {'Ours':>6}"
print(nhdr)
print(f"  {'-' * (len(nhdr) - 2)}")
for i, m in enumerate(models):
    print(f"  {m:<45} {fid_norm[i]:>6.3f} {human_norm[i]:>6.3f} {ours_norm[i]:>6.3f}")
print()


# ============================================================
# 5. Summary Table
# ============================================================
print("=" * 80)
print("5. SUMMARY")
print("=" * 80)
print(f"  {'Metric':<20} {'Spearman rho':>12} {'Pearson r':>10} {'NRMSE':>8}")
print(f"  {'-' * 52}")
print(f"  {'FID (negated)':<20} {rho_fid:>+12.4f} {r_fid:>+10.4f} {nrmse_fid:>8.4f}")
print(f"  {'Ours':<20} {rho_ours:>+12.4f} {r_ours:>+10.4f} {nrmse_ours:>8.4f}")
print()
print("  Conclusion:")
wins = sum([rho_ours > rho_fid, r_ours > r_fid, nrmse_ours < nrmse_fid])
if wins == 3:
    print("  >> THEMIS outperforms FID on ALL three metrics:")
    print(f"     - Spearman rho (rank agreement):  {rho_ours:.4f}  >  {rho_fid:.4f}")
    print(f"     - Pearson r   (spacing fidelity):  {r_ours:.4f}  >  {r_fid:.4f}")
    print(f"     - NRMSE        (absolute fit):     {nrmse_ours:.4f}  <  {nrmse_fid:.4f}")
else:
    print(f"  >> THEMIS wins on {wins}/3 metrics. See details above.")
print()
print("  Note: With only 5 models, p-values are inherently limited.")
print("        Focus on the correlation magnitudes and NRMSE for comparison.")


# ============================================================
# Save results to JSON
# ============================================================
results = {
    "models": models,
    "scores": {
        "FID":   fid.tolist(),
        "Human": human.tolist(),
        "Ours":  ours.tolist(),
    },
    "rankings": {
        "FID":   fid_rank.tolist(),
        "Human": human_rank.tolist(),
        "Ours":  ours_rank.tolist(),
    },
    "spearman": {
        "FID_vs_Human":  {"rho": rho_fid,  "p_value": p_fid},
        "Ours_vs_Human": {"rho": rho_ours, "p_value": p_ours},
    },
    "pearson": {
        "FID_vs_Human":  {"r": r_fid,  "p_value": p_r_fid},
        "Ours_vs_Human": {"r": r_ours, "p_value": p_r_ours},
    },
    "normalized_rmse": {
        "FID_vs_Human":  nrmse_fid,
        "Ours_vs_Human": nrmse_ours,
    },
    "normalized_values": {
        "FID":   fid_norm.tolist(),
        "Human": human_norm.tolist(),
        "Ours":  ours_norm.tolist(),
    },
}

output_path = "rank_correlation_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nResults saved to {output_path}")

#!/usr/bin/env python3
"""
experiments/exp_bm3_cwru_leakfree/step5_gain_vs_coherence_leakfree.py

BATCH6 item 2: regenerate the A1 gain-vs-coherence scatter (S1,
supplementary material) with the leak-free, BATCH5-verified gain values.
Copied from experiments/step5_regen_coherence_plot.py (not modified; that
script and its output results/figures/gain_vs_coherence_n8.{png,pdf}
remain untouched). Coherence values are unchanged (intrinsic signal
property, independent of epoch-selection protocol) -- same source as the
original script.

Gain values: CWRU 5 points from tab:b2 (leak-free, n=8 at -4/-6/-8dB,
n=5 at -2/0dB); XJTU LOBO/Cross from tab:xjtu (BATCH5-verified, n=8,
nokin). Spearman uses exact permutation p-values (not scipy's asymptotic
formula, which breaks down at |r|=1 with small n -- see
EVIDENCE_REPORT_bm3_fix_addendum2.md and BATCH3_RESULTS.md).

Does not overwrite results/figures/gain_vs_coherence_n8.{png,pdf}.
Writes results/figures/gain_vs_coherence_n8_leakfree.{png,pdf}.
"""
import csv
import pathlib
from itertools import permutations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 9, "figure.dpi": 150,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

OUT_DIR = pathlib.Path("<REPO_ROOT>/results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Coherence values unchanged from the original script (intrinsic signal
# property). Gains: leak-free, matching tab:b2 / tab:xjtu (main text).
points = [
    ("CWRU", "SNR=0dB",   0.013335292227566242,  0.03),
    ("CWRU", "SNR=-2dB",  0.008194568566977978,  0.42),
    ("CWRU", "SNR=-4dB",  0.0038763377815485,    0.59),
    ("CWRU", "SNR=-6dB",  0.0014647903153672814, 0.62),
    ("CWRU", "SNR=-8dB",  0.0008575583924539387, 2.71),
    ("XJTU", "LOBO(Cond3)", 0.41167497634887695, -18.5),
    ("XJTU", "Cross(Cond2->Cond3)", 0.4285101542870204, 26.6),
]

new_csv = OUT_DIR.parent / "a1_coherence_20260708-2358" / "gain_vs_coherence_points_n8_leakfree.csv"
new_csv.parent.mkdir(parents=True, exist_ok=True)
with open(new_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "condition", "coherence", "gain_pp"])
    for row in points:
        w.writerow(row)


def exact_spearman_p(x, y):
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    r_obs, _ = spearmanr(x, y)
    rx = rankdata(x)
    count = total = 0
    for perm in permutations(range(n)):
        ry = rankdata(y)[list(perm)]
        r, _ = spearmanr(rx, ry)
        total += 1
        if abs(r) >= abs(r_obs) - 1e-9:
            count += 1
    return r_obs, count / total


cwru = [(c, g) for ds, cond, c, g in points if ds == "CWRU"]
xjtu = [(c, g) for ds, cond, c, g in points if ds == "XJTU"]

coh_cwru = [c for c, g in cwru]
gain_cwru = [g for c, g in cwru]
r5, p5 = exact_spearman_p(coh_cwru, gain_cwru)

coh_all = [p[2] for p in points]
gain_all = [p[3] for p in points]
r7, p7 = exact_spearman_p(coh_all, gain_all)
print(f"CWRU-only (n=5, exact): r={r5:.3f}, p={p5:.4f}")
print(f"Pooled (n=7, exact): r={r7:.3f}, p={p7:.4f}")

fig, ax = plt.subplots(figsize=(4.6, 3.9))
ax.scatter(coh_cwru, gain_cwru, color="#1f77b4", label="CWRU (AWGN sweep)", zorder=3)
ax.scatter([c for c, g in xjtu], [g for c, g in xjtu], color="#d62728",
           marker="s", label="XJTU-SY (natural)", zorder=3)
ax.axhline(0, color="#aaaaaa", lw=0.7, ls=":")
ax.set_xlabel("Fault-band channel coherence")
ax.set_ylabel("Dual $-$ single gain (pp)")
ax.set_title(f"Gain vs. coherence (leakage-free, n=8 CWRU + BATCH5-verified XJTU)\n"
             f"CWRU-only exact Spearman r={r5:.3f}, p={p5:.4f}; pooled r={r7:.3f}, p={p7:.4f}",
             fontsize=7.5)
ax.set_xlim(-0.03, 0.50)
ax.set_ylim(-22, 30)

label_pos = {
    "SNR=-8dB":  (0.075, 10.5),
    "SNR=-6dB":  (0.075, 6.5),
    "SNR=-4dB":  (0.075, 2.5),
    "SNR=0dB":   (0.075, -1.5),
    "SNR=-2dB":  (0.075, -5.5),
    "LOBO(Cond3)":         (0.30, -18.5),
    "Cross(Cond2->Cond3)": (0.20, 24.0),
}
for ds, cond, c, g in points:
    tx, ty = label_pos[cond]
    ax.annotate(
        cond, xy=(c, g), xytext=(tx, ty), fontsize=6.5,
        ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6,
                         shrinkA=2, shrinkB=4),
        zorder=4,
    )

ax.legend(fontsize=7, loc="center", bbox_to_anchor=(0.55, 0.42))
fig.tight_layout()
fig.savefig(OUT_DIR / "gain_vs_coherence_n8_leakfree.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "gain_vs_coherence_n8_leakfree.pdf", bbox_inches="tight")
print("Saved -> gain_vs_coherence_n8_leakfree.{png,pdf}")

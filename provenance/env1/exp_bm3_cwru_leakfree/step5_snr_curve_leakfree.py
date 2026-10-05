#!/usr/bin/env python3
"""
Step 5 / SNR curve (leak-free): accuracy vs SNR for BM3 CE-only / BM3 +L_kin / BM2 CE-only.

C2 "variance stabilisation" WITHDRAWN under leak-free data (EDITS_C2_withdraw.md,
EVIDENCE_REPORT_bm3_fix_addendum2.md [1]): mean-difference and std-ratio no longer
move in a consistent direction across the SNR grid. This script (copied from
experiments/step5_snr_curve.py, see BATCH4 diff) drops the -4dB "-62%"/annotation
entirely rather than recomputing a corrected percentage — there is no single
number left to highlight.

Data: results/bm3_cwru_leakfree_20260911-leakfree/exp02_snr*/summary.json (BM3)
      and .../exp04_mamba2_snr*/summary.json (BM2), field test_accs_at_best_val
SNR grid: {-8, -6, -4, -2, 0, +10} dB (unchanged)

Usage:
    source venv/bin/activate
    python experiments/exp_bm3_cwru_leakfree/step5_snr_curve_leakfree.py
"""
import json
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT    = pathlib.Path("<REPO_ROOT>/results/bm3_cwru_leakfree_20260911-leakfree")
OUT_DIR = pathlib.Path("<REPO_ROOT>/results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SNRS = [-8, -6, -4, -2, 0, 10]   # SNR grid


def load_accs(exp_dir: pathlib.Path):
    """Return (mean%, std%) from summary.json, or (None, None) if missing."""
    p = exp_dir / "summary.json"
    if not p.exists():
        return None, None
    d = json.loads(p.read_text())
    seeds = d.get("test_accs_at_best_val", [])
    if not seeds:
        return None, None
    arr = np.array(seeds) * 100
    return float(arr.mean()), float(arr.std(ddof=1))


# ── load data ─────────────────────────────────────────────────────────────────
bm3_nokin_m, bm3_nokin_s = [], []
bm3_kin_m,   bm3_kin_s   = [], []
bm2_m,       bm2_s       = [], []

for snr in SNRS:
    m, s = load_accs(ROOT / f"exp02_snr{snr}_nokin");  bm3_nokin_m.append(m); bm3_nokin_s.append(s)
    m, s = load_accs(ROOT / f"exp02_snr{snr}_kin");    bm3_kin_m.append(m);   bm3_kin_s.append(s)
    m, s = load_accs(ROOT / f"exp04_mamba2_snr{snr}"); bm2_m.append(m);       bm2_s.append(s)

snrs    = np.array(SNRS, dtype=float)
x_ticks = snrs.copy()

# Convert to numpy, handle None
def arr(lst):
    return np.array([v if v is not None else np.nan for v in lst])

bm3_nokin_m = arr(bm3_nokin_m); bm3_nokin_s = arr(bm3_nokin_s)
bm3_kin_m   = arr(bm3_kin_m);   bm3_kin_s   = arr(bm3_kin_s)
bm2_m       = arr(bm2_m);       bm2_s       = arr(bm2_s)

print("SNR data loaded:")
for i, snr in enumerate(SNRS):
    print(f"  SNR={snr:+3d}: nokin={bm3_nokin_m[i]:.2f}±{bm3_nokin_s[i]:.2f}  "
          f"kin={bm3_kin_m[i]:.2f}±{bm3_kin_s[i]:.2f}  "
          f"bm2={bm2_m[i]:.2f}±{bm2_s[i]:.2f}")

# ── figure: 2-panel ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6),
                          gridspec_kw={"width_ratios": [2.2, 1.0], "wspace": 0.35})

# ── panel (a): accuracy curves with std bands ─────────────────────────────────
ax = axes[0]
ax.set_title("(a) Accuracy vs. SNR — BM3 ±L_kin and BM2 baseline")

# BM2 reference (orange, thinner, semi-transparent)
ax.fill_between(snrs, bm2_m - bm2_s, bm2_m + bm2_s,
                alpha=0.18, color="#ff7f0e")
ax.plot(snrs, bm2_m, color="#ff7f0e", lw=1.4, ls="--",
        marker="s", markersize=5, markerfacecolor="white", markeredgewidth=1.2,
        label="BM2 CE-only (baseline)")

# BM3 nokin (blue dashed)
ax.fill_between(snrs, bm3_nokin_m - bm3_nokin_s, bm3_nokin_m + bm3_nokin_s,
                alpha=0.20, color="#1f77b4")
ax.plot(snrs, bm3_nokin_m, color="#1f77b4", lw=1.5, ls="--",
        marker="o", markersize=5, markerfacecolor="white", markeredgewidth=1.2,
        label="BM3 CE-only (w/o L_kin)")

# BM3 kin (blue solid, narrower band)
ax.fill_between(snrs, bm3_kin_m - bm3_kin_s, bm3_kin_m + bm3_kin_s,
                alpha=0.35, color="#1f77b4")
ax.plot(snrs, bm3_kin_m, color="#1f77b4", lw=2.0, ls="-",
        marker="o", markersize=5,
        label="BM3 +L_kin (λ=0.01, cover)")

ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Accuracy (%)")
ax.set_xlim(-9.5, 11.5)
ax.set_xticks(SNRS)
ax.set_xticklabels([f"{s:+d}" for s in SNRS])
ax.set_ylim(82, 101.5)
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.grid(axis="y", ls=":", alpha=0.45)
ax.grid(axis="x", ls=":", alpha=0.25)
ax.legend(loc="lower right", framealpha=0.88)

# NOTE (EDITS_C2_withdraw.md E4): the -4dB "std reduction" annotation that used to
# live here is deleted, not fixed. Under leak-free data the mean-difference /
# std-ratio no longer moves in a consistent direction across the SNR grid (see
# EVIDENCE_REPORT_bm3_fix_addendum2.md [1]), so there is no single "-XX%" figure
# left to highlight — the withdrawn C2 variance-stabilisation claim is retired,
# not re-computed with a corrected sign.

# ── panel (b): std (variance) curves only ─────────────────────────────────────
ax2 = axes[1]
ax2.set_title("(b) Std across 5 seeds")

ax2.plot(snrs, bm3_nokin_s, color="#1f77b4", lw=1.5, ls="--",
         marker="o", markersize=5, markerfacecolor="white", markeredgewidth=1.2,
         label="BM3 w/o L_kin")
ax2.plot(snrs, bm3_kin_s, color="#1f77b4", lw=2.0, ls="-",
         marker="o", markersize=5,
         label="BM3 +L_kin")
ax2.plot(snrs, bm2_s, color="#ff7f0e", lw=1.4, ls="--",
         marker="s", markersize=5, markerfacecolor="white", markeredgewidth=1.2,
         label="BM2")

ax2.set_xlabel("SNR (dB)")
ax2.set_ylabel("Std (%)")
ax2.set_xlim(-9.5, 11.5)
ax2.set_xticks(SNRS)
ax2.set_xticklabels([f"{s:+d}" for s in SNRS])
ax2.set_ylim(bottom=-0.05)
ax2.grid(axis="y", ls=":", alpha=0.45)
ax2.grid(axis="x", ls=":", alpha=0.25)
ax2.legend(loc="upper right", framealpha=0.88)

# NOTE (EDITS_C2_withdraw.md E4): the hardcoded "-62%" annotation is deleted
# (it was a fixed string, not derived from data — see addendum for why the
# withdrawal, not just the sign, is the correct fix).

# ── finalize ─────────────────────────────────────────────────────────────────
fig.suptitle(
    "CWRU 4-class / BearMamba-3 SISO / 5 seeds",
    fontsize=8, color="#444444"
)
fig.tight_layout(rect=[0, 0, 1, 0.95])

out_stem = "snr_curve_cwru_leakfree"
fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / f"{out_stem}.png", bbox_inches="tight", dpi=200)
print(f"\nSaved → {OUT_DIR / out_stem}.{{pdf,png}}")
plt.close(fig)

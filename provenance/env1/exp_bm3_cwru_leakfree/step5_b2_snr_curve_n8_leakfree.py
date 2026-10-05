#!/usr/bin/env python3
"""
Step 5 / B2 (leak-free n=8 update, BATCH4): single vs dual-sensor SNR curve,
merging leak-free seeds 0-4 (main 42-config campaign) with the leak-free E6
new-seed extension (seeds 5-7, BATCH1) at the three "flagship" SNR points
(-4/-6/-8 dB). -2/0 dB remain leak-free n=5 (not extended in BATCH1).

Copied from experiments/step5_b2_snr_curve_n8.py (see BATCH4 diff). That
script's own n=8 finding (-4dB nokin p=0.641, "false positive") was computed
on the LEAKY pipeline; this script recomputes the same n=8 merge on
leak-free data (EVIDENCE_REPORT_bm3_fix.md / BATCH1_RESULTS.md /
BATCH2_RESULTS.md give the actual leak-free numbers — -4dB nokin p=0.133,
-8dB nokin p=0.0039; kin is asymmetric, see BATCH2_RESULTS.md §8). No
hardcoded p-value or percentage is baked into this script; the Wilcoxon
p-values are computed at run time from whatever data is on disk, same as
the original.

Does NOT overwrite results/figures/b2_snr_curve_dual_n8.{pdf,png} (results/
guardrail: append-only). Output is a new file,
results/figures/b2_snr_curve_dual_n8_leakfree.{pdf,png}.
"""
import json
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

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

SNRS = [-8, -6, -4, -2, 0]         # +10dB excluded (single already 100%)
N8_SNRS = {-8, -6, -4}             # E6 new-seed extension only covers these


def load_accs(exp_dir: pathlib.Path):
    p = exp_dir / "summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    accs = np.array(d.get("test_accs_at_best_val", [])) * 100
    return accs if len(accs) else None


def get_old_dual_dir(kind: str, snr: int) -> pathlib.Path:
    if snr == 0:
        return ROOT / f"exp_b2_dual_{kind}"
    tag = f"snrm{abs(snr)}" if snr < 0 else f"snr{snr}"
    return ROOT / f"exp_b2_dual_{kind}_{tag}"


def get_old_single_dir(kind: str, snr: int) -> pathlib.Path:
    return ROOT / f"exp02_snr{snr}_{kind}"


def get_e6_dir(arm: str, kind: str, snr: int) -> pathlib.Path:
    return ROOT / f"exp_e6_{arm}_{kind}_snrm{abs(snr)}_newseed_leakfree"


def merged_seeds(arm: str, kind: str, snr: int):
    """arm='single'|'dual'. Returns concatenated seed-accuracy array (n=5 or n=8)."""
    old_dir = get_old_single_dir(kind, snr) if arm == "single" else get_old_dual_dir(kind, snr)
    old = load_accs(old_dir)
    if old is None:
        return None
    if snr in N8_SNRS:
        new = load_accs(get_e6_dir(arm, kind, snr))
        if new is not None:
            return np.concatenate([old, new])
    return old


# ── load merged data ──────────────────────────────────────────────────────────
data = {}
for cond in ("nokin", "kin"):
    data[cond] = {"single": {"seeds": [], "mean": [], "std": [], "n": []},
                  "dual":   {"seeds": [], "mean": [], "std": [], "n": []}}
    for snr in SNRS:
        for arm in ("single", "dual"):
            seeds = merged_seeds(arm, cond, snr)
            data[cond][arm]["seeds"].append(seeds)
            data[cond][arm]["mean"].append(float(seeds.mean()) if seeds is not None else np.nan)
            data[cond][arm]["std"].append(float(seeds.std(ddof=1)) if seeds is not None else np.nan)
            data[cond][arm]["n"].append(len(seeds) if seeds is not None else 0)

print("Sample sizes per SNR (single/dual, nokin):",
      list(zip(SNRS, data["nokin"]["single"]["n"], data["nokin"]["dual"]["n"])))

# Wilcoxon (dual vs single, per SNR, paired by seed index — valid for the
# n=8 points because seeds [0..4] are shared across old runs and [5..7] are
# shared across the E6 new-seed batch; for n=5 points this reduces to the
# original pairing).
p_nokin, p_kin = [], []
for i, snr in enumerate(SNRS):
    sn, dn = data["nokin"]["single"]["seeds"][i], data["nokin"]["dual"]["seeds"][i]
    sk, dk = data["kin"]["single"]["seeds"][i], data["kin"]["dual"]["seeds"][i]
    try:
        p_nokin.append(stats.wilcoxon(dn, sn, alternative="greater").pvalue)
    except Exception:
        p_nokin.append(float("nan"))
    try:
        p_kin.append(stats.wilcoxon(dk, sk, alternative="greater").pvalue)
    except Exception:
        p_kin.append(float("nan"))

print("B2(n8) Wilcoxon p-values (dual > single, one-sided):")
for snr, pn, pk, n in zip(SNRS, p_nokin, p_kin, data["nokin"]["dual"]["n"]):
    print(f"  {snr:+3d}dB (n={n})  nokin p={pn:.4f}  kin p={pk:.4f}")

snrs = np.array(SNRS, dtype=float)

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6),
                          gridspec_kw={"width_ratios": [2.0, 1.0], "wspace": 0.38})

COLOR = {"nokin": "#1f77b4", "kin": "#2ca02c"}
LABEL = {"nokin": "CE-only", "kin": "+L_kin"}

# ── panel (a): accuracy curves ────────────────────────────────────────────────
ax = axes[0]
ax.set_title("(a) Single vs Dual Sensor — Accuracy vs SNR (n=8 @ -4/-6/-8dB)")

for cond in ("nokin", "kin"):
    c = COLOR[cond]
    lbl = LABEL[cond]
    m_s = np.array(data[cond]["single"]["mean"])
    s_s = np.array(data[cond]["single"]["std"])
    m_d = np.array(data[cond]["dual"]["mean"])
    s_d = np.array(data[cond]["dual"]["std"])

    ax.fill_between(snrs, m_s - s_s, m_s + s_s, alpha=0.18, color=c)
    ax.plot(snrs, m_s, color=c, lw=1.4, ls="--",
            marker="o", markersize=5, markerfacecolor="white", markeredgewidth=1.2,
            label=f"Single {lbl}")
    ax.fill_between(snrs, m_d - s_d, m_d + s_d, alpha=0.30, color=c)
    ax.plot(snrs, m_d, color=c, lw=2.0, ls="-",
            marker="o", markersize=5,
            label=f"Dual {lbl}")

ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Accuracy (%)")
ax.set_xlim(-9.5, 1.5)
ax.set_xticks(SNRS)
ax.set_xticklabels([f"{s:+d}" for s in SNRS])
ax.set_ylim(82, 101.5)
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.grid(axis="y", ls=":", alpha=0.45)
ax.grid(axis="x", ls=":", alpha=0.25)
ax.legend(loc="lower right", framealpha=0.88, ncol=2)

# ── panel (b): gain (dual - single), monotonic-trend annotation (no
#     significance claim — replaces the retired p=0.031 callout) ─────────────
ax2 = axes[1]
ax2.set_title("(b) Gain: Dual − Single (pp)")

for cond in ("nokin", "kin"):
    c = COLOR[cond]
    lbl = LABEL[cond]
    gain = np.array(data[cond]["dual"]["mean"]) - np.array(data[cond]["single"]["mean"])
    ax2.plot(snrs, gain, color=c, lw=1.8, ls="-" if cond == "nokin" else "--",
             marker="o", markersize=5,
             label=f"{lbl}")
    for i, (snr, g) in enumerate(zip(SNRS, gain)):
        if abs(g) > 0.3:
            # NOTE (BATCH4 diff): was f"+{g:.2f}" — a hardcoded "+" produces a
            # double-sign glyph ("+-0.50") if a leak-free gain is negative.
            # Signed format handles both directions correctly.
            ax2.text(snr + 0.15, g + 0.05, f"{g:+.2f}", fontsize=6.5,
                     color=c, va="bottom")

ax2.axhline(0, color="#aaaaaa", lw=0.8, ls=":")
ax2.annotate(
    # BATCH4_FIGURES.md fix: the blanket "not significant" this text
    # previously carried was wrong — nokin -8dB (n=8) reaches p=0.0039
    # (see BATCH2_RESULTS.md / addendum2). Point to the table instead of
    # asserting a significance verdict here.
    "direction-consistent trend across SNR;\nper-point significance varies, see Table",
    xy=(0.97, 0.04), xycoords="axes fraction",
    ha="right", va="bottom",
    fontsize=7, color="#444444",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="#888888"),
)
ax2.set_xlabel("SNR (dB)")
ax2.set_ylabel("Accuracy gain (pp)")
ax2.set_xlim(-9.5, 1.5)
ax2.set_xticks(SNRS)
ax2.set_xticklabels([f"{s:+d}" for s in SNRS])
ax2.set_ylim(-0.6, 2.6)
ax2.grid(axis="y", ls=":", alpha=0.45)
ax2.legend(loc="upper right", framealpha=0.88)

fig.suptitle(
    "CWRU SISO→Dual (DE+FE) / BearMamba-3 — n=8 at -4/-6/-8dB, n=5 at -2/0dB",
    fontsize=8, color="#444444"
)
fig.tight_layout(rect=[0, 0, 1, 0.94])

out_stem = "b2_snr_curve_dual_n8_leakfree"
fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / f"{out_stem}.png", bbox_inches="tight", dpi=200)
print(f"\nSaved -> {OUT_DIR / out_stem}.{{pdf,png}}")
plt.close(fig)

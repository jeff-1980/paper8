#!/usr/bin/env python3
"""
experiments/exp_bm3_cwru_leakfree/analyze_b2_leakfree_n8.py

BATCH 1 follow-up analysis: merges the leak-free n=5 pool (seeds 0-4, main
42-config campaign) with the leak-free n=8 extension (seeds 5-7, BATCH 1)
at the three flagship SNR points (-4/-6/-8 dB), for single vs dual sensor,
CE-only ("nokin"). Pairing logic mirrors
experiments/step5_b2_snr_curve_n8.py::merged_seeds() (np.concatenate of the
two seed ranges), but reads `test_accs_at_best_val` from the leak-free
results tree instead of `best_val_accs` from the leaky one.

kin condition: only dual_kin was extended to seeds 5-7 in BATCH 1 (per user
spec — single_kin was NOT rerun at n=8). A paired Wilcoxon at matched n=8
is therefore NOT computable for kin (single_kin stays at n=5); this script
reports dual_kin's n=8 descriptive stats only, with an explicit gap note,
rather than fabricating a mismatched-n test.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).parent.parent.parent
LEAKFREE_ROOT = ROOT / "results" / "bm3_cwru_leakfree_20260911-leakfree"

SNRS = [-4, -6, -8]


def snr_tag(snr):
    return f"snrm{abs(int(snr))}"


def load(name):
    p = LEAKFREE_ROOT / name / "summary.json"
    d = json.loads(p.read_text())
    return np.array(d["test_accs_at_best_val"]), d["seeds"]


def main():
    print("=" * 90)
    print("  B2 flagship points, leak-free n=8 pool (seeds 0-4 main campaign + 5-7 BATCH1)")
    print("=" * 90)

    results = {}
    for snr in SNRS:
        tag = snr_tag(snr)

        s_nk_5, s_nk_5_seeds = load(f"exp02_snr{snr}_nokin")
        s_nk_3, s_nk_3_seeds = load(f"exp_e6_single_nokin_{tag}_newseed_leakfree")
        d_nk_5, d_nk_5_seeds = load(f"exp_b2_dual_nokin_{tag}")
        d_nk_3, d_nk_3_seeds = load(f"exp_e6_dual_nokin_{tag}_newseed_leakfree")

        s_nk_8 = np.concatenate([s_nk_5, s_nk_3])
        d_nk_8 = np.concatenate([d_nk_5, d_nk_3])
        seeds_8 = s_nk_5_seeds + s_nk_3_seeds
        assert seeds_8 == d_nk_5_seeds + d_nk_3_seeds == [0, 1, 2, 3, 4, 5, 6, 7]

        p_nk = wilcoxon(d_nk_8, s_nk_8, alternative="greater").pvalue
        delta_nk = float((d_nk_8 - s_nk_8).mean()) * 100

        # kin: dual extended to n=8, single stays n=5 — no valid paired n=8 test
        d_k_5, _ = load(f"exp_b2_dual_kin_{tag}")
        d_k_3, _ = load(f"exp_e6_dual_kin_{tag}_newseed_leakfree")
        d_k_8 = np.concatenate([d_k_5, d_k_3])
        s_k_5, _ = load(f"exp02_snr{snr}_kin")  # still n=5 — NOT extended in BATCH1

        results[snr] = {
            "single_nokin_n8": s_nk_8, "dual_nokin_n8": d_nk_8,
            "p_nokin_n8": p_nk, "delta_nokin_n8": delta_nk,
            "single_kin_n5": s_k_5, "dual_kin_n8": d_k_8,
        }

        print(f"\n--- SNR = {snr:+d} dB ---")
        print(f"  single_nokin (n=8): {[f'{v*100:.2f}' for v in s_nk_8]}")
        print(f"  dual_nokin   (n=8): {[f'{v*100:.2f}' for v in d_nk_8]}")
        print(f"  mean single_nokin = {s_nk_8.mean()*100:.2f} ± {s_nk_8.std(ddof=1)*100:.2f}%")
        print(f"  mean dual_nokin   = {d_nk_8.mean()*100:.2f} ± {d_nk_8.std(ddof=1)*100:.2f}%")
        print(f"  Delta (dual - single) = {delta_nk:+.2f}pp")
        print(f"  Wilcoxon(dual, single, alternative='greater'), n=8: p = {p_nk:.4f}")
        print(f"  positive pairs: {(d_nk_8 - s_nk_8 > 0).sum()}/8")
        print()
        print(f"  [kin, n=8 dual only — single_kin stays n=5, no valid paired n=8 test]")
        print(f"  dual_kin (n=8): {[f'{v*100:.2f}' for v in d_k_8]}")
        print(f"  mean dual_kin (n=8) = {d_k_8.mean()*100:.2f} ± {d_k_8.std(ddof=1)*100:.2f}%")
        print(f"  mean single_kin (n=5, unchanged) = {s_k_5.mean()*100:.2f} ± {s_k_5.std(ddof=1)*100:.2f}%")
        print(f"  descriptive delta (n=8 dual - n=5 single) = "
              f"{(d_k_8.mean()-s_k_5.mean())*100:+.2f}pp  [NOT a valid paired test — mismatched n]")

    # ── summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  Summary (nokin, n=8, valid paired test)")
    print("=" * 90)
    print(f"  {'SNR':>5}  {'Delta(pp)':>10}  {'p (one-sided)':>14}  {'pos/8':>6}")
    for snr in SNRS:
        r = results[snr]
        pos = int((r["dual_nokin_n8"] - r["single_nokin_n8"] > 0).sum())
        print(f"  {snr:+4d}dB  {r['delta_nokin_n8']:>+9.2f}  {r['p_nokin_n8']:>14.4f}  {pos:>4d}/8")

    out = {}
    for snr in SNRS:
        r = results[snr]
        out[str(snr)] = {
            "single_nokin_n8": r["single_nokin_n8"].tolist(),
            "dual_nokin_n8": r["dual_nokin_n8"].tolist(),
            "delta_nokin_n8_pp": r["delta_nokin_n8"],
            "p_nokin_n8_one_sided": float(r["p_nokin_n8"]),
            "dual_kin_n8": r["dual_kin_n8"].tolist(),
            "single_kin_n5_unchanged": r["single_kin_n5"].tolist(),
            "kin_paired_n8_test": "NOT COMPUTABLE — single_kin not extended past n=5 in BATCH1",
        }
    out_path = LEAKFREE_ROOT / "b2_n8_leakfree_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()

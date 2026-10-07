# TABLE_MAP — from every main-text number to its files

> **Manuscript numbering, fifth draft (release mst-v3).** Section headings below use the working labels
> of `aggregate/aggregate_main.py`; match by LaTeX label. Main text: Table 1 = `tab:litcomp` (not computed),
> Table 2 = `tab:influence` (not computed), **Table 3 = `tab:xc`** (heading "Table 1" below),
> **Table 4 = `tab:gated`** (round 3), **Table 5 = `tab:loco`** ("Table 2"), **Table 6 = `tab:lobo`**
> ("Table 3"), **Table 7 = `tab:cov`** ("Table 4"), **Table 8 = `tab:onset`** (round 3),
> **Table 9 = `tab:cwru`** and **Table 10 = `tab:corr`** (round 4, corrected noise implementation; see the
> section "Round 4" at the end), Table 11 = `tab:claims` (not computed). Figures 1–3 keep their numbers.
> Supplementary: S24–S30 (section S8) are the per-seed round-3 XJTU-SY tables, S31 (section S9) the per-seed
> round-4 CWRU table.
>
> **Superseded noise protocol.** Every CWRU and Paderborn result in this repository that involves injected
> noise and is *not* under `results/mst_round4_noisefix_20261007/` was obtained with the earlier noise
> implementation (`noise_key: v1`: noise seeded by the index within a split, reused across splits). These
> results, and the blocks of `aggregate_main.py` and of this file that refer to them (marked
> "SUPERSEDED" below), are kept for provenance and are not used to support the current manuscript.
> Noise-free results (all XJTU-SY results) are unaffected.
>
> Late fusion: both variants (weight from the source hold-out applied to the hold-out sub-models, or to the
> full-data models) are printed for both designs, as reported in the manuscript.

Every value below is printed by

```bash
python aggregate/aggregate_main.py          # full output: aggregate/expected_output.txt
```

The "section" column names the block of `expected_output.txt` in which the value appears.
Result paths are relative to `results/`, configuration paths relative to `configs/`. "Seeds" are the
training seeds (initialisation and data-loader order) on fixed splits. Metrics: macro-F1 for the
held-out-condition designs, present-class recall for LOBO folds, window accuracy on the test bearing
for the coverage sweeps, test accuracy at the validation-selected epoch (`test_acc_at_best_val`) for
CWRU. All XJTU-SY numbers use the fixed final epoch (50).

Environment 1 trainers: `experiments/exp01_cwru_baseline/train_leakfree.py` (CWRU),
`experiments/exp_bm3_cwru_leakfree/train_xjtu_leakfree.py` (XJTU cross-condition),
`experiments/exp_e3_lobo_leakfree/train_lobo_leakfree.py` (LOBO),
`experiments/exp_bm3_cwru_leakfree/train_p14_coverage_leakfree.py` (IR coverage sweep).
Environment 2 trainers: `scripts/xjtu/train_xjtu_leakfree{,_v2,_v3}.py` + `scripts/extra/eval_ckpts_cross{,_v3}.py`
(per-bearing), `scripts/lobo/train_lobo_leakfree_v2.py` + `scripts/extra/eval_ckpts_lobo.py`,
`scripts/extra/fusion_holdout.py` + `scripts/extra/build_fusion_report.py` (late fusion), and the
environment-1 CWRU trainer (byte-identical copy) for the CWRU environment-2 arms.

## Table 1 (`tab:xc`) — XJTU-SY condition 2 → 3, macro-F1

Output block: `TABLE 1 (tab:xc)`.

| row | env | seeds | configs | result dirs |
|---|---|---|---|---|
| A: 1D-CNN H | 1 | 0–4, 5–7 | `env1/xjtu_cross/exp_mext_e13_1dcnn_xjtu_cross_leakfree.yaml`, `..._newseed_leakfree.yaml` | `env1/xjtu_cross/exp_mext_e13_1dcnn_xjtu_cross_leakfree/`, `..._newseed_leakfree/` |
| A: 1D-CNN H+V | 1 | 0–4, 5–7 | `env1/xjtu_cross/exp_e1b_xjtu_cross_dual_cnn_leakfree.yaml`, `..._newseed_leakfree.yaml` | `env1/xjtu_cross/exp_e1b_xjtu_cross_dual_cnn_leakfree/`, `..._newseed_leakfree/` |
| A: 1D-CNN V | see note a | 0–7 | `env2/xjtu_cross/cnn_V_only.yaml` | `env2/xjtu_cross/cnn_V_only/` (log: `env2/xjtu_cross/reports/log_cnn_V_only.txt`) |
| A: Mamba-3 H, V, H+V | 2 | 0–4 + 5–7 | `env2/xjtu_cross/xc_bm3_{H,V,HV}.yaml` (seeds 0–4), `env2/xjtu_cross/q1_bm3_{H,V,HV}.yaml` (seeds 5–7) | `env2/xjtu_cross/xc_bm3_{H,V,HV}/`, `env2/xjtu_cross/q1_bm3_{H,V,HV}/` |
| A: Mamba-2 H, V | 2 | 0–7 | `env2/xjtu_cross/r2_bm2_{H,V}.yaml` | `env2/xjtu_cross/r2_bm2_{H,V}/` |
| B: Mamba-2 H | 1 | 0–4, 5–7 | `env1/xjtu_cross/exp_e1b_xjtu_cross_single_bm2_leakfree.yaml`, `..._newseed_leakfree.yaml` | `env1/xjtu_cross/exp_e1b_xjtu_cross_single_bm2_leakfree/`, `..._newseed_leakfree/` |
| B: Mamba-2 H+V | 1 | 0–4, 5–7 | `env1/xjtu_cross/exp_e1_xjtu_cross_dual_bm2_leakfree.yaml`, `..._newseed_leakfree.yaml` | `env1/xjtu_cross/exp_e1_xjtu_cross_dual_bm2_leakfree/`, `..._newseed_leakfree/` |
| B: Mamba-3 H | 1 | 0–4, 5–7 | `env1/xjtu_cross/exp_xjtu_cross_nokin_leakfree.yaml`, `..._nokin_newseed_leakfree.yaml` | `env1/xjtu_cross/exp_xjtu_cross_nokin_leakfree/`, `..._nokin_newseed_leakfree/` |
| B: Mamba-3 H+V | 1 | 0–4, 5–7 | `env1/xjtu_cross/exp_xjtu_cross_dual_nokin_leakfree.yaml`, `..._dual_nokin_newseed_leakfree.yaml` | `env1/xjtu_cross/exp_xjtu_cross_dual_nokin_leakfree/`, `..._newseed_leakfree/` |
| footnote a (1D-CNN bridge) | 2 | 0–4 | `env2/xjtu_cross/xc_cnn_{H,HV}.yaml` | `env2/xjtu_cross/xc_cnn_{H,HV}/` vs the environment-1 dirs above |
| A: Mamba-2 H+V (round 3) | 2 | 0–7 | `mst_round3_20261007/cfg_r3/e1_bm2_HV.yaml` | `mst_round3_20261007/e1_bm2_HV/` |
| ~~footnote b (Mamba-2, 17.9 pp, cross-environment)~~ | — | — | superseded in the fourth draft by row A Mamba-2 H+V (same environment, paired) | — |

Note a — the 1D-CNN V arm was trained after the H and H+V arms with the same script
(`train_xjtu_leakfree.py` with `sensor_channel: 1`); its log shows it ran in the environment-2 software
stack. The 1D-CNN uses no Mamba kernel, and the H and H+V arms re-trained in environment 2 reproduce the
environment-1 values seed by seed (footnote a; `xc_cnn_V` seeds 0–4 equal `cnn_V_only` seeds 0–4).
The manuscript labels the row "env. 1".

Contrast footnote below the table (V − H for the three learners, Mamba-3 H+V − V p = 0.0156, 7/8):
same files; printed in the same block.

## Section 4.1 — numbers quoted in the text

Output blocks: `TABLE 1` (panel B lines print the inner-race recall) and `Section 4.1`.

| quoted value | files |
|---|---|
| 1D-CNN / Mamba-2 H macro-F1 40.2 / 43.3, IR recall 0.2 / 3.1 % (env 1) | Table 1 panel-B dirs, field `final_per_class_recall[1]` |
| V-only IR recall 72.5 % = 1264/1744 (Bearing3_4 share); per-bearing 3_3 = 0, 3_4 = 100 | `env2/xjtu_cross/cnn_V_only/seed_*.json`; per bearing `env2/xjtu_cross/xc_cnn_V/per_bearing.json` (seeds 0–4), `env2/xjtu_cross/r2_bm2_V/per_bearing.json` |
| Mamba-3 H+V − V +8.2 [4.1, 12.5], 7/8; IR recall 79.0 vs 78.9; per bearing 23/100 vs 93/74, Bearing3_5 100 vs 74 | `env2/xjtu_cross/{xc,q1}_bm3_{V,HV}/seed_*.json`, `per_bearing.json` |
| late fusion 86.5 ± 5.1, −8.1 [−12.0, −4.3] vs early fusion (seeds 0–4); 1D-CNN late fusion ≈ V-only | `env2/xjtu_cross/late_fusion/fusion_cross_bm3.json` (`fusion_sub`, sub-models trained without the hold-out), `fusion_cross_cnn1d.json`; configs `env2/xjtu_cross/fus_xc_{bm3,cnn}.yaml`; report `env2/xjtu_cross/reports/summary_fusion.md` |
| channel-attention variant (supplement only) | `env2/xjtu_cross/xc_cnn_ATT/`, `late_fusion/attn_weights.json` |
| section 3.4: SD of paired differences 6.5 pp, s/√n 2.3 pp | as Mamba-3 H+V − V |
| section 3.1: OR:IR = 4,624 : 624 = 7.4 : 1 | `train_class_counts` in any condition-2 → 3 `seed_*.json` |

## Table 2 (`tab:loco`) — leave-one-condition-out, macro-F1, environment 2

Output block: `TABLE 2 (tab:loco)`.

| row | seeds | configs (`env2/xjtu_loco/`) | result dirs (`env2/xjtu_loco/`) |
|---|---|---|---|
| Cond. 2 Mamba-3 | 0–2 + 3–7 | `bm3_L2_{H,V,HV}.yaml`, `r1_bm3_L2_{H,V,HV}.yaml` | `bm3_L2_*`, `r1_bm3_L2_*` |
| Cond. 2 1D-CNN | 0–2 + 3–4 | `cnn_L2_*.yaml`, `r1_cnn_L2_*.yaml` | `cnn_L2_*`, `r1_cnn_L2_*` |
| Cond. 3 Mamba-3 | 0–2 + 3–7 | `q3_bm3_L3_*.yaml`, `r1_bm3_L3_*.yaml` | `q3_bm3_L3_*`, `r1_bm3_L3_*` |
| Cond. 3 1D-CNN | 0–2 + 3–4 | `q3_cnn_L3_*.yaml`, `r1_cnn_L3_*.yaml` | `q3_cnn_L3_*`, `r1_cnn_L3_*` |
| Cond. 1 (both learners), OR recall 98–100 per bearing | 0–2 | `bm3_L1_*.yaml`, `cnn_L1_*.yaml` | `bm3_L1_*`, `cnn_L1_*` |

Section 4.2 text values (p = 0.0156 and 7/8; 1D-CNN −22.3, 5/5; Bearing2_5 H 4, V 100, dual 74;
per-bearing dual − V +7, −1, 0, −26 at condition 2 and +76, −32, +33, 0 at condition 3; +8.6 [3.7, 13.5],
p = 0.039, 6/8) are printed in the same block from `seed_*.json` and `per_bearing.json`.
Earlier analysis reports: `env2/xjtu_loco/reports/an_loco_results.md`, `an_q3_results.md`, `an_r1_results.md`.
Pre-specified protocol: `protocols/prereg_independent_validation.md`.

## Figure 2 (`fig:forest`)

Output block: `FIGURE 2 (fig:forest)`. Rows are the dual − single contrasts of Table 1 (condition 2 → 3:
1D-CNN dual − H from panel B, environment 1; 1D-CNN dual − V and Mamba-3 rows from panel A; both Mamba-2 rows
from round 3, `e1_bm2_HV` vs `env2/xjtu_cross/r2_bm2_{H,V}`, environment 2),
Table 2 (held-out conditions 2 and 3; dual − H and dual − V) and Table 3 (LOBO OR-fold mean recall,
environment 2). Plotted values and sources: manuscript supplementary section S7 and
`paper_mst/fig2_forest_data.csv` (fourth draft). The earlier cross-environment Mamba-2 point is no longer plotted.

## Section 4.3 — LOBO main comparison (environment 1)

Output block: `Section 4.3 -- LOBO main comparison`. Mamba-3 macro-recall
(OR-fold mean + IR-fold mean)/2, H 86.4 ± 2.4, H+V 67.9 ± 9.7, −18.5 [−25.5, −12.0], 8/8, p = 0.0078.

| arm | seeds | configs (`env1/xjtu_lobo/`) | result dirs (`env1/xjtu_lobo/`) |
|---|---|---|---|
| H | 0–4, 5–7 | `config_lobo_nokin_leakfree.yaml`, `config_lobo_nokin_leakfree_n8ext.yaml` | `single_nokin/`, `single_nokin_n8ext/` |
| H+V | 0–4, 5–7 | `config_lobo_dual_nokin_leakfree.yaml`, `config_lobo_dual_nokin_leakfree_n8ext.yaml` | `dual_nokin/`, `dual_nokin_n8ext/` |

Pre-specified protocols: `protocols/prereg_lobo_leakfree.md`, `protocols/prereg_bm3_leakfree.md`.

## Table 3 (`tab:lobo`) — condition-3 LOBO controls, environment 2

Output block: `TABLE 3 (tab:lobo)`.

| row | seeds | configs (`env2/xjtu_lobo/`) | result dirs (`env2/xjtu_lobo/`) |
|---|---|---|---|
| Mamba-3 H, V, H+V | 0–2, 3–7 | `lobo_bm3_{H,V,HV}.yaml` (seeds 3–7 via `--seeds`) | `lobo_bm3_{H,V,HV}/seeds0-2/`, `lobo_bm3_{H,V,HV}/seeds3-7/` |
| Mamba-3 late fusion | 0–2 | `fus_lobo_bm3.yaml` | `late_fusion/fusion_lobo_bm3.json` (field `fusion_full`: weight selected on the source hold-out, applied to the full-data H and V models) |
| 1D-CNN H, V, H+V | 0–4 | `lobo_cnn_{H,V,HV}.yaml` | `lobo_cnn_{H,V,HV}/` |

Section 4.3 text: OR-fold recall 34.2 vs 75.7 / 76.0; −41.5 [−48.8, −35.7] and −41.9 [−48.8, −34.3], 8/8,
p = 0.0078, negative on both OR test bearings; late fusion 77.1; 1D-CNN 94.6 / 45.3 / 59.1 — same block.
Reports: `env2/xjtu_lobo/reports/summary_lobo_bm3_8seed.md` (8 seeds, `scripts/extra/an_lobo8.py`),
`summary_lobo_bm3.md`, `summary_lobo_cnn.md`.

## Section 4.3 — pre-specified IR (other-class) coverage sweep (supplementary table S19)

Output block: `Section 4.3 -- pre-specified IR (other-class) coverage sweep`. Environment 1, Mamba-3,
seeds 0–4, test bearing Bearing3_1; configs `env1/xjtu_ir_coverage_p14/config_p14_{single,dual}.yaml`;
results `env1/xjtu_ir_coverage_p14/{single,dual}/k*_seed*.json`. "30–48 pp below H in 5/5 seeds per
subset" = the three subsets containing the single OR training bearing (Bearing3_5) plus one or two IR
bearings.

## Table 4 (`tab:cov`) and Figure 3 (`fig:cov`) — OR coverage sweep, environment 2

Output block: `TABLE 4 (tab:cov) and FIGURE 3 (fig:cov)`. Mamba-3, seeds 0–2, IR training bearings
2_1, 3_3, 3_4 fixed. Configs and results: `env2/xjtu_or_coverage/q2_T{3_1,2_2,1_1}_{SAME,OTHER}_k{1,2,3}_{H,V,HV}`
(`.yaml` in `configs/`, directories with `seed_*.json` and `per_bearing.json` in `results/`).
Δ_best = H+V mean − max(H mean, V mean), from unrounded seed means. Figure 3 points are printed per
seed for every cell. The first (pre-specified, one test bearing, seeds 0–1) sweep is
`env2/xjtu_or_coverage/bm3_COV{1,2,4,7}_{H,HV}` (supplement). Report: `env2/xjtu_or_coverage/reports/an_q2_results.md`.

## SUPERSEDED — former Table 5 (`tab:cwru`) — CWRU, AWGN, seeds 0–4 (noise_key v1; current Table 9: see "Round 4")

Output block: `TABLE 5 (tab:cwru)`.

| row | env | configs | result dirs |
|---|---|---|---|
| Mamba-3 DE-only (fourth draft, Table 9) | 2 | `mst_round3_20261007/cfg_r3/e4_bm3_DE_snr{-4,-6,-8}.yaml` | `mst_round3_20261007/e4_bm3_DE_snr{-4,-6,-8}/` |
| Mamba-3 DE+FE (fourth draft, Table 9) | 2 | `mst_round3_20261007/cfg_r3/e4_bm3_DEFE_snr{-4,-6,-8}.yaml` | `mst_round3_20261007/e4_bm3_DEFE_snr{-4,-6,-8}/` |
| Mamba-3 DE-only, environment 1 (first measurement, quoted in text and S6) | 1 | `env1/cwru/exp02_snr-{4,6,8}_nokin.yaml` | `env1/cwru/exp02_snr-{4,6,8}_nokin/` |
| Mamba-3 DE+FE, environment 1 (first measurement, quoted in text and S6) | 1 | `env1/cwru/exp_b2_dual_nokin_snrm{4,6,8}.yaml` | `env1/cwru/exp_b2_dual_nokin_snrm{4,6,8}/` |
| Mamba-3 FE-only | 2 | `env2/cwru/bm3_FE_snr-{4,6,8}.yaml` | `env2/cwru/bm3_FE_snr-{4,6,8}/` |
| 1D-CNN FE-only, DE-only, DE+FE | 2 | `env2/cwru/cnn_{FE,DE,DEFE}_snr-{4,6,8}.yaml` | `env2/cwru/cnn_{FE,DE,DEFE}_snr-{4,6,8}/` |
| environment bridge (Mamba-3 DE-only −8 dB, −0.06 pp) | 2 | `env2/cwru/bm3_DE_snr-8.yaml` | `env2/cwru/bm3_DE_snr-8/` (report `env2/cwru/reports/summary_cwru_fe_full.md`) |

Section 4.4 text, same block: 1D-CNN gains +0.1, +0.7, +4.3; environment-1 Mamba-3 gains +1.2, +1.5, +2.8
(first measurement). Environment-2 Mamba-3 gains (+0.52, +1.88, +3.27) and FE − DE are printed by
`mst_round3_20261007/an_r3.py` (block E4).

## SUPERSEDED — former section 4.4 — eight-seed pool, pink noise, coherence (noise_key v1; not used)

| quoted value | output block | env, seeds | result dirs |
|---|---|---|---|
| 8-seed gains +0.59, +0.62, +2.71, one-sided p 0.13, 0.32, 0.0039 | `eight-seed Mamba-3 pool` | 1, 0–7 | `env1/cwru/exp02_snr-{4,6,8}_nokin/` + `exp_e6_single_nokin_snrm{4,6,8}_newseed_leakfree/`; `exp_b2_dual_nokin_snrm{4,6,8}/` + `exp_e6_dual_nokin_snrm{4,6,8}_newseed_leakfree/` (cross-check `env1/cwru/b2_n8_leakfree_summary.json`) |
| pink gains −0.21, −0.45, −0.39, two-sided p ≥ 0.19 | `pink noise` | 1, 0–4 | `env1/cwru/exp_e5_{single,dual}_pink_snr-{4,6,8}_leakfree/` |
| DE-only at −8 dB: pink 97.6, AWGN 85.8 | `pink noise` | 1; pink 0–4, AWGN 0–7 | as above |
| matched accuracy 0.98 [−0.28, 2.33] | `pink noise` | 1 | AWGN −4 dB 8-seed gain vs pink −8 dB 5-seed gain (independent-group bootstrap) |
| in-band SNR 2–5 kHz +0.1 dB vs −5.8 dB | `pink noise` | — | stored output `env2/cwru_noise_analysis/inband.json` (script `provenance/env2/analysis/inband.py`, needs raw CWRU data) |
| coherence 0.013 (0 dB), 0.001 (−8 dB) | `pink noise` | — | stored output `env1/cwru_coherence/gain_vs_coherence_points_n8_leakfree.csv` (script `provenance/env1/run_a1_coherence.py`, needs raw CWRU data) |
| section 3.4: SD of paired differences in the n = 8 CWRU cells 1.3–2.4 pp | `eight-seed Mamba-3 pool` | 1 | as above |
| section 3.4: power 0.99 / 0.61 / 0.31 | — | — | stored output `env2/cwru_noise_analysis/power.json` (script `provenance/env2/analysis/analysis3.py`) |

Pre-specified protocol for the CWRU seed extension and the dual-sensor baselines:
`protocols/prereg_e1b_n8.md`, `protocols/prereg_dual_baselines.md`, `protocols/prereg_bm3_leakfree.md`.

## Not recomputed by `aggregate_main.py`

* In-band SNR, DE–FE coherence and the power simulation require the raw CWRU signals and noise
  simulation; their stored outputs are read and printed, and the producing scripts are in
  `provenance/`.
* Figure 1 is a schematic (no data).


## Round 3 (pre-registered, environment 2) — `results/mst_round3_20261007/`

Protocol: `protocols/prereg_mst_round3.md` (committed before the first run; Amendment 1 appended after the
window-gate condition 2 → 3 runs and before any token-gate run). Run order and wall time: `wall.txt`
(the first CWRU pass failed on an import path and was rerun in full; only the rerun is used). Bridge
check before the runs: `smoke_bm2_H_s0/` reproduces `env2/xjtu_cross/r2_bm2_H/seed_0.json` exactly.
Reproduce: `cd results/mst_round3_20261007 && python an_r3.py && python gen_r3_tex.py && python gen_r3_supp.py`.

| manuscript item | round-3 dirs (under `mst_round3_20261007/`) | comparators |
|---|---|---|
| Table 3 panel A, Mamba-2 H+V and contrasts | `e1_bm2_HV/` | `env2/xjtu_cross/r2_bm2_{H,V}/` |
| Table 4 panel A (gated fusion, condition 2 → 3, seeds 0–4) | `e3_{cnn,bm3}_gated/`, `e3t_{cnn,bm3}_tgated/` (+ `gate_stats.json`) | `env2/xjtu_cross/xc_{cnn,bm3}_{H,V,HV}/` |
| Table 4 panel B (gated fusion, LOBO OR folds, seeds 0–2) | `e3_lobo_bm3_gated/`, `e3t_lobo_bm3_tgated/` | `env2/xjtu_lobo/lobo_bm3_{H,V,HV}/` |
| Table 8 (onset rule V / last 15 %, seeds 0–2) | `e2_{V,last15}_{xc,lobo}_bm3_{H,V,HV}/` | H rule: `env2/xjtu_cross/xc_bm3_*`, `env2/xjtu_lobo/lobo_bm3_*` (seeds 0–2); segment counts `audit/onset_audit.csv` |
| SUPERSEDED (noise_key v1) — fourth-draft Table 9 Mamba-3 DE-only and DE+FE (seeds 0–4) | `e4_bm3_{DE,DEFE}_snr{-4,-6,-8}/` | FE-only: `env2/cwru/bm3_FE_snr{-4,-6,-8}/` |
| SUPERSEDED (noise_key v1) — fourth-draft Table 10 (correlated noise, seeds 0–4) | `e5_bm3_DEFE_snr{-6,-8}_rho{0.5,0.9}/`; ρ = 0 is `e4_bm3_DEFE_snr{-6,-8}/` | DE-only `e4_bm3_DE_snr{-6,-8}/` |
| Figure 2 Mamba-2 rows | `e1_bm2_HV/` | `env2/xjtu_cross/r2_bm2_{H,V}/` |
| Fourth-draft supplementary S29–S36 (now S24–S30; the CWRU table is archived) | all of the above | — |

Code used: `bearmamba3/data_xjtu.py` (`XJTU_ONSET_RULE` = `H` (default) / `V` / `last15`) with
`environments/patches/env2_data_xjtu_channel.diff` applied; `bearmamba3/data_cwru_leakfree.py`
(noise type `awgn_corr<ρ>`; `awgn` unchanged); `baselines/gated_fusion.py`;
`scripts/xjtu/train_xjtu_leakfree_v3.py` (backbones `*_gated`, `*_tgated`); `scripts/lobo/train_lobo_leakfree_v3.py`
(option `only_folds`); CWRU: the environment-1 trainer. Mamba-2 arm: also apply
`environments/patches/env2_mamba2_unfused_conv.diff`.

## Round 4 (CWRU noise-key correction, environment 2) — `results/mst_round4_noisefix_20261007/`

Protocol `protocols/prereg_mst_round4_noisefix.md` (committed before the runs). Corrected generator
`noise_key: v2` in `bearmamba3/data_cwru_leakfree.py`: seeded by (split, recording, window start sample,
realisation), one row per physical channel, ρ applied to the FE row; train/val/test noise independent, test
noise identical across seeds and configurations. Unit tests: `r4/unit_tests_noisefix.py` (run from the
repository root). 13 configurations × seeds 0–4, Mamba-3, all exit 0 (`wall.txt`); only the
validation-selected checkpoint was kept (not released). Printed by `an_r4.py` (`an_r4_results.md`,
`an_r4_results.json`); the LaTeX rows of Tables 9 and 10 and of table S31 are written by `gen_r4_tex.py`
(`r4_tex.json`). Both scripts run in place and reproduce the stored outputs byte for byte.

| manuscript item | result folders | configs |
|---|---|---|
| Table 9 (FE-only, DE-only, DE+FE, −4/−6/−8 dB) and the FE−DE text | `r4_{FE,DE,DEFE_rho0}_snr{-4,-6,-8}/` | `configs/round4/` (repo-relative; copies used for the runs in `cfg_r4/`) |
| Table 10 (ρ = 0, 0.5, 0.9 at −6/−8 dB; ΔG) | `r4_DEFE_rho{0,0.5,0.9}_snr{-6,-8}/`; DE-only `r4_DE_snr{-6,-8}/` | as above |
| Supplementary table S31 | all 13 folders | — |

Re-run: `bash reproduce/run_round.sh round4 [config ...]` from the repository root (see `reproduce/README.md`).

# Pre-specified protocols

Preregistration documents written before the corresponding runs (dates inside each file).

| file | scope |
|---|---|
| `prereg_bm3_leakfree.md` | leak-free re-run protocol (CWRU time-axis split, XJTU-SY/Paderborn fixed final epoch) |
| `prereg_lobo_leakfree.md` | condition-3 LOBO, leak-free |
| `prereg_e1b_n8.md` | extension of the condition 2 -> 3 H and H+V arms from seeds 0-4 to 0-7 |
| `prereg_dual_baselines.md`, `prereg_cnn_ln.md` | dual-sensor baselines and the LayerNorm 1D-CNN (supplement) |
| `prereg_independent_validation.md` | leave-one-condition-out design and the first OR coverage sweep |
| `prereg_mst_round3.md` | round 3 of the MST revision (E1–E5): Mamba-2 dual arm in environment 2, degraded-phase sensitivity, gated fusion, CWRU arms in environment 2, correlated inter-channel noise. Committed to the authors' working repository on 2026-10-06 19:49 (+08:00), before the first run (20:42 completion of the first arm is in `results/mst_round3_20261007/wall.txt`); Amendment 1 appended 2026-10-06 21:31, after the window-gate condition 2 → 3 runs and before any token-gate run. |

For this public release, author-tool attributions and references to internal project notes were replaced by `[redacted]` / `[internal project notes, not released]`; no protocol content was changed.
* `prereg_mst_round4_noisefix.md` — CWRU noise-implementation correction; committed to the authors' working
  repository (commit `9f3cffb`, 2026-10-07 17:58 UTC+8) before the first round-4 run started.

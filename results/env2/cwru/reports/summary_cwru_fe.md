# CWRU fan-end (FE) single-channel control — BM3 and 1D-CNN, AWGN -4/-6/-8 dB

## Deviations from the brief / run notes
- **5 seeds (0-4)** only, for every arm (brief: 5; the plan's "8 seeds if possible" was not attempted). Provisional in the sense that n=5 gives a two-sided exact Wilcoxon floor of p = 0.0625: no comparison can reach p<0.05; report effect sizes and bootstrap CIs.
- **Added file**: `scripts/cwru/train.py` (original exp01_cwru_baseline/train.py) was copied in, because the supplied package's `train_leakfree.py` imports `build_model` from it and it was missing from the zip. Trainer `train_leakfree.py` itself unchanged.
- **Added driver** `run_multi.py` runs several configs sequentially in one process (via runpy of the unchanged trainer) to avoid repeated Mamba-3 JIT. No hyperparameter changed (d_model 64, d_state 128, 4 layers, AdamW 3e-4, wd 1e-4, batch 64, 50 epochs, cosine, clip 1). Configs differ from the originals only in data_dir (absolute), num_workers 0, results_dir, channels, backbone (cnn1d for the CNN arms).
- **Environment check (BM3 DE-only, -8 dB, seeds 0-4)**: 85.03 vs 85.09 original, difference -0.06 pt (<1 pt), so the **original BM3 DE-only and DE+FE result files** (exp02_snr{-4,-6,-8}_nokin, exp_b2_dual_nokin_snrm{4,6,8}; seeds 0-4) are used as comparators; **BM3 DE-only (-4,-6) and DE+FE were NOT rerun here**. Seed-paired differences therefore pair runs from different environments/kernel builds; this is justified only by the environment check at a single SNR (-8 dB) for a single arm, and per-seed differences there reach 1.67 pt.
- **1D-CNN (added per the plan step, not in the brief)**: FE, DE and DE+FE all trained here at the 3 SNRs (9 arms, 5 seeds), because no leak-free original CNN DE+FE exists (exp_mext_e14 used the earlier random split). CNN DE-only vs original leak-free files: differences -0.18/0.00/-0.03 pt at -8/-6/-4 dB.
- **Failed/interrupted run**: the first CNN batch was killed by a daemon/kernel restart while running cnn_FE_snr-6 (seeds 0-1 finished); the six remaining CNN arms, including cnn_FE_snr-6 from scratch, were rerun in a second process (out/log_batch3). cnn_FE_snr-8, cnn_DE_snr-8, cnn_DEFE_snr-8 come from the first process. No results were discarded selectively.
- Per-epoch checkpoints (~285 MB per BM3 arm) are omitted from the zip; per-seed JSON, test-prediction NPZ, split index, source snapshots are included. Original-file test predictions used for per-class recall (BM3 DE, DE+FE) were read from the original directory (read-only).
- "Better single channel" is chosen by mean **validation** accuracy (never test); with this rule it is DE in every case, and it coincides with the better-by-test choice.
- Bootstrap: 20000 resamples of the 5 seed-paired differences (seed 12345); Wilcoxon: scipy exact, two-sided (CNN -4 dB DE+FE−DE has one zero difference, dropped, so n=4, floor 0.125).
- Not run: 8-seed extension; BM3 reruns of DE/DE+FE in this environment; pink noise or other noise types.

## 1. Environment check (this environment vs original result files, test_acc_at_best_val, seeds 0-4)

| backbone | arm | this env (mean ± SD) | original (mean ± SD) | mean diff (pts) | max abs per-seed diff (pts) |
|---|---|---|---|---|---|
| bm3 | DE-only -8 dB | 85.03 ± 0.66 | 85.09 ± 1.76 | -0.06 | 1.67 |
| cnn | DE-only -8 dB | 94.45 ± 0.79 | 94.64 ± 1.31 | -0.18 | 0.91 |
| cnn | DE-only -6 dB | 99.18 ± 0.14 | 99.18 ± 0.44 | -0.00 | 0.45 |
| cnn | DE-only -4 dB | 99.85 ± 0.11 | 99.88 ± 0.13 | -0.03 | 0.15 |

BM3: |diff| < 1 pt at -8 dB, so original DE-only and DE+FE files are used as comparators (BM3 DE/DE+FE were NOT rerun). CNN: DE-only rerun at all three SNRs and DE+FE rerun here (no leak-free original exists for CNN DE+FE; exp_mext_e14 used the pre-leak-free random split).

## 2. BM3 (Mamba-3, CE only, lambda_kin=0)

Accuracy = test_acc_at_best_val (checkpoint chosen by validation accuracy), mean ± SD over seeds 0-4, %.

| SNR | FE-only | DE-only | DE+FE | source of DE / DE+FE |
|---|---|---|---|---|
| -4 dB | 92.55 ± 0.91 | 97.12 ± 0.99 | 98.33 ± 0.47 | original files |
| -6 dB | 84.42 ± 0.96 | 92.67 ± 1.79 | 94.12 ± 1.13 | original files |
| -8 dB | 76.09 ± 2.20 | 85.09 ± 1.76 | 87.88 ± 1.80 | original files |

Paired differences (percentage points, seed-paired, n=5). Columns: mean diff, bootstrap 95% CI (20000 resamples over seed-paired differences), exact two-sided Wilcoxon p (**n=5 floor is 2/32 = 0.0625, so p<0.05 is unattainable**), #seeds positive/negative/zero.

| SNR | comparison | mean diff (pts) | bootstrap 95% CI | Wilcoxon p (exact) | +/-/0 |
|---|---|---|---|---|---|
| -4 dB | DE+FE − DE | +1.21 | [+0.52, +2.21] | 0.0625 | 5/0/0 |
| -4 dB | DE+FE − better single (by mean val-acc: DE) | +1.21 | [+0.52, +2.21] | 0.0625 | 5/0/0 |
| -4 dB | FE − DE | -4.58 | [-5.91, -3.24] | 0.0625 | 0/5/0 |
| -6 dB | DE+FE − DE | +1.45 | [-0.27, +3.18] | 0.4375 | 3/2/0 |
| -6 dB | DE+FE − better single (by mean val-acc: DE) | +1.45 | [-0.27, +3.18] | 0.4375 | 3/2/0 |
| -6 dB | FE − DE | -8.24 | [-9.73, -6.55] | 0.0625 | 0/5/0 |
| -8 dB | DE+FE − DE | +2.79 | [+0.58, +5.45] | 0.0625 | 5/0/0 |
| -8 dB | DE+FE − better single (by mean val-acc: DE) | +2.79 | [+0.58, +5.45] | 0.0625 | 5/0/0 |
| -8 dB | FE − DE | -9.00 | [-10.39, -7.42] | 0.0625 | 0/5/0 |

Per-class recall (%), mean over seeds 0-4 (test set, checkpoint chosen by val). Classes: normal, inner, ball, outer.

| SNR | arm | normal | inner | ball | outer |
|---|---|---|---|---|---|
| -4 dB | FE | 99.9 | 91.9 | 87.5 | 86.7 |
| -4 dB | DE | 100.0 | 95.9 | 96.5 | 94.4 |
| -4 dB | DE+FE | 100.0 | 96.1 | 98.1 | 98.2 |
| -6 dB | FE | 99.6 | 75.2 | 81.1 | 73.2 |
| -6 dB | DE | 100.0 | 87.9 | 91.2 | 87.4 |
| -6 dB | DE+FE | 100.0 | 88.8 | 93.3 | 91.0 |
| -8 dB | FE | 97.6 | 60.3 | 69.6 | 64.6 |
| -8 dB | DE | 100.0 | 78.2 | 77.6 | 76.0 |
| -8 dB | DE+FE | 100.0 | 80.1 | 82.6 | 81.8 |

Per-seed test accuracy (%):

| SNR | arm | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|
| -4 dB | FE | 91.67 | 93.18 | 93.79 | 92.27 | 91.82 |
| -4 dB | DE | 97.73 | 95.76 | 96.67 | 98.33 | 97.12 |
| -4 dB | DEFE | 98.33 | 98.94 | 97.88 | 98.64 | 97.88 |
| -6 dB | FE | 85.15 | 84.09 | 85.00 | 82.88 | 85.00 |
| -6 dB | DE | 90.45 | 91.21 | 93.94 | 93.03 | 94.70 |
| -6 dB | DEFE | 94.09 | 95.30 | 94.39 | 92.27 | 94.55 |
| -8 dB | FE | 77.42 | 73.64 | 73.94 | 76.82 | 78.64 |
| -8 dB | DE | 83.64 | 83.18 | 85.30 | 85.76 | 87.58 |
| -8 dB | DEFE | 88.03 | 90.61 | 86.67 | 85.91 | 88.18 |

## 3. 1D-CNN (baselines/cnn1d)

Accuracy = test_acc_at_best_val (checkpoint chosen by validation accuracy), mean ± SD over seeds 0-4, %.

| SNR | FE-only | DE-only | DE+FE | source of DE / DE+FE |
|---|---|---|---|---|
| -4 dB | 96.48 ± 0.91 | 99.85 ± 0.11 | 99.97 ± 0.07 | rerun here |
| -6 dB | 91.76 ± 0.93 | 99.18 ± 0.14 | 99.91 ± 0.08 | rerun here |
| -8 dB | 84.18 ± 1.52 | 94.45 ± 0.79 | 98.73 ± 0.35 | rerun here |

Paired differences (percentage points, seed-paired, n=5). Columns: mean diff, bootstrap 95% CI (20000 resamples over seed-paired differences), exact two-sided Wilcoxon p (**n=5 floor is 2/32 = 0.0625, so p<0.05 is unattainable**), #seeds positive/negative/zero.

| SNR | comparison | mean diff (pts) | bootstrap 95% CI | Wilcoxon p (exact) | +/-/0 |
|---|---|---|---|---|---|
| -4 dB | DE+FE − DE | +0.12 | [+0.06, +0.15] | 0.1250 | 4/0/1 |
| -4 dB | DE+FE − better single (by mean val-acc: DE) | +0.12 | [+0.06, +0.15] | 0.1250 | 4/0/1 |
| -4 dB | FE − DE | -3.36 | [-4.12, -2.67] | 0.0625 | 0/5/0 |
| -6 dB | DE+FE − DE | +0.73 | [+0.58, +0.88] | 0.0625 | 5/0/0 |
| -6 dB | DE+FE − better single (by mean val-acc: DE) | +0.73 | [+0.58, +0.88] | 0.0625 | 5/0/0 |
| -6 dB | FE − DE | -7.42 | [-8.18, -6.85] | 0.0625 | 0/5/0 |
| -8 dB | DE+FE − DE | +4.27 | [+3.64, +5.00] | 0.0625 | 5/0/0 |
| -8 dB | DE+FE − better single (by mean val-acc: DE) | +4.27 | [+3.64, +5.00] | 0.0625 | 5/0/0 |
| -8 dB | FE − DE | -10.27 | [-11.79, -8.76] | 0.0625 | 0/5/0 |

Per-class recall (%), mean over seeds 0-4 (test set, checkpoint chosen by val). Classes: normal, inner, ball, outer.

| SNR | arm | normal | inner | ball | outer |
|---|---|---|---|---|---|
| -4 dB | FE | 100.0 | 98.5 | 93.9 | 91.5 |
| -4 dB | DE | 100.0 | 99.4 | 99.9 | 100.0 |
| -4 dB | DE+FE | 100.0 | 99.9 | 100.0 | 100.0 |
| -6 dB | FE | 100.0 | 92.1 | 89.0 | 81.1 |
| -6 dB | DE | 100.0 | 98.2 | 99.4 | 98.6 |
| -6 dB | DE+FE | 100.0 | 99.9 | 100.0 | 99.7 |
| -8 dB | FE | 99.7 | 74.8 | 82.9 | 70.4 |
| -8 dB | DE | 100.0 | 86.9 | 96.1 | 91.7 |
| -8 dB | DE+FE | 100.0 | 97.9 | 99.0 | 97.2 |

Per-seed test accuracy (%):

| SNR | arm | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|
| -4 dB | FE | 96.97 | 95.30 | 95.76 | 96.97 | 97.42 |
| -4 dB | DE | 100.00 | 99.85 | 99.85 | 99.85 | 99.70 |
| -4 dB | DEFE | 100.00 | 100.00 | 100.00 | 100.00 | 99.85 |
| -6 dB | FE | 91.67 | 90.30 | 92.88 | 91.97 | 91.97 |
| -6 dB | DE | 99.09 | 99.09 | 99.39 | 99.24 | 99.09 |
| -6 dB | DEFE | 100.00 | 99.85 | 99.85 | 99.85 | 100.00 |
| -8 dB | FE | 86.52 | 84.85 | 83.03 | 82.88 | 83.64 |
| -8 dB | DE | 94.39 | 93.94 | 95.45 | 95.00 | 93.48 |
| -8 dB | DEFE | 98.94 | 98.18 | 98.79 | 98.64 | 99.09 |

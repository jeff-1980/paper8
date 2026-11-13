# Protocol: round 6 (rewrite of MST-140500), written 2026-11-04 before any round-6 run

Scope: Jeff's decision of 2026-11-04 (`paper_mst/UNFREEZE_MST-140500.md`). Reviewer item numbers refer to
the decision letter (R1.x = referee 1, R2.x = referee 2).

## 0. Common settings
- Code: release mst-v3 plus the round-6 changes listed in section 7 (all opt-in; defaults reproduce mst-v3).
- Environment 2 (Python 3.11.15, PyTorch 2.13.0+cu130, Triton 3.7.1, mamba_ssm 2.3.2.post1 build a14b1df),
  same GPU, bf16, num_workers 0, `BM_DETERMINISTIC=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` for every run.
- Training hyper-parameters identical to the corresponding mst-v3 configurations; only the fields named
  here change. Every XJTU run keeps its final-epoch checkpoint; every CWRU run keeps its
  validation-selected checkpoint.
- **Within-round comparisons only.** Round-6 arms are never paired with stored results of earlier rounds.
- **Use in the manuscript (fixed now).** In the rewritten manuscript the XJTU-SY condition-2 -> 3 table
  (the former table 3), the onset-rule table, the gated-fusion table, the CWRU table and the
  correlated-noise table report round-6 values. Earlier values move to the supplement as history and are
  never pooled with round-6 seeds. This choice is made before any result is seen and does not depend on
  whether round-6 values are more or less favourable.

## 1. Statistics (all blocks)
- Unit for seed statistics: seed; seed-paired difference d. Report mean, percentile bootstrap 95% interval
  (20,000 resamples, `default_rng(12345)`), exact Wilcoxon signed-rank p (two-sided unless stated),
  positive/negative seed counts.
- Unit for bearing statements: test bearing (accuracy on the bearing's windows, mean over seeds). As in
  rounds 3-4: no bearing-level claim for a contrast whose sign differs across test bearings.
- Metrics: macro-F1 (window level, as before); macro-recall (= class-balanced accuracy); bearing-equal
  balanced accuracy (mean over classes of the mean accuracy of that class's test bearings).
- **Primary contrasts** (Holm correction over this family, alpha = 0.05; everything else is exploratory
  and reported without multiplicity correction, labelled as such):
  P1-P3 (block A, one per learner): V - H on bearing-equal balanced accuracy, H rule, condition 2 -> 3.
  P4 (block B): Mamba-3 V - H on the common test set (section 2), last-15% rule.
  P5 (block D): Mamba-3 accuracy, trained at rho = 0.9 and tested at rho = 0.9 minus tested at rho = 0, -8 dB.
  P6 (block D): DIFF-only minus DE-only accuracy at rho = 0.9, -8 dB.
  P7 (block F): coverage, mean over units of (H+V - H) at k = 3 minus at k = 1.
- Numerical-noise floor: block G repeats configurations with identical seeds; the 95th percentile of
  |repeat - original| (macro-F1 or accuracy) is reported, and contrasts smaller than it are called
  "within numerical variation" whatever their p-value.

## 2. Block A — onset rules x learners, XJTU-SY condition 2 -> 3 (R1.2, R1.3, R1.7, R2.1)
Arms H, V, H+V; rules H (original), V, last15 (`XJTU_ONSET_RULE`); learners Mamba-3 and 1D-CNN seeds 0-7,
Mamba-2 seeds 0-4. 27 configurations. Predictions: (A1) under the H rule, V > H for all three learners on
macro-F1, macro-recall and bearing-equal balanced accuracy, and V > H+V for 1D-CNN and Mamba-2, i.e. the
ordering survives class balancing; (A2) the V rule keeps every H-rule direction for every learner; (A3)
the last-15% rule reverses Mamba-3 V - H (replication of round 3); no prediction for the other learners.
Common test set: windows from files at or after max(onset_H, onset_V, floor(0.85 n)) of each test bearing
(windows every rule labels faulty); every model is also evaluated there.

## 3. Block B — decomposition of the last-15% effect (R2.2)
Training-side variants of the last-15% rule (resampled with `resample_seed` 12345, fixed across seeds):
(b) "seg": per-bearing training counts equal to the H-rule counts (2_1: 624, 2_2: 1808, 2_4: 240,
2_5: 2576); (c) "size": H-rule total (5248) with last-15% proportions (2_1: 2474, 2_2: 836, 2_4: 234,
2_5: 1704). Arms H, V, H+V; Mamba-3 seeds 0-4, 1D-CNN seeds 0-7. 12 configurations.
For any contrast C (V - H, H+V - H, H+V - V), evaluated on the common test set (subscript c):
test composition = C_own - C_c (each model); training segmentation = C_c(seg) - C_c(H rule);
bearing weighting = C_c(size) - C_c(seg); sample size = C_c(last15) - C_c(size). The last three add up to
C_c(last15) - C_c(H rule). Prediction (B1): the Mamba-3 reversal is mainly test composition, i.e. on the
common test set Mamba-3 V - H has the H-rule sign under the last-15% rule (P4).

## 4. Block C — LOBO onset sensitivity, Mamba-3 (R1.3, R1.7)
OR folds (test 3_1 and 3_5), arms H, V, H+V, rules H and last15, seeds 0-4 (the V rule is not extended).
6 configurations. Prediction: under the H rule H+V OR recall is below both single sensors (replication);
no prediction for last15.

## 5. Block D — common-noise controls, CWRU Mamba-3, seeds 0-4, noise key v2 (R1.5, R2.4)
- D0 baselines at -8 dB: DE-only; DE+FE at rho 0 and 0.9 (realisation 0).
- D1 mismatch (evaluation only): round-4 checkpoints (-6 and -8 dB): rho 0.9-trained tested at rho 0;
  rho 0-trained tested at rho 0.9; and the D0 checkpoints likewise. Prediction (P5): accuracy of the
  rho 0.9-trained model drops by more than 2 pp when tested at rho 0, 5/5 seeds.
- D2 fixed linear combinations: single-channel input SUM = (x_DE + x_FE)/sqrt2 or DIFF = (x_DE - x_FE)/sqrt2
  of the z-scored noisy channels, at rho 0 and 0.9, -8 dB. Predictions: (P6) DIFF at rho 0.9 above DE-only
  at rho 0.9; DIFF(rho 0.9) above DIFF(rho 0); SUM(rho 0.9) not above SUM(rho 0).
- D3 delayed common component at rho 0.9, -8 dB: delay 4 and 64 samples. Prediction: the DE+FE - DE gain
  decreases with delay, and at 64 samples is closer to the rho = 0 gain than to the rho 0.9 gain.
- D4 noise realisations 1 and 2 for DE-only, DE+FE rho 0 and rho 0.9, -8 dB. Prediction: both gains keep
  their sign with intervals excluding zero.
16 training configurations.

## 6. Block E — gated fusion and fixed-gate ablation (R1.6, R2.3)
Window- and token-gated Mamba-3 and 1D-CNN, condition 2 -> 3, seeds 0-7; gated Mamba-3 on the LOBO OR folds,
seeds 0-4 (concatenation baselines come from blocks A and C). Ablation (evaluation only, same script for
both): gate replaced by its mean over the training windows. A gate is "functional" if the ablation changes
macro-F1 (cross-condition) or OR recall (LOBO) by at least 1 pp in mean with the interval excluding zero.
No directional prediction. The adaptivity criterion of round 3 is reported again unchanged.

## 7. Block F — controlled coverage (R2.5)
Rule H, Mamba-3, arms H and H+V, seeds 0-1. Test bearings Bearing1_1, Bearing2_2, Bearing3_1. Training: IR
bearings 2_1, 3_3, 3_4 with 320 windows each; OR bearings: k in {1, 2, 3} bearings drawn from the 7 OR
bearings other than the test bearing, 3 distinct subsets per (test bearing, k) drawn with
`default_rng(2026)`; 1200 OR windows in total split equally over the k bearings. Training set size (2160
windows), epochs (50) and optimisation steps are therefore identical across k. Unit: (test bearing,
subset), mean over seeds; 9 units per k. Prediction (P7): H+V - H OR recall rises with k (k = 3 minus k = 1
positive, bootstrap over units). The number of distinct operating conditions among the training OR
bearings is reported as a covariate.

## 8. Block G — numerical replicates (R1.7)
Repeat with identical seeds: Mamba-3 H+V, H rule, condition 2 -> 3, seeds 0-7; Mamba-2 H, H rule, seeds 0-4;
CWRU DE+FE rho 0.9, -8 dB, seeds 0-4.

## 9. Analyses without training
- Class-balanced and bearing-level reporting (R1.2, R2.1): from block A.
- Independent fault indicator for the degraded phase (R1.3): for every file and both channels, squared
  envelope spectrum of the 1-10 kHz band-passed signal (4th-order Butterworth, zero phase); indicator =
  sum of the spectrum peaks within +-2% of 1x, 2x, 3x the class's characteristic frequency (BPFO for OR,
  BPFI for IR bearings; LDK UER204: 8 balls, d = 7.92 mm, D = 34.55 mm, contact angle 0) divided by the
  median spectrum level in 10-500 Hz. Indicator onset = first file where the 5-file rolling mean exceeds
  baseline mean + 3 SD (baseline: first min(20, n/5) files). Reported per bearing and channel: indicator
  onset, its offset from the H-rule, V-rule and last-15% onsets, and the fraction of each rule's faulty
  segment that lies before the indicator onset. No retraining depends on this analysis.

## 10. Order and stopping
Order: A, B, E, D, F, C, G. Runs killed by GPU contention are restarted from scratch. A configuration is
used only if it completed and its evaluation completed. Nothing is dropped because of its result.
Estimated 28-30 GPU hours.

## Deviation 1 (2026-11-05, before the rerun): first execution lost
The first execution of this protocol started 2026-11-04 12:26 (+0800) in a temporary working directory
that the analysis platform deleted on 2026-11-05 around 08:57; the queue was killed and all outputs,
configurations and modified code were lost. Results seen before the loss: the four completed
configurations' completion records, and per-seed macro-F1 of block A, Mamba-3, H rule, H arm (8 seeds);
the envelope-indicator analysis of section 9 (its result was reported to Jeff). No other result was
inspected and no analysis of any contrast was run. The rerun uses this protocol unchanged: the code was
restored verbatim from the session record, the data-level unit tests and one-epoch smoke tests of every
new path were repeated and passed, and the configurations were regenerated by the same generator (the
block-F subsets, drawn with default_rng(2026), are identical). The working directory is now
~/论文8/round6 on the authors' disk; outputs are written there as each configuration completes.
Timing of the first execution: about 37 configurations in about 20.5 h, so the full queue is expected to
take about 45-55 h rather than the 28-30 h stated in section 10.

### Correction to deviation 1 (2026-11-06)
Two statements in deviation 1 are inaccurate. "Deleted on 2026-11-05 around 08:57" was the time the loss was
noticed, not the time of deletion; and "about 37 configurations in about 20.5 h" divided by that interval.
The runner's log shows that the first execution's last completed configuration was B_cnn_size_V, after
which every later configuration failed immediately because its directory no longer existed. In the rerun
the same configuration completed 13.3 h after the start, so the deletion of the first execution's
directory most likely happened around 2026-11-05 01:45 (+0800), about 13 h after the 12:26 start; the
exact time cannot be recovered. The run-time estimate in deviation 1 (45-55 h) was therefore too high; the
rerun took about 32 h. Nothing else in deviation 1 changes.

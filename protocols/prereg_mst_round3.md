# Pre-registration: MST revision, round 3 (written 2026-10-06, before any of these runs)

Context: simulated MST review of 2026-10-05, "must" and "high-priority" items. All runs are in
software environment 2 (Python 3.11.15, PyTorch 2.13.0+cu130, Triton 3.7.1, mamba_ssm 2.3.2.post1 @
a14b1df, Mamba-2 unfused-conv patch). Bridge check done before writing this file: Mamba-2 H,
condition 2->3, seed 0 reproduced the existing environment-2 result bit for bit (macro-F1 0.453033).
All hyperparameters are those of the existing environment-2 arms; only the items named below change.
Comparators are existing environment-2 arms with the same seeds. No test-set information is used for
selection (fixed final epoch 50 on XJTU-SY; validation-selected epoch on CWRU, as before).

## E1 Mamba-2 dual-sensor arm, condition 2->3
- New arm: Mamba-2 H+V, seeds 0-7 (config = r2_bm2_H with n_sensors=2).
- Contrasts (seed-paired, n=8): H+V - V, H+V - H, against r2_bm2_V and r2_bm2_H.
- Reporting: mean, bootstrap 95% CI (20,000 resamples), exact two-sided Wilcoxon, sign counts,
  per-bearing accuracy. Rule: "V-only above dual for Mamba-2 (same environment)" is stated only if
  the CI of H+V - V excludes 0 and at least 7/8 seeds are negative; otherwise the result is reported
  as not confirming the earlier cross-environment ordering.

## E2 Degraded-phase (onset) sensitivity
- Plan wording was "joint H/V onset and fixed last-15% segment". The read-only audit shows the joint
  onset equals the H onset for every condition-2/3 bearing except Bearing3_3 (338 vs 341 files, 48
  windows), so a joint-rule rerun would be nearly identical to the existing arms. It is therefore
  replaced by the **V-defined rule** (same kurtosis/RMS rule on the vertical channel), which is the
  direct test of the "H-defined selection" concern. Second rule: **last15** (last 15% of files of
  every bearing, independent of either channel). Window counts per bearing for both rules were
  verified against onset_audit.csv before the runs.
- Arms per rule: Mamba-3 condition 2->3 H, V, H+V (seeds 0-2); Mamba-3 LOBO OR folds (test Bearing3_1
  and Bearing3_5) H, V, H+V (seeds 0-2). Comparator: the H-rule environment-2 arms, seeds 0-2.
- Quantities: condition 2->3 macro-F1 contrasts V - H and H+V - V; LOBO OR-fold recall contrasts
  H+V - H and H+V - V.
- Rule: a direction is called "stable under rule r" if the seed-mean contrast under r has the same
  sign as under the H rule and at least 2 of 3 seeds share that sign. Otherwise "not stable". With
  3 seeds no p values are reported. The magnitude change is reported for every contrast.

## E3 Sample-adaptive gated fusion baseline
- Model: one Conv1d embedding branch per channel (same kernel/stride as the learner's shared
  embedding), a gate MLP (input: mean of GELU(h) and std over time of each branch, 4d -> 32 -> 2,
  softmax) giving per-sample weights; weighted sum fed to the unchanged backbone.
  Unit test: forward pass for 1D-CNN (109,412 parameters) and Mamba-3 (186,740) on GPU.
- Arms: 1D-CNN-gated and Mamba-3-gated, condition 2->3, seeds 0-4; Mamba-3-gated LOBO OR folds,
  seeds 0-2.
- Validity check (fixed in advance): the gate is called sample-adaptive only if the standard
  deviation of the channel-1 gate weight over the test windows exceeds 0.01 in every seed; otherwise
  it is reported as non-adaptive in this setting and not used as an adaptive-fusion control.
- Contrasts: gated - early fusion (concatenation), gated - V, gated - H (cross, macro-F1, n=5; existing
  env-2 xc_* arms seeds 0-4); LOBO OR-fold recall gated vs H, V, H+V (seeds 0-2).

## E4 CWRU three arms in one environment
- New arms: Mamba-3 DE-only and DE+FE at -4, -6, -8 dB, seeds 0-4, environment 2 (existing FE-only
  environment-2 arms are the third arm). Contrasts: DE+FE - DE, FE - DE, n=5, exact two-sided
  Wilcoxon (floor 0.0625) and the one-sided test of the pre-declared direction DE+FE > DE (floor 0.031).

## E5 Correlated sensor noise at fixed per-channel power
- Noise model: for each window, z0, z1 ~ N(0,1) i.i.d. (same draw as the existing AWGN),
  z1' = rho z0 + sqrt(1 - rho^2) z1; each channel scaled to the nominal SNR of that channel. rho = 0
  reproduces the existing AWGN exactly (unit-tested); single-channel inputs are unaffected
  (unit-tested). Empirical check: corr 0.002 / 0.501 / 0.900 and per-channel SNR -6.0 dB for
  rho = 0 / 0.5 / 0.9.
- Arms: Mamba-3 DE+FE at rho = 0.5 and 0.9, -6 and -8 dB, seeds 0-4. rho = 0 and DE-only come from E4.
- Hypothesis (spatial averaging of independent noise): the gain DE+FE - DE decreases as rho increases.
  Rule: "consistent with the hypothesis" if, at both SNRs, the seed-mean gain is ordered
  g(0) > g(0.5) > g(0.9) and g(0.9) < g(0) in at least 4 of 5 seeds. Any other pattern is reported as
  not consistent. This is a synthetic noise model; no claim about real sensor noise is made from it.

## General
- All results are reported whatever their direction. All contrasts are exploratory with respect to
  multiplicity (no correction). Seeds quantify training stochasticity on fixed splits only.
- Runs that crash or are killed for GPU contention are rerun from scratch; only complete runs are
  analysed. Any departure from this file is listed in the manuscript's deviation section.

## Amendment 1 (2026-10-06, after the E3 condition 2->3 window-gate runs, before any token-gate run)
- Outcome of the pre-specified check: the window-level gate is **not sample-adaptive**: SD of the
  channel-1 gate weight over test windows was 0.0075/0.0101/0.0315/0.0028/0.0075 (1D-CNN) and
  0.0016/0.0038/0.0040/0.0023/0.0067 (Mamba-3), i.e. not > 0.01 in every seed. Per the rule it is
  reported as non-adaptive and not used as an adaptive-fusion control. Its LOBO arm is still run
  as queued and reported descriptively.
- Post hoc variant (designed after seeing this outcome, hence post hoc): **token-level gate**,
  g_t = softmax(MLP([h1_t, h2_t])) over channels for every embedding token, weighted sum fed to the
  unchanged backbone (1D-CNN 105,316 and Mamba-3 182,644 parameters). Same arms as E3 (condition
  2->3 seeds 0-4 for both learners; Mamba-3 LOBO OR folds seeds 0-2), same contrasts.
  Adaptivity rule: SD over test windows of the window-mean channel-1 gate > 0.01 in every seed; the
  mean within-window (token-level) SD is reported as well.

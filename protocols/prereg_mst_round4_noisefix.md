# Protocol: correction of the CWRU synthetic-noise implementation (round 4; written 2026-10-07, before any of these runs)

## Why
The CWRU (and Paderborn) loaders seeded the injected noise with `hash((idx, noise_snr_db))`, where `idx` is
the index within each split after a seed-dependent shuffle. Consequences: (i) train, validation and test
windows with the same local index received the same standard-normal noise waveform; (ii) changing the
training seed re-paired test windows with different noise, so CWRU seed spreads mixed training
stochasticity with noise-realisation variation. This round reruns the existing Mamba-3 AWGN and
correlated-AWGN conditions with a corrected noise key. It corrects existing experiments; it adds no new
design factor. Scope fixed by the authors on 2026-10-07.

## Corrected noise key (`noise_key: v2`, `bearmamba3/data_cwru_leakfree.py`)
- Generator seeded by `(split code, file id, absolute window start sample, realisation)`, split codes
  train = 1, val = 2, test = 3, realisation = 0 for all splits.
- Two standard-normal rows are always drawn, row 0 for the drive-end (DE) channel and row 1 for the
  fan-end (FE) channel; correlation is applied as z_FE' = ρ z_DE + sqrt(1−ρ²) z_FE; rows are then selected
  by the configured physical channels. DE-only uses row 0, FE-only row 1, the dual system both; all arms and
  all ρ share the same underlying draws. Each channel is scaled to the nominal SNR from its own window power
  (unchanged).
- Training, validation and test noise are independent streams. The validation and test noise banks are
  fixed (do not depend on the seed or on sample order). Training noise is fixed per window across epochs
  (as in the original implementation) and, unlike the original implementation, does not change with the
  seed.
- Unit tests (`r4/unit_tests_noisefix.py`, all passing before launch): `noise_key: v1` bit-identical to the
  original; cross-split |corr| ≈ 0.02 (v1: 1.0); test noise seed-invariant; DE-only = dual DE row; FE-only
  = dual FE row at ρ = 0; DE row shared across ρ; `awgn` ≡ `awgn_corr0.0`; empirical ρ 0.900 / −0.002;
  per-channel SNR within 0.02 dB.

## Runs (Mamba-3, environment 2, seeds 0–4, all other settings identical to the round-3 CWRU configs)
| block | SNR | arms | configs |
|---|---|---|---|
| AWGN | −8, −6, −4 dB | DE-only, FE-only, DE+FE (ρ = 0) | 9 |
| correlated AWGN | −8, −6 dB | DE+FE, ρ = 0.5 and 0.9 | 4 |
13 configs, 65 runs. The DE-only arm at each SNR is the baseline for every ρ (not retrained per ρ).
Order: the −8 dB block first (5 configs, 25 runs) as a pipeline check (all runs exit 0, disjointness
check passes, 660 test windows, noise key v2 logged in every source snapshot); then the remaining 8 configs
**regardless of whether the earlier advantage is reproduced**. The validation-selected checkpoint of every
run is kept.

## Not run in this round
Pink noise; train/test ρ mismatch; delay of the common component; additional architectures; new XJTU-SY
experiments; any other CWRU or Paderborn noise condition.

## Reported quantities and statistics
Test accuracy at the validation-selected epoch. Per SNR: DE+FE(ρ=0) − DE, FE − DE. Per SNR and ρ ∈ {0.5, 0.9}:
DE+FE(ρ) − DE and the change of the gain, ΔG(ρ) = [DE+FE(ρ) − DE] − [DE+FE(0) − DE] = DE+FE(ρ) − DE+FE(0).
All differences seed-paired: mean, percentile bootstrap 95% interval (20,000 resamples,
`default_rng(12345)`), positive/negative seed counts, exact Wilcoxon signed-rank p (one-sided for the
gain over DE as in the manuscript, two-sided for ΔG; floors 0.031 and 0.0625 at n = 5). These are
exploratory contrasts without multiplicity correction. Intervals describe training repeatability
conditional on the fixed noise realisations, not uncertainty over future noise realisations or over
bearings.

## Use in the manuscript
The corrected results replace the current CWRU Mamba-3 AWGN and correlated-noise results. They are never
pooled with earlier seeds. CWRU/Paderborn noise results that are not rerun (environment-1 eight-seed pool,
pink noise, 1D-CNN CWRU arms, CWRU backbone comparisons, Paderborn noise grid, gain–coherence analysis) are
removed from the current evidence and kept in a historical archive labelled "superseded noise protocol;
not used to support the current manuscript's conclusions". Mechanism wording stays at "consistent with …
not directly tested", and any effect size remains specific to the synchronous synthetic noise model.

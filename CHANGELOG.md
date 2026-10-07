# Changelog

## mst-v3 (2026-10-07) — fifth manuscript draft

Fixed
* CWRU synthetic noise reused across training, validation and test splits (noise seeded by the index within
  a split) and re-paired with test windows by the training seed. New `noise_key: v2` in
  `bearmamba3/data_cwru_leakfree.py`; `noise_key: v1` (default) reproduces the old generator bit for bit.
  `experiments/exp01_cwru_baseline/train_leakfree.py` passes split, channels and noise key.

Added
* `results/mst_round4_noisefix_20261007/`: 13 corrected Mamba-3 CWRU configurations (65 runs), per-seed
  JSON, source snapshots, unit tests, analysis and table-generating scripts (reproduce stored outputs byte
  for byte), runner and queue.
* `protocols/prereg_mst_round4_noisefix.md` (committed before the runs).
* `configs/round3/`, `configs/round4/` with repository-relative paths and `reproduce/run_round.sh`,
  `reproduce/README.md` (the round-3 runner of mst-v2 referred to directories that do not exist in the
  repository and to placeholder paths).
* `results/mst_round3_20261007/SUPERSEDED_CWRU.md`.

Changed
* `TABLE_MAP.md`: fifth-draft numbering; Tables 9 and 10 from round 4; superseded noise-protocol blocks marked.
* `README.md`.

Superseded (kept, not used by the manuscript): every noise-injected CWRU and Paderborn result outside
`results/mst_round4_noisefix_20261007/`, including the round-3 CWRU arms, the environment-1 eight-seed pool,
pink noise, the gain–coherence analysis, CWRU learner comparisons and the Paderborn noise grid.

## mst-v2 (2026-10-07) — fourth manuscript draft

Added
* `results/mst_round3_20261007/`: the five pre-registered round-3 experiments (29 configurations, all
  completed), with configurations, per-seed/per-bearing results, run log, analysis and table generators.
  Re-running `an_r3.py`, `gen_r3_tex.py` and `gen_r3_supp.py` in place reproduces the committed outputs
  byte for byte.
* `protocols/prereg_mst_round3.md` (with Amendment 1).
* `baselines/gated_fusion.py` (window-level and token-level gated early fusion);
  `scripts/lobo/train_lobo_leakfree_v3.py` (option `only_folds`).

Changed (backward compatible; defaults reproduce mst-v1)
* `bearmamba3/data_xjtu.py`: environment variable `XJTU_ONSET_RULE` (`H` default, `V`, `last15`).
  The environment-2 `channel` patch still applies unchanged.
* `bearmamba3/data_cwru_leakfree.py`: noise type `awgn_corr<ρ>`; `awgn` and single-channel inputs unchanged
  (ρ = 0 reproduces `awgn` exactly).
* `scripts/xjtu/train_xjtu_leakfree_v3.py`: backbones `mamba3_gated`, `cnn1d_gated`, `mamba3_tgated`,
  `cnn1d_tgated`.
* `TABLE_MAP.md`: fourth-draft table numbering (Tables 1–11), round-3 mapping, Figure 2 and Table 9
  sources updated; the cross-environment Mamba-2 footnote of mst-v1 is marked superseded.
* `README.md`, `protocols/README.md`.

Unchanged: every file under `results/env1/`, `results/env2/`, `aggregate/` and `configs/`.

## mst-v1 (2026-10-05) — third manuscript draft
Initial release.

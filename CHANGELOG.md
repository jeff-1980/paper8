# Changelog

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

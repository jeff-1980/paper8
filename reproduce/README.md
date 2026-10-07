# Re-running the revision experiments

Run from the repository root, with the data laid out as in the top-level README (`data/cwru_12k_de`,
`data/XJTU-SY_Bearing_Datasets`) and environment 2 (`environments/`).

```bash
export MAMBA_SSM_BUILD=/path/to/mamba_ssm/build/lib.linux-x86_64-cpython-311   # Mamba-3 kernel build
bash reproduce/run_round.sh round4                       # all 13 corrected CWRU configurations
bash reproduce/run_round.sh round4 r4_DEFE_rho0.9_snr-8  # one configuration
bash reproduce/run_round.sh round3 e1_bm2_HV             # a round-3 XJTU-SY configuration
```

* Configurations: `configs/round3/`, `configs/round4/` (repository-relative paths; results go to
  `out_r3/<name>`, `out_r4/<name>`). The copies under `results/*/cfg_r*` are the ones used for the runs,
  with machine-specific paths replaced by placeholders, and are kept for provenance only.
* Round 3 needs the environment-2 patches in `environments/patches/` (see `TABLE_MAP.md`); the round-3 CWRU
  configurations (`e4_*`, `e5_*`) reproduce the superseded noise implementation and are kept only for audit.
* Compare a rerun with the stored per-seed JSON files in `results/mst_round3_20261007/` or
  `results/mst_round4_noisefix_20261007/`; recompute tables with `an_r3.py`/`gen_r3_*.py` or
  `an_r4.py`/`gen_r4_tex.py` in those folders.
* Checked: `run_round.sh round4 r4_DEFE_rho0.9_snr-8` with 1 epoch / 1 seed ran end to end from a clean
  clone (exit 0) before release.

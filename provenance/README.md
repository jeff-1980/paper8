# provenance/

Original batch drivers, run manifests, runner shell scripts and analysis scripts, kept for the
record. They reference the original directory layout (absolute paths replaced by the placeholders
`<REPO_ROOT>`, `<ENV2_WORKDIR>`, `<DATA_ROOT>`, `<MAMBA_SSM_BUILD>`, `<HOME>`) and are **not** needed to
reproduce any number: use the trainers with `configs/` (README) and `aggregate/aggregate_main.py`.

* `env1/exp_bm3_cwru_leakfree/` — `gen_configs*.py` (generated the leak-free configs from the
  pre-correction ones), `run_*.py` batch drivers and `run_manifest*.json` (which config ran into which
  results directory, with seeds), figure scripts `step5_*_leakfree.py` (supplementary figures).
* `env1/exp_e3_lobo_leakfree/` — LOBO launcher and config audit.
* `env1/run_a1_coherence.py` — DE–FE coherence analysis (output in `results/env1/cwru_coherence/`).
* `env2/runners/` — shell runners used in environment 2.
* `env2/extended_validation*/`, `env2/independent_validation/` — analysis scripts that produced the
  reports in `results/env2/*/reports/`.
* `env2/analysis/` — XJTU paired effects, CWRU AWGN-vs-pink gains, power simulation, in-band SNR
  (`inband.py`; output `results/env2/cwru_noise_analysis/`).
* `env2/cwru/` — config generator, multi-config driver and analysis for the CWRU fan-end control.

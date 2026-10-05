# paper8 — code and results for "When does a second vibration sensor help?"

This repository is the frozen code-and-results release (version **mst-v1**) accompanying the
manuscript

> Y. Wang and Y. Tang, *When does a second vibration sensor help? A controlled case study of
> sensor configuration for deep-learning bearing fault diagnosis*, submitted to
> *Measurement Science and Technology* (MST).

It contains the leakage-free training code, every configuration file, the split definitions, the
pre-specified protocols, the per-seed and per-bearing result files behind every number in the main
text, the degraded-phase (onset) audit, and one aggregation script that recomputes all main-text
tables and figure values from those files. `TABLE_MAP.md` maps every main-text table, figure and
quoted number to its environment, seeds, configuration files, result directories and the
aggregation command.

**Superseded earlier repository.** The earlier repository
[jeff-1980/BM3-MU](https://github.com/jeff-1980/BM3-MU) contains superseded protocols
(overlapping CWRU windows split at window level; XJTU-SY and Paderborn checkpoints selected on the
test fold). It is not the source of any number reported in the manuscript. In this repository the
corresponding superseded entry points are kept only under `historical/` (see
`historical/HISTORICAL.md`); no default entry point uses them.

**Licence:** to be decided by the authors.

## Quick check (no data, no GPU)

```bash
pip install numpy scipy            # the aggregator needs only numpy and scipy
python aggregate/aggregate_main.py # prints every main-text value
diff <(python aggregate/aggregate_main.py) aggregate/expected_output.txt
```

## Repository layout

| path | content |
|---|---|
| `bearmamba3/`, `baselines/` | model and data modules (repository versions used in environment 1) |
| `experiments/` | environment-1 trainers (leak-free CWRU, XJTU-SY cross-condition, LOBO, IR coverage sweep, Paderborn) |
| `scripts/` | environment-2 trainers (`xjtu/`, `lobo/`) and evaluation / late-fusion / analysis helpers (`extra/`) |
| `configs/env1/`, `configs/env2/` | every configuration used, grouped by design (`cwru`, `xjtu_cross`, `xjtu_lobo`, `xjtu_loco`, `xjtu_or_coverage`, `xjtu_ir_coverage_p14`, `pu`) |
| `results/env1/`, `results/env2/` | per-seed JSON (`seed_*.json`, `fold*_seed*.json`), `per_bearing.json`, `summary*.json`, source snapshots and run reports; no checkpoints |
| `aggregate/` | `aggregate_main.py`, its output `expected_output.txt` and `aggregate_values.json` |
| `audit/` | read-only degraded-phase onset audit (`onset_audit.py`, `onset_audit.csv`) |
| `splits/` | split manifests (CWRU per recording; XJTU-SY per design) and the script that writes them |
| `environments/` | software environments and the two environment-2 patches as unified diffs |
| `protocols/` | pre-specified protocols (preregistration documents) |
| `provenance/` | original run drivers, run manifests and analysis scripts, kept for the record (paths refer to the original layout; not needed to reproduce) |
| `historical/` | superseded entry points (see `historical/HISTORICAL.md`) |
| `archive/` | placeholder for `archive_supplement.pdf` (archived non-peer-reviewed results) |

## Software environments

Two software environments were used; every comparison reported in the manuscript is made within one
environment, except two comparisons that are labelled descriptive (see `environments/ENVIRONMENT.md`).

* **Environment 1** — Python 3.12.3 venv, `mamba_ssm` 2.3.2.post1 from state-spaces/mamba commit
  `b3cae1baee273b6617b9b1dffa52c75703b4ced0`, tilelang 0.1.8, PyTorch installed from the cu121 wheel
  index (the setup log shows torch 2.5.1+cu121 downloaded); torch/triton are no longer present in the
  venv, so the exact final versions are unrecoverable. bf16, RTX A5000 Laptop GPU.
  Set-up script: `environments/setup_env1.sh`.
* **Environment 2** — Python 3.11.15, PyTorch 2.13.0+cu130 (CUDA 13.0), Triton 3.7.1, `mamba_ssm`
  2.3.2.post1 built from commit `a14b1dff0454a3bc27d9eb31355dc01e4b2490ec` (clean checkout).
  Patches: (i) `data_xjtu.py` adds a `channel` argument so single-sensor runs can read column 1
  (vertical); (ii) `baselines/mamba2.py` sets `mamba_ssm.modules.mamba2.causal_conv1d_fn=None` and
  `use_mem_eff_path=False` (the fused causal-conv1d rejected unaligned strides; falls back to
  `nn.Conv1d`+SiLU, same math). `num_workers=0`. Same GPU, bf16.

The code in `bearmamba3/` and `baselines/` is the environment-1 repository version. For
environment-2 runs apply the two patches first:

```bash
git apply environments/patches/env2_data_xjtu_channel.diff    # needed by every scripts/ trainer
git apply environments/patches/env2_mamba2_unfused_conv.diff  # Mamba-2 arms in environment 2 only
```

The `channel` patch leaves two-channel and horizontal-only behaviour unchanged (default `channel=0`).

## Data

Download the three public datasets and place them under `data/` in the repository root (or set the
`DATA_ROOT` environment variable for the audit and split scripts; the configs use the relative paths
shown below and the trainers are run from the repository root).

```
data/
├── cwru_12k_de/                      # CWRU Bearing Data Center, 12 kHz drive-end files
│   ├── 97.mat 98.mat 99.mat 100.mat  # normal, 0-3 hp
│   ├── 105-108, 169-172, 209-212     # inner race 0.007/0.014/0.021 in
│   ├── 118-121, 185-188, 222-225     # ball
│   └── 130-133, 197-200, 234-237     # outer race @6:00
├── XJTU-SY_Bearing_Datasets/         # XJTU-SY run-to-failure data (IEEE DataPort)
│   ├── 35Hz12kN/Bearing1_1/1.csv ... # columns: horizontal, vertical (25.6 kHz, 32768 samples/file)
│   ├── 37.5Hz11kN/Bearing2_1/...
│   └── 40Hz10kN/Bearing3_1/...
└── paderborn/                        # Paderborn KAt-DataCenter (supplementary experiments only)
    └── {COND}_{BEARING}_{N}.mat      # flat folder, e.g. N15_M07_F10_KA04_1.mat
```

* CWRU: <https://engineering.case.edu/bearingdatacenter> — 40 recordings (ids in
  `bearmamba3/data_cwru.py`, `MANIFEST`); each file contains the `X<id>_DE_time` and `X<id>_FE_time`
  channels.
* XJTU-SY: Rolling Element Bearing Accelerated Life Test Datasets, IEEE DataPort.
* Paderborn: <https://mb.uni-paderborn.de/kat/forschung/kat-datacenter> (bearings K001, K002,
  KA04, KA15, KI01, KI03, KI05).

## How to run

All commands are run from the repository root. Each config lists its seeds; `--seeds` overrides
them. Results are written to the config's `results_dir` (`runs/env1/...` for environment 1;
`out/`, `out_q/`, `out_loco/` for environment 2, the layout the environment-2 helper scripts
expect). Checkpoints are written there too and are excluded by `.gitignore`.

### Environment 1

```bash
# CWRU (leak-free time-axis split, validation-selected epoch); e.g. Mamba-3 DE-only and DE+FE at -4 dB
python experiments/exp01_cwru_baseline/train_leakfree.py --config configs/env1/cwru/exp02_snr-4_nokin.yaml
python experiments/exp01_cwru_baseline/train_leakfree.py --config configs/env1/cwru/exp_b2_dual_nokin_snrm4.yaml
#   seeds 5-7: configs/env1/cwru/exp_e6_{single,dual}_nokin_snrm{4,6,8}_newseed_leakfree.yaml
#   pink noise: configs/env1/cwru/exp_e5_{single,dual}_pink_snr-{4,6,8}_leakfree.yaml

# XJTU-SY condition 2 -> 3 (fixed final epoch 50); e.g. Mamba-3 H and H+V, seeds 0-4 and 5-7
python experiments/exp_bm3_cwru_leakfree/train_xjtu_leakfree.py --config configs/env1/xjtu_cross/exp_xjtu_cross_nokin_leakfree.yaml
python experiments/exp_bm3_cwru_leakfree/train_xjtu_leakfree.py --config configs/env1/xjtu_cross/exp_xjtu_cross_dual_nokin_newseed_leakfree.yaml

# XJTU-SY condition-3 LOBO (4 folds); results directory given on the command line
python experiments/exp_e3_lobo_leakfree/train_lobo_leakfree.py --config configs/env1/xjtu_lobo/config_lobo_nokin_leakfree.yaml --results-dir runs/env1/xjtu_lobo/single_nokin
python experiments/exp_e3_lobo_leakfree/train_lobo_leakfree.py --config configs/env1/xjtu_lobo/config_lobo_dual_nokin_leakfree_n8ext.yaml --results-dir runs/env1/xjtu_lobo/dual_nokin_n8ext

# pre-specified IR (other-class) coverage sweep at Bearing3_1
python experiments/exp_bm3_cwru_leakfree/train_p14_coverage_leakfree.py --config configs/env1/xjtu_ir_coverage_p14/config_p14_dual.yaml

# Paderborn (supplement)
python experiments/exp_bm3_cwru_leakfree/train_pu_leakfree.py --config configs/env1/pu/exp06_pu_nokin_leakfree.yaml
```

### Environment 2 (apply the patches first; `mamba_ssm` must be importable, e.g. via `PYTHONPATH`)

```bash
# XJTU-SY condition 2 -> 3, leave-one-condition-out and OR coverage sweep: one trainer (v3) + per-bearing evaluation
python scripts/xjtu/train_xjtu_leakfree_v3.py --config configs/env2/xjtu_cross/xc_bm3_V.yaml
python scripts/extra/eval_ckpts_cross_v3.py configs/env2/xjtu_cross/xc_bm3_V.yaml          # writes per_bearing.json
python scripts/xjtu/train_xjtu_leakfree_v3.py --config configs/env2/xjtu_loco/r1_bm3_L2_HV.yaml
python scripts/xjtu/train_xjtu_leakfree_v3.py --config configs/env2/xjtu_or_coverage/q2_T3_1_SAME_k1_HV.yaml

# condition-3 LOBO single-sensor controls (Mamba-3, 1D-CNN)
python scripts/lobo/train_lobo_leakfree_v2.py --config configs/env2/xjtu_lobo/lobo_bm3_V.yaml --seeds 3 4 5 6 7
python scripts/extra/eval_ckpts_lobo.py configs/env2/xjtu_lobo/lobo_bm3_V.yaml

# validation-selected late fusion (hold-out sub-models, then weight selection and report)
python scripts/extra/fusion_holdout.py configs/env2/xjtu_cross/fus_xc_bm3.yaml cross
python scripts/extra/fusion_holdout.py configs/env2/xjtu_lobo/fus_lobo_bm3.yaml lobo
python scripts/extra/eval_ckpts_cross.py configs/env2/xjtu_cross/xc_bm3_H.yaml configs/env2/xjtu_cross/xc_bm3_V.yaml  # full-data test logits
python scripts/extra/build_fusion_report.py cross_bm3 lobo_bm3 cross_cnn lobo_cnn   # writes out/fusion_*.json, out/summary_fusion.md

# CWRU fan-end-only control and 1D-CNN arms (same leak-free trainer as environment 1)
python experiments/exp01_cwru_baseline/train_leakfree.py --config configs/env2/cwru/bm3_FE_snr-4.yaml
```

`scripts/xjtu/train_xjtu_leakfree_v3.py` is a strict superset of the `train_xjtu_leakfree.py`
(environment-2 `channel` support) and `_v2.py` (adds the channel-attention baseline) versions that
produced the earlier environment-2 arms; the additions are new branches only (explicit
`train_bearings`/`test_bearings` lists in v3), so configs without those keys run identically. All three
versions are included. The exact version used by each run is in its `source_snapshot/`.

## Reproducing the numbers

`python aggregate/aggregate_main.py` recomputes Table 1 (panels A, B), Tables 2–5, the values plotted
in Figures 2 and 3, and the numbers quoted in sections 4.1–4.4 from `results/`. Bootstrap: 20,000
percentile resamples of the seed-paired differences, `numpy.random.default_rng(12345)` per contrast;
exact Wilcoxon signed-rank test by full enumeration (ties handled with mid-ranks). Interval end points
can differ from the printed manuscript values by Monte-Carlo error of the bootstrap (about 0.1 pp,
at most 0.2 pp for the widest intervals).

`audit/onset_audit.py` (raw XJTU-SY CSVs) and `splits/make_split_manifests.py` (CWRU .mat files)
need the data; their outputs are included.

## Notes on the result files

* Hard-coded absolute paths inside copied result files (e.g. `checkpoint_path`, source-snapshot
  manifests) were replaced by placeholders: `<REPO_ROOT>` (original repository root),
  `<ENV2_WORKDIR>` (working directory of the environment-2 runs), `<DATA_ROOT>`,
  `<MAMBA_SSM_BUILD>`. Numeric content is unchanged.
* Checkpoints (`*.pt`) and test-logit/prediction arrays (`*.npz`) are not included (tens of GB).

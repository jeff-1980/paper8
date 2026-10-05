# Software environments

The facts below are reproduced verbatim from the verified environment record (2026-10-05) that
underlies the manuscript's supplementary environment section.

```
# Facts for supplement (verified 2026-10-05 from files; use verbatim numbers)
Environment 1: Python 3.12.3 venv (~/论文8/venv), mamba_ssm 2.3.2.post1 from state-spaces/mamba commit b3cae1baee273b6617b9b1dffa52c75703b4ced0, tilelang 0.1.8, PyTorch installed from the cu121 wheel index (setup log shows torch 2.5.1+cu121 downloaded); torch/triton no longer present in the venv, exact final versions unrecoverable. bf16, RTX A5000 Laptop GPU.
Environment 2: Python 3.11.15, PyTorch 2.13.0+cu130 (CUDA 13.0), Triton 3.7.1, mamba_ssm 2.3.2.post1 built from commit a14b1dff0454a3bc27d9eb31355dc01e4b2490ec (clean checkout). Patches: (i) data_xjtu.py adds `channel` argument so single-sensor runs can read column 1 (vertical); (ii) baselines/mamba2.py sets mamba_ssm.modules.mamba2.causal_conv1d_fn=None and use_mem_eff_path=False (fused causal-conv1d rejected unaligned strides; falls back to nn.Conv1d+SiLU, same math). num_workers=0. Same GPU, bf16.
Bridging checks (condition 2->3, macro-F1 %, final epoch):
- 1D-CNN H, seeds 0-4: env1 = env2 identical per seed (40.22, 40.22, 40.16, 40.04, 40.10).
- 1D-CNN H+V, seeds 0-4: env1 43.23/49.61/42.10/40.28/50.49 vs env2 43.17/49.61/42.16/40.35/50.39 (|diff| <= 0.1 pp).
- Mamba-2 H, seeds 0-7: env2 minus env1 per seed = +0.40, +0.06, +1.01, +0.98, +0.83, -0.12, +6.95, +0.12 (mean |diff| 1.31, max 6.95). Seed-0 check with patched kernel path: 44.906 vs 44.91.
- Mamba-3 H seed 0: env2 54.38 vs env1 63.45 (does not reproduce).
- CWRU Mamba-3 DE-only -8 dB re-run in env2 differed by -0.06 pp (single bridge).
Parameter counts (two-class): 1D-CNN 100,546 single / 101,058 dual (+512, k=8, 0.51%); Mamba-2 173,850 / 174,298 (+448, k=7, 0.26%); Mamba-3 177,938 single (four-class 178,068 minus 130) +448 (k=7, 0.25%).
Head (code bearmamba3/model.py line 73, model2.py line 40): h = mean_t( LayerNorm(H) ), i.e. layer norm per token, then mean over tokens.
Protocol-correction table (old S7, XJTU H cross-condition): original-protocol (best-epoch on test fold) Mamba-3 H = 80.89 (seeds 0-4) / 80.37 (seeds 0-7); 1D-CNN H = 41.31 (seeds 0-4) / 41.99 (seeds 0-7). Leakage-free final-epoch: Mamba-3 63.09 (8 seeds), 1D-CNN 40.23 (8) / 40.15 (seeds 0-4). So "+38.4" = 80.37-41.99 (8 seeds) and "+22.9" = 63.09-40.23 (8 seeds). Selection inflated the 1D-CNN H by 1.16 pp (seeds 0-4) / 1.76 pp (8 seeds); Mamba-3 by 17.80 (5) / 17.28 (8).
Onset audit: onset_audit.csv (this folder) produced by onset_audit.py (read-only, raw CSVs). Windows = files x 16 (32768 samples per file / 2048). No fallback used for any bearing. Joint onset (min of H and V) equals the H onset except Bearing1_3 (100 vs 105) and Bearing3_3 (338 vs 341).
```

## Patches used in environment 2 (unified diffs against this repository)

| file | patch | used by |
|---|---|---|
| `bearmamba3/data_xjtu.py` | `patches/env2_data_xjtu_channel.diff` — adds `channel` (0 = horizontal, 1 = vertical) for single-sensor runs; default 0 keeps the environment-1 behaviour | every environment-2 XJTU-SY run (`scripts/xjtu/*`, `scripts/lobo/*`, `scripts/extra/*` pass `channel=`) |
| `baselines/mamba2.py` | `patches/env2_mamba2_unfused_conv.diff` — `mamba_ssm.modules.mamba2.causal_conv1d_fn = None` and `use_mem_eff_path=False` (unfused `nn.Conv1d` + SiLU path, same math) | Mamba-2 arms trained in environment 2 (`r2_bm2_*`, `bm2_V_only`, `bm2_H_repro`) |

Apply from the repository root with `git apply environments/patches/<file>.diff`
(`git apply -R` reverts).

## Environment 1 set-up

`setup_env1.sh` is the original set-up script (paths made relative). It installs PyTorch from the
cu121 wheel index and builds `mamba_ssm` from the state-spaces/mamba repository; to reproduce
environment 1 exactly, check out commit `b3cae1baee273b6617b9b1dffa52c75703b4ced0` before building.

## Environment 2 set-up

Python 3.11, PyTorch 2.13.0 (CUDA 13.0 wheels), Triton 3.7.1; build `mamba_ssm` 2.3.2.post1 from a
clean checkout of commit `a14b1dff0454a3bc27d9eb31355dc01e4b2490ec` and make it importable
(e.g. `export PYTHONPATH=<mamba build dir>:$(pwd)`); set `num_workers: 0` (already set in all
`configs/env2/` files). Additional Python packages: numpy, scipy, pandas, pyyaml, scikit-learn.

# Environment 2 as used for round 6 (MST-140500 rewrite, runs 2026-11-05/06)

| item | value |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.13.0+cu130 (CUDA 13.0, cuDNN 92000) |
| Triton | 3.7.1 |
| mamba_ssm | 2.3.2.post1, source build at commit `a14b1df` (`<MAMBA_SSM_BUILD>` on `PYTHONPATH`) |
| NumPy / SciPy | 2.4.6 / 1.17.1 |
| GPU / driver | NVIDIA RTX A5000 Laptop GPU, driver 596.47 (the October driver version was not recorded) |
| Precision | bf16 autocast; `num_workers: 0` |
| Determinism | `BM_DETERMINISTIC=1` (switch added in round 6, off by default: cuDNN deterministic, no benchmark, `torch.use_deterministic_algorithms(True, warn_only=True)`), `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| Patches | `environments/patches/env2_data_xjtu_channel.diff`, `env2_mamba2_unfused_conv.diff` (as for mst-v2/v3) |
| Full package list | `environments/env2_round6_pip_freeze.txt` |

Reproducibility found in round 6: with these settings three repeated configurations (block G) reproduced
every seed exactly. Without the deterministic switch, and against values stored in October, results were not
bit-identical (Mamba-2 H seed 0: macro-F1 0.452467 vs 0.453033) although code and libraries were unchanged;
round-6 arms are therefore compared only with one another.

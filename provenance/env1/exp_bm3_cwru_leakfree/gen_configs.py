#!/usr/bin/env python3
"""
experiments/exp_bm3_cwru_leakfree/gen_configs.py

STEP 2/5 support script for 故障诊断Wiki/_tasks/bm3-cwru-leakage-fix.md,
scope 分档C (per user decision 2026-09-11): exp01-05 主表 + B2 (C3 statistical
evidence).

Copies every affected config VERBATIM (P3: no hyperparameter changes) except
`results_dir`, which is redirected into a new timestamped root under
results/ so the guardrail's "results/** 只增不覆盖" rule is respected — this
never writes into an existing results/<dir>.

B2 configs are not static YAML files upstream (they are built in-memory by
experiments/exp_b2_dual_sensor/run_b2_snr_curve.py::make_config); this
script re-derives them with the identical field values (verified by diffing
against that function) and writes them out as YAML so train_leakfree.py can
consume them the same way as the static exp01-05 configs.

Outputs:
  experiments/exp_bm3_cwru_leakfree/configs/*.yaml   (generated configs)
  experiments/exp_bm3_cwru_leakfree/run_manifest.json (config_path -> results_dir, driver)
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent  # <REPO_ROOT>
CFG_OUT = Path(__file__).parent / "configs"
CFG_OUT.mkdir(exist_ok=True)

TS = "20260911-leakfree"
NEW_RESULTS_ROOT = f"<REPO_ROOT>/results/bm3_cwru_leakfree_{TS}"

TRAIN_LEAKFREE = str(ROOT / "experiments/exp01_cwru_baseline/train_leakfree.py")

# ── exp01-05: static configs, redirect results_dir only ──────────────────────
STATIC_CONFIGS = [
    "experiments/exp01_cwru_baseline/config.yaml",
    "experiments/exp01_cwru_baseline/config_kin.yaml",
    *sorted((ROOT / "experiments/exp02_snr_ablation").glob("config_*.yaml")),
    *sorted((ROOT / "experiments/exp03_10class").glob("config_*.yaml")),
    *sorted((ROOT / "experiments/exp04_mamba2").glob("config_*.yaml")),
    *sorted((ROOT / "experiments/exp05_lambda_scan").glob("config_*.yaml")),
]


def basename_of(results_dir: str) -> str:
    return Path(results_dir).expanduser().name


def redirect(cfg: dict) -> dict:
    old_dir = cfg["results_dir"]
    base = basename_of(old_dir)
    cfg = dict(cfg)  # shallow copy, values untouched (P3)
    cfg["results_dir"] = f"{NEW_RESULTS_ROOT}/{base}"
    return cfg, base


def gen_static():
    manifest = []
    for entry in STATIC_CONFIGS:
        path = Path(entry) if not isinstance(entry, Path) else entry
        if not path.is_absolute():
            path = ROOT / path
        with open(path) as f:
            cfg = yaml.safe_load(f)
        new_cfg, base = redirect(cfg)
        out_path = CFG_OUT / f"{base}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(new_cfg, f, sort_keys=False)
        manifest.append({
            "orig_config": str(path.relative_to(ROOT)),
            "leakfree_config": str(out_path.relative_to(ROOT)),
            "results_dir": new_cfg["results_dir"],
            "driver": "train_leakfree.py",
        })
    return manifest


# ── B2: re-derive from run_b2_snr_curve.py::make_config (dual-sensor) ────────
SNR_GRID = [-8, -6, -4, -2, 0]


def snr_tag(snr: float) -> str:
    return f"snrm{abs(int(snr))}" if snr < 0 else f"snr{int(snr)}"


def make_b2_config(kind: str, snr: float) -> dict:
    lk = 0.0 if kind == "nokin" else 0.01
    tag = snr_tag(snr)
    name = f"exp_b2_dual_{kind}_{tag}" if snr != 0 else f"exp_b2_dual_{kind}"
    cfg = {
        "name": name,
        "data_dir": "<DATA_ROOT>/cwru_12k_de",
        "channels": ["DE", "FE"],
        "win_len": 2048,
        "stride": 1024,
        "val_ratio": 0.2,
        "batch_size": 64,
        "num_workers": 4,
        "d_model": 64,
        "d_state": 128,
        "n_layers": 4,
        "n_classes": 4,
        "conv_stride": 2,
        "epochs": 50,
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "grad_clip": 1.0,
        "scheduler": "cosine",
        "noise_snr_db": float(snr),
        "lambda_kin": lk,
        "kin_variant": "cover",
        **({"bearing_kwargs": {"n_balls": 9, "d": 7.938, "D": 39.040}} if lk > 0 else {}),
        "seeds": [0, 1, 2, 3, 4],
        "results_dir": f"{NEW_RESULTS_ROOT}/exp_b2_dual_{kind}_{tag}" if snr != 0
                        else f"{NEW_RESULTS_ROOT}/exp_b2_dual_{kind}",
    }
    return cfg


def gen_b2():
    manifest = []
    for kind in ("nokin", "kin"):
        for snr in SNR_GRID:
            cfg = make_b2_config(kind, snr)
            base = Path(cfg["results_dir"]).name
            out_path = CFG_OUT / f"{base}.yaml"
            with open(out_path, "w") as f:
                yaml.dump(cfg, f, sort_keys=False)
            manifest.append({
                "orig_config": "experiments/exp_b2_dual_sensor/run_b2_snr_curve.py::make_config"
                               f"({kind!r}, {snr})",
                "leakfree_config": str(out_path.relative_to(ROOT)),
                "results_dir": cfg["results_dir"],
                "driver": "train_leakfree.py",
            })
    return manifest


def main():
    manifest = gen_static() + gen_b2()
    manifest_path = Path(__file__).parent / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated {len(manifest)} leak-free configs → {CFG_OUT}")
    print(f"Manifest → {manifest_path}")
    for row in manifest:
        print(f"  {row['leakfree_config']:60s} -> {row['results_dir']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
experiments/exp_bm3_cwru_leakfree/gen_configs_batch6_dualbm2.py

BATCH6 item 4 (P13): CWRU dual-BM2 leak-free config generator. Copies each
of the existing exp_e1_dual_baselines/cwru_dual_bm2 configs VERBATIM (P3
discipline) except results_dir, redirected into the main leak-free results
root. Driver: experiments/exp01_cwru_baseline/train_leakfree.py (already
verified; reuses build_model from train.py, which already supports
backbone=mamba2 with n_sensors=len(channels)=2 -- no code changes needed).
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent
CFG_OUT = Path(__file__).parent / "configs_batch6_dualbm2"
CFG_OUT.mkdir(exist_ok=True)

NEW_RESULTS_ROOT = "<REPO_ROOT>/results/bm3_cwru_leakfree_20260911-leakfree"

SOURCES = [
    ("-4dB", "experiments/exp_e1_dual_baselines/cwru_dual_bm2/config_snr-4.yaml"),
    ("-6dB", "experiments/exp_e1_dual_baselines/cwru_dual_bm2/config_snr-6.yaml"),
    ("-8dB", "experiments/exp_e1_dual_baselines/cwru_dual_bm2/config_snr-8.yaml"),
]


def main():
    manifest = []
    for label, rel in SOURCES:
        src = ROOT / rel
        with open(src) as f:
            cfg = yaml.safe_load(f)
        base = Path(cfg["results_dir"]).name
        cfg = dict(cfg)
        cfg["results_dir"] = f"{NEW_RESULTS_ROOT}/{base}_leakfree"
        out_path = CFG_OUT / f"{base}_leakfree.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)
        manifest.append({
            "label": label,
            "orig_config": str(src.relative_to(ROOT)),
            "leakfree_config": str(out_path.relative_to(ROOT)),
            "results_dir": cfg["results_dir"],
            "driver": "experiments/exp01_cwru_baseline/train_leakfree.py",
            "seeds": cfg.get("seeds", []),
        })

    manifest_path = Path(__file__).parent / "run_manifest_batch6_dualbm2.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_runs = sum(len(row["seeds"]) for row in manifest)
    print(f"Generated {len(manifest)} leak-free dual-BM2 configs ({total_runs} runs) -> {CFG_OUT}")
    for row in manifest:
        print(f"  {row['label']:6s} seeds={row['seeds']}  -> {row['results_dir']}")


if __name__ == "__main__":
    main()

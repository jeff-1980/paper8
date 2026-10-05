#!/usr/bin/env python3
"""
experiments/exp_bm3_cwru_leakfree/gen_configs_batch5_pu.py

P12 PU leak-free config generator. Copies each of the 7 exp06_pu configs
VERBATIM (P3 discipline: no hyperparameter changes) except results_dir
(redirected into a new leak-free PU results root) plus two new explicit
fields: val_frac=0.2, split_seed=42 (P12-approved defaults, written
explicitly rather than left implicit).
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent
CFG_OUT = Path(__file__).parent / "configs_batch5_pu"
CFG_OUT.mkdir(exist_ok=True)

NEW_RESULTS_ROOT = "<REPO_ROOT>/results/pu_leakfree_20260918"

SOURCES = [
    ("nokin",       "experiments/exp06_pu/config_nokin.yaml"),
    ("kin",         "experiments/exp06_pu/config_kin.yaml"),
    ("bm2_nokin",   "experiments/exp06_pu/config_bm2_nokin.yaml"),
    ("nokin_snr0",  "experiments/exp06_pu/config_nokin_snr0.yaml"),
    ("kin_snr0",    "experiments/exp06_pu/config_kin_snr0.yaml"),
    ("nokin_snrm4", "experiments/exp06_pu/config_nokin_snrm4.yaml"),
    ("kin_snrm4",   "experiments/exp06_pu/config_kin_snrm4.yaml"),
]


def main():
    manifest = []
    for family, rel in SOURCES:
        src = ROOT / rel
        with open(src) as f:
            cfg = yaml.safe_load(f)
        base = Path(cfg["results_dir"]).name
        cfg = dict(cfg)
        cfg["results_dir"] = f"{NEW_RESULTS_ROOT}/{base}_leakfree"
        cfg["val_frac"] = 0.2
        cfg["split_seed"] = 42
        out_path = CFG_OUT / f"{base}_leakfree.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)
        manifest.append({
            "family": family,
            "orig_config": str(src.relative_to(ROOT)),
            "leakfree_config": str(out_path.relative_to(ROOT)),
            "results_dir": cfg["results_dir"],
            "driver": "experiments/exp_bm3_cwru_leakfree/train_pu_leakfree.py",
            "seeds": cfg.get("seeds", []),
        })

    manifest_path = Path(__file__).parent / "run_manifest_batch5_pu.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_runs = sum(len(row["seeds"]) for row in manifest)
    print(f"Generated {len(manifest)} leak-free PU configs ({total_runs} runs) -> {CFG_OUT}")
    for row in manifest:
        print(f"  {row['family']:14s} seeds={row['seeds']}  -> {row['results_dir']}")


if __name__ == "__main__":
    main()

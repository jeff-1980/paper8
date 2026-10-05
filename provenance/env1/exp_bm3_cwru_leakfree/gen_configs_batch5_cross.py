#!/usr/bin/env python3
"""
experiments/exp_bm3_cwru_leakfree/gen_configs_batch5_cross.py

BATCH5 Cross-condition config generator. Copies every source config
VERBATIM (P3-style discipline: no hyperparameter changes) except
`results_dir`, redirected into a new leak-free XJTU results root. Driver
is train_xjtu_leakfree.py (P10(a) fixed-final-epoch fix), not the original
exp_xjtu/train.py.
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent
CFG_OUT = Path(__file__).parent / "configs_batch5_cross"
CFG_OUT.mkdir(exist_ok=True)

NEW_RESULTS_ROOT = "<REPO_ROOT>/results/xjtu_leakfree_20260917"

SOURCES = [
    ("BM3_single_nokin", "experiments/exp_xjtu/config_cross_nokin.yaml"),
    ("BM3_single_kin",   "experiments/exp_xjtu/config_cross_kin.yaml"),
    ("BM3_dual_nokin",   "experiments/exp_xjtu/config_cross_dual_nokin.yaml"),
    ("BM3_dual_kin",     "experiments/exp_xjtu/config_cross_dual_kin.yaml"),
    ("BM3_single_nokin_n8", "experiments/exp_xjtu/config_cross_nokin_newseed.yaml"),
    ("BM3_dual_nokin_n8",   "experiments/exp_xjtu/config_cross_dual_nokin_newseed.yaml"),
    ("BM2_single", "experiments/exp_e1_dual_baselines/xjtu_cross_single_bm2/config_cross_single_bm2.yaml"),
    ("BM2_dual",   "experiments/exp_e1_dual_baselines/xjtu_cross_dual_bm2/config_cross_dual_bm2.yaml"),
    ("BM2_single_n8", "experiments/exp_e1_dual_baselines/xjtu_cross_single_bm2/config_cross_single_bm2_newseed.yaml"),
    ("BM2_dual_n8",   "experiments/exp_e1_dual_baselines/xjtu_cross_dual_bm2/config_cross_dual_bm2_newseed.yaml"),
    ("1DCNN_single", "experiments/exp_mext_e13_1dcnn_xjtu_cross/config.yaml"),
    ("1DCNN_dual",   "experiments/exp_e1_dual_baselines/xjtu_cross_dual_cnn/config_cross_dual_cnn.yaml"),
    ("1DCNN_single_n8", "experiments/exp_mext_e13_1dcnn_xjtu_cross/config_newseed.yaml"),
    ("1DCNN_dual_n8",   "experiments/exp_e1_dual_baselines/xjtu_cross_dual_cnn/config_cross_dual_cnn_newseed.yaml"),
    ("BM3BN", "experiments/exp_mext_e23_bm3bn_xjtu_cross/config.yaml"),
    ("CNN_LN", "experiments/exp_p2_cnn1d_ln/xjtu_cross/config.yaml"),
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
        out_path = CFG_OUT / f"{base}_leakfree.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)
        manifest.append({
            "family": family,
            "orig_config": str(src.relative_to(ROOT)),
            "leakfree_config": str(out_path.relative_to(ROOT)),
            "results_dir": cfg["results_dir"],
            "driver": "train_xjtu_leakfree.py",
            "seeds": cfg.get("seeds", []),
        })

    manifest_path = Path(__file__).parent / "run_manifest_batch5_cross.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_runs = sum(len(row["seeds"]) for row in manifest)
    print(f"Generated {len(manifest)} leak-free Cross-condition configs ({total_runs} runs) -> {CFG_OUT}")
    for row in manifest:
        print(f"  {row['family']:20s} seeds={row['seeds']}  -> {row['results_dir']}")


if __name__ == "__main__":
    main()

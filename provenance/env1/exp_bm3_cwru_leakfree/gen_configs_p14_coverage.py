#!/usr/bin/env python3
"""
experiments/exp_bm3_cwru_leakfree/gen_configs_p14_coverage.py

P14 (prereg_bm3_leakfree.md): generates the 2 configs (single/dual, BM3
CE-only) for the training-bearing-coverage sweep. Each config is copied
VERBATIM from the corresponding existing LOBO leak-free config (P3
discipline — no hyperparameter changes), changing only name/results_dir/
seeds. seeds=[0,1,2,3,4] (5 seeds x 7 combos x 2 arms = 70 runs, per
prereg's corrected run count).
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent
CFG_OUT = Path(__file__).parent / "configs_p14_coverage"
CFG_OUT.mkdir(exist_ok=True)

SOURCES = [
    ("BM3_single_nokin", "experiments/exp_e3_lobo_leakfree/config_lobo_nokin_leakfree.yaml"),
    ("BM3_dual_nokin",   "experiments/exp_e3_lobo_leakfree/config_lobo_dual_nokin_leakfree.yaml"),
]

RESULTS_ROOT = ROOT / "results" / "e3_p14_coverage_leakfree_20260928"


def main():
    manifest = []
    for family, rel in SOURCES:
        src = ROOT / rel
        with open(src) as f:
            cfg = yaml.safe_load(f)
        assert cfg.get("lambda_kin", 0.0) == 0.0, f"{rel}: expected CE-only (lambda_kin=0)"

        cfg = dict(cfg)
        cfg["name"] = f"p14_{family}"
        cfg["seeds"] = [0, 1, 2, 3, 4]
        arm = "single" if "single" in family else "dual"
        results_dir = RESULTS_ROOT / arm
        cfg["results_dir"] = str(results_dir)

        out_path = CFG_OUT / f"config_p14_{arm}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)
        manifest.append({
            "family": family,
            "arm": arm,
            "orig_config": str(src.relative_to(ROOT)),
            "p14_config": str(out_path.relative_to(ROOT)),
            "results_dir": str(results_dir),
            "driver": "experiments/exp_bm3_cwru_leakfree/train_p14_coverage_leakfree.py",
            "seeds": cfg["seeds"],
        })

    manifest_path = Path(__file__).parent / "run_manifest_p14_coverage.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_runs = sum(len(row["seeds"]) * 7 for row in manifest)  # 7 combos/seed
    print(f"Generated {len(manifest)} P14 coverage-sweep configs ({total_runs} combo-runs, "
          f"5 seeds x 7 combos each) -> {CFG_OUT}")
    for row in manifest:
        print(f"  {row['family']:16s} -> {row['results_dir']}")


if __name__ == "__main__":
    main()

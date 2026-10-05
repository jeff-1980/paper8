#!/usr/bin/env python3
"""
experiments/exp_bm3_cwru_leakfree/gen_configs_batch5_lobo_n8.py

P11 (2026-09-17, user-scoped): LOBO n=8 extension. Copies each of the 4
existing exp_e3_lobo_leakfree configs VERBATIM except seeds -> [5,6,7] and
results_dir -> a sibling "_n8ext" directory (kept separate from the n=5
run so existing provenance is never touched; n=8 pooling happens at
analysis time by merging both directories' per-seed macro values).
No new backbones (user decision 2026-09-17): BM3 only, same 4 arms
(single/dual x nokin/kin), same train_lobo_leakfree.py / fixed_epoch
selection_mode.
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent
CFG_OUT = Path(__file__).parent / "configs_batch5_lobo_n8"
CFG_OUT.mkdir(exist_ok=True)

SOURCES = [
    ("BM3_single_nokin", "experiments/exp_e3_lobo_leakfree/config_lobo_nokin_leakfree.yaml"),
    ("BM3_single_kin",   "experiments/exp_e3_lobo_leakfree/config_lobo_kin_leakfree.yaml"),
    ("BM3_dual_nokin",   "experiments/exp_e3_lobo_leakfree/config_lobo_dual_nokin_leakfree.yaml"),
    ("BM3_dual_kin",     "experiments/exp_e3_lobo_leakfree/config_lobo_dual_kin_leakfree.yaml"),
]


def main():
    manifest = []
    for family, rel in SOURCES:
        src = ROOT / rel
        with open(src) as f:
            cfg = yaml.safe_load(f)
        orig_dir = Path(cfg["results_dir"]).expanduser()
        cfg = dict(cfg)
        cfg["seeds"] = [5, 6, 7]
        cfg["results_dir"] = str(orig_dir.parent / f"{orig_dir.name}_n8ext")
        out_path = CFG_OUT / f"{Path(rel).stem}_n8ext.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)
        manifest.append({
            "family": family,
            "orig_config": str(src.relative_to(ROOT)),
            "n8_config": str(out_path.relative_to(ROOT)),
            "n5_results_dir": str(orig_dir),
            "n8_results_dir": cfg["results_dir"],
            "driver": "experiments/exp_e3_lobo_leakfree/train_lobo_leakfree.py",
            "seeds": cfg["seeds"],
        })

    manifest_path = Path(__file__).parent / "run_manifest_batch5_lobo_n8.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_runs = sum(len(row["seeds"]) * 4 for row in manifest)  # 4 folds/seed
    print(f"Generated {len(manifest)} LOBO n8-extension configs ({total_runs} fold-runs, "
          f"3 seeds x 4 folds each) -> {CFG_OUT}")
    for row in manifest:
        print(f"  {row['family']:16s} seeds={row['seeds']}  -> {row['n8_results_dir']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
experiments/exp_bm3_cwru_leakfree/gen_configs_batch2.py

BATCH 2 GPU-queue config generator (per user instruction, 2026-09-16).
Copies every source config VERBATIM (P3: no hyperparameter changes) except
`results_dir`, redirected into the leak-free results root. Queue order is
fixed by this script's own ordering (single_kin first, transformer1d last),
matching the user's specified sequence.

Sources:
  single_kin      <- experiments/exp_e6_newseeds/single_kin/config_snr{-4,-6,-8}.yaml (seeds [5,6,7], completes BATCH1's n=8 kin gap)
  cnn1d           <- experiments/exp07_baselines/config_cnn1d_{clean,snr-2,snr-4,snr-6,snr-8,snr0}.yaml
  1dcnn_nobn      <- experiments/exp_mext_e21b_1dcnn_nobn_cwru/config_{snr-8,snr-6,snr-4,snr-2,snr0,clean}.yaml
  cnn1d_ln        <- experiments/exp_p2_cnn1d_ln/cwru/config_{clean,snr-2,snr-4,snr-6,snr-8,snr0}.yaml
  bm3bn           <- experiments/exp_mext_e21_bm3bn_cwru/config_{snr-8,snr-6,snr-4,snr-2,snr0,clean}.yaml
  bm3bn_kin       <- experiments/exp_mext_e22_bm3bn_kin_cwru/config_{snr-8,snr-6,snr-4,snr-2,snr0,clean}.yaml
  e5_single       <- experiments/exp_e5_pink_noise/single/config_snr{-4,-6,-8}.yaml
  e5_dual         <- experiments/exp_e5_pink_noise/dual/config_snr{-4,-6,-8}.yaml
  transformer1d   <- experiments/exp07_baselines/config_transformer1d_{clean,snr-2,snr-4,snr-6,snr-8,snr0}.yaml
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent
CFG_OUT = Path(__file__).parent / "configs_batch2"
CFG_OUT.mkdir(exist_ok=True)

NEW_RESULTS_ROOT = "<REPO_ROOT>/results/bm3_cwru_leakfree_20260911-leakfree"


def redirect(src_path: Path, name_override: str | None = None):
    with open(src_path) as f:
        cfg = yaml.safe_load(f)
    base = name_override or Path(cfg["results_dir"]).name
    if not base.endswith("_leakfree"):
        base = f"{base}_leakfree"
    cfg = dict(cfg)
    cfg["results_dir"] = f"{NEW_RESULTS_ROOT}/{base}"
    out_path = CFG_OUT / f"{base}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)
    return out_path, cfg["results_dir"], cfg.get("seeds", [0, 1, 2, 3, 4])


def main():
    manifest = []

    def add(family, src_path, name_override=None):
        out_path, results_dir, seeds = redirect(src_path, name_override)
        manifest.append({
            "family": family,
            "orig_config": str(src_path.relative_to(ROOT)),
            "leakfree_config": str(out_path.relative_to(ROOT)),
            "results_dir": results_dir,
            "driver": "train_leakfree.py",
            "seeds": seeds,
        })

    # 1. single_kin (9 runs) — completes BATCH1's n=8 kin gap
    for snr in [-4, -6, -8]:
        add("single_kin", ROOT / f"experiments/exp_e6_newseeds/single_kin/config_snr{snr}.yaml")

    # 2. cnn1d (30 runs)
    for tag in ["clean", "snr-2", "snr-4", "snr-6", "snr-8", "snr0"]:
        add("cnn1d", ROOT / f"experiments/exp07_baselines/config_cnn1d_{tag}.yaml")

    # 3. 1dcnn_nobn (30 runs)
    for tag in ["snr-8", "snr-6", "snr-4", "snr-2", "snr0", "clean"]:
        add("1dcnn_nobn", ROOT / f"experiments/exp_mext_e21b_1dcnn_nobn_cwru/config_{tag}.yaml")

    # 4. cnn1d_ln (30 runs)
    for tag in ["clean", "snr-2", "snr-4", "snr-6", "snr-8", "snr0"]:
        add("cnn1d_ln", ROOT / f"experiments/exp_p2_cnn1d_ln/cwru/config_{tag}.yaml")

    # 5. bm3bn nokin (30 runs)
    for tag in ["snr-8", "snr-6", "snr-4", "snr-2", "snr0", "clean"]:
        add("bm3bn", ROOT / f"experiments/exp_mext_e21_bm3bn_cwru/config_{tag}.yaml")

    # 6. bm3bn_kin (30 runs)
    for tag in ["snr-8", "snr-6", "snr-4", "snr-2", "snr0", "clean"]:
        add("bm3bn_kin", ROOT / f"experiments/exp_mext_e22_bm3bn_kin_cwru/config_{tag}.yaml")

    # 7. e5_single (15 runs)
    for snr in [-4, -6, -8]:
        add("e5_single", ROOT / f"experiments/exp_e5_pink_noise/single/config_snr{snr}.yaml")

    # 8. e5_dual (15 runs)
    for snr in [-4, -6, -8]:
        add("e5_dual", ROOT / f"experiments/exp_e5_pink_noise/dual/config_snr{snr}.yaml")

    # 9. transformer1d (30 runs)
    for tag in ["clean", "snr-2", "snr-4", "snr-6", "snr-8", "snr0"]:
        add("transformer1d", ROOT / f"experiments/exp07_baselines/config_transformer1d_{tag}.yaml")

    manifest_path = Path(__file__).parent / "run_manifest_batch2.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_runs = sum(len(row["seeds"]) for row in manifest)
    print(f"Generated {len(manifest)} leak-free BATCH2 configs ({total_runs} runs) -> {CFG_OUT}")
    print(f"Manifest -> {manifest_path}")
    by_family = {}
    for row in manifest:
        by_family.setdefault(row["family"], []).append(row)
    for fam, rows in by_family.items():
        n = sum(len(r["seeds"]) for r in rows)
        print(f"  {fam:15s} {len(rows)} configs, {n} runs")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
experiments/exp_bm3_cwru_leakfree/gen_configs_batch1_n8.py

BATCH 1 of the B2-n8 leak-free follow-up (per user instruction, 2026-09-15):
B2 flagship SNR points (-4/-6/-8 dB) x {single_nokin, dual_nokin, dual_kin}
x seeds {5,6,7} = 9 configs x 3 seeds = 27 runs.

Configs are derived VERBATIM from experiments/exp_e6_newseeds/{arm}/config_snr{snr}.yaml
(the original leaky-pipeline seed-5-7 configs) — every field copied unchanged except:
  - results_dir: redirected to a new leak-free results root
  - (seeds field is already [5,6,7] in the source configs — left untouched)

This does NOT touch exp_e6_newseeds/ or its results/ (read-only source).
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent  # <REPO_ROOT>
CFG_OUT = Path(__file__).parent / "configs_batch1_n8"
CFG_OUT.mkdir(exist_ok=True)

NEW_RESULTS_ROOT = "<REPO_ROOT>/results/bm3_cwru_leakfree_20260911-leakfree"

ARMS = ["single_nokin", "dual_nokin", "dual_kin"]
SNRS = [-4, -6, -8]

SRC_ROOT = ROOT / "experiments/exp_e6_newseeds"


def main():
    manifest = []
    for arm in ARMS:
        for snr in SNRS:
            src = SRC_ROOT / arm / f"config_snr{snr}.yaml"
            with open(src) as f:
                cfg = yaml.safe_load(f)
            assert cfg["seeds"] == [5, 6, 7], f"{src}: unexpected seeds {cfg['seeds']}"
            old_name = Path(cfg["results_dir"]).name  # e.g. exp_e6_single_nokin_snrm4_newseed
            cfg = dict(cfg)
            cfg["results_dir"] = f"{NEW_RESULTS_ROOT}/{old_name}_leakfree"
            out_path = CFG_OUT / f"{old_name}_leakfree.yaml"
            with open(out_path, "w") as f:
                yaml.dump(cfg, f, sort_keys=False)
            manifest.append({
                "orig_config": str(src.relative_to(ROOT)),
                "leakfree_config": str(out_path.relative_to(ROOT)),
                "results_dir": cfg["results_dir"],
                "driver": "train_leakfree.py",
                "arm": arm,
                "snr": snr,
                "seeds": cfg["seeds"],
            })

    manifest_path = Path(__file__).parent / "run_manifest_batch1_n8.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated {len(manifest)} leak-free BATCH1 configs -> {CFG_OUT}")
    print(f"Manifest -> {manifest_path}")
    for row in manifest:
        print(f"  {row['leakfree_config']:70s} -> {row['results_dir']}")


if __name__ == "__main__":
    main()

#!/bin/bash
# Re-run the round-3 or round-4 experiments from the repository root.
#   usage: bash reproduce/run_round.sh round3|round4 [config-name ...]
# Requirements: data under data/ (see README "Data"), environment 2 (environments/), and
#   export MAMBA_SSM_BUILD=/path/to/mamba_ssm/build/lib...   (the Mamba-3 kernel build; not needed for 1D-CNN)
# For round 3 the environment-2 patches in environments/patches/ must be applied first (see TABLE_MAP.md).
# Results are written to out_r3/<name> or out_r4/<name>; compare with results/mst_round3_20261007/ or
# results/mst_round4_noisefix_20261007/.
set -e
cd "$(dirname "$0")/.."
R=$1; shift || true
[ "$R" = round3 ] || [ "$R" = round4 ] || { echo "first argument: round3 or round4"; exit 2; }
export PYTHONPATH=${MAMBA_SSM_BUILD:-}:$(pwd):$(pwd)/experiments/exp01_cwru_baseline
Q=reproduce/queue_$R.txt; mkdir -p out_r${R#round}
names="$*"; [ -z "$names" ] && names=$(awk '{print $1}' $Q)
for name in $names; do
  line=$(grep -E "^$name( |$)" $Q || true); [ -z "$line" ] && { echo "unknown config $name"; exit 2; }
  kind=$(echo $line | awk '{print $2}'); rule=$(echo $line | awk '{print $3}')
  [ "$R" = round4 ] && kind=cwru
  cfg=configs/$R/$name.yaml
  case $kind in
    xc)   XJTU_ONSET_RULE=${rule:-H} python -u scripts/xjtu/train_xjtu_leakfree_v3.py --config $cfg
          XJTU_ONSET_RULE=${rule:-H} python -u scripts/extra/eval_ckpts_cross_v3.py $cfg ;;
    lobo) XJTU_ONSET_RULE=${rule:-H} python -u scripts/lobo/train_lobo_leakfree_v3.py --config $cfg ;;
    cwru) python -u experiments/exp01_cwru_baseline/train_leakfree.py --config $cfg ;;
    *) echo "unknown kind $kind for $name"; exit 2 ;;
  esac
done

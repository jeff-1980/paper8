#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH=<MAMBA_SSM_BUILD>:$(pwd)
for c in $(cat loco_names.txt); do
  [ -f out_loco/$c/done.flag ] && continue
  s=$(date +%s)
  python -u scripts/xjtu/train_xjtu_leakfree_v3.py --config cfg_loco/$c.yaml > out_loco/log_$c.txt 2>&1
  r=$?
  if [ $r -eq 0 ]; then python -u scripts/extra/eval_ckpts_cross_v3.py cfg_loco/$c.yaml > out_loco/logeval_$c.txt 2>&1; r=$?; fi
  echo "$c exit=$r wall_s=$(( $(date +%s)-s ))" >> out_loco/wall.txt
  [ $r -eq 0 ] && touch out_loco/$c/done.flag
done
echo ALLDONE >> out_loco/wall.txt

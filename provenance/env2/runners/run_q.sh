#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH=<MAMBA_SSM_BUILD>:$(pwd)
for c in $(cat q_names.txt); do
  [ -f out_q/$c/done.flag ] && continue
  s=$(date +%s)
  python -u scripts/xjtu/train_xjtu_leakfree_v3.py --config cfg_q/$c.yaml > out_q/log_$c.txt 2>&1
  r=$?
  if [ $r -eq 0 ]; then python -u scripts/extra/eval_ckpts_cross_v3.py cfg_q/$c.yaml > out_q/logeval_$c.txt 2>&1; r=$?; fi
  echo "$c exit=$r wall_s=$(( $(date +%s)-s ))" >> out_q/wall.txt
  [ $r -eq 0 ] && touch out_q/$c/done.flag
done
echo ALLDONE >> out_q/wall.txt

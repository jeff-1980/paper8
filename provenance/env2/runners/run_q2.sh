#!/bin/bash
# GPU-contention-aware runner: waits for a free GPU, kills a config if a foreign process pushes total memory > 11 GB, then retries it.
cd "$(dirname "$0")"
export PYTHONPATH=<MAMBA_SSM_BUILD>:$(pwd)
mem(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
wait_free(){ ok=0; while [ $ok -lt 3 ]; do [ "$(mem)" -lt 5500 ] && ok=$((ok+1)) || ok=0; sleep 60; done; }
for c in $(cat q_names.txt); do
  [ -f out_q/$c/done.flag ] && continue
  while true; do
    wait_free
    s=$(date +%s)
    python -u scripts/xjtu/train_xjtu_leakfree_v3.py --config cfg_q/$c.yaml > out_q/log_$c.txt 2>&1 &
    pid=$!; bad=0; killed=0
    while kill -0 $pid 2>/dev/null; do
      sleep 30
      if [ "$(mem)" -gt 12500 ]; then bad=$((bad+1)); else bad=0; fi
      if [ $bad -ge 4 ]; then kill $pid; sleep 5; killed=1; echo "$c killed (GPU contention) $(date)" >> out_q/wall.txt; break; fi
    done
    wait $pid 2>/dev/null; r=$?
    [ $killed -eq 1 ] && continue
    if [ $r -eq 0 ]; then python -u scripts/extra/eval_ckpts_cross_v3.py cfg_q/$c.yaml > out_q/logeval_$c.txt 2>&1; r=$?; fi
    echo "$c exit=$r wall_s=$(( $(date +%s)-s ))" >> out_q/wall.txt
    [ $r -eq 0 ] && touch out_q/$c/done.flag
    break
  done
done
echo ALLDONE >> out_q/wall.txt

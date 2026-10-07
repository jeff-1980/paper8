#!/bin/bash
# Round-3 queue runner. One process per config (all seeds). Waits for a quiet GPU; kills a run if
# foreign memory pushes the card above 12.5 GB for 2 min and reruns it from scratch. done flag only on success.
cd "$(dirname "$0")"
export PYTHONPATH=<MAMBA_SSM_BUILD>:$(pwd)
mem(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
wait_free(){ ok=0; while [ $ok -lt 2 ]; do [ "$(mem)" -lt 8500 ] && ok=$((ok+1)) || ok=0; sleep 30; done; }
while read name kind rule; do
  [ -z "$name" ] && continue
  [ -f out_r3/done_$name ] && continue
  case $kind in
    xc)   cmd="python -u scripts/xjtu/train_xjtu_leakfree_v3.py --config cfg_r3/$name.yaml";;
    lobo) cmd="python -u scripts/lobo/train_lobo_leakfree_v3.py --config cfg_r3/$name.yaml";;
    cwru) cmd="env PYTHONPATH=$PYTHONPATH:$(pwd)/scripts/cwru python -u scripts/cwru/train_leakfree.py --config cfg_r3/$name.yaml";;
  esac
  while true; do
    wait_free; st=$(date +%s)
    XJTU_ONSET_RULE=$rule $cmd > out_r3/log_$name.txt 2>&1 &
    pid=$!; bad=0; killed=0
    while kill -0 $pid 2>/dev/null; do
      sleep 30
      if [ "$(mem)" -gt 12500 ]; then bad=$((bad+1)); else bad=0; fi
      if [ $bad -ge 4 ]; then kill $pid; sleep 5; killed=1; echo "$name killed (GPU contention) $(date +%m-%d_%H:%M)" >> out_r3/wall.txt; break; fi
    done
    wait $pid 2>/dev/null; r=$?
    [ $killed -eq 1 ] && { rm -rf out_r3/$name; continue; }
    if [ $r -eq 0 ] && [ $kind = xc ]; then XJTU_ONSET_RULE=$rule python -u scripts/extra/eval_ckpts_cross_v3.py cfg_r3/$name.yaml > out_r3/logeval_$name.txt 2>&1; r=$?; fi
    [ $r -eq 0 ] && [ $kind = cwru ] && rm -rf out_r3/$name/checkpoints
    echo "$name exit=$r wall_s=$(( $(date +%s)-st )) $(date +%m-%d_%H:%M)" >> out_r3/wall.txt
    [ $r -eq 0 ] && touch out_r3/done_$name
    break
  done
done < queue_r3.txt
echo ALLDONE $(date +%m-%d_%H:%M) >> out_r3/wall.txt

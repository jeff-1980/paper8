#!/bin/bash
# Noise-fix rerun queue (CWRU, Mamba-3, noise_key=v2). One process per config (5 seeds).
# Waits for a quiet GPU; kills and restarts a config from scratch on foreign GPU contention.
# On success keeps only the validation-selected checkpoint of each seed. done flag only on success.
cd "$(dirname "$0")"
export PYTHONPATH=<MAMBA_SSM_BUILD>:$(pwd):$(pwd)/scripts/cwru
mem(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
wait_free(){ ok=0; while [ $ok -lt 2 ]; do [ "$(mem)" -lt 8500 ] && ok=$((ok+1)) || ok=0; sleep 30; done; }
while read name; do
  [ -z "$name" ] && continue
  [ -f out_r4/done_$name ] && continue
  while true; do
    wait_free; st=$(date +%s)
    python -u scripts/cwru/train_leakfree.py --config cfg_r4/$name.yaml > out_r4/log_$name.txt 2>&1 &
    pid=$!; bad=0; killed=0
    while kill -0 $pid 2>/dev/null; do
      sleep 30
      if [ "$(mem)" -gt 12500 ]; then bad=$((bad+1)); else bad=0; fi
      if [ $bad -ge 4 ]; then kill $pid; sleep 5; killed=1; echo "$name killed (GPU contention) $(date +%m-%d_%H:%M)" >> out_r4/wall.txt; break; fi
    done
    wait $pid 2>/dev/null; r=$?
    [ $killed -eq 1 ] && { rm -rf out_r4/$name; continue; }
    if [ $r -eq 0 ]; then python r4/keep_best_ckpt.py out_r4/$name || r=99; fi
    echo "$name exit=$r wall_s=$(( $(date +%s)-st )) $(date +%m-%d_%H:%M)" >> out_r4/wall.txt
    [ $r -eq 0 ] && touch out_r4/done_$name
    break
  done
done < queue_r4.txt
echo ALLDONE $(date +%m-%d_%H:%M) >> out_r4/wall.txt

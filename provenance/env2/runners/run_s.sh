#!/bin/bash
# LOBO seed extension 3->8 (seeds 3-7), identical to the original per-seed chain runs except seeds/results_dir.
# Waits for a quiet GPU; kills a seed-run if foreign memory pushes the card past 12.5 GB for 2 min, then retries it.
cd "$(dirname "$0")"
export PYTHONPATH=<MAMBA_SSM_BUILD>:$(pwd)
mem(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
wait_free(){ ok=0; while [ $ok -lt 3 ]; do [ "$(mem)" -lt 8500 ] && ok=$((ok+1)) || ok=0; sleep 60; done; }
for s in 3 4 5 6 7; do for a in H V HV; do
  c=lobo_bm3_$a
  [ -f out_s/done_${a}_$s ] && continue
  while true; do
    wait_free
    st=$(date +%s)
    python -u scripts/lobo/train_lobo_leakfree_v2.py --config cfg_s/$c.yaml --seeds $s > out_s/log_${c}_seed$s.txt 2>&1 &
    pid=$!; bad=0; killed=0
    while kill -0 $pid 2>/dev/null; do
      sleep 30
      if [ "$(mem)" -gt 12500 ]; then bad=$((bad+1)); else bad=0; fi
      if [ $bad -ge 4 ]; then kill $pid; sleep 5; killed=1; echo "$c seed$s killed (GPU contention) $(date +%H:%M)" >> out_s/wall.txt; break; fi
    done
    wait $pid 2>/dev/null; r=$?
    [ $killed -eq 1 ] && continue
    echo "$c [--seeds $s] exit=$r wall_s=$(( $(date +%s)-st ))" >> out_s/wall.txt
    [ $r -eq 0 ] && touch out_s/done_${a}_$s
    break
  done
done; done
for a in H V HV; do python -u scripts/extra/eval_ckpts_lobo.py cfg_s/lobo_bm3_$a.yaml > out_s/logeval_lobo_bm3_$a.txt 2>&1; echo "eval $a exit=$?" >> out_s/wall.txt; done
echo ALLDONE >> out_s/wall.txt

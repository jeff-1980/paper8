#!/bin/bash
# Round 6 queue (prereg_mst_round6.md). One process per configuration; deterministic switch on.
# Waits for a quiet GPU; kills and restarts a configuration from scratch on foreign GPU contention.
# done flag only when training AND evaluation succeeded.
cd "$(dirname "$0")"
export PYTHONPATH=<MAMBA_SSM_BUILD>:$(pwd):$(pwd)/experiments/exp01_cwru_baseline
export BM_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TRITON_CACHE_DIR=<REPO_ROOT>/round6/cache/triton TILELANG_CACHE_DIR=<REPO_ROOT>/round6/cache/tilelang   # persistent kernel caches (rerun 2026-11-05)
mem(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
wait_free(){ ok=0; while [ $ok -lt 2 ]; do [ "$(mem)" -lt 8500 ] && ok=$((ok+1)) || ok=0; sleep 20; done; }
guard(){ "$@" & pid=$!; bad=0
  while kill -0 $pid 2>/dev/null; do sleep 30; if [ "$(mem)" -gt 12500 ]; then bad=$((bad+1)); else bad=0; fi
    if [ $bad -ge 4 ]; then kill $pid; sleep 5; return 77; fi; done; wait $pid; }
while read -r name kind arg; do
  [ -z "$name" ] && continue
  tag=$name; [ "$kind" = evalcwru ] && tag=${name}_eval_${arg}
  [ -f out_r6/done_$tag ] && continue
  while true; do
    wait_free; st=$(date +%s); r=0
    case $kind in
      xc)   rm -rf out_r6/$name; XJTU_ONSET_RULE=$arg guard python -u scripts/xjtu/train_xjtu_leakfree_v3.py --config cfg_r6/$name.yaml > out_r6/log_$name.txt 2>&1; r=$?
            [ $r -eq 0 ] && { XJTU_ONSET_RULE=$arg python -u scripts/extra/eval_r6_xjtu.py cfg_r6/$name.yaml > out_r6/logeval_$name.txt 2>&1; r=$?; } ;;
      lobo) rm -rf out_r6/$name; XJTU_ONSET_RULE=$arg guard python -u scripts/lobo/train_lobo_leakfree_v3.py --config cfg_r6/$name.yaml > out_r6/log_$name.txt 2>&1; r=$?
            [ $r -eq 0 ] && { XJTU_ONSET_RULE=$arg python -u scripts/extra/eval_r6_xjtu.py cfg_r6/$name.yaml > out_r6/logeval_$name.txt 2>&1; r=$?; } ;;
      cwru) rm -rf out_r6/$name; guard python -u experiments/exp01_cwru_baseline/train_leakfree.py --config cfg_r6/$name.yaml > out_r6/log_$name.txt 2>&1; r=$?
            [ $r -eq 0 ] && { python r6tests/keep_best_ckpt.py out_r6/$name > /dev/null; r=$?; } ;;
      evalcwru) R6_EVAL_DIR=out_r6/evalcwru python -u scripts/extra/eval_r6_cwru.py cfg_r6/$name.yaml $arg > out_r6/logeval_$tag.txt 2>&1; r=$? ;;
    esac
    if [ $r -eq 77 ]; then echo "$tag killed (GPU contention) $(date +%m-%d_%H:%M)" >> out_r6/wall.txt; continue; fi
    echo "$tag exit=$r wall_s=$(( $(date +%s)-st )) $(date +%m-%d_%H:%M)" >> out_r6/wall.txt
    [ $r -eq 0 ] && touch out_r6/done_$tag
    break
  done
done < queue_r6.txt
echo "ALLDONE $(date +%m-%d_%H:%M)" >> out_r6/wall.txt

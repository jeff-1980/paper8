#!/bin/bash
# usage: chain.sh <trainer> "<extra args>" <cfg...>   sequential, wall-clock logged to out/wall.txt
cd "$(dirname "$0")"
export PYTHONPATH=<MAMBA_SSM_BUILD>:$(pwd)
T=$1; X=$2; shift; shift
for c in "$@"; do
  s=$(date +%s)
  python -u $T --config cfg_run/$c.yaml $X > out/log_${c}_$(echo $X | tr -d ' -').txt 2>&1
  echo "$c [$X] exit=$? wall_s=$(( $(date +%s)-s ))" >> out/wall.txt
done

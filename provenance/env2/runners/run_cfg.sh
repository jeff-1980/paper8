#!/bin/bash
# usage: run_cfg.sh <trainer-relpath> <cfg-name...>  (runs sequentially, one process at a time)
cd "$(dirname "$0")"
export PYTHONPATH=<MAMBA_SSM_BUILD>:$(pwd)
T=$1; shift
for c in "$@"; do
  s=$(date +%s)
  python -u $T --config cfg_run/$c.yaml > out/log_$c.txt 2>&1
  echo "$c exit=$? wall_s=$(( $(date +%s)-s ))" >> out/wall.txt
done

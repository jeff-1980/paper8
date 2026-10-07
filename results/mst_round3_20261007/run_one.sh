#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH=<MAMBA_SSM_BUILD>:$(pwd)
python -u scripts/xjtu/train_xjtu_leakfree_v3.py --config cfg_r3/$1.yaml > out_r3/log_$1.txt 2>&1
echo "exit=$?"

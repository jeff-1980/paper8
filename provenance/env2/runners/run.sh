#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH=<MAMBA_SSM_BUILD>:$(pwd)
python -u scripts/xjtu/train_xjtu_leakfree.py --config cfg/$1.yaml > out/log_$1.txt 2>&1

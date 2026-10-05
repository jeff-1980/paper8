#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH=<MAMBA_SSM_BUILD>:$(pwd)
python -u run_multi.py "$@"

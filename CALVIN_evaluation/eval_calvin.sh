#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Activate the CALVIN environment first; pass --calvin_root and --dataset_path.
python -m CALVIN_evaluation.main \
    --host 127.0.0.1 --port 9000 \
    --out_path CALVIN_evaluation/result \
    --calvin_root ~/work/users/luyuxiang/code/calvin \
    --dataset_path ~/work/users/luyuxiang/code/calvin/dataset/task_ABC_D \
    --save_name evo1_80k

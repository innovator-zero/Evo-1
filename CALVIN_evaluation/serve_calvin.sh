#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Activate the evo1 environment first. Override checkpoint/device with arguments.
python -m CALVIN_evaluation.server \
    --checkpoint Evo_1/checkpoints/evo1_calvin/stage2/step_80000 \
    --host 127.0.0.1 --port 9000 --device cuda:0 \
    "$@"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Activate the evo1 environment before running. Edit parameters here as needed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
VLM_PATH="${VLM_PATH:-$PWD/.cache/huggingface/hub/models--OpenGVLab--InternVL3-1B/snapshots/4415a3b810e636d11dfa86b0e9ba40bb00535aa8}"

accelerate launch \
    --use_deepspeed --deepspeed_config_file ds_config/zero2.json \
    --num_processes "${NUM_PROCESSES:-1}" --num_machines 1 --machine_rank 0 \
    --main_process_port "${MASTER_PORT:-29571}" \
    --mixed_precision bf16 --dynamo_backend no --gradient_accumulation_steps 1 \
    scripts/train.py \
    --dataset_config_path dataset/config_calvin.yaml \
    --vlm_name "$VLM_PATH" \
    --cache_dir dataset/dataset_cache/calvin \
    --batch_size 16 --horizon 50 --per_action_dim 24 --state_dim 24 \
    --image_size 448 --num_layers 8 \
    --lr 1e-5 --dropout 0.2 --weight_decay 0.001 \
    --warmup_steps 1000 --ckpt_interval 10000 --log_interval 100 \
    --num_workers 4 --prefetch_factor 2 --video_backend av \
    --finetune_action_head --use_augmentation \
    --disable_wandb --disable_swanlab --seed 42 \
    --run_name calvin_stage1 --save_dir checkpoints/evo1_calvin/stage1 \
    --max_steps 20000 \
    "$@"

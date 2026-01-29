NPROC_PER_NODE=${SENSECORE_ACCELERATE_DEVICE_COUNT:-1}

. /mnt/luyuxiang/miniconda3/etc/profile.d/conda.sh

conda activate evo1

export HF_HOME="/mnt/luyuxiang/hf"
export HF_ENDPOINT="https://hf-mirror.com"

cd Evo_1/

dataset=$1

accelerate launch \
    --num_processes $NPROC_PER_NODE \
    --deepspeed_config_file ds_config.json \
    scripts/train.py \
    --run_name Evo1_${dataset}_stage1 \
    --use_augmentation \
    --dropout 0.2 \
    --weight_decay 1e-3 \
    --max_steps 5000 \
    --ckpt_interval 2500 \
    --warmup_steps 1000 \
    --horizon 50 \
    --finetune_action_head \
    --disable_wandb \
    --dataset_config_path dataset/config_${dataset}.yaml \
    --per_action_dim 24 \
    --state_dim 24 \
    --save_dir checkpoints/Evo1_${dataset}_stage1

accelerate launch \
    --num_processes $NPROC_PER_NODE \
    --deepspeed_config_file ds_config.json \
    scripts/train.py \
    --run_name Evo1_${dataset}_stage2 \
    --use_augmentation \
    --dropout 0.2 \
    --weight_decay 1e-3 \
    --max_steps 80000 \
    --ckpt_interval 10000 \
    --warmup_steps 1000 \
    --horizon 50 \
    --finetune_vlm \
    --finetune_action_head \
    --disable_wandb \
    --dataset_config_path dataset/config_${dataset}.yaml \
    --per_action_dim 24 \
    --state_dim 24 \
    --save_dir checkpoints/Evo1_${dataset}_stage2 \
    --resume \
    --resume_pretrain \
    --resume_path checkpoints/Evo1_${dataset}_stage1/step_5000
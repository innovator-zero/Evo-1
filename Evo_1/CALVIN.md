# CALVIN training

Run on hw with the existing environment:

```bash
conda activate /home/ma-user/work/users/luyuxiang/envs/evo1
cd /home/ma-user/work/users/luyuxiang/code/Evo-1/Evo_1

# Stage 1: frozen VLM, train action head (20,000 steps).
bash scripts/train_calvin.sh

# Stage 2: initialize from stage1/step_final; train VLM and head (80,000 steps).
# Run after Stage 1 finishes.
bash scripts/train_calvin_stage2.sh

# Two GPUs; batch size is per GPU.
CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 bash scripts/train_calvin.sh

# Pass train.py arguments directly, or edit the command in the script.
bash scripts/train_calvin.sh --batch_size 2 --max_steps 5 --warmup_steps 1 \
  --ckpt_interval 2 --max_episodes 8 --num_workers 0 \
  --save_dir checkpoints/calvin/short_stage1 --verify_updates
```

The scripts call `accelerate launch` directly. There is no Python launcher,
argument translation, automatic model download, or generated dataset YAML.
Training flags use underscores, exactly as in `scripts/train.py`.
Edit `dataset/config_calvin.yaml` to change the data path.
`VLM_PATH` can override the already downloaded local model path.
Set both `CUDA_VISIBLE_DEVICES` and `NUM_PROCESSES` when changing GPU count.

Both scripts use the unchanged `ds_config/zero2.json` (ZeRO-2, BF16), batch 8
per GPU, horizon 50, learning rate 1e-5 and gradient accumulation 1.
Choose a distinct `--save_dir` for each new experiment; the simple scripts do
not check for existing outputs. To capture stdout/stderr, use shell `tee`:

```bash
mkdir -p logs
set -o pipefail
bash scripts/train_calvin.sh 2>&1 | tee logs/calvin_stage1.log
```

## Checkpoint loading

- Stage 1 to Stage 2: `--resume --resume_pretrain --resume_path <stage1 checkpoint>`
  loads weights and resets optimizer, scheduler and step.
- Same-stage continuation: `--resume --resume_path <checkpoint>` restores optimizer,
  step and scheduler. For Stage 2 continuation, **remove `--resume_pretrain` from
  `train_calvin_stage2.sh`**, and point `--resume_path` at the Stage 2 checkpoint.
  Keep `max_steps` and `warmup_steps` unchanged.
- Standard checkpoints retry non-strict loading when strict loading fails.
  DeepSpeed retries weights-only loading when state restoration fails and emits
  a warning; this fallback is not a full optimizer-state recovery.

Outputs include training logs, `metrics.jsonl`, and `step_<N>/` / `step_final/`
checkpoints with config and normalization stats. `--verify_updates` also writes
`update_check.json`. DeepSpeed momentum norm is omitted when unavailable.
W&B and SwanLab are disabled by the commands above.

## Data contract

The default source is `/home/ma-user/work/dataset/lerobot/InternData-Calvin_ABC`.
Source data and stats are read-only. Channel order matches
`better_openpi/src/openpi/policies/calvin_policy.py`:

| Input | Ordered fields |
| --- | --- |
| State | `state.ee_pos` (3), `state.ee_rot` (3), `state.gripper` (1) |
| Action | `action.delta_ee_pos` (3), `action.delta_ee_rot` (3), `action.gripper` (1) |
| Images | `video.image_base`, `video.image_wrist` |

Actions are already relative commands; no state subtraction is applied.
Per-field stats are concatenated in the same order. Bounds normalization precedes
padding to 24 dimensions; seven channels are valid. State/action outputs retain
the original BF16 dtype. The third image is zero padded and masked. Windows stay
within episodes and repeat the final action at the boundary.

## Checks

```bash
python -m unittest discover -s tests -v
```

The hw environment uses Torch 2.5.1+cu124 and DeepSpeed 0.16.9; the compatibility
overlay is in `requirements-calvin.txt`. Previous short GPU validation is recorded
in `checkpoints/calvin/validation_20260906.json` (before launcher simplification
and BF16 restoration). It is not a convergence test or CALVIN rollout evaluation.

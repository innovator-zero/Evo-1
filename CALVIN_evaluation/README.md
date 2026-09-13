# Evo-1 CALVIN evaluation

Standalone two-process evaluator adapted from
`better_openpi/examples/calvin/main.py` and `calvin_env_wrapper.py`. It uses CALVIN's
sequence generator, initial states, language annotations and task oracle. It does
not import better_openpi or openpi-client. Existing Evo-1 benchmark servers are unchanged.

## 1. Policy server (evo1 environment)

From the Evo-1 repository root on hw:

```bash
conda activate /home/ma-user/work/users/luyuxiang/envs/evo1
cd /home/ma-user/work/users/luyuxiang/code/Evo-1
CUDA_VISIBLE_DEVICES=1 bash CALVIN_evaluation/serve_calvin.sh
```

Choose an idle physical GPU. The script defaults to
`Evo_1/checkpoints/evo1_calvin/stage2/step_final`. To use another checkpoint:

```bash
bash CALVIN_evaluation/serve_calvin.sh \
  --checkpoint Evo_1/checkpoints/evo1_calvin/stage2/step_best
```

The checkpoint must contain `config.json`, `norm_stats.json`, and either
`mp_rank_00_model_states.pt` (ZeRO-2) or `checkpoint.pt` (standard training).
Only model weights are loaded, strictly; ZeRO-3 shards are not supported.
Only load trusted PyTorch checkpoints. `--model_path` overrides the base VLM path
saved in the config. `--inference_steps` overrides the saved sampling steps
(currently 50), independently of the predicted action horizon (also currently 50).
The server uses the evo1 dependencies plus `websockets>=14,<17`.

The default listener is `127.0.0.1:9000`; use `--host 0.0.0.0` only when clients
must connect from another machine. The JSON protocol is intended for trusted networks.

## 2. Evaluator (existing CALVIN environment)

Install CALVIN and its simulator dependencies according to its own checkout's
instructions, then install this small client overlay in that environment:

```bash
python -m pip install -r CALVIN_evaluation/requirements-client.txt
bash CALVIN_evaluation/eval_calvin.sh \
  --calvin_root /absolute/path/to/calvin \
  --dataset_path /absolute/path/to/task_ABC_D \
  --save_name evo1_stage2_final
```

Both paths are required. `dataset_path` contains the official `validation/`
directory and its simulator configuration. The LeRobot `InternData-Calvin_ABC`
training dataset cannot substitute for this directory. CALVIN's own Torch, Hydra,
OmegaConf, Gym and PyBullet dependencies belong to the simulator environment.
Set `EGL_VISIBLE_DEVICES` if needed; an explicit value is preserved.

Default protocol: ABC→D, 1000 sequences, five chained tasks each, 720 environment
steps per task, seed 0. Each task executes the complete predicted action chunk
before requesting another; `--execute_steps 10` executes only the first ten.
A new task clears the old action queue but does not reset the environment.
Task failure ends that chain. Oracle success is checked after every environment step.

For a short **simulation** run (requires the CALVIN environment):

```bash
bash CALVIN_evaluation/eval_calvin.sh \
  --calvin_root /absolute/path/to/calvin \
  --dataset_path /absolute/path/to/task_ABC_D \
  --num_sequences 2 --save_name evo1_short --save_videos
```

`--ep_len` changes the per-task limit. Nondefault sequence counts/limits are short
tests, not the full benchmark. `--host`, `--port`, `--out_path` and all other flags
can be passed directly; scripts are editable commands with no environment management.

## Input/output contract

The client sends raw RGB static/gripper images and state ordered as
`robot_obs[:3] + robot_obs[3:6] + robot_obs[-1:]`. The server applies the training
bicubic resize/ToTensor path with no augmentation, color swap or image flip; it
adds a zero third view and `[1,1,0]` mask. State uses checkpoint bounds statistics,
24D padding and the training BF16 rounding. Only seven action channels are valid.
Responses are denormalized `[horizon,7]` relative actions. There is no state
subtraction or extra motion scaling. The client maps gripper `<0` to `-1`, otherwise `+1`.

Wire request: `{"image": HWC_RGB, "wrist_image": HWC_RGB, "state": [7 values], "prompt": "..."}`.
Wire response: `{"actions": [[7 values], ...]}`; server errors use `{"error": "..."}`.
Invalid/non-finite inputs or outputs terminate the request/evaluation explicitly.

## Outputs and checks

Outputs go to `outputs/calvin/<save_name>/` by default. Choose a new name for each
run; existing result files are not overwritten. Every completed sequence updates:

- `result.json`: `avg_seq_len`, `chain_sr` (fractions for chains 1–5), attempted-task
  `task_info`, completed/requested counts, status and evaluation arguments.
- `sequences.jsonl`: sequence index, task list and consecutive success count.
- `success_rate.txt`: running chain rates and average length.
- Optional double-view MP4s per subtask when `--save_videos` is enabled.

Transport/inference errors preserve completed results with `status=error` rather
than recording a task failure; Ctrl-C records `interrupted`. Scores on an incomplete
run cover completed sequences only. Environment and socket are closed on exit.

```bash
# Run from repository root, in the evo1 environment. No CALVIN installation needed.
python -m unittest discover -s CALVIN_evaluation/tests -v

# Real checkpoint, real LeRobot frames/state/prompt, two WebSocket requests.
# This creates its own temporary loopback server; no separate server is needed.
CUDA_VISIBLE_DEVICES=1 python -m CALVIN_evaluation.smoke \
  --checkpoint Evo_1/checkpoints/evo1_calvin/stage2/step_final \
  --data_root /home/ma-user/work/dataset/lerobot/InternData-Calvin_ABC \
  --output Evo_1/checkpoints/calvin_eval_smoke/result.json
```

The smoke report records latency, peak allocated GPU memory and finite action
shapes. It verifies model inference and transport, **not** simulator rollout,
CALVIN success rates, or policy convergence.

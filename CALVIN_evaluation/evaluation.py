"""Chain evaluation adapted from better_openpi/examples/calvin/main.py.

CALVIN dependencies are injected by main.py so rollout and metrics can be tested
without installing the simulator. The task oracle determines success per step.
"""
from collections import Counter, deque
import json
import logging
from pathlib import Path

import numpy as np

from CALVIN_evaluation.protocol import validate_actions


def summarize(results, sequences):
    succeeded, failed = Counter(), Counter()
    for result, (_, tasks) in zip(results, sequences):
        succeeded.update(tasks[:result])
        if result < len(tasks):
            failed.update([tasks[result]])
    totals = succeeded + failed
    return {
        "num_completed": len(results),
        "avg_seq_len": float(np.mean(results)) if results else 0.0,
        "chain_sr": {str(i): float(np.mean(np.asarray(results) >= i)) if results else 0.0
                     for i in range(1, 6)},
        "task_info": {task: {"success": succeeded[task], "total": totals[task]}
                      for task in sorted(totals)},
    }


def save_summary(directory, data):
    # Replace the summary atomically so an interrupted write leaves valid JSON.
    temporary = directory / "result.json.tmp"
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(directory / "result.json")


def save_video(frames, directory, sequence_id, subtask_id, task, status):
    import imageio.v2 as imageio
    for camera, images in frames.items():
        if images:
            filename = f"{sequence_id:04d}-{subtask_id}-{task}-{camera}-{status}.mp4"
            imageio.mimsave(directory / filename, images, fps=30)


def rollout(env, model, oracle, task, instruction, max_steps=720, execute_steps=None,
            frames=None):
    obs, start_info = env.get_obs(), env.get_info()
    queue = deque()
    for _ in range(max_steps):
        if not queue:
            chunk = validate_actions(model.step(obs, instruction))
            if execute_steps is not None:
                if execute_steps > len(chunk):
                    raise ValueError(f"execute_steps={execute_steps} exceeds model horizon={len(chunk)}")
                chunk = chunk[:execute_steps]
            queue.extend(chunk)
        action = queue.popleft().copy()
        action[-1] = -1 if action[-1] < 0 else 1
        obs, _, _, info = env.step(action)
        if frames is not None:
            for camera in frames:
                frames[camera].append(np.asarray(obs["rgb_obs"][camera]).copy())
        if oracle.get_task_info_for_set(start_info, info, {task}):
            return True
    return False


def evaluate_sequence(env, model, oracle, initial, tasks, annotations, state_resolver,
                      directory, sequence_id, max_steps=720, execute_steps=None,
                      save_videos=False):
    robot_obs, scene_obs = state_resolver(initial)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    model.reset()
    successes = 0
    for subtask_id, task in enumerate(tasks):
        # Each rollout owns its queue: never replay an old instruction's actions.
        frames = {"rgb_static": [], "rgb_gripper": []} if save_videos else None
        success = rollout(env, model, oracle, task, annotations[task][0], max_steps,
                          execute_steps, frames)
        if frames is not None:
            save_video(frames, directory, sequence_id, subtask_id, task,
                       "success" if success else "failure")
        if not success:
            break
        successes += 1
    return successes


def evaluate_policy(env, model, oracle, sequences, annotations, state_resolver,
                    directory, max_steps=720, execute_steps=None, save_videos=False,
                    metadata=None):
    if max_steps <= 0 or (execute_steps is not None and execute_steps <= 0):
        raise ValueError("max_steps and execute_steps must be positive")
    sequences = list(sequences)
    if not sequences:
        raise ValueError("At least one evaluation sequence is required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for filename in ("result.json", "sequences.jsonl", "success_rate.txt"):
        if (directory / filename).exists():
            raise FileExistsError(f"Existing evaluation output: {directory / filename}; choose a new save_name")
    results = []
    metadata = dict(metadata or {})

    def snapshot(status, error=None):
        data = summarize(results, sequences)
        data.update(status=status, num_requested=len(sequences), metadata=metadata)
        if error:
            data["error"] = error
        save_summary(directory, data)
        return data

    snapshot("running")
    try:
        for index, (initial, tasks) in enumerate(sequences):
            count = evaluate_sequence(env, model, oracle, initial, tasks, annotations,
                                      state_resolver, directory, index, max_steps,
                                      execute_steps, save_videos)
            results.append(count)
            with (directory / "sequences.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"sequence_id": index, "tasks": list(tasks),
                                         "success_count": count}) + "\n")
            data = snapshot("running")
            line = (f"{index + 1}/{len(sequences)}: "
                    + " | ".join(f"{data['chain_sr'][str(i)]:.3f}" for i in range(1, 6))
                    + f" | avg_seq_len={data['avg_seq_len']:.3f}")
            with (directory / "success_rate.txt").open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
            logging.info(line)
    except BaseException as exc:
        snapshot("interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                 f"{type(exc).__name__}: {exc}")
        raise
    return snapshot("complete")

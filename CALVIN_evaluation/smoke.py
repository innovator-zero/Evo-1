"""Real checkpoint + real LeRobot observation + WebSocket inference, without CALVIN."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from websockets.asyncio.server import serve

from CALVIN_evaluation.protocol import ClientModel
from CALVIN_evaluation.server import handle_connection


def load_observation(dataset):
    import av
    import pandas as pd
    dataset = Path(dataset)
    info = json.loads((dataset / "meta/info.json").read_text())
    episode = 0
    parts = dict(episode_chunk=episode // info["chunks_size"], episode_index=episode)
    rows = pd.read_parquet(dataset / info["data_path"].format(**parts))
    row = rows.iloc[0]
    images = {}
    for camera, key in (("rgb_static", "video.image_base"), ("rgb_gripper", "video.image_wrist")):
        path = dataset / info["video_path"].format(**parts, video_key=key)
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                if float(frame.time) >= float(row["timestamp"]) - 1e-6:
                    images[camera] = frame.to_ndarray(format="rgb24")
                    break
        if camera not in images:
            raise RuntimeError(f"No frame at {row['timestamp']} in {path}")
    tasks = [json.loads(line) for line in (dataset / "meta/tasks.jsonl").read_text().splitlines()]
    prompt = next(task["task"] for task in tasks if task["task_index"] == int(row["task_index"]))
    state = np.concatenate([np.asarray(row[key]).reshape(-1)
                            for key in ("state.ee_pos", "state.ee_rot", "state.gripper")])
    return {"robot_obs": state, "rgb_obs": images}, prompt


async def smoke(args):
    import torch
    from accelerate.utils import set_seed
    from CALVIN_evaluation.policy import CalvinPolicy
    set_seed(0)
    stats_file = args.data_root / "meta/stats.json"
    before = hashlib.sha256(stats_file.read_bytes()).hexdigest()
    obs, prompt = load_observation(args.data_root)
    torch.cuda.set_device(torch.device(args.device))
    torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    policy = CalvinPolicy.load(args.checkpoint, args.device)
    load_seconds = time.perf_counter() - started
    async with serve(lambda ws: handle_connection(ws, policy), "127.0.0.1", 0,
                     max_size=100_000_000, ping_interval=None) as server:
        port = server.sockets[0].getsockname()[1]

        def requests():
            client = ClientModel(port=port)
            measurements = []
            try:
                for _ in range(2):
                    start = time.perf_counter()
                    actions = client.step(obs, prompt)
                    expected = (policy.config.action_horizon or policy.config.horizon, 7)
                    if actions.shape != expected:
                        raise AssertionError((actions.shape, expected))
                    measurements.append({"shape": list(actions.shape),
                                         "seconds": time.perf_counter() - start,
                                         "min": float(actions.min()), "max": float(actions.max())})
            finally:
                client.close()
            return measurements

        measurements = await asyncio.to_thread(requests)
    assert hashlib.sha256(stats_file.read_bytes()).hexdigest() == before
    result = {"checkpoint": str(args.checkpoint.resolve()), "device": args.device,
              "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(args.device),
              "model_load_seconds": load_seconds, "requests": measurements,
              "peak_allocated_gib": torch.cuda.max_memory_allocated(args.device) / 1024**3,
              "source_stats_sha256": before, "calvin_rollout_tested": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(smoke(parser.parse_args()))


if __name__ == "__main__":
    main()

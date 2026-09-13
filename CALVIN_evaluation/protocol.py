"""JSON protocol shared by the evaluator and policy server; no CALVIN imports."""
import json

import numpy as np


def rgb_image(value):
    image = np.asarray(value)
    if (image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) == 0
            or not np.issubdtype(image.dtype, np.number)
            or not np.isfinite(image).all() or np.any(image < 0)
            or np.any(image > 255) or np.any(image != np.floor(image))):
        raise ValueError("Image must be a nonempty HxWx3 RGB array of integer values in [0, 255]")
    return image.astype(np.uint8)


def validate_request(data):
    state = np.asarray(data["state"], dtype=np.float32)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("state must contain seven finite values")
    if not isinstance(data["prompt"], str) or not data["prompt"].strip():
        raise ValueError("prompt must be a nonempty string")
    return rgb_image(data["image"]), rgb_image(data["wrist_image"]), state


def observation_request(obs, instruction):
    robot = np.asarray(obs["robot_obs"])
    if robot.ndim != 1 or robot.size < 7:
        raise ValueError("robot_obs must contain position, rotation and gripper state")
    data = {
        "image": rgb_image(obs["rgb_obs"]["rgb_static"]).tolist(),
        "wrist_image": rgb_image(obs["rgb_obs"]["rgb_gripper"]).tolist(),
        "state": np.concatenate((robot[:3], robot[3:6], robot[-1:])).tolist(),
        "prompt": instruction,
    }
    validate_request(data)
    return data


def validate_actions(value):
    actions = np.asarray(value, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 7:
        raise ValueError(f"Expected a nonempty [horizon, 7] action chunk; got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Action chunk contains non-finite values")
    return actions.copy()


class ClientModel:
    def __init__(self, host="127.0.0.1", port=9000, timeout=120):
        from websockets.sync.client import connect
        self.connection = connect(f"ws://{host}:{port}", max_size=100_000_000,
                                  open_timeout=timeout)
        self.timeout = timeout

    def reset(self):
        # Evo-1 inference is stateless; action queues belong to each subtask.
        pass

    def step(self, obs, instruction):
        self.connection.send(json.dumps(observation_request(obs, instruction), allow_nan=False))
        response = json.loads(self.connection.recv(timeout=self.timeout))
        if "error" in response:
            raise RuntimeError(f"Policy server failed: {response['error']}")
        return validate_actions(response["actions"])

    def close(self):
        self.connection.close()

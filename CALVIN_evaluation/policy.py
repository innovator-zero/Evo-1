"""Checkpoint loading and CALVIN input adaptation for Evo-1."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from CALVIN_evaluation.protocol import validate_actions, validate_request

EVO_ROOT = Path(__file__).resolve().parents[1] / "Evo_1"
sys.path.insert(0, str(EVO_ROOT))


def checkpoint_weights(directory):
    directory = Path(directory)
    for filename, key in (("mp_rank_00_model_states.pt", "module"),
                          ("checkpoint.pt", "model_state_dict")):
        path = directory / filename
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if key not in payload:
                raise ValueError(f"{path}: missing model key {key!r}")
            return payload[key]
    raise FileNotFoundError(f"{directory}: expected mp_rank_00_model_states.pt or checkpoint.pt")


def validate_stats(stats, arm, dataset, mode):
    try:
        group = stats[arm]
        selected = group if "observation.state" in group else group[dataset]
        keys = {"bounds": ("min", "max"), "bounds_q99": ("q01", "q99"),
                "normal": ("mean", "std")}[mode]
        for field in ("observation.state", "action"):
            values = [np.asarray(selected[field][key], dtype=np.float32) for key in keys]
            if any(v.shape != (7,) or not np.isfinite(v).all() for v in values):
                raise ValueError(f"{field}: expected seven finite values for {keys}")
            if mode != "normal" and np.any(values[1] < values[0]):
                raise ValueError(f"{field}: upper bound is less than lower bound")
            if mode == "normal" and np.any(values[1] < 0):
                raise ValueError(f"{field}: negative standard deviation")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid norm_stats.json for {arm}/{dataset} ({mode}): {exc}") from exc


class CalvinPolicy:
    def __init__(self, model, normalizer, config, arm="calvin_franka_delta",
                 dataset="InternData-Calvin_ABC"):
        self.model, self.normalizer, self.config = model, normalizer, config
        self.arm, self.dataset = arm, dataset
        self.device = torch.device(config.device)
        self.transform = T.Compose([
            T.Resize((config.image_size, config.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])

    @classmethod
    def load(cls, checkpoint, device="cuda:0", model_path=None, inference_steps=None,
             arm="calvin_franka_delta", dataset="InternData-Calvin_ABC"):
        # The existing normalizer is shared without changing other benchmark servers.
        from config import EvoConfig
        from scripts.Evo1 import EVO1
        from scripts.Evo1_server import Normalizer

        directory = Path(checkpoint).resolve()
        raw = json.loads((directory / "config.json").read_text())
        stats = json.loads((directory / "norm_stats.json").read_text())
        mode = raw.get("normalization_type", "bounds")
        validate_stats(stats, arm, dataset, mode)
        raw.update(device=device, enable_gradient_checkpointing=False,
                   finetune_vlm=False, finetune_language_model=False,
                   finetune_vision_model=False, finetune_action_head=False)
        if model_path:
            raw["vlm_name"] = model_path
        if inference_steps is not None:
            if inference_steps <= 0:
                raise ValueError("inference_steps must be positive")
            raw["num_inference_timesteps"] = inference_steps
        config = EvoConfig.from_dict(raw)
        if config.state_dim != 24 or config.per_action_dim != 24:
            raise ValueError("CALVIN adapter requires state_dim=per_action_dim=24")
        weights = checkpoint_weights(directory)
        model = EVO1(config)
        model.load_state_dict(weights, strict=True)
        model.eval().requires_grad_(False)
        return cls(model, Normalizer(stats, normalization_type=mode), config, arm, dataset)

    def prepare_inputs(self, data):
        static, wrist, state = validate_request(data)
        images = [self.transform(Image.fromarray(image)).to(self.device)
                  for image in (static, wrist)]
        images.append(torch.zeros_like(images[0]))
        state = torch.as_tensor(state, device=self.device).unsqueeze(0)
        state = self.normalizer.normalize_state(state, self.arm, self.dataset)
        # Training returns BF16 state from the dataset, then casts to FP32 in train.py.
        state = state.to(torch.bfloat16).float()
        return dict(images=images, image_mask=torch.tensor([1, 1, 0], device=self.device),
                    state_input=state, prompt=data["prompt"],
                    action_mask=torch.tensor([[1] * 7 + [0] * 17], device=self.device))

    @torch.inference_mode()
    def infer(self, data):
        inputs = self.prepare_inputs(data)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            action = self.model.run_inference(**inputs)
        horizon = self.config.action_horizon or self.config.horizon
        action = action.float().reshape(horizon, 24)
        action = self.normalizer.denormalize_action(action, self.arm, self.dataset)
        return {"actions": validate_actions(action[:, :7].cpu().numpy()).tolist()}

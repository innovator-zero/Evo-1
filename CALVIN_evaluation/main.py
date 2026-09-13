"""CALVIN ABC->D evaluation, adapted from better_openpi/examples/calvin/main.py."""
import argparse
import json
import logging
from pathlib import Path
import random
import sys

import numpy as np

from CALVIN_evaluation.evaluation import evaluate_policy
from CALVIN_evaluation.protocol import ClientModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calvin_root", type=Path, required=True)
    parser.add_argument("--dataset_path", type=Path, required=True, help="Official task_ABC_D directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--out_path", type=Path, default=Path("outputs/calvin"))
    parser.add_argument("--save_name", required=True)
    parser.add_argument("--num_sequences", type=int, default=1000)
    parser.add_argument("--ep_len", type=int, default=720)
    parser.add_argument("--execute_steps", type=int, help="Default: execute the full predicted chunk")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_videos", action="store_true")
    args = parser.parse_args()
    if args.num_sequences <= 0 or args.ep_len <= 0 or (args.execute_steps is not None and args.execute_steps <= 0):
        parser.error("Sequence count and step limits must be positive")
    root = args.calvin_root.resolve()
    validation = args.dataset_path.resolve() / "validation"
    conf = root / "calvin_models/conf"
    for path in (validation, conf / "callbacks/rollout/tasks/new_playtable_tasks.yaml",
                 conf / "annotations/new_playtable_validation.yaml"):
        if not path.exists():
            raise FileNotFoundError(path)
    sys.path[:0] = [str(root / "calvin_models"), str(root / "calvin_env")]
    # Delay simulator imports so --help and unit tests work in the model environment.
    import hydra
    from omegaconf import OmegaConf
    import torch
    from calvin_agent.evaluation.multistep_sequences import get_sequences
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
    from CALVIN_evaluation.calvin_env_wrapper import CalvinEnvWrapperRaw

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sequences = get_sequences(args.num_sequences)
    oracle = hydra.utils.instantiate(OmegaConf.load(conf / "callbacks/rollout/tasks/new_playtable_tasks.yaml"))
    annotations = OmegaConf.load(conf / "annotations/new_playtable_validation.yaml")
    env = model = None
    try:
        env = CalvinEnvWrapperRaw(validation)
        model = ClientModel(args.host, args.port)
        result = evaluate_policy(
            env, model, oracle, sequences, annotations, get_env_state_for_initial_condition,
            args.out_path / args.save_name, args.ep_len, args.execute_steps, args.save_videos,
            metadata={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        )
        print(json.dumps(result, indent=2))
    finally:
        try:
            if model is not None:
                model.close()
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    main()

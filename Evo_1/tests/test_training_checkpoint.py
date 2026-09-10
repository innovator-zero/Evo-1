from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from accelerate import DistributedType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train import (save_checkpoint, load_checkpoint_standard, get_lr_lambda,
                   prepare_dataloader, load_checkpoint_with_deepspeed, log_training_step)
from config import EvoConfig


class LocalAccelerator:
    distributed_type = DistributedType.NO
    is_main_process = True
    process_index = 0

    def wait_for_everyone(self):
        pass

    def unwrap_model(self, model):
        return model


class CheckpointTests(unittest.TestCase):
    def test_standard_checkpoint_allows_missing_keys(self):
        model = torch.nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "step_best"
            folder.mkdir()
            torch.save({"model_state_dict": {"weight": torch.ones_like(model.weight)}},
                       folder / "checkpoint.pt")
            load_checkpoint_standard(model, None, tmp, LocalAccelerator())
            self.assertTrue(torch.equal(model.weight, torch.ones_like(model.weight)))

    def test_deepspeed_retries_weights_only(self):
        engine = Mock()
        engine.load_checkpoint.side_effect = [RuntimeError("optimizer mismatch"),
                                              ("loaded", {"global_step": 12})]
        step, _ = load_checkpoint_with_deepspeed(engine, "checkpoint", LocalAccelerator())
        self.assertEqual(step, 12)
        self.assertTrue(engine.load_checkpoint.call_args_list[0].kwargs["load_optimizer_states"])
        self.assertFalse(engine.load_checkpoint.call_args.kwargs["load_optimizer_states"])
        self.assertTrue(engine.load_checkpoint.call_args.kwargs["load_module_only"])

    def test_unavailable_momentum_is_omitted_but_real_zero_is_logged(self):
        scheduler = Mock()
        scheduler.get_last_lr.return_value = [1e-5]
        with patch("train.wandb.log") as log, patch("train.SWANLAB_ENABLED", False):
            for momentum in (None, torch.tensor(0.0)):
                log_training_step(1, torch.tensor(1.0), 1.0, momentum,
                                  scheduler, [0], LocalAccelerator())
                payload = log.call_args.args[0]
                self.assertEqual("optimizer/momentum_norm" in payload, momentum is not None)

    def test_zero_workers_and_empty_batches(self):
        config = EvoConfig(batch_size=1, num_workers=0)
        loader = prepare_dataloader([{}], config)
        self.assertFalse(loader.persistent_workers)
        self.assertIsNone(loader.prefetch_factor)
        with self.assertRaisesRegex(ValueError, "No complete batches"):
            prepare_dataloader([], config)

    def test_final_tag_has_numeric_step_and_restorable_optimizer_schedule(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda(1, 5))
        for _ in range(2):
            optimizer.zero_grad()
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            scheduler.step()
        acc = LocalAccelerator()
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(tmp, 2, model, 1.0, acc, optimizer, scheduler, EvoConfig(), {}, tag="step_final")
            restored = torch.nn.Linear(2, 1)
            opt2 = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            step, payload = load_checkpoint_standard(restored, opt2, tmp, acc, tag="step_final")
            self.assertEqual(step, 2)
            self.assertEqual(payload["global_step"], 2)
            self.assertTrue(torch.equal(restored.weight, model.weight))
            self.assertEqual(opt2.state_dict()["state"][0]["step"], 2)
            sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, get_lr_lambda(1, 5))
            sched2.load_state_dict(payload["scheduler_state_dict"])
            for group, lr in zip(opt2.param_groups, sched2.get_last_lr()):
                group["lr"] = lr
            self.assertEqual(optimizer.param_groups[0]["lr"], opt2.param_groups[0]["lr"])
            optimizer.step(); scheduler.step()
            opt2.step(); sched2.step()
            self.assertEqual(scheduler.get_last_lr(), sched2.get_last_lr())
            # Stage transition loads weights, leaving the new optimizer empty.
            fresh_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            load_checkpoint_standard(restored, fresh_opt, tmp, acc, tag="step_final", load_optimizer_states=False)
            self.assertFalse(fresh_opt.state)


if __name__ == "__main__":
    unittest.main()

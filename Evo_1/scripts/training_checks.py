"""Small, opt-in update and rank-consistency checks for real training smoke tests."""
import json
from pathlib import Path

import torch


class UpdateCheck:
    def __init__(self, model, accelerator):
        self.accelerator = accelerator
        self.groups = {}
        self.handles = []
        for name, module in (("vision", model.embedder.model.vision_model),
                             ("language", model.embedder.model.language_model),
                             ("action_head", model.action_head)):
            # Exclude embedding tables: most vocabulary entries need not change.
            candidates = [(n, p) for n, p in module.named_parameters()
                          if p.ndim > 1 and "embed_tokens" not in n][:4]
            group = {"trainable": any(p.requires_grad for p in module.parameters()),
                     "samples": [], "grad_max": 0.0}
            for param_name, p in candidates:
                stride = max(1, p.numel() // 4096)
                group["samples"].append((param_name, p, stride, p.detach().flatten()[::stride].float().clone()))
                if p.requires_grad:
                    def hook(grad, group=group):
                        if not torch.isfinite(grad).all():
                            raise FloatingPointError("Non-finite gradient in smoke test")
                        group["grad_max"] = max(group["grad_max"], grad.detach().float().abs().max().item())
                    self.handles.append(p.register_hook(hook))
            self.groups[name] = group

    def finish(self, save_dir):
        report = {}
        passed = True
        for name, group in self.groups.items():
            max_change = 0.0
            consistent = True
            for _, p, stride, before in group["samples"]:
                after = p.detach().flatten()[::stride].float().contiguous()
                max_change = max(max_change, (after - before).abs().max().item())
                gathered = self.accelerator.gather(after.unsqueeze(0))
                consistent &= bool(torch.allclose(gathered, gathered[0].expand_as(gathered), atol=1e-6, rtol=1e-5))
            updated = max_change > 0
            ok = consistent and (updated and group["grad_max"] > 0 if group["trainable"] else not updated)
            passed &= ok
            report[name] = {"trainable": group["trainable"], "sampled_max_change": max_change,
                            "sampled_grad_max": group["grad_max"], "ranks_consistent": consistent, "passed": ok}
        for handle in self.handles:
            handle.remove()
        if self.accelerator.is_main_process:
            Path(save_dir, "update_check.json").write_text(json.dumps(report, indent=2))
        if not passed:
            raise AssertionError(f"Parameter update/rank checks failed: {report}")

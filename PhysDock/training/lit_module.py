"""LightningModule: training with PhysDockLoss, validation = diffusion loss + sampling metrics, fused AdamW + AF3 LR schedule.

Logged (Boltz style): train/<loss terms> and train/lr every step; every log_every_n_steps the gradient statistics
(train/grad_norm, grad_abs_max, grad_abs_min, param_norm, per module train/grad_norm/<module>, and with
grad_stats="parameter" grad/<parameter>/{norm,abs_max,abs_min}); per validation the means over all validation
systems and DDP ranks of val/loss, val/loss_<term> and the sampling metrics val/<metric> (training/metrics.py).
"""
import lightning.pytorch as L
import torch
from torchmetrics import MeanMetric

from PhysDock.data import Sample
from PhysDock.models.loss import PhysDockLoss
from PhysDock.models.model import PhysDock
from PhysDock.training.metrics import sample_metrics
from PhysDock.utils.import_weights import load_checkpoint_state_dict

SAMPLE_METRICS = ("lrmsd", "success", "best_success", "ligand_mse", "slddt", "within_slddt", "ligand_bond_error",
                  "inter_clash")


def af3_lr_lambda(warmup_steps: int, start_decay_after: int, decay_every: int, decay_factor: float):
    """LR factor: linear warmup, plateau, then decay_factor every decay_every steps (AF3 / PhysDock2 schedule)."""
    def factor(step: int) -> float:
        if step <= warmup_steps:
            return step / max(1, warmup_steps)
        if step > start_decay_after:
            return decay_factor ** ((step - start_decay_after) // decay_every + 1)
        return 1.0
    return factor


def _module_of(name: str) -> str:
    """Module a parameter is reported under: e.g. dit.token_dit, diffusion_conditioning.token_embedder.pairformer."""
    parts = name.split(".")
    depth = 3 if parts[:2] == ["diffusion_conditioning", "token_embedder"] else 2
    return ".".join(parts[:min(depth, len(parts) - 1)])


class PhysDockLitModule(L.LightningModule):
    """
    Args:
        model: model config (configs/model/<size>.yaml); with a `pomelo` entry (configs/model/pomelo_mini.yaml) the network is Pomelo (PhysDock.pomelo),
            the same interface, the rest of the config (sigma_data, diffusion, num_augmentation_sample) used as for PhysDock
        loss: loss config (configs/model/loss.yaml)
        optim: {lr, betas, eps, weight_decay, scheduler: {warmup_steps, start_decay_after, decay_every, decay_factor}}
        init_from: optional checkpoint to start from (params.pt, Lightning or Uni-Core checkpoint)
        val: {num_samples, steps} of the validation sampling
        grad_stats: "total", "module" or "parameter" (see module docstring)
    """

    def __init__(self, model: dict, loss: dict, optim: dict, init_from: str = None, val: dict = None,
                 grad_stats: str = "parameter") -> None:
        super().__init__()
        if grad_stats not in ("total", "module", "parameter"):
            raise ValueError(f"grad_stats must be total, module or parameter, got {grad_stats!r}")
        self.save_hyperparameters(logger=False)         # the full run config is logged by scripts/train/train.py
        if model.get("pomelo"):
            from PhysDock.pomelo import Pomelo
            self.model = Pomelo(**model)
        elif model.get("singletrack"):
            from PhysDock.singletrack import SingleTrack
            self.model = SingleTrack(**model)
        else:
            self.model = PhysDock(**model)
        if init_from:
            self.model.load_state_dict(load_checkpoint_state_dict(init_from, mmap=False))
        self.loss_fn = PhysDockLoss(**loss)
        self.optim = optim
        self.val = val or {"num_samples": 5, "steps": 20}
        self.grad_stats = grad_stats
        self.skipped_steps = 0                  # updates skipped because of a non-finite gradient
        # MeanMetric is synced over the DDP ranks at epoch end, also for ranks with no (or skipped) systems
        names = ["loss"] + [f"loss_{k}" for k in self.loss_fn.weights] + list(SAMPLE_METRICS)
        self.val_metrics = torch.nn.ModuleDict({k: MeanMetric() for k in names})

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if batch is None:
            return None
        if isinstance(batch, Sample):           # validation: features + metadata
            return Sample({k: v.to(device) for k, v in batch.features.items()}, batch.meta)
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}      # pinned by the train loader

    # ------------------------------------------------------------------ training
    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)
        loss, logs = self.loss_fn(outputs, batch)
        self.log("train/loss", logs.pop("loss"), prog_bar=True, batch_size=1)
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, batch_size=1)
        self.log_dict({f"train/rms_{n}": v for n, v in zip("asz", outputs["stream_rms"])}, batch_size=1)      # scale of the encoder streams
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], batch_size=1)
        return loss

    @torch.no_grad()
    def on_before_optimizer_step(self, optimizer) -> None:
        """Gradient statistics (before clipping) every log_every_n_steps, logged by the global-zero rank."""
        if self.logger is None or self.global_step % self.trainer.log_every_n_steps != 0:
            return
        named = [(n, p) for n, p in self.model.named_parameters() if p.grad is not None]
        if not named:
            return
        grads = [p.grad.detach() for _, p in named]
        flat = torch.cat([g.reshape(-1) for g in grads]).abs_()
        abs_min = torch.segment_reduce(flat, "min", lengths=torch.tensor([g.numel() for g in grads], device=flat.device))
        stats = torch.stack([
            torch.stack(torch._foreach_norm(grads)),
            torch.stack(torch._foreach_norm(grads, float("inf"))),          # abs max
            abs_min,
            torch.stack(torch._foreach_norm([p.detach() for _, p in named])),
        ]).float().cpu()
        g_norm, g_max, g_min, p_norm = stats
        metrics = {"train/grad_norm": g_norm.norm().item(), "train/grad_abs_max": g_max.max().item(),
                   "train/grad_abs_min": g_min.min().item(), "train/param_norm": p_norm.norm().item()}
        if self.grad_stats in ("module", "parameter"):
            sq = {}
            for (name, _), v in zip(named, g_norm.tolist()):
                sq[_module_of(name)] = sq.get(_module_of(name), 0.) + v * v
            metrics.update({f"train/grad_norm/{m}": s ** 0.5 for m, s in sq.items()})
        if self.grad_stats == "parameter":
            for (name, _), n, mx, mn in zip(named, g_norm.tolist(), g_max.tolist(), g_min.tolist()):
                metrics[f"grad/{name}/norm"], metrics[f"grad/{name}/abs_max"], metrics[f"grad/{name}/abs_min"] = n, mx, mn
        self.logger.log_metrics(metrics, step=self.global_step)

    # ------------------------------------------------------------------ validation
    def validation_step(self, batch, batch_idx):
        if batch is None:                       # system that could not be featurised (on_error="skip")
            return
        f = batch.features
        _, logs = self.loss_fn(self.model.forward(f), f)        # .forward: eager even if the model is compiled
        for k, v in logs.items():
            self.val_metrics["loss" if k == "loss" else f"loss_{k}"].update(v)
        x = self.model.sample_diffusion(f, num_sample=self.val["num_samples"], steps=self.val["steps"],
                                        align_ref_pos=False)
        for k, v in sample_metrics(x, f).items():
            self.val_metrics[k].update(v)

    def on_validation_epoch_end(self) -> None:
        for name, metric in self.val_metrics.items():
            value = metric.compute()            # synced over the ranks
            metric.reset()
            if torch.isfinite(value):           # e.g. inter_clash stays empty if every system has one chain
                self.log(f"val/{name}", value, prog_bar=name in ("loss", "lrmsd"))

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None) -> None:
        """Gradient norm clipping (trainer.gradient_clip_val) without Lightning's precision plugin, which refuses fused
        optimizers under mixed precision. With fp32 / bf16 the gradients are not loss-scaled; with 16-mixed the optimizer is not
        fused (configure_optimizers), so Lightning has unscaled the gradients before this hook."""
        if not gradient_clip_val:
            return
        if self.trainer.precision == "16-true":
            raise ValueError("16-true is not supported, use precision 32, bf16-mixed or 16-mixed")
        if gradient_clip_algorithm not in (None, "norm"):
            raise ValueError(f"only norm clipping is supported, got {gradient_clip_algorithm!r}")
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_val)
        if not torch.isfinite(norm):          # a bad batch: skip the update (Adam leaves parameters without a gradient alone) instead of poisoning the weights
            optimizer.zero_grad(set_to_none=True)
            if self.trainer.precision == "16-mixed":          # an overflow of the loss scale: the GradScaler skips the step and lowers the scale
                return
            self.skipped_steps += 1
            print(f"[train] step {self.global_step}: non-finite gradient norm, update skipped ({self.skipped_steps} so far)", flush=True)
        self.log("train/skipped_steps", float(self.skipped_steps), batch_size=1)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        decay = self.optim.get("weight_decay", 0.0)
        groups = [{"params": [p for p in params if p.ndim >= 2], "weight_decay": decay},        # weight matrices
                  {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}]           # biases, norm scales, other vectors
        optimizer = torch.optim.AdamW(groups, lr=self.optim["lr"], betas=tuple(self.optim["betas"]), eps=self.optim["eps"],
                                      fused=self.device.type == "cuda" and self.trainer.precision != "16-mixed")    # 16-mixed: Lightning unscales before clipping
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, af3_lr_lambda(**self.optim["scheduler"]))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

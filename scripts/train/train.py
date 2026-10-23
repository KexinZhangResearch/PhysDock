#!/usr/bin/env python3
"""Train PhysDock with Lightning (config: configs/train.yaml; logs to wandb, run `wandb login` once).

    python scripts/train/train.py                                   # medium size (the current structure), 1 GPU
    python scripts/train/train.py trainer=ddp                       # all GPUs of the node
    python scripts/train/train.py model=toy data.samples_per_epoch=100 trainer.max_steps=200
    python scripts/train/train.py model=released init_from=data/params/params.pt   # fine-tune the released weights (their structure)
    python scripts/train/train.py hydra.run.dir=logs/train/<run>   # a fixed run directory: started again, the run resumes from its newest checkpoint
    python scripts/train/train.py ckpt_path=<file>.ckpt             # resume this checkpoint
    python scripts/train/train.py compile=true logger.offline=true  # torch.compile, wandb offline (wandb sync later)
    python scripts/train/train.py model=pomelo_mini                 # the Pomelo network (PhysDock/pomelo) in place of the encoder and decoder
(scripts/train/launch.sh starts it under torchrun, on one or several nodes, and restarts it after a crash)
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PHYSDOCK_ROOT", str(ROOT))

import hydra  # noqa: E402
import torch  # noqa: E402
import lightning.pytorch as L  # noqa: E402
from lightning.pytorch.callbacks import TQDMProgressBar  # noqa: E402
from lightning.pytorch.loggers import WandbLogger  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from PhysDock.configs import to_dict  # noqa: E402
from PhysDock.models.primitives import set_checkpointing  # noqa: E402
from PhysDock.utils.precision import set_matmul_precision  # noqa: E402
from PhysDock.training import EMA, PhysDockDataModule, PhysDockLitModule, compile_training  # noqa: E402
from PhysDock.training.compile import _compile_method  # noqa: E402
from PhysDock.training.resume import NewestCheckpoint, latest_checkpoint, wandb_online, wandb_run_id  # noqa: E402


class TextLog(L.Callback):
    """every `every` steps one line: the means of x_mse and smooth lDDT, the gradient norm before clipping, the time per step"""

    def __init__(self, every: int = 100) -> None:
        self.every, self.rows = every, []

    def on_train_start(self, trainer, module) -> None:
        self.t0, self.start = time.time(), trainer.global_step

    def on_before_optimizer_step(self, trainer, module, optimizer) -> None:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        self.gnorm = float(torch.stack(torch._foreach_norm(grads)).norm()) if grads else float("nan")

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx) -> None:
        m = trainer.callback_metrics
        self.rows.append((float(m.get("train/x_mse", float("nan"))), float(m.get("train/smooth_lddt", float("nan"))), self.gnorm))
        step = trainer.global_step
        if self.every and step % self.every == 0 and trainer.is_global_zero:
            h = torch.tensor(self.rows[-self.every:]).mean(0)
            print(f"step {step} x_mse {h[0]:.4f} slddt {h[1]:.4f} gnorm(pre-clip) {h[2]:.1f}  {(time.time() - self.t0) / max(1, step - self.start):.2f} s/step", flush=True)


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_matmul_precision(cfg.matmul_precision)
    L.seed_everything(cfg.seed)
    optim = to_dict(cfg.optim)
    batch = cfg.trainer.num_nodes * (cfg.trainer.devices if isinstance(cfg.trainer.devices, int) and cfg.trainer.devices > 0 else
                                     max(1, torch.cuda.device_count())) * cfg.trainer.accumulate_grad_batches
    if optim["lr"] is None:                       # the learning rate of this batch size from the reference (batch, lr) of AlphaFold 3
        optim["lr"] = optim["reference_lr"] * (batch / optim["reference_batch"]) ** 0.5
    print(f"[train] {batch} structure(s) per optimiser step: lr {optim['lr']:.3e} (reference {optim['reference_lr']:.1e} at batch "
          f"{optim['reference_batch']}), weight decay {optim['weight_decay']}", flush=True)
    cfg.optim.lr = optim["lr"]
    datamodule = PhysDockDataModule(features=to_dict(cfg.features), val_features=to_dict(cfg.val_features),
                                    **to_dict(cfg.data))
    module = PhysDockLitModule(model=to_dict(cfg.model), loss=to_dict(cfg.loss), optim=optim,
                               init_from=cfg.init_from, val=to_dict(cfg.val), grad_stats=cfg.grad_stats)
    if cfg.model.get("pomelo"):                                  # Pomelo: its own blocks (no Stack), compiled per block; no activation checkpointing
        if cfg.trainer.precision in ("16-mixed", "bf16-mixed"):
            module.model.set_attention_backend("efficient", "efficient")      # half precision: the memory-efficient SDPA kernel (no [N, N] logits)
        module.model.prepare_training(compile=cfg.compile)
        if cfg.compile:
            _compile_method(module.loss_fn, "terms")
    else:
        if cfg.trainer.precision == "16-mixed":
            module.model.dit.half = True                         # float16: the whole network under autocast (the decoder too), the loss in float32
        set_checkpointing(module.model, to_dict(cfg.checkpointing))
        if cfg.compile:
            compile_training(module.model, module.loss_fn)      # region by region, memory as eager
    print(f"[train] {type(module.model).__name__}: {sum(p.numel() for p in module.model.parameters()) / 1e6:.1f}M parameters", flush=True)

    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
    ckpt_path = latest_checkpoint(ckpt_dir) if cfg.ckpt_path == "auto" else cfg.ckpt_path
    print(f"[train] output_dir {cfg.output_dir}: " + (f"resuming {ckpt_path}" if ckpt_path else "no checkpoint, starting from scratch"), flush=True)

    logger = False
    if cfg.logger is not None:
        kwargs = to_dict(cfg.logger)
        if kwargs.get("offline") == "auto":                      # no API key (wandb login / WANDB_API_KEY): log locally, wandb sync later
            kwargs["offline"] = not wandb_online()
        if kwargs.get("id") is None:                             # the same wandb run over restarts of this output directory
            kwargs["id"] = wandb_run_id(cfg.output_dir)
        kwargs.setdefault("resume", "allow")
        logger = WandbLogger(save_dir=cfg.output_dir, **kwargs)
        logger.log_hyperparams(to_dict(cfg))
    callbacks = [EMA(**to_dict(cfg.ema)), TQDMProgressBar(refresh_rate=cfg.progress_refresh), TextLog(cfg.get("text_log_every", 100)),
                 NewestCheckpoint(dirpath=ckpt_dir, filename="step_{step}", auto_insert_metric_name=False, **to_dict(cfg.checkpoint))]
    trainer = L.Trainer(default_root_dir=cfg.output_dir, callbacks=callbacks, logger=logger, **to_dict(cfg.trainer))
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()

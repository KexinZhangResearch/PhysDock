#!/usr/bin/env python3
"""Train PhysDock with Lightning (config: configs/train.yaml; logs to wandb, run `wandb login` once).

    python scripts/train/train.py                                   # released model size, 1 GPU
    python scripts/train/train.py trainer=ddp                       # all GPUs of the node
    python scripts/train/train.py model=toy data.samples_per_epoch=100 trainer.max_steps=200
    python scripts/train/train.py init_from=data/params/params.pt   # fine-tune the released weights
    python scripts/train/train.py ckpt_path=logs/train/<run>/checkpoints/last.ckpt
    python scripts/train/train.py compile=true logger.offline=true  # torch.compile, wandb offline (wandb sync later)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PHYSDOCK_ROOT", str(ROOT))

import hydra  # noqa: E402
import lightning.pytorch as L  # noqa: E402
import torch  # noqa: E402
from lightning.pytorch.callbacks import ModelCheckpoint  # noqa: E402
from lightning.pytorch.loggers import WandbLogger  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from PhysDock.configs import to_dict  # noqa: E402
from PhysDock.models.primitives import compile_blocks  # noqa: E402
from PhysDock.utils.precision import set_matmul_precision  # noqa: E402
from PhysDock.training import EMA, PhysDockDataModule, PhysDockLitModule  # noqa: E402


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_matmul_precision(cfg.matmul_precision)
    L.seed_everything(cfg.seed)
    datamodule = PhysDockDataModule(features=to_dict(cfg.features), val_features=to_dict(cfg.val_features),
                                    **to_dict(cfg.data))
    module = PhysDockLitModule(model=to_dict(cfg.model), loss=to_dict(cfg.loss), optim=to_dict(cfg.optim),
                               init_from=cfg.init_from, val=to_dict(cfg.val), grad_stats=cfg.grad_stats)
    if cfg.compile not in (False, True, "full"):
        raise ValueError(f"compile must be false, true (every block) or full, got {cfg.compile!r}")
    if cfg.compile == "full":
        # compiles model.__call__ (the training forward); validation calls model.forward / sampling eagerly,
        # parameter names and checkpoints are unchanged. Dynamo's DDP graph splitting does not support the
        # activation checkpointing of the blocks (torch 2.4), so DDP syncs one bucket for the whole graph.
        torch._dynamo.config.optimize_ddp = False
        module.model.compile()
    elif cfg.compile:
        compile_blocks(module.model)                # every block, memory as eager

    logger = False
    if cfg.logger is not None:
        logger = WandbLogger(save_dir=cfg.output_dir, **to_dict(cfg.logger))
        logger.log_hyperparams(to_dict(cfg))
    callbacks = [EMA(**to_dict(cfg.ema)),
                 ModelCheckpoint(dirpath=os.path.join(cfg.output_dir, "checkpoints"), filename="step_{step}",
                                 auto_insert_metric_name=False, **to_dict(cfg.checkpoint))]
    trainer = L.Trainer(default_root_dir=cfg.output_dir, callbacks=callbacks, logger=logger, **to_dict(cfg.trainer))
    trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()

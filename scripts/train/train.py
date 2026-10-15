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
from lightning.pytorch.callbacks import ModelCheckpoint  # noqa: E402
from lightning.pytorch.loggers import WandbLogger  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from PhysDock.configs import to_dict  # noqa: E402
from PhysDock.models.primitives import set_checkpointing  # noqa: E402
from PhysDock.utils.precision import set_matmul_precision  # noqa: E402
from PhysDock.training import EMA, PhysDockDataModule, PhysDockLitModule, compile_training  # noqa: E402


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_matmul_precision(cfg.matmul_precision)
    L.seed_everything(cfg.seed)
    datamodule = PhysDockDataModule(features=to_dict(cfg.features), val_features=to_dict(cfg.val_features),
                                    **to_dict(cfg.data))
    module = PhysDockLitModule(model=to_dict(cfg.model), loss=to_dict(cfg.loss), optim=to_dict(cfg.optim),
                               init_from=cfg.init_from, val=to_dict(cfg.val), grad_stats=cfg.grad_stats)
    set_checkpointing(module.model, to_dict(cfg.checkpointing))
    if cfg.compile:
        compile_training(module.model, module.loss_fn)          # region by region, memory as eager

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

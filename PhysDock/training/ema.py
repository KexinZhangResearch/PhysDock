"""Exponential moving average of the weights as a Lightning callback (after Boltz / NVIDIA NeMo).

The EMA weights are updated after every optimizer step, used for validation / testing / prediction, and saved in
the checkpoint under "ema" (PhysDock.utils.import_weights.load_checkpoint_state_dict reads them for inference).
"""
import lightning.pytorch as L
import torch


class EMA(L.Callback):
    """ema = decay * ema + (1 - decay) * weights after every optimizer step.

    Args:
        decay: EMA decay
        warm_start: use min(decay, (1 + step) / (10 + step)) early in training
        eval_with_ema: validate / test with the EMA weights
    """

    def __init__(self, decay: float = 0.999, warm_start: bool = True, eval_with_ema: bool = True) -> None:
        if not 0 <= decay <= 1:
            raise ValueError(f"EMA decay must be in [0, 1], got {decay}")
        self.decay, self.warm_start, self.eval_with_ema = decay, warm_start, eval_with_ema
        self.weights = None             # {name: tensor} of the LightningModule state dict
        self.step = None                # global step of the last update
        self._backup = None             # training weights while the EMA weights are swapped in

    def on_train_start(self, trainer, pl_module) -> None:
        if self.weights is None:
            self.weights = {k: v.detach().clone() for k, v in pl_module.state_dict().items()}
        self.weights = {k: v.to(pl_module.device) for k, v in self.weights.items()}

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = trainer.global_step
        if step == self.step:           # no optimizer step yet (gradient accumulation)
            return
        self.step = step
        decay = min(self.decay, (1 + step) / (10 + step)) if self.warm_start else self.decay
        for k, w in pl_module.state_dict().items():
            if w.dtype.is_floating_point:
                self.weights[k].lerp_(w, 1 - decay)

    # ---------------------------------------------------------------- evaluation with the EMA weights
    def _swap_in(self, pl_module) -> None:
        if self.weights is not None and self.eval_with_ema:
            self._backup = {k: v.detach().clone() for k, v in pl_module.state_dict().items()}
            pl_module.load_state_dict(self.weights, strict=False)

    def _swap_out(self, pl_module) -> None:
        if self._backup is not None:
            pl_module.load_state_dict(self._backup, strict=False)
            self._backup = None

    def on_validation_start(self, trainer, pl_module):
        self._swap_in(pl_module)

    def on_validation_end(self, trainer, pl_module):
        self._swap_out(pl_module)

    def on_test_start(self, trainer, pl_module):
        self._swap_in(pl_module)

    def on_test_end(self, trainer, pl_module):
        self._swap_out(pl_module)

    # ---------------------------------------------------------------- checkpoint
    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        if self.weights is not None:
            checkpoint["ema"] = {"weights": self.weights, "step": self.step, "decay": self.decay}
        if self._backup is not None:    # saved during evaluation: keep the training weights as the state dict
            checkpoint["state_dict"] = self._backup

    def on_load_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        if "ema" in checkpoint:
            self.weights, self.step = checkpoint["ema"]["weights"], checkpoint["ema"]["step"]

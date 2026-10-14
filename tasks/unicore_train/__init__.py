"""Uni-Core reference training task (task "stfold", loss "stfoldloss", arch "stfold") on the released data/v1 set."""
import logging
import os
import sys

import torch
from unicore import metrics
from unicore.data import UnicoreDataset
from unicore.losses import UnicoreLoss, register_loss
from unicore.models import BaseUnicoreModel, build_model, register_model, register_model_architecture
from unicore.tasks import UnicoreTask, register_task

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PhysDock import PhysDock, PhysDockConfig, PhysDockLoss  # noqa: E402
from PhysDock.data import FeatureConfig, PhysDockDataset  # noqa: E402
from PhysDock.utils.import_weights import import_unicore_ckpt  # noqa: E402

logger = logging.getLogger(__name__)


class STFoldUnicoreDataset(UnicoreDataset):
    """Every item is a weighted random draw from data/v1 (the index is ignored); failed systems are resampled."""

    def __init__(self, args) -> None:
        super().__init__()
        config = FeatureConfig(crop_size=args.crop_size, atom_crop_size=8 * args.crop_size,
                               max_msa_clusters=args.max_msa_clusters)
        self.dataset = PhysDockDataset.from_train_val(config=config, training=True, on_error="resample")
        weights = self.dataset.sample_weights
        self.probabilities = weights / weights.sum() if weights is not None else torch.ones(len(self.dataset), dtype=torch.double)
        self.final_len = 256 * 120000

    def __getitem__(self, item):
        return self.dataset[torch.multinomial(self.probabilities, 1).item()].features

    def __len__(self):
        return self.final_len

    @staticmethod
    def collater(samples):
        return samples[0]       # batch size 1


@register_model("stfold")
class STFoldUnicoreModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        pass

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.config = PhysDockConfig(model_name=args.model_name, num_augmentation_sample=args.num_augmentation_sample)
        self.model = PhysDock(self.config)
        self.loss_fn = PhysDockLoss(self.config)
        if args.init_from_ckpt != "none":
            import_unicore_ckpt(self.model, args.init_from_ckpt)
        if args.compile:
            self.model.compile()
            self.loss_fn.compile()

    def half(self):
        return self

    def bfloat16(self):
        return self

    @classmethod
    def build_model(cls, args, task):
        return cls(args)

    def forward(self, batch, **kwargs):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16 if self.args.use_bf16 else torch.float32):
            outputs = self.model(batch)
        return self.loss_fn(outputs, batch)


@register_model_architecture("stfold", "stfold")
def base_architecture(args):
    pass


@register_loss("stfoldloss")
class STFoldUnicoreLoss(UnicoreLoss):
    def forward(self, model, batch):
        loss, logs = model(batch)
        logs["sample_size"] = 1
        return loss, 1, logs

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        return True

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("loss", sum(log.get("loss", 0) for log in logging_outputs) / sample_size, sample_size, round=4)
        for key in logging_outputs[0]:
            if key not in ("sample_size", "bsz"):
                value = sum(log.get(key, 0) for log in logging_outputs) / sample_size
                metrics.log_scalar(key, value, sample_size, round=4)


@register_task("stfold")
class STFoldTask(UnicoreTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--model-name", type=str, default="full", help="toy / tiny / small / medium / full")
        parser.add_argument("--crop-size", type=int, default=256, help="tokens per crop (atoms: 8 x crop size)")
        parser.add_argument("--max-msa-clusters", type=int, default=128)
        parser.add_argument("--num-augmentation-sample", type=int, default=48)
        parser.add_argument("--use-bf16", action="store_true")
        parser.add_argument("--compile", action="store_true")
        parser.add_argument("--init-from-ckpt", type=str, default="none")
        parser.add_argument("--num-processors", type=int, default=1)
        parser.add_argument("--matmul-precision", type=str, default="high")

    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = args
        self.seed = args.seed
        torch.set_float32_matmul_precision(args.matmul_precision)

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def load_dataset(self, split, combine=False, **kwargs):
        self.datasets[split] = STFoldUnicoreDataset(self.args)

    def build_model(self, args):
        return build_model(args, self)

    def disable_shuffling(self) -> bool:
        return True

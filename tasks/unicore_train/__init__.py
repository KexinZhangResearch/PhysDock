import random
import sys
import os
import numpy as np
import torch
import logging
from typing import List, Dict, Union

logger = logging.getLogger(__name__)

from unicore.data import UnicoreDataset
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture, build_model
from unicore.tasks import UnicoreTask, register_task
from unicore.losses import UnicoreLoss, register_loss
from unicore import metrics

stfold_path = os.path.split(os.path.split(os.path.split(__file__)[0])[0])[0]
sys.path.append(stfold_path)

from PhysDock.configs_old import model_config
from PhysDock.models.model import PhysDock as RFFold
from PhysDock.data.feature_loader_plinder import FeatureLoader
from PhysDock.models.loss_module3 import RFFoldLoss
from PhysDock.utils.import_weights import import_unicore_ckpt


class STFoldUnicoreDataset(UnicoreDataset):
    def __init__(
            self,
            config,
            mode="train",
            num_processors=1,
    ):
        super().__init__()
        self.config = config
        self.mode = mode
        self.num_processors = num_processors

        self.feature_loader = FeatureLoader(
            atom_crop_size=256 * 8,
            token_crop_size=256,
            inference_mode=False
        )

        self.final_len = int(256 * 120000)

    def __getitem__(self, item):
        while True:
            try:
                tensors, infer_meta_data = self.feature_loader.weighted_random_load()
                break
            except Exception as e:
                # print(sample_id)
                logging.warning(e)
                continue
        return tensors

    def __len__(self):
        return self.final_len

    @staticmethod
    def collater(samples):
        # Only this first sample of batch is used in AlphaFold3
        # feats = tensor_tree_map(lambda t: t[..., 0], samples[0])
        return samples[0]


@register_model("stfold")
class STFoldUnicoreModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # parser.add_argument(
        #     "--model-name",
        #     help="choose the model config",
        # )

    def __init__(
            self,
            args,
    ):
        super().__init__()
        base_architecture(args)
        self.args = args

        self.config = model_config(
            model_name=self.args.model_name,
            crop_size=self.args.crop_size,
            max_recycling_iters=self.args.max_recycling_iters,
            max_msa_clusters=self.args.max_msa_clusters,
            num_augmentation_sample=self.args.num_augmentation_sample,
            alpha_confifdence=1e-4,
            alpha_diffusion=4,
            alpha_bond=self.args.alpha_bond,
            alpha_distogram=3e-2,
            alpha_pae=self.args.alpha_pae,
            use_template=self.args.use_template,
            use_mini_rollout=self.args.use_mini_rollout,
            use_flash_attn=self.args.use_flash_attn,
            custom_rel_token=42,
            ref_dim=167,
            mini_rollout_steps=self.args.mini_rollout_steps,
            atom_attention_type=self.args.atom_attention_type,
            interaction_aware=self.args.interaction_aware,
            templ_dim=self.args.templ_dim,
        )
        self.config.data.crop_size = args.crop_size

        ###############################
        self.model = RFFold(self.config)
        self.loss_fn = RFFoldLoss(self.config)
        if args.init_from_ckpt != "none":
            import_unicore_ckpt(self.model, args.init_from_ckpt)
        if self.args.compile:
            self.model.compile()
            self.loss_fn.compile()

    def half(self):
        # self.model = self.model.half()
        return self

    def bfloat16(self):
        # self.model = self.model.bfloat16()
        return self

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args)

    # The Training Step
    def forward(self, batch, **kwargs):
        # dtype = torch.float32
        # if self.args.fp16:
        #     dtype = torch.float16
        # elif self.args.bf16:
        #     dtype = torch.bfloat16
        if self.args.use_bf16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        with torch.amp.autocast("cuda", dtype=dtype):
            outputs = self.model(batch)
        loss, loss_dict = self.loss_fn(outputs, batch)
        # loss = loss3
        return loss, loss_dict


@register_model_architecture("stfold", "stfold")
def base_architecture(args):
    pass


@register_loss("stfoldloss")
class STFoldUnicoreLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, batch):

        loss, loss_breakdown = model(batch)
        logging_output = loss_breakdown
        sample_size = 1
        logging_output["sample_size"] = sample_size
        return loss, sample_size, logging_output

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # print(logging_outputs)
        # print(loss_sum,sample_size)
        # input()
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=4)
        # input()
        for key in logging_outputs[0]:
            if key in ["sample_size", "bsz"]:
                continue
            loss_sum = sum(log.get(key, 0) for log in logging_outputs)
            metrics.log_scalar(key, loss_sum / sample_size, sample_size, round=4)


@register_task("stfold")
class STFoldTask(UnicoreTask):

    @staticmethod
    def add_args(parser):
        #####################################################
        #                   Add Config Args                 #
        #####################################################

        parser.add_argument(
            "--model-name",
            help="choose the model config",
            type=str,
            default="full",
        )
        parser.add_argument(
            "--crop-size",
            type=int,
            default=256,
        )
        parser.add_argument(
            "--max-recycling-iters",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--max-msa-clusters",
            type=int,
            default=128,
        )
        parser.add_argument(
            "--templ-dim",
            type=int,
            default=108,
        )
        parser.add_argument(
            "--num-augmentation-sample",
            type=int,
            default=48,
        )
        parser.add_argument(
            "--alpha-bond",
            type=int,
            default=0,
        )
        parser.add_argument(
            "--alpha-pae",
            type=int,
            default=1,
        )
        parser.add_argument("--use-template", action="store_true")
        parser.add_argument("--use-bf16", action="store_true")
        parser.add_argument("--use-mini-rollout", action="store_true")
        parser.add_argument(
            "--mini-rollout-steps",
            type=int,
            default=20,
        )
        parser.add_argument("--use-flash-attn", action="store_true")
        parser.add_argument("--interaction-aware", action="store_true")
        parser.add_argument("--compile", action="store_true")

        parser.add_argument(
            "--init-from-ckpt",
            type=str,
            default="none",
        )
        parser.add_argument(
            "--num-processors",
            type=int,
            default=1,
        )

        parser.add_argument(
            "--matmul-precision",
            type=str,
            default="high",
        )
        parser.add_argument(
            "--atom-attention-type",
            type=str,
            default="sequence",
        )

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.seed = args.seed

        torch.set_float32_matmul_precision(self.args.matmul_precision)

        self.config = model_config(
            model_name=self.args.model_name,
            crop_size=self.args.crop_size,
            max_recycling_iters=self.args.max_recycling_iters,
            max_msa_clusters=self.args.max_msa_clusters,
            num_augmentation_sample=self.args.num_augmentation_sample,
            alpha_confifdence=1e-4,
            alpha_diffusion=4,
            alpha_bond=self.args.alpha_bond,
            alpha_distogram=3e-2,
            alpha_pae=self.args.alpha_pae,
            use_template=self.args.use_template,
            use_mini_rollout=self.args.use_mini_rollout,
            use_flash_attn=self.args.use_flash_attn,
            custom_rel_token=42,
            ref_dim=167,
            mini_rollout_steps=self.args.mini_rollout_steps,
            atom_attention_type=self.args.atom_attention_type,
            interaction_aware=self.args.interaction_aware,
            templ_dim=self.args.templ_dim,
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if split == "train":
            dataset = STFoldUnicoreDataset(
                config=self.config,
                mode="train",
                num_processors=self.args.num_processors,
            )
        else:
            dataset = STFoldUnicoreDataset(
                config=self.config,
                mode="eval",
                num_processors=self.args.num_processors,
            )

        self.datasets[split] = dataset

    def build_model(self, args):
        model = build_model(args, self)
        return model

    def disable_shuffling(self) -> bool:
        return True

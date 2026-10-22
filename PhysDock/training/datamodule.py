"""LightningDataModule: weighted random training systems (data/v1); validation on a fixed random subset of the same
systems (for now), featurised with the inference settings."""
import random
from pathlib import Path

import lightning.pytorch as L
import torch

from PhysDock.data import CCDLibrary, FeatureConfig, PhysDockDataloader, PhysDockDataset


def _features_only(samples):
    return samples[0].features      # batch size 1, training needs no metadata


class PhysDockDataModule(L.LightningDataModule):
    """
    Args:
        features, val_features: FeatureConfig fields of training / validation
        root: released training set (train_val/, msa_features/, uniprot_msa_features/, sample weights)
        samples_per_epoch: weighted random training systems per epoch
        num_workers: training dataloader workers
        num_val_systems: validation systems, a fixed random (seed) subset of the training systems; null = all
        val_num_workers: validation dataloader workers
        ccd: CCD library path
        seed: seed of the validation subset and the training sampler
        systems: text file with system names (one per line, in root/train_val): train on these systems only, drawn uniformly, and validate on the first
            num_val_systems of them (overfitting tests); null = all systems, weighted
    """

    def __init__(self, features: dict, val_features: dict, root: str, samples_per_epoch: int, num_workers: int,
                 num_val_systems: int = 32, val_num_workers: int = 2, ccd: str = None, seed: int = 0, systems: str = None) -> None:
        super().__init__()
        self.features, self.val_features = FeatureConfig(**features), FeatureConfig(**val_features)
        self.root, self.samples_per_epoch, self.num_workers = Path(root), samples_per_epoch, num_workers
        self.num_val_systems, self.val_num_workers = num_val_systems, val_num_workers
        self.ccd_path, self.seed, self.systems = ccd, seed, systems
        self.train_set = self.val_set = None

    def setup(self, stage: str = None) -> None:
        if self.train_set is not None:
            return
        ccd = CCDLibrary(self.ccd_path)           # loaded once, shared with the (forked) workers
        if self.systems is None:
            self.train_set = PhysDockDataset.from_train_val(self.root, config=self.features, ccd=ccd)
            entries = self.train_set.entries
            if self.num_val_systems is not None and self.num_val_systems < len(entries):
                entries = sorted(random.Random(self.seed).sample(entries, self.num_val_systems), key=lambda e: e.path)
        else:
            paths = [str(self.root / "train_val" / f"{name}.pkl.gz") for name in Path(self.systems).read_text().split()]
            self.train_set = PhysDockDataset(paths, self.root / "msa_features", self.root / "uniprot_msa_features", ccd=ccd,
                                             config=self.features, training=True, sample_weights=torch.ones(len(paths)),
                                             on_error="resample")
            entries = self.train_set.entries[:self.num_val_systems]
        self.val_set = PhysDockDataset(entries, self.root / "msa_features", self.root / "uniprot_msa_features",
                                       ccd=ccd, config=self.val_features, on_error="skip")

    def train_dataloader(self):
        # weighted draws split over the DDP ranks; Lightning must not replace this sampler (use_distributed_sampler). A resumed run starts
        # from the step of its checkpoint: its draws are seeded with that step too, so it does not repeat the systems the epoch began with.
        start = self.trainer.global_step if self.trainer is not None else 0
        print(f"[data] training sampler seeded from step {start}", flush=True)
        return PhysDockDataloader(self.train_set, weighted_sampling=True, num_samples=self.samples_per_epoch,
                                  num_workers=self.num_workers, collate_fn=_features_only, distributed=True,
                                  seed=self.seed + 7919 * start, pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        # every rank validates its own slice of the systems (metrics are synced at the end of the epoch)
        dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank, world = (torch.distributed.get_rank(), torch.distributed.get_world_size()) if dist else (0, 1)
        return PhysDockDataloader(self.val_set, num_workers=self.val_num_workers,
                                  sampler=list(range(rank, len(self.val_set), world)))

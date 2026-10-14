"""PhysDockDataset, PhysDockCollator and PhysDockDataloader."""
import glob
import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from PhysDock.data.ccd import CCDLibrary
from PhysDock.data.features import FEATURES
from PhysDock.data.featurizer import FeatureConfig, Sample, SystemEntry, SystemFeaturizer
from PhysDock.utils.io_utils import convert_md5_string, load_json

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


class PhysDockDataset(Dataset):
    """Dataset of systems.

    Args:
        entries: SystemEntry objects or pickle paths
        msa_features_dir: MSA features dir (or a dir with msa_features/ and uniprot_msa_features/)
        uniprot_msa_features_dir: UniProt MSA features dir
        ccd: CCDLibrary or its path (default data/ccd_id_meta_data.pkl.gz)
        config: FeatureConfig
        training: training mode (random settings, padding)
        sample_weights: per-entry weights for weighted sampling
        on_error: "raise", "skip" (return None) or "resample" (featurise a random other entry)
    """

    def __init__(self, entries, msa_features_dir, uniprot_msa_features_dir=None, ccd=None, config: FeatureConfig = None,
                 training: bool = False, sample_weights=None, on_error: str = "raise", max_retries: int = 10) -> None:
        if on_error not in ("raise", "skip", "resample"):
            raise ValueError(f"on_error must be 'raise', 'skip' or 'resample', got {on_error!r}")
        self.entries = [e if isinstance(e, SystemEntry) else SystemEntry(str(e)) for e in entries]
        if sample_weights is not None and len(sample_weights) != len(self.entries):
            raise ValueError(f"{len(sample_weights)} sample weights for {len(self.entries)} entries")
        msa_dir = Path(msa_features_dir)
        if uniprot_msa_features_dir is not None:
            uni_dir = Path(uniprot_msa_features_dir)
        elif (msa_dir / "msa_features").is_dir() and (msa_dir / "uniprot_msa_features").is_dir():
            msa_dir, uni_dir = msa_dir / "msa_features", msa_dir / "uniprot_msa_features"
        else:
            raise ValueError(f"uniprot_msa_features_dir is required ({msa_dir} has no msa_features/ and uniprot_msa_features/)")
        for d in (msa_dir, uni_dir):
            if not d.is_dir():
                raise FileNotFoundError(f"MSA feature directory not found: {d}")
        self.ccd = ccd if isinstance(ccd, CCDLibrary) else CCDLibrary(ccd)
        self.featurizer = SystemFeaturizer(self.ccd, msa_dir, uni_dir, config, training)
        self.config, self.training = self.featurizer.config, training
        self.sample_weights = None if sample_weights is None else torch.as_tensor(sample_weights, dtype=torch.double)
        self.on_error, self.max_retries = on_error, max_retries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Optional[Sample]:
        entry = self.entries[index]
        for _ in range(1 + (self.max_retries if self.on_error == "resample" else 0)):
            try:
                return self.featurizer(entry)
            except Exception as error:  # noqa: BLE001
                if self.on_error == "raise":
                    raise
                logger.warning("failed to featurise %s: %s: %s", entry.path, type(error).__name__, error)
                if self.on_error == "skip":
                    return None
                entry = self.entries[random.randrange(len(self.entries))]
        raise RuntimeError(f"no system could be featurised in {self.max_retries} attempts after {self.entries[index].path}")

    @classmethod
    def from_directory(cls, systems_dir, msa_features_dir, uniprot_msa_features_dir=None, **kwargs) -> "PhysDockDataset":
        """All *.pkl.gz below a directory (sorted)."""
        paths = sorted(glob.glob(os.path.join(str(systems_dir), "**", "*.pkl.gz"), recursive=True))
        if not paths:
            raise FileNotFoundError(f"no *.pkl.gz system in {systems_dir}")
        return cls(paths, msa_features_dir, uniprot_msa_features_dir, **kwargs)

    @classmethod
    def for_screening(cls, system_path, smiles, msa_features_dir, uniprot_msa_features_dir=None, **kwargs) -> "PhysDockDataset":
        """One entry per SMILES (system id md5(smiles))."""
        entries = [SystemEntry(str(system_path), smiles=s, system_id=convert_md5_string(s)) for s in smiles]
        return cls(entries, msa_features_dir, uniprot_msa_features_dir, **kwargs)

    @classmethod
    def from_train_val(cls, root=_REPO_ROOT / "data" / "v1", **kwargs) -> "PhysDockDataset":
        """Released training set: train_val/, MSA features and train_val_samples_weights.json (training, resample)."""
        root = Path(root)
        paths = sorted(glob.glob(str(root / "train_val" / "*.pkl.gz")))
        if not paths:
            raise FileNotFoundError(f"no system in {root / 'train_val'}")
        weights = None
        if (root / "train_val_samples_weights.json").is_file():
            table = load_json(root / "train_val_samples_weights.json")
            names = [SystemEntry(p).name for p in paths]
            missing = [n for n in names if n not in table]
            if missing:
                raise ValueError(f"{len(missing)} systems have no sampling weight, e.g. {missing[:3]}")
            weights = [table[n] for n in names]
        kwargs.setdefault("training", True)
        kwargs.setdefault("on_error", "resample")
        return cls(paths, root / "msa_features", root / "uniprot_msa_features", sample_weights=weights, **kwargs)


class PhysDockCollator:
    """Batch size 1: return the sample unchanged (no batch dim). Larger: pad every symbolic dim and stack."""

    def __call__(self, samples):
        if len(samples) == 1:
            return samples[0]
        samples = [s for s in samples if s is not None]
        if not samples:
            return None
        keys = set(samples[0].features)
        if any(set(s.features) != keys for s in samples):
            raise ValueError("samples with different feature keys cannot be batched")
        sizes = {}
        for s in samples:
            for key, value in s.features.items():
                for dim, size in zip(FEATURES[key].shape, value.shape):
                    if isinstance(dim, str):
                        sizes[dim] = max(sizes.get(dim, 0), size)
        out = {}
        for key in keys:
            padded = []
            for s in samples:
                value, pad = s.features[key], []
                for dim, size in reversed(list(zip(FEATURES[key].shape, value.shape))):
                    pad += [0, sizes[dim] - size if isinstance(dim, str) else 0]
                padded.append(F.pad(value, pad) if any(pad) else value)
            out[key] = torch.stack(padded, dim=0)
        return Sample(out, [s.meta for s in samples])


def seed_worker(worker_id: int) -> None:
    """worker_init_fn: seed NumPy per worker (torch only seeds random / torch) and use one torch thread."""
    np.random.seed(torch.initial_seed() % 2 ** 32)
    torch.set_num_threads(1)


class PhysDockDataloader(DataLoader):
    """DataLoader with PhysDockCollator, per-worker seeding and optional weighted sampling (dataset.sample_weights)."""

    def __init__(self, dataset: PhysDockDataset, batch_size: int = 1, shuffle: bool = False, num_workers: int = 0,
                 weighted_sampling: bool = False, num_samples: int = None, seed: int = None, **kwargs) -> None:
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        sampler = None
        if weighted_sampling:
            if dataset.sample_weights is None:
                raise ValueError("weighted_sampling needs a dataset with sample_weights")
            sampler = WeightedRandomSampler(dataset.sample_weights, num_samples or len(dataset), replacement=True,
                                            generator=generator)
            shuffle = False
        kwargs.setdefault("collate_fn", PhysDockCollator())
        kwargs.setdefault("worker_init_fn", seed_worker)
        kwargs.setdefault("persistent_workers", num_workers > 0)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                         generator=generator, **kwargs)

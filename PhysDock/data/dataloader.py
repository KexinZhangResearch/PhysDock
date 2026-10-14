"""Featurisation of PhysDock systems: :class:`PhysDockDataset`, :class:`PhysDockCollator`, :class:`PhysDockDataloader`.

A *system* is a pickled ``{chain_id: chain}`` dict as written by ``prepare_system.py``. Chains with a numeric id are
ligand chains, all others receptor chains. Every chain holds ``ccds`` (component ids), ``all_atom_positions`` and
``all_atom_mask`` (one ``[n_atoms_of_ccd, 3]`` / ``[n_atoms_of_ccd]`` array per component, atoms in the order of the
CCD library) and optionally the PLIP interaction flags (``"salt bridges"``, ...). Protein chains additionally need their
MSA features ``<md5>.pkl.gz`` in the MSA directories (``md5("protein:" + one-letter sequence)``).

The featurisation reproduces the released PhysDock pipeline exactly: with the same states of Python's ``random``,
NumPy's global and torch's global random number generator it produces bit-identical tensors (only the key names and
the removal of redundant keys differ, see :data:`PhysDock.data.FEATURES`). Several behaviours that look odd are part
of what the released parameters were trained on and are therefore kept; they are marked ``[legacy]`` below:

* the merged MSA is padded to 16384 rows by sampling rows with replacement before the profile is computed and before
  the rows of ``msa_feat`` are drawn; the deletion "mean" is ``atan(sum(deletions) / 3) * 2 / pi`` (a sum, not a mean);
* every additional MSA sample (``num_msa_samples > 1``) is drawn from the rows of the previous sample;
* the key-residue flags are randomly masked in inference too;
* ``token_bonds`` only holds bonds inside one non-standard component;
* the template features treat peptide-like ligand chains as non-protein.

Differences to the old ``FeatureLoader`` (all of them only affect inputs on which it crashed or silently produced
wrong features; every other system gives identical features):

* chains that are empty after dropping incomplete residues are removed instead of crashing;
* standard residues also need their token centre and pseudo-beta atom (only changes nucleotides; the old rule tested
  atom positions 1 and 4, i.e. CA/CB of amino acids but P/O5' of nucleotides);
* unknown components (``UNK``, ``UNL``, ``N``, ``DN``) are dropped: they have no reference conformer and used to
  produce misaligned atom/token maps;
* receptor chains without usable atoms are skipped when the pocket is computed;
* receptor chains are kept when the system has no ligand chain (the old loader dropped them);
* when the ligand is replaced by a SMILES (screening), the crop is centred on the reference ligand of the system
  instead of the RDKit conformer of the SMILES, which sits at the origin;
* ``pocket_dist_type="ligand_centre"`` measures the distance to the ligand centre (the old code used the minimum
  corner of the bounding box) and the training option is spelled ``"ligand_centre"`` (was ``"ligand_cetre"``);
* the training pocket options are drawn correctly (``random.choices(...)[0]``; the old code compared lists to strings
  and raised ``NotImplementedError`` for every training system with a ligand);
* the deletion matrix of a ligand-only system is zero (it was a copy of the MSA residue codes);
* a peptide ligand chain with the sequence of a receptor chain gets its own entity when both survive the crop (the
  MSA merge stacked their MSAs into one entity and crashed);
* the inter-chain bond search, whose result no model used, is not computed (its random draw is still consumed).
"""
from __future__ import annotations

import glob
import logging
import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from functools import reduce
from operator import add
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from PhysDock.data import FEATURES, PDB_CHAIN_IDS
from PhysDock.data import constants as rc
from PhysDock.data.constants.PDBData import protein_letters_3to1_extended
from PhysDock.data.constants.periodic_table import PeriodicTable
from PhysDock.data.tools.feature_processing_multimer import pair_and_merge
from PhysDock.data.tools.rdkit import get_features_from_smi
from PhysDock.utils.io_utils import convert_md5_string, load_json, load_pkl
from PhysDock.utils.tensor_utils import centre_random_augmentation_np_apply, dgram_from_positions

__all__ = [
    "CCDConformer", "CCDLibrary", "FeatureConfig", "SystemEntry", "SystemMeta", "Sample", "SystemFeaturizer",
    "PhysDockDataset", "PhysDockCollator", "PhysDockDataloader", "default_ccd_path", "seed_worker", "msa_feat_for_round",
]

logger = logging.getLogger(__name__)
PathLike = Union[str, os.PathLike]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIGAND_SMILES_CCD = "XXX"           # component id of a ligand built from SMILES
_LIGAND_SMILES_CHAIN = "99"          # chain id of a ligand built from SMILES
_MSA_ROWS = 16384                    # rows of the merged MSA (tools/feature_processing_multimer.MSA_CROP_SIZE)
_UNPADDED_KEYS_TOKEN = ("restype", "residue_index", "is_protein", "is_short_poly", "is_ligand", "asym_id", "entity_id",
                        "sym_id", "token_mask", "token_to_centre_atom", "token_to_pseudo_beta_atom", "split_sizes",
                        "is_pocket")


def default_ccd_path() -> Path:
    """Location of the CCD library: ``<repo>/data/ccd_id_meta_data.pkl.gz`` (or ``<repo>/params/...``).

    Returns:
        The first existing candidate.

    Raises:
        FileNotFoundError: If neither exists.
    """
    for path in (_REPO_ROOT / "data" / "ccd_id_meta_data.pkl.gz", _REPO_ROOT / "params" / "ccd_id_meta_data.pkl.gz"):
        if path.is_file():
            return path
    raise FileNotFoundError("CCD library ccd_id_meta_data.pkl.gz not found in <repo>/data or <repo>/params")


# ======================================================================================
# CCD library
# ======================================================================================
@dataclass(frozen=True)
class CCDConformer:
    """Reference data of one chemical component.

    Attributes:
        ref_feat: Per-atom reference features ``[n_atoms, 167]`` (float32), coordinates centred on the conformer.
        pair_feat: Pair features ``[n_atoms, n_atoms, 42]`` (float32); ``None`` for standard residues (unused there).
        bonds: Covalent bonds ``[n_atoms, n_atoms]`` (float32).
        atom_names: Atom names in library order.
        elements: Element index of every atom (``PeriodicTable[element]`` is the symbol).
    """

    ref_feat: np.ndarray
    pair_feat: Optional[np.ndarray]
    bonds: np.ndarray
    atom_names: Tuple[str, ...]
    elements: np.ndarray

    @property
    def num_atoms(self) -> int:
        """Number of atoms of the reference conformer."""
        return self.ref_feat.shape[0]

    def index(self, atom_name: str) -> Optional[int]:
        """Position of an atom name in the conformer, or ``None``."""
        try:
            return self.atom_names.index(atom_name)
        except ValueError:
            return None

    @staticmethod
    def from_arrays(feats: Mapping[str, Any], ref_pos: np.ndarray, atom_names: Sequence[str], with_pairs: bool) -> "CCDConformer":
        """Build the conformer from the raw per-atom / per-pair arrays of the CCD library or of a SMILES.

        Args:
            feats: Mapping with ``ref_charge``, ``ref_element``, ``ref_is_aromatic``, ``ref_degree``, ``ref_hybridization``,
                ``ref_implicit_valence``, ``ref_chirality``, ``ref_in_ring_of_3`` .. ``_8``, ``d_token``, ``bond_type``,
                ``token_bonds``, ``bond_as_double``, ``bond_in_ring``, ``bond_is_conjugated``, ``bond_is_aromatic``.
            ref_pos: Reference coordinates ``[n_atoms, 3]``.
            atom_names: Atom names.
            with_pairs: Also build ``pair_feat``.

        Returns:
            The conformer.
        """
        ref_feat = np.concatenate([
            ref_pos,
            feats["ref_charge"][..., None],
            rc.eye_128[feats["ref_element"]].astype(np.float32),
            feats["ref_is_aromatic"].astype(np.float32)[..., None],
            rc.eye_9[feats["ref_degree"]].astype(np.float32),
            rc.eye_7[feats["ref_hybridization"]].astype(np.float32),
            rc.eye_9[feats["ref_implicit_valence"]].astype(np.float32),
            rc.eye_3[feats["ref_chirality"]].astype(np.float32),
            feats["ref_in_ring_of_3"].astype(np.float32)[..., None],
            feats["ref_in_ring_of_4"].astype(np.float32)[..., None],
            feats["ref_in_ring_of_5"].astype(np.float32)[..., None],
            feats["ref_in_ring_of_6"].astype(np.float32)[..., None],
            feats["ref_in_ring_of_7"].astype(np.float32)[..., None],
            feats["ref_in_ring_of_8"].astype(np.float32)[..., None],
        ], axis=-1).astype(np.float32)
        pair_feat = None
        if with_pairs:
            pair_feat = np.concatenate([
                rc.eye_32[feats["d_token"]].astype(np.float32),
                rc.eye_5[feats["bond_type"]].astype(np.float32),
                feats["token_bonds"].astype(np.float32)[..., None],
                feats["bond_as_double"].astype(np.float32)[..., None],
                feats["bond_in_ring"].astype(np.float32)[..., None],
                feats["bond_is_conjugated"].astype(np.float32)[..., None],
                feats["bond_is_aromatic"].astype(np.float32)[..., None],
            ], axis=-1).astype(np.float32)
        return CCDConformer(ref_feat=ref_feat, pair_feat=pair_feat, bonds=np.asarray(feats["token_bonds"]).astype(np.float32),
                            atom_names=tuple(atom_names), elements=np.asarray(feats["ref_element"]))


class CCDLibrary:
    """The chemical-component library (``ccd_id_meta_data.pkl.gz``) with a cache of derived conformer arrays.

    Loading the library takes a few seconds; build it once and share it (DataLoader workers started with ``fork``
    share the memory).

    Args:
        source: Path to the pickled library, or an already loaded ``{ccd: entry}`` mapping. Defaults to
            :func:`default_ccd_path`.
        cache_size: Number of non-standard components whose derived arrays are cached (standard residues are always
            cached).
    """

    def __init__(self, source: Union[PathLike, Mapping[str, Mapping[str, Any]], None] = None, cache_size: int = 256) -> None:
        if source is None:
            source = default_ccd_path()
        if isinstance(source, Mapping):
            self.path: Optional[str] = None
            self._entries = source
        else:
            self.path = str(source)
            entries = load_pkl(self.path)
            if not isinstance(entries, Mapping):
                raise ValueError(f"{self.path} is not a CCD library (expected a pickled dict)")
            self._entries = entries
        self.cache_size = cache_size
        self._standard: Dict[str, CCDConformer] = {}
        self._cache: "OrderedDict[str, CCDConformer]" = OrderedDict()

    def __contains__(self, ccd: object) -> bool:
        return ccd in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def entry(self, ccd: str) -> Mapping[str, Any]:
        """The raw library entry of a component.

        Raises:
            KeyError: If the component is not in the library.
        """
        try:
            return self._entries[ccd]
        except KeyError:
            raise KeyError(f"component {ccd!r} is not in the CCD library{f' {self.path}' if self.path else ''}") from None

    def conformer(self, ccd: str) -> CCDConformer:
        """Derived arrays of a component (cached)."""
        if ccd in self._standard:
            return self._standard[ccd]
        hit = self._cache.get(ccd)
        if hit is not None:
            self._cache.move_to_end(ccd)
            return hit
        entry = self.entry(ccd)
        ref_pos = entry["ref_pos"]
        conf = CCDConformer.from_arrays(entry, ref_pos - np.mean(ref_pos, axis=0, keepdims=True), entry["ref_atom_name_chars"],
                                        with_pairs=not rc.is_standard(ccd))
        if rc.is_standard(ccd):
            self._standard[ccd] = conf
        elif self.cache_size > 0:
            self._cache[ccd] = conf
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return conf

    def ref_mol(self, ccd: str) -> Optional[Chem.Mol]:
        """A copy of the RDKit molecule of a component (``None`` if the library has none)."""
        mol = self.entry(ccd).get("ref_mol")
        return Chem.Mol(mol) if mol is not None else None


# ======================================================================================
# Configuration, entries, metadata
# ======================================================================================
@dataclass(frozen=True)
class FeatureConfig:
    """Featurisation settings (the defaults are those of the released pipeline).

    Args:
        crop_size: Maximum number of tokens; ``None`` disables cropping (then ``atom_crop_size`` is ignored).
        atom_crop_size: Maximum number of atoms of a crop.
        max_msa_clusters: Number of MSA rows in ``msa_feat`` (the query plus randomly drawn rows).
        num_msa_samples: Number of MSA samples; samples 2.. go to ``msa_feat_resampled`` (multi-round redocking).
        key_res_random_mask_ratio: Probability that a key-residue flag is masked (also in inference [legacy]).
        pocket_type: Inference: ``"atom"`` (any atom of a residue) or ``"ca"`` (its CA) decides whether a residue is in
            the pocket.
        pocket_cutoff: Inference: distance cutoff (Angstrom) of the pocket.
        pocket_dist_type: Inference: distance to any ligand atom (``"ligand"``) or to the ligand centre
            (``"ligand_centre"``).
        use_pocket: Inference: provide the pocket feature.
        use_key_res: Inference: provide the key-residue feature.
        use_x_gt_ligand_as_ref_pos: Ablation only: use the ground-truth ligand as its reference conformer.
        train_pocket_type_atom_ratio: Training: probability of ``pocket_type="atom"``.
        train_pocket_cutoff_ligand_min: Training: lower bound of the pocket cutoff for ``"ligand"``.
        train_pocket_cutoff_ligand_max: Training: upper bound of the pocket cutoff for ``"ligand"``.
        train_pocket_cutoff_ligand_centre_min: Training: lower bound of the pocket cutoff for ``"ligand_centre"``.
        train_pocket_cutoff_ligand_centre_max: Training: upper bound of the pocket cutoff for ``"ligand_centre"``.
        train_pocket_dist_type_ligand_ratio: Training: probability of ``pocket_dist_type="ligand"``.
        train_use_pocket_ratio: Training: probability that the pocket feature is provided.
        train_use_key_res_ratio: Training: probability that the key-residue feature is provided.
        train_shuffle_sym_id: Training: shuffle the order of the chains of an entity.
        train_spatial_crop_ligand_ratio: Training: probability of a crop centred on a ligand atom.
        train_spatial_crop_interface_ratio: Training: probability of an interface crop (also used in inference when the
            system has no single-component ligand chain [legacy]).
        train_spatial_crop_interface_threshold: Training: distance (Angstrom) that defines interface tokens.
        train_charility_augmentation_ratio: Training: probability that the chirality of ligand atoms is randomly
            hidden.
        train_use_template_ratio: Training: probability that the template is *switched off* [legacy name].
        train_template_mask_max_ratio: Training: bounds the fraction of randomly masked template residues.

    Raises:
        ValueError: For inconsistent settings.
    """

    crop_size: Optional[int] = 256
    atom_crop_size: int = 256 * 8
    max_msa_clusters: int = 128
    num_msa_samples: int = 1
    key_res_random_mask_ratio: float = 0.5

    pocket_type: Literal["atom", "ca"] = "atom"
    pocket_cutoff: float = 6.0
    pocket_dist_type: Literal["ligand", "ligand_centre"] = "ligand"
    use_pocket: bool = True
    use_key_res: bool = True
    use_x_gt_ligand_as_ref_pos: bool = False

    train_pocket_type_atom_ratio: float = 0.5
    train_pocket_cutoff_ligand_min: float = 6.0
    train_pocket_cutoff_ligand_max: float = 12.0
    train_pocket_cutoff_ligand_centre_min: float = 10.0
    train_pocket_cutoff_ligand_centre_max: float = 16.0
    train_pocket_dist_type_ligand_ratio: float = 0.5
    train_use_pocket_ratio: float = 0.5
    train_use_key_res_ratio: float = 0.5
    train_shuffle_sym_id: bool = True
    train_spatial_crop_ligand_ratio: float = 0.2
    train_spatial_crop_interface_ratio: float = 0.4
    train_spatial_crop_interface_threshold: float = 15.0
    train_charility_augmentation_ratio: float = 0.1
    train_use_template_ratio: float = 0.75
    train_template_mask_max_ratio: float = 0.4

    def __post_init__(self) -> None:
        if self.crop_size is not None and (self.crop_size < 1 or self.atom_crop_size < 1):
            raise ValueError(f"crop sizes must be positive, got crop_size={self.crop_size}, atom_crop_size={self.atom_crop_size}")
        if self.max_msa_clusters < 1 or self.num_msa_samples < 1:
            raise ValueError("max_msa_clusters and num_msa_samples must be at least 1")
        if self.pocket_type not in ("atom", "ca"):
            raise ValueError(f"pocket_type must be 'atom' or 'ca', got {self.pocket_type!r}")
        if self.pocket_dist_type not in ("ligand", "ligand_centre"):
            raise ValueError(f"pocket_dist_type must be 'ligand' or 'ligand_centre', got {self.pocket_dist_type!r}")


@dataclass(frozen=True)
class SystemEntry:
    """One item of a :class:`PhysDockDataset`.

    Args:
        path: System pickle.
        smiles: If given, the ligand chains are replaced by this molecule (screening). The reference ligand of the
            system still defines the pocket and the crop centre.
        system_id: Name of the system; defaults to the file name without ``.pkl.gz`` / ``.pkl``.
        remove_ligand: Remove the ligand chains (implied by ``smiles``).
    """

    path: str
    smiles: Optional[str] = None
    system_id: Optional[str] = None
    remove_ligand: bool = False

    @property
    def name(self) -> str:
        """The system id."""
        if self.system_id is not None:
            return self.system_id
        base = os.path.basename(self.path)
        for suffix in (".pkl.gz", ".pkl"):
            if base.endswith(suffix):
                return base[: -len(suffix)]
        return base


@dataclass
class SystemMeta:
    """Non-tensor information of a featurised system, needed to write structures.

    Per-conformer lists follow the order of the features (a conformer is a residue or a ligand).

    Attributes:
        system_id: Name of the system.
        ref_mol: RDKit molecule of the ligand if the system has exactly one ligand chain with one component (or a
            SMILES ligand); atoms in library order. ``None`` otherwise.
        ccds: Component id of every conformer.
        conformer_sizes: Number of atoms of every conformer.
        residue_index: Position of every conformer in its input chain.
        asym_id: Chain index of every conformer.
        conformer_is_ligand: Whether every conformer belongs to a ligand chain (written as HETATM).
        chain_ids: Input chain id of every chain index.
        atom_names: Name of every atom.
        atom_elements: Element symbol of every atom.
        atom_to_conformer_atom: Index of every atom inside its library conformer.
    """

    system_id: str
    ref_mol: Optional[Chem.Mol]
    ccds: List[str]
    conformer_sizes: np.ndarray
    residue_index: np.ndarray
    asym_id: np.ndarray
    conformer_is_ligand: List[bool]
    chain_ids: Dict[int, str]
    atom_names: List[str]
    atom_elements: List[str]
    atom_to_conformer_atom: np.ndarray
    _pdb_prefix: Optional[List[str]] = None
    _pdb_suffix: Optional[List[str]] = None
    _is_hetatm: Optional[np.ndarray] = None

    @property
    def num_atoms(self) -> int:
        """Number of (unpadded) atoms."""
        return len(self.atom_names)

    def _pdb_fields(self) -> None:
        prefix, suffix, het = [], [], []
        atom = 0
        for conf, (ccd, size) in enumerate(zip(self.ccds, self.conformer_sizes.tolist())):
            record = "HETATM" if self.conformer_is_ligand[conf] else "ATOM"
            chain = PDB_CHAIN_IDS[int(self.asym_id[conf]) % len(PDB_CHAIN_IDS)]
            res_name = ccd.split()[0][-3:]
            res_id = int(self.residue_index[conf]) + 1
            for _ in range(size):
                name = self.atom_names[atom]
                name = name if len(name) == 4 else f" {name}"
                prefix.append(f"{record:<6}{atom + 1:>5} {name:<4}{'':>1}{res_name:>3} {chain:>1}{res_id:>4}{'':>1}   ")
                suffix.append(f"{1.00:>6.2f}{70.:>6.2f}          {self.atom_elements[atom]:>2}{0:>2}")
                het.append(record == "HETATM")
                atom += 1
        self._pdb_prefix, self._pdb_suffix, self._is_hetatm = prefix, suffix, np.array(het, dtype=bool)

    def to_pdb(self, coords: Union[torch.Tensor, np.ndarray], receptor_only: bool = False, ligand_only: bool = False) -> str:
        """Write coordinates as a PDB block (identical to the output of the old ``FeatureLoader.write_pdb_block``).

        Args:
            coords: Coordinates ``[N_atom, 3]``; rows beyond the real atoms (padding) are ignored.
            receptor_only: Only receptor atoms (ATOM records).
            ligand_only: Only ligand atoms (HETATM records).

        Returns:
            The PDB text, starting with ``MODEL 1`` and ending with ``END``.

        Raises:
            ValueError: If both selections are requested or ``coords`` has too few rows.
        """
        if receptor_only and ligand_only:
            raise ValueError("receptor_only and ligand_only are mutually exclusive")
        if self._pdb_prefix is None:
            self._pdb_fields()
        if isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu()
        xyz = coords[: self.num_atoms].tolist()
        if len(xyz) < self.num_atoms:
            raise ValueError(f"got {len(xyz)} coordinates for {self.num_atoms} atoms")
        lines = []
        for i, ((x, y, z), het) in enumerate(zip(xyz, self._is_hetatm)):
            if (receptor_only and het) or (ligand_only and not het):
                continue
            lines.append(f"{self._pdb_prefix[i]}{x:>8.3f}{y:>8.3f}{z:>8.3f}{self._pdb_suffix[i]}")
        return f"MODEL     1\n" + "\n".join(lines) + "\nTER\nENDMDL\nEND"

    def write_pdb(self, path: PathLike, coords: Union[torch.Tensor, np.ndarray], receptor_only: bool = False,
                  ligand_only: bool = False) -> None:
        """Write :meth:`to_pdb` to a file."""
        Path(path).write_text(self.to_pdb(coords, receptor_only=receptor_only, ligand_only=ligand_only))


class Sample(NamedTuple):
    """Output of :class:`PhysDockDataset`: model inputs and metadata."""

    features: Dict[str, torch.Tensor]
    meta: SystemMeta


def msa_feat_for_round(features: Mapping[str, torch.Tensor], round_id: int) -> torch.Tensor:
    """MSA features of a redocking round: ``msa_feat`` for round 0, ``msa_feat_resampled[round_id - 1]`` after.

    Args:
        features: Features of one sample (``msa_feat_resampled`` needs ``num_msa_samples > round_id``).
        round_id: Zero-based round.

    Returns:
        ``[N_msa, N_token, 34]`` (or with the batch dimension of ``features``).

    Raises:
        IndexError: If the features hold fewer MSA samples.
    """
    if round_id == 0:
        return features["msa_feat"]
    resampled = features.get("msa_feat_resampled")
    n = 1 if resampled is None else 1 + resampled.shape[-4]
    if round_id < 0 or round_id >= n:
        raise IndexError(f"round {round_id} needs num_msa_samples > {round_id}, the features hold {n} MSA samples")
    return resampled[..., round_id - 1, :, :, :]


class MissingMSAError(FileNotFoundError):
    """Raised when the MSA features of a protein chain are missing."""


# ======================================================================================
# Featuriser
# ======================================================================================
class SystemFeaturizer:
    """Turns a system pickle into model inputs (see the module docstring for the exact behaviour).

    Randomness comes from the global random number generators of Python, NumPy and torch, in the same order as in the
    released pipeline. :class:`PhysDockDataloader` seeds them per worker.

    Args:
        ccd: The CCD library.
        msa_features_dir: Directory with the MSA features ``<md5>.pkl.gz`` (keys ``msa``, ``deletion_matrix``).
        uniprot_msa_features_dir: Directory with the UniProt MSA features ``<md5>.pkl.gz`` (keys ``msa_all_seq``,
            ``deletion_matrix_all_seq``, ``msa_species_identifiers_all_seq``).
        config: Featurisation settings.
        training: Training mode (random pocket / key-residue settings, crop augmentation, chirality augmentation,
            template masking, padding to the crop sizes).
        msa_cache_size: Number of MSA feature files kept in memory.
    """

    def __init__(
            self,
            ccd: CCDLibrary,
            msa_features_dir: PathLike,
            uniprot_msa_features_dir: PathLike,
            config: Optional[FeatureConfig] = None,
            training: bool = False,
            msa_cache_size: int = 8,
    ) -> None:
        self.ccd = ccd
        self.msa_features_dir = str(msa_features_dir)
        self.uniprot_msa_features_dir = str(uniprot_msa_features_dir)
        self.config = config if config is not None else FeatureConfig()
        self.training = training
        if training and self.config.crop_size is None:
            raise ValueError("training needs a crop size (features are padded to crop_size / atom_crop_size)")
        self.msa_cache_size = msa_cache_size
        self._msa_cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()

    # ------------------------------------------------------------------ public
    def __call__(self, entry: SystemEntry) -> Sample:
        """Featurise one system.

        Args:
            entry: The system.

        Returns:
            ``Sample(features, meta)``.
        """
        return self.featurize(entry)

    def featurize(self, entry: SystemEntry) -> Sample:
        """See :meth:`__call__`."""
        cfg, training = self.config, self.training
        # ---- 1. pocket / key-residue settings -----------------------------------------------------------------
        if training:
            pocket_type = random.choices(["atom", "ca"], [cfg.train_pocket_type_atom_ratio, 1 - cfg.train_pocket_type_atom_ratio])[0]
            pocket_dist_type = random.choices(["ligand", "ligand_centre"], [cfg.train_pocket_dist_type_ligand_ratio,
                                                                            1 - cfg.train_pocket_dist_type_ligand_ratio])[0]
            if pocket_dist_type == "ligand":
                pocket_cutoff = cfg.train_pocket_cutoff_ligand_min + random.random() * (
                        cfg.train_pocket_cutoff_ligand_max - cfg.train_pocket_cutoff_ligand_min)
            else:
                pocket_cutoff = cfg.train_pocket_cutoff_ligand_centre_min + random.random() * (
                        cfg.train_pocket_cutoff_ligand_centre_max - cfg.train_pocket_cutoff_ligand_centre_min)
            use_pocket = random.random() < cfg.train_use_pocket_ratio
            use_key_res = random.random() < cfg.train_use_key_res_ratio
        else:
            pocket_type, pocket_cutoff, pocket_dist_type = cfg.pocket_type, cfg.pocket_cutoff, cfg.pocket_dist_type
            use_pocket, use_key_res = cfg.use_pocket, cfg.use_key_res

        # ---- 2. read the system -----------------------------------------------------------------------------------
        data = load_pkl(entry.path)
        if not isinstance(data, Mapping) or not data:
            raise ValueError(f"{entry.path}: not a system pickle (expected a non-empty {{chain_id: chain}} dict)")
        for chain_id, chain in data.items():
            if not isinstance(chain_id, str):
                raise ValueError(f"{entry.path}: chain ids must be strings, got {chain_id!r}")
            for key in ("ccds", "all_atom_positions", "all_atom_mask"):
                if key not in chain:
                    raise ValueError(f"{entry.path}: chain {chain_id!r} has no {key!r}")
            if not (len(chain["ccds"]) == len(chain["all_atom_positions"]) == len(chain["all_atom_mask"])):
                raise ValueError(f"{entry.path}: chain {chain_id!r} has inconsistent numbers of components")
            for c_id, (ccd, pos, mask) in enumerate(zip(chain["ccds"], chain["all_atom_positions"], chain["all_atom_mask"])):
                if rc.is_unk(ccd):
                    continue                                  # dropped below; the library entries are inconsistent
                if np.ndim(pos) != 2 or np.shape(pos)[-1] != 3 or np.shape(mask) != (len(pos),):
                    raise ValueError(f"{entry.path}: chain {chain_id!r} component {c_id}: positions {np.shape(pos)} and "
                                     f"mask {np.shape(mask)} do not match")
        receptor_ids = [c for c in data if not c.isdigit()]
        ligand_ids = [c for c in data if c.isdigit()]
        remove_ligand = entry.remove_ligand or entry.smiles is not None

        # ---- 3. pocket of every receptor chain ------------------------------------------------------------------
        labels: "OrderedDict[str, Mapping[str, Any]]" = OrderedDict()
        pockets: Dict[str, np.ndarray] = {}
        ligand_atoms = [self._present_atoms(data[c]) for c in ligand_ids]
        for chain_id in receptor_ids:
            labels[chain_id] = data[chain_id]
            if ligand_ids:
                pockets[chain_id] = self._pocket(data[chain_id], ligand_atoms, pocket_type, pocket_dist_type, pocket_cutoff)

        # ---- 4. ligand chains (or the SMILES ligand) ------------------------------------------------------------
        conformers: Dict[str, CCDConformer] = {}
        ref_mol = None
        crop_centre = None
        if remove_ligand:
            if entry.smiles is not None:
                label, conformers[_LIGAND_SMILES_CCD], ref_mol = self._smiles_ligand(entry.smiles)
                labels[_LIGAND_SMILES_CHAIN] = label
                if ligand_atoms and sum(len(a) for a in ligand_atoms):
                    crop_centre = np.mean(np.concatenate(ligand_atoms, axis=0), axis=0)[None]
        else:
            for chain_id in ligand_ids:
                labels[chain_id] = data[chain_id]
            if len(ligand_ids) == 1 and len(data[ligand_ids[0]]["ccds"]) == 1:
                ref_mol = self.ccd.ref_mol(data[ligand_ids[0]]["ccds"][0])

        # ---- 5. per-chain features ------------------------------------------------------------------------------
        chains: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        seq3: "OrderedDict[str, str]" = OrderedDict()
        chain_class: Dict[str, str] = {}
        for chain_id, label in labels.items():
            for ccd in label["ccds"]:
                if ccd not in conformers and not rc.is_unk(ccd):
                    conformers[ccd] = self.ccd.conformer(ccd)
            feats = self._chain_features(chain_id, label, conformers, pockets.get(chain_id), use_pocket, use_key_res, entry.path)
            if feats is None:
                logger.warning("%s: chain %s has no complete component and is skipped", entry.name, chain_id)
                continue
            chains[chain_id] = feats
            seq3[chain_id] = "-".join(label["ccds"])
            chain_class[chain_id] = feats["chain_class"]
        if not chains:
            raise ValueError(f"{entry.name}: no chain with a complete component")
        if ref_mol is not None and entry.smiles is None and ligand_ids[0] not in chains:
            ref_mol = None                                    # the ligand has no atom with a position

        # ---- 6. chain / entity / symmetry ids --------------------------------------------------------------------
        asym_to_chain = self._assembly(chains, seq3)

        # ---- 7. spatial crop ------------------------------------------------------------------------------------
        if cfg.crop_size is not None:
            self._crop(chains, conformers, crop_centre, entry.name,
                       first_chain=_LIGAND_SMILES_CHAIN if entry.smiles is not None else None)

        # ---- 8. MSA pairing / merging and the conformer-level arrays ---------------------------------------------
        raw, order = self._pair_and_merge(chains, chain_class, asym_to_chain)
        conformer_class = []
        for chain_id in order:
            conformer_class += [chain_class[chain_id]] * len(chains[chain_id]["ccds"])
        ccds = reduce(add, [chains[chain_id]["ccds"] for chain_id in order])

        # ---- 9. atoms and tokens ----------------------------------------------------------------------------------
        feats, meta = self._tokens(raw, ccds, conformers, conformer_class, asym_to_chain, entry.name, ref_mol)

        # ---- 10. tensors: target / MSA features, template, padding ------------------------------------------------
        return Sample(self._transform(feats), meta)

    # ------------------------------------------------------------------ inputs
    @staticmethod
    def _present_atoms(chain: Mapping[str, Any]) -> np.ndarray:
        """Coordinates of all atoms with a position of a chain ``[n, 3]``."""
        if not len(chain["all_atom_positions"]):
            return np.zeros([0, 3], dtype=np.float32)
        pos = np.concatenate(chain["all_atom_positions"], axis=0)
        return pos[np.concatenate(chain["all_atom_mask"], axis=0).astype(np.bool_)]

    @staticmethod
    def _pocket(chain: Mapping[str, Any], ligand_atoms: Sequence[np.ndarray], pocket_type: str, dist_type: str,
                cutoff: float, chunk: int = 4096) -> np.ndarray:
        """Pocket flag of every component of a receptor chain ``[n_components]`` (float32)."""
        ccds = chain["ccds"]
        coords, owner = [], []
        for i, (ccd, pos, mask) in enumerate(zip(ccds, chain["all_atom_positions"], chain["all_atom_mask"])):
            if not rc.is_standard(ccd) or rc.is_unk(ccd):    # unknown residues are dropped from the features
                continue
            mask = np.asarray(mask).astype(np.bool_)
            if mask.shape[0] < 2 or not mask[1]:          # position 1 is CA (amino acids) / P (nucleotides) [legacy]
                continue
            if pocket_type == "atom":
                coords.append(pos[mask])
                owner += [i] * int(mask.sum())
            else:
                coords.append(pos[1][None])
                owner.append(i)
        pocket = np.zeros([len(ccds)], dtype=np.float32)
        if not coords:
            return pocket
        coords = np.concatenate(coords, axis=0)
        owner = np.asarray(owner)
        used = []
        for lig in ligand_atoms:
            if len(lig) == 0:
                continue
            if dist_type == "ligand":
                close = np.zeros(len(coords), dtype=np.bool_)
                for s in range(0, len(coords), chunk):   # chunked: identical float32 distances, bounded memory
                    block = coords[s:s + chunk]
                    close[s:s + chunk] = np.any(np.linalg.norm(block[:, None] - lig[None], axis=-1) < cutoff, axis=-1)
            elif dist_type == "ligand_centre":
                close = np.linalg.norm(coords - np.mean(lig, axis=0, keepdims=True), axis=-1) < cutoff
            else:
                raise ValueError(f"unknown pocket distance type {dist_type!r}")
            used.append(owner[close])
        used = sorted(set(np.concatenate(used, axis=0).tolist())) if used else []
        pocket[used] = 1.
        return pocket

    @staticmethod
    def _smiles_ligand(smiles: str) -> Tuple[Dict[str, Any], CCDConformer, Chem.Mol]:
        """Chain dict, conformer and molecule of a ligand given as SMILES."""
        label_feature, conf_feature, ref_mol = get_features_from_smi(smiles)
        names = [f"{PeriodicTable[ele] + str(i):<4}" for i, ele in enumerate(conf_feature["ref_element"])]
        conformer = CCDConformer.from_arrays(conf_feature, conf_feature["ref_pos"], names, with_pairs=True)
        label = {"all_atom_positions": label_feature["x_gt"][None], "all_atom_mask": label_feature["x_exists"][None],
                 "ccds": [_LIGAND_SMILES_CCD]}
        return label, conformer, ref_mol

    def _load_msa(self, md5: str, sequence: str, chain_id: str, path: str) -> Dict[str, np.ndarray]:
        hit = self._msa_cache.get(md5)
        if hit is not None:
            self._msa_cache.move_to_end(md5)
            return hit
        files = (os.path.join(self.msa_features_dir, f"{md5}.pkl.gz"), os.path.join(self.uniprot_msa_features_dir, f"{md5}.pkl.gz"))
        missing = [f for f in files if not os.path.isfile(f)]
        if missing:
            raise MissingMSAError(f"{path}: MSA features of chain {chain_id!r} not found: {missing}; "
                                  f"generate them with run_homo_search.py for\n>{md5}\n{sequence}")
        msa = dict(load_pkl(files[0]))
        msa.update(load_pkl(files[1]))
        for key in ("msa", "deletion_matrix", "msa_all_seq", "deletion_matrix_all_seq", "msa_species_identifiers_all_seq"):
            if key not in msa:
                raise ValueError(f"MSA features {md5} lack {key!r}")
        for key in ("msa", "deletion_matrix", "msa_all_seq", "deletion_matrix_all_seq"):
            if msa[key].ndim != 2 or msa[key].shape[1] != len(sequence):
                raise ValueError(f"MSA features {md5} ({key} {msa[key].shape}) do not match the {len(sequence)} residues "
                                 f"of chain {chain_id!r} in {path}")
        if self.msa_cache_size > 0:
            self._msa_cache[md5] = msa
            if len(self._msa_cache) > self.msa_cache_size:
                self._msa_cache.popitem(last=False)
        return msa

    # ------------------------------------------------------------------ chains
    def _chain_features(
            self,
            chain_id: str,
            chain: Mapping[str, Any],
            conformers: Mapping[str, CCDConformer],
            pocket: Optional[np.ndarray],
            use_pocket: bool,
            use_key_res: bool,
            path: str,
    ) -> Optional[Dict[str, Any]]:
        """Complete components of a chain and their features; ``None`` if no component is complete."""
        ccds_ori = list(chain["ccds"])
        n = len(ccds_ori)
        if n == 0:
            return None
        chain_class = "protein" if not chain_id.isdigit() else "ligand"
        if chain_class == "protein":
            sequence = "".join(protein_letters_3to1_extended.get(ccd, "X") for ccd in ccds_ori)
            msa_feats = self._load_msa(convert_md5_string(f"protein:{sequence}"), sequence, chain_id, path)
        else:
            codes = np.array([[rc.standard_ccds.index(ccd) if ccd in rc.standard_ccds else 20 for ccd in ccds_ori]] * 2,
                             dtype=np.int8)
            msa_feats = {"msa": codes, "deletion_matrix": np.zeros_like(codes)}

        if "salt bridges" in chain and use_key_res:
            key_res_feat = np.stack([
                chain["salt bridges"], chain["pi-cation interactions"], chain["hydrophobic interactions"],
                chain["pi-stacking"], chain["hydrogen bonds"], chain["metal complexes"], np.zeros_like(chain["salt bridges"]),
            ], axis=-1).astype(np.float32)
        else:
            key_res_feat = np.zeros([n, 7], dtype=np.float32)
        key_res_feat = key_res_feat * (np.random.random([n, 7]) > self.config.key_res_random_mask_ratio)  # [legacy] also in inference
        if pocket is None or not use_pocket:
            pocket = np.zeros([n], dtype=np.float32)

        kept = np.zeros([n], dtype=np.bool_)
        x_gt, atom_index, residue_index, sizes, restype, ccds = [], [], [], [], [], []
        for c_id, ccd in enumerate(ccds_ori):
            if rc.is_unk(ccd):
                continue
            conf = conformers[ccd]
            mask = np.asarray(chain["all_atom_mask"][c_id]).astype(np.bool_)
            if mask.shape[0] != conf.num_atoms or len(chain["all_atom_positions"][c_id]) != conf.num_atoms:
                raise ValueError(f"{path}: chain {chain_id!r} component {c_id} ({ccd!r}) has {mask.shape[0]} atom slots, "
                                 f"the CCD library has {conf.num_atoms}")
            if not mask.any():
                continue
            if rc.is_standard(ccd):
                if not mask[1] or (ccd != "GLY" and not mask[4]):                       # [legacy] CA and CB
                    continue
                centre = conf.index(rc.standard_ccd_to_token_centre_atom_name[ccd])
                beta = conf.index(rc.standard_ccd_to_token_pseudo_beta_atom_name[ccd])
                if centre is None or beta is None or not mask[centre] or not mask[beta]:
                    continue
            kept[c_id] = True
            x_gt.append(np.asarray(chain["all_atom_positions"][c_id])[mask])
            atom_index.append(np.arange(conf.num_atoms)[mask])
            residue_index.append(c_id)
            sizes.append(int(mask.sum()))
            restype.append(rc.standard_ccds.index(ccd) if ccd in rc.standard_ccds else 20)
            ccds.append(ccd)
        if not kept.any():
            return None
        for key in ("msa", "deletion_matrix"):
            if msa_feats[key].shape[1] != n:
                raise ValueError(f"{path}: {key} of chain {chain_id!r} has {msa_feats[key].shape[1]} columns for {n} components")

        out: Dict[str, Any] = {
            "x_gt": np.concatenate(x_gt, axis=0),
            "atom_to_conformer_atom": np.concatenate(atom_index, axis=0, dtype=np.int32),
            "residue_index": np.array(residue_index, dtype=np.int64),
            "conformer_sizes": np.array(sizes, dtype=np.int64),
            "restype": np.array(restype, dtype=np.int64),
            "ccds": ccds,
            "msa": msa_feats["msa"][:, kept],
            "deletion_matrix": msa_feats["deletion_matrix"][:, kept],
            "chain_class": chain_class,
            "key_res_feat": key_res_feat[kept],
            "is_pocket": pocket[kept],
        }
        num = len(ccds)
        out["is_protein"] = np.array([chain_class == "protein"] * num).astype(np.float32)
        out["is_ligand"] = np.array([chain_class != "protein"] * num).astype(np.float32)
        out["is_short_poly"] = np.array([chain_class != "protein" and num >= 2 and rc.is_standard(ccd) for ccd in ccds]).astype(np.float32)
        if "msa_all_seq" in msa_feats:
            out["msa_all_seq"] = msa_feats["msa_all_seq"][:, kept]
            out["deletion_matrix_all_seq"] = msa_feats["deletion_matrix_all_seq"][:, kept]
            out["msa_species_identifiers_all_seq"] = msa_feats["msa_species_identifiers_all_seq"]
        return out

    def _assembly(self, chains: "OrderedDict[str, Dict[str, Any]]", seq3: Mapping[str, str]) -> Dict[int, str]:
        """Assign ``asym_id`` / ``sym_id`` / ``entity_id``; returns ``{asym_id: chain_id}``."""
        entities: "OrderedDict[str, List[str]]" = OrderedDict()
        for chain_id, s in seq3.items():
            entities.setdefault(s, []).append(chain_id)
        asym_to_chain: Dict[int, str] = {}
        asym = 0
        for entity_id, s in enumerate(entities):
            chain_ids = list(entities[s])
            if self.training and self.config.train_shuffle_sym_id:
                random.shuffle(chain_ids)
            for sym_id, chain_id in enumerate(chain_ids):
                num = len(chains[chain_id]["ccds"])
                chains[chain_id]["asym_id"] = np.full([num], fill_value=asym, dtype=np.int32)
                chains[chain_id]["sym_id"] = np.full([num], fill_value=sym_id, dtype=np.int32)
                chains[chain_id]["entity_id"] = np.full([num], fill_value=entity_id, dtype=np.int32)
                chains[chain_id]["sequence_3"] = s
                asym_to_chain[asym] = chain_id
                asym += 1
        return asym_to_chain

    def _crop(self, chains: "OrderedDict[str, Dict[str, Any]]", conformers: Mapping[str, CCDConformer],
              centre_override: Optional[np.ndarray], name: str, first_chain: Optional[str] = None) -> None:
        """Spatial crop around a ligand / interface / random token (in place).

        Args:
            chains: Per-chain features.
            conformers: Conformers of all components.
            centre_override: Crop centre (inference only).
            name: System id (for errors).
            first_chain: Chain whose tokens are taken before all others (the SMILES ligand, whose coordinates are
                those of its RDKit conformer and say nothing about its distance to the pocket).
        """
        cfg = self.config
        order = list(chains)
        x_gt = np.concatenate([chains[c]["x_gt"] for c in order], axis=0)
        tok_atom, tok_conf, tok_size, tok_std, tok_asym, tok_first = [], [], [], [], [], []
        ligand_atoms = []
        atom = conf_id = 0
        for chain_id in order:
            chain = chains[chain_id]
            n_tok_before = len(tok_atom)
            if chain_id.isdigit() and len(chain["ccds"]) == 1:
                ligand_atoms.append(chain["x_gt"])
            offset = 0
            for ccd, size, asym in zip(chain["ccds"], chain["conformer_sizes"].tolist(), chain["asym_id"].tolist()):
                if rc.is_standard(ccd):
                    centre = conformers[ccd].index(rc.standard_ccd_to_token_centre_atom_name[ccd])
                    pos = int(np.flatnonzero(chain["atom_to_conformer_atom"][offset:offset + size] == centre)[0])
                    tok_atom.append(atom + pos); tok_conf.append(conf_id); tok_size.append(size); tok_std.append(True); tok_asym.append(asym)
                else:
                    tok_atom += range(atom, atom + size); tok_conf += [conf_id] * size; tok_size += [size] * size
                    tok_std += [False] * size; tok_asym += [asym] * size
                atom += size
                offset += size
                conf_id += 1
            tok_first += [chain_id == first_chain] * (len(tok_atom) - n_tok_before)
        x_gt_ca = x_gt[tok_atom]
        asym_ca = np.array(tok_asym)

        seed = random.random()
        if centre_override is not None and not self.training:
            centre = centre_override
        elif not self.training and len(ligand_atoms) == 1:
            centre = np.mean(np.concatenate(ligand_atoms, axis=0), axis=0)[None]
        elif seed < (cfg.train_spatial_crop_ligand_ratio if self.training else 1.0) and len(ligand_atoms) > 0:
            centre = random.choice(np.concatenate(ligand_atoms, axis=0))[None]
        elif seed < cfg.train_spatial_crop_ligand_ratio + cfg.train_spatial_crop_interface_ratio and len(set(asym_ca.tolist())) > 1:
            chain_same = asym_ca[None] * asym_ca[:, None]                                   # [legacy] product of the ids
            dist = np.linalg.norm(x_gt_ca[:, None] - x_gt_ca[None], axis=-1) + chain_same * 100
            mask = np.any(dist < cfg.train_spatial_crop_interface_threshold, axis=-1)
            centre = random.choice(x_gt_ca[mask])[None] if mask.sum() > 0 else random.choice(x_gt_ca)[None]
        else:
            centre = random.choice(x_gt_ca)[None]
        dist = np.linalg.norm(x_gt_ca - centre, axis=-1)
        if first_chain is not None:
            dist = np.where(np.array(tok_first, dtype=np.bool_), -1.0, dist)
        token_order = np.argsort(dist)

        selected = np.zeros([conf_id], dtype=np.bool_)
        n_atoms = n_tokens = 0
        for t in token_order.tolist():
            c = tok_conf[t]
            if selected[c]:
                continue
            size = tok_size[t]
            if n_atoms + size > cfg.atom_crop_size:
                break
            add_tokens = 1 if tok_std[t] else size
            if n_tokens + add_tokens > cfg.crop_size:
                break
            selected[c] = True
            n_atoms += size
            n_tokens += add_tokens
        if not selected.any():
            raise ValueError(f"{name}: the component closest to the crop centre does not fit into the crop "
                             f"(crop_size={cfg.crop_size}, atom_crop_size={cfg.atom_crop_size})")

        start = 0
        for chain_id in order:
            chain = chains[chain_id]
            num = len(chain["ccds"])
            conf_mask = selected[start:start + num]
            start += num
            if not conf_mask.any():
                del chains[chain_id]
                continue
            atom_mask = np.repeat(conf_mask, chain["conformer_sizes"])
            chain["x_gt"] = chain["x_gt"][atom_mask]
            chain["atom_to_conformer_atom"] = chain["atom_to_conformer_atom"][atom_mask]
            for key in ("restype", "residue_index", "conformer_sizes", "key_res_feat", "is_pocket", "is_protein",
                        "is_short_poly", "is_ligand", "asym_id", "sym_id", "entity_id"):
                chain[key] = chain[key][conf_mask]
            chain["ccds"] = [ccd for ccd, keep in zip(chain["ccds"], conf_mask) if keep]
            for key in ("msa", "deletion_matrix", "msa_all_seq", "deletion_matrix_all_seq"):
                if key in chain:
                    chain[key] = chain[key][:, conf_mask]

    # ------------------------------------------------------------------ MSA
    @staticmethod
    def _profile(msa: np.ndarray, rows_per_chunk: int = 2048) -> np.ndarray:
        """Per-column residue frequencies ``[n_cols, 32]`` without a one-hot of the whole MSA (exact counts)."""
        n_rows, n_cols = msa.shape
        if msa.size and (msa.min() < 0 or msa.max() > 31):
            raise ValueError(f"MSA residue codes must be in [0, 31], got [{msa.min()}, {msa.max()}]")
        counts = np.zeros(n_cols * 32, dtype=np.int64)
        offsets = (np.arange(n_cols, dtype=np.int64) * 32)[None]
        for s in range(0, n_rows, rows_per_chunk):
            counts += np.bincount((msa[s:s + rows_per_chunk].astype(np.int64) + offsets).ravel(), minlength=n_cols * 32)
        return (torch.from_numpy(counts.reshape(n_cols, 32).astype(np.float32)) / n_rows).numpy()

    @staticmethod
    def _deletion_mean(deletion_matrix: np.ndarray) -> np.ndarray:
        """``atan(sum(deletions) / 3) * 2 / pi`` per column [legacy: a sum, not a mean]."""
        return (torch.atan(torch.from_numpy(np.sum(deletion_matrix, axis=0, dtype=np.int64)) / 3.0) * (2.0 / torch.pi)).numpy()

    @staticmethod
    def _split_mixed_entities(chains: "OrderedDict[str, Dict[str, Any]]", chain_ids: Sequence[str]) -> None:
        """Give ligand chains that share an entity with a receptor chain their own entity (in place).

        A peptide ligand chain with the sequence of a receptor chain gets the receptor's entity id. The MSA merge
        stacks the MSAs of all chains of an entity, which fails for a receptor MSA and a ligand pseudo-MSA (the old
        loader crashed whenever both chains survived the crop). Only such entities are changed: their ligand chains
        get a new entity id (after all others) and sym ids 0, 1, ... in chain order.
        """
        classes: Dict[int, set] = {}
        for c in chain_ids:
            classes.setdefault(int(chains[c]["entity_id"][0]), set()).add(chains[c]["chain_class"])
        next_entity = max(int(chains[c]["entity_id"].max()) for c in chain_ids) + 1
        for entity in sorted(e for e, cls in classes.items() if len(cls) > 1):
            ligand_chains = [c for c in chain_ids if int(chains[c]["entity_id"][0]) == entity and chains[c]["chain_class"] == "ligand"]
            ligand_chains.sort(key=lambda c: int(chains[c]["asym_id"][0]))
            for sym_id, c in enumerate(ligand_chains):
                chains[c]["entity_id"] = np.full_like(chains[c]["entity_id"], next_entity)
                chains[c]["sym_id"] = np.full_like(chains[c]["sym_id"], sym_id)
            next_entity += 1

    def _pair_and_merge(
            self,
            chains: "OrderedDict[str, Dict[str, Any]]",
            chain_class: Mapping[str, str],
            asym_to_chain: Mapping[int, str],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Pair and merge the MSAs of all chains and concatenate the conformer-level arrays in the merged order."""
        chain_ids = list(chains)
        concat_keys = ("x_gt", "atom_to_conformer_atom", "residue_index", "conformer_sizes", "restype", "is_protein",
                       "is_short_poly", "is_ligand", "is_pocket", "key_res_feat")
        if len(chain_ids) == 1 and chain_class[chain_ids[0]] == "ligand":
            chain = chains[chain_ids[0]]
            raw = {k: chain[k] for k in concat_keys + ("asym_id", "sym_id", "entity_id")}
            raw["msa"] = np.repeat(chain["msa"][:1], 256, axis=0)
            raw["deletion_matrix"] = np.repeat(chain["deletion_matrix"][:1], 256, axis=0)
            order = chain_ids
        else:
            self._split_mixed_entities(chains, chain_ids)
            homo = {}
            for chain_id in chain_ids:
                chain = chains[chain_id]
                homo[chain_id] = {
                    "asym_id": chain["asym_id"].copy(), "sym_id": chain["sym_id"].copy(), "entity_id": chain["entity_id"].copy(),
                    "chain_class": chain["chain_class"], "sequence_3": chain["sequence_3"],
                    "msa": chain["msa"], "deletion_matrix": chain["deletion_matrix"],
                }
                if "msa_all_seq" in chain:
                    for key in ("msa_all_seq", "deletion_matrix_all_seq", "msa_species_identifiers_all_seq"):
                        homo[chain_id][key] = chain[key]
            merged = pair_and_merge(homo, is_homomer_or_monomer=False)
            asym_order = list(OrderedDict.fromkeys(merged["asym_id"].tolist()))
            order = [asym_to_chain[a] for a in asym_order]
            raw = {"msa": merged["msa"], "deletion_matrix": merged["deletion_matrix"], "asym_id": merged["asym_id"],
                   "sym_id": merged["sym_id"], "entity_id": merged["entity_id"]}
            for key in concat_keys:
                raw[key] = np.concatenate([chains[c][key] for c in order], axis=0)
        raw["profile"] = self._profile(raw["msa"])
        raw["deletion_mean"] = self._deletion_mean(raw["deletion_matrix"])
        return raw, order

    # ------------------------------------------------------------------ atoms and tokens
    def _tokens(
            self,
            raw: Dict[str, Any],
            ccds: List[str],
            conformers: Mapping[str, CCDConformer],
            conformer_class: List[str],
            asym_to_chain: Mapping[int, str],
            name: str,
            ref_mol: Optional[Chem.Mol],
    ) -> Tuple[Dict[str, np.ndarray], SystemMeta]:
        """Atom / token index maps, reference features, token pair features and the metadata."""
        sizes = raw["conformer_sizes"]
        n_conf = len(ccds)
        is_std = np.array([rc.is_standard(ccd) for ccd in ccds], dtype=np.bool_)
        atom_index = raw["atom_to_conformer_atom"]

        tokens_per_conf = np.where(is_std, 1, sizes)
        atom_start = np.cumsum(sizes) - sizes
        token_start = np.cumsum(tokens_per_conf) - tokens_per_conf
        n_atom, n_token = int(sizes.sum()), int(tokens_per_conf.sum())
        atom_to_conformer = np.repeat(np.arange(n_conf, dtype=np.int64), sizes)
        token_to_conformer = np.repeat(np.arange(n_conf, dtype=np.int64), tokens_per_conf)
        local_atom = np.arange(n_atom, dtype=np.int64) - atom_start[atom_to_conformer]
        atom_to_token = token_start[atom_to_conformer] + np.where(is_std[atom_to_conformer], 0, local_atom)
        local_token = np.arange(n_token, dtype=np.int64) - token_start[token_to_conformer]
        split_sizes = np.where(is_std[token_to_conformer], sizes[token_to_conformer], 1).astype(np.int64)
        centre = atom_start[token_to_conformer] + local_token
        beta = centre.copy()
        ref_feat, atom_names, atom_elements = [], [], []
        for c, ccd in enumerate(ccds):
            conf = conformers[ccd]
            inner = atom_index[atom_start[c]:atom_start[c] + sizes[c]]
            ref_feat.append(conf.ref_feat[inner])
            atom_names += [conf.atom_names[i] for i in inner]
            atom_elements += [PeriodicTable[conf.elements[i]] for i in inner]
            if is_std[c]:
                t = token_start[c]
                centre[t] = atom_start[c] + int(np.flatnonzero(inner == conf.index(rc.standard_ccd_to_token_centre_atom_name[ccd]))[0])
                beta[t] = atom_start[c] + int(np.flatnonzero(inner == conf.index(rc.standard_ccd_to_token_pseudo_beta_atom_name[ccd]))[0])
        ref_feat = (np.concatenate(ref_feat, axis=0) if len(ref_feat) > 1 else ref_feat[0]).astype(np.float32)
        ref_pos = ref_feat[..., :3]
        if self.config.use_x_gt_ligand_as_ref_pos:
            lig = raw["is_ligand"][atom_to_conformer].astype(np.bool_)
            ref_pos[lig] = raw["x_gt"][lig] - np.mean(raw["x_gt"][lig], axis=0, keepdims=True)

        meta = SystemMeta(
            system_id=name, ref_mol=ref_mol, ccds=list(ccds), conformer_sizes=sizes.copy(), residue_index=raw["residue_index"].copy(),
            asym_id=raw["asym_id"].copy(), conformer_is_ligand=[c == "ligand" for c in conformer_class],
            chain_ids=dict(asym_to_chain), atom_names=atom_names, atom_elements=atom_elements, atom_to_conformer_atom=atom_index.copy(),
        )

        feats = {"x_gt": raw["x_gt"]}
        for key in ("is_protein", "is_short_poly", "is_ligand", "residue_index", "restype", "asym_id", "entity_id", "sym_id",
                    "deletion_mean", "profile", "is_pocket", "key_res_feat"):
            feats[key] = raw[key][token_to_conformer]
        feats["msa"] = raw["msa"][:, token_to_conformer]
        feats["deletion_matrix"] = raw["deletion_matrix"][:, token_to_conformer]

        ref_pos = centre_random_augmentation_np_apply(ref_pos, atom_to_conformer).astype(np.float32)
        ref_feat[:, :3] = ref_pos

        token_bonds = np.zeros([n_token, n_token], dtype=np.float32)
        pair_feat = np.zeros([n_token, n_token, 42], dtype=np.float32)
        for c in np.flatnonzero(~is_std).tolist():
            conf = conformers[ccds[c]]
            inner = atom_index[atom_start[c]:atom_start[c] + sizes[c]]
            t0, t1 = token_start[c], token_start[c] + sizes[c]
            token_bonds[t0:t1, t0:t1] = conf.bonds[np.ix_(inner, inner)]
            pair_feat[t0:t1, t0:t1] = conf.pair_feat[np.ix_(inner, inner)]

        if self.training:
            if random.random() < self.config.train_charility_augmentation_ratio:
                chirality = ref_feat[:, 158:161]
                replace = np.zeros_like(chirality)
                replace[:, 2] = 1
                is_ligand_atom = feats["is_ligand"][atom_to_token]
                hide = (np.random.randint(0, 2, [len(is_ligand_atom)]) * is_ligand_atom).astype(np.bool_)
                ref_feat[:, 158:161] = np.where(hide[:, None], replace, chirality)

        feats.update({
            "atom_to_token": atom_to_token.astype(np.int64), "atom_to_conformer": atom_to_conformer, "ref_feat": ref_feat,
            "ref_pos": ref_pos, "split_sizes": split_sizes, "token_to_centre_atom": centre.astype(np.int64),
            "token_to_pseudo_beta_atom": beta.astype(np.int64), "token_bonds": token_bonds, "ligand_pair_feat": pair_feat,
            "atom_mask": np.ones_like(raw["x_gt"][..., 0]).astype(np.float32),
            "token_mask": np.ones_like(feats["asym_id"]).astype(np.float32),
        })
        return feats, meta

    # ------------------------------------------------------------------ tensors
    def _transform(self, feats: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Target / MSA features, padding (training), template and the final type correction."""
        cfg = self.config
        t = {k: torch.from_numpy(v) for k, v in feats.items()}
        t["target_feat"] = torch.cat([
            F.one_hot(t["restype"].long(), 32).float(),
            t.pop("profile").float(),
            t.pop("deletion_mean")[..., None].float(),
        ], dim=-1)

        msa, deletion = t.pop("msa"), t.pop("deletion_matrix")
        pi = torch.acos(torch.zeros(1)) * 2
        samples = []
        for _ in range(cfg.num_msa_samples):
            inds = [0] + torch.randperm(len(msa))[:cfg.max_msa_clusters - 1].tolist()
            msa, deletion = msa[inds], deletion[inds]                     # [legacy] later samples come from earlier ones
            has_deletion = torch.clamp(deletion.float(), min=0., max=1.)
            deletion_value = torch.atan(deletion / 3.) * (2. / pi)
            samples.append(torch.cat([F.one_hot(msa.long(), 32).float(), has_deletion[..., None].float(),
                                      deletion_value[..., None].float()], dim=-1))
        t["msa_feat"] = samples[0]
        if len(samples) > 1:
            t["msa_feat_resampled"] = torch.stack(samples[1:], dim=0)
        random.random()          # [legacy] draw of the removed inter-chain bond search; keeps the random stream aligned

        if self.training:
            self._pad(t)

        token_mask = t["token_mask"]
        z_mask = token_mask[None] * token_mask[:, None]
        x_beta = t["x_gt"][t["token_to_pseudo_beta_atom"]]
        protein2d = t["is_protein"][None] * t["is_protein"][:, None]
        dgram = dgram_from_positions(x_beta, no_bins=39) * protein2d[..., None] * z_mask[..., None]
        if self.training and random.random() > cfg.train_use_template_ratio:
            t["templ_mask"] = torch.tensor(1, dtype=torch.float32)
            bert_mask = torch.rand([len(x_beta)]) > random.random() * (1 - cfg.train_template_mask_max_ratio)
            pair_mask = (bert_mask[None] * bert_mask[:, None]) * z_mask * protein2d
        elif self.training:
            t["templ_mask"] = torch.tensor(0, dtype=torch.float32)
            pair_mask = z_mask * protein2d
        else:
            t["templ_mask"] = torch.tensor(1, dtype=torch.float32)
            pair_mask = z_mask * protein2d
        t["templ_feat"] = torch.cat([dgram * pair_mask[..., None], pair_mask[..., None]], dim=-1).float()

        is_short_poly = t.pop("is_short_poly")
        t["is_protein"] = t["is_protein"] + is_short_poly
        t["is_ligand"] = t["is_ligand"] - is_short_poly
        return t

    def _pad(self, t: Dict[str, torch.Tensor]) -> None:
        """Pad token and atom dimensions to the crop sizes (training)."""
        pad_atom = self.config.atom_crop_size - len(t["x_gt"])
        pad_token = self.config.crop_size - len(t["residue_index"])
        if pad_token > 0:
            for k in _UNPADDED_KEYS_TOKEN:
                t[k] = F.pad(t[k], [0, pad_token])
            for k in ("target_feat", "msa_feat", "key_res_feat", "msa_feat_resampled"):
                if k in t:
                    t[k] = F.pad(t[k], [0, 0, 0, pad_token])
            t["token_bonds"] = F.pad(t["token_bonds"], [0, pad_token, 0, pad_token])
            t["ligand_pair_feat"] = F.pad(t["ligand_pair_feat"], [0, 0, 0, pad_token, 0, pad_token])
        if pad_atom > 0:
            for k in ("atom_mask", "atom_to_conformer", "atom_to_token"):
                t[k] = F.pad(t[k], [0, pad_atom])
            for k in ("x_gt", "ref_feat", "ref_pos"):
                t[k] = F.pad(t[k], [0, 0, 0, pad_atom])


# ======================================================================================
# Dataset, collator, dataloader
# ======================================================================================
class PhysDockDataset(Dataset):
    """Map-style dataset of PhysDock systems.

    Args:
        entries: Systems, as :class:`SystemEntry` or paths.
        msa_features_dir: Directory with the MSA features. If ``uniprot_msa_features_dir`` is not given and this
            directory contains ``msa_features/`` and ``uniprot_msa_features/``, those are used.
        uniprot_msa_features_dir: Directory with the UniProt MSA features.
        ccd: The CCD library (or its path); defaults to :func:`default_ccd_path`.
        config: Featurisation settings.
        training: Training mode, see :class:`SystemFeaturizer`.
        sample_weights: Optional sampling weight of every entry (used by ``PhysDockDataloader(weighted_sampling=True)``).
        on_error: What to do if a system fails: ``"raise"``, ``"skip"`` (return ``None``; the collator passes it on) or
            ``"resample"`` (log and featurise a random other entry instead; for training).
        max_retries: Attempts for ``on_error="resample"``.
    """

    def __init__(
            self,
            entries: Sequence[Union[SystemEntry, PathLike]],
            msa_features_dir: PathLike,
            uniprot_msa_features_dir: Optional[PathLike] = None,
            ccd: Union[CCDLibrary, PathLike, None] = None,
            config: Optional[FeatureConfig] = None,
            training: bool = False,
            sample_weights: Optional[Sequence[float]] = None,
            on_error: Literal["raise", "skip", "resample"] = "raise",
            max_retries: int = 10,
    ) -> None:
        if on_error not in ("raise", "skip", "resample"):
            raise ValueError(f"on_error must be 'raise', 'skip' or 'resample', got {on_error!r}")
        self.entries = [e if isinstance(e, SystemEntry) else SystemEntry(str(e)) for e in entries]
        if sample_weights is not None and len(sample_weights) != len(self.entries):
            raise ValueError(f"{len(sample_weights)} sample weights for {len(self.entries)} entries")
        msa_dir = Path(msa_features_dir)
        if uniprot_msa_features_dir is None:
            if (msa_dir / "msa_features").is_dir() and (msa_dir / "uniprot_msa_features").is_dir():
                msa_dir, uniprot_dir = msa_dir / "msa_features", msa_dir / "uniprot_msa_features"
            else:
                raise ValueError(f"uniprot_msa_features_dir is required ({msa_dir} has no msa_features/ and uniprot_msa_features/)")
        else:
            uniprot_dir = Path(uniprot_msa_features_dir)
        for d in (msa_dir, uniprot_dir):
            if not d.is_dir():
                raise FileNotFoundError(f"MSA feature directory not found: {d}")
        self.ccd = ccd if isinstance(ccd, CCDLibrary) else CCDLibrary(ccd)
        self.featurizer = SystemFeaturizer(self.ccd, msa_dir, uniprot_dir, config, training)
        self.sample_weights = None if sample_weights is None else torch.as_tensor(sample_weights, dtype=torch.double)
        self.on_error = on_error
        self.max_retries = max_retries

    @property
    def config(self) -> FeatureConfig:
        """The featurisation settings."""
        return self.featurizer.config

    @property
    def training(self) -> bool:
        """Whether the dataset is in training mode."""
        return self.featurizer.training

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Optional[Sample]:
        entry = self.entries[index]
        try:
            return self.featurizer(entry)
        except Exception as error:  # noqa: BLE001 - the policy decides
            if self.on_error == "raise":
                raise
            logger.warning("failed to featurise %s: %s: %s", entry.path, type(error).__name__, error)
            if self.on_error == "skip":
                return None
        for _ in range(self.max_retries):
            other = self.entries[random.randrange(len(self.entries))]
            try:
                return self.featurizer(other)
            except Exception as error:  # noqa: BLE001
                logger.warning("failed to featurise %s: %s: %s", other.path, type(error).__name__, error)
        raise RuntimeError(f"no system could be featurised in {self.max_retries} attempts after {entry.path}")

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_directory(cls, systems_dir: PathLike, msa_features_dir: PathLike, uniprot_msa_features_dir: Optional[PathLike] = None,
                       recursive: bool = True, **kwargs: Any) -> "PhysDockDataset":
        """All ``*.pkl.gz`` systems of a directory (sorted)."""
        pattern = os.path.join(str(systems_dir), "**", "*.pkl.gz") if recursive else os.path.join(str(systems_dir), "*.pkl.gz")
        paths = sorted(glob.glob(pattern, recursive=recursive))
        if not paths:
            raise FileNotFoundError(f"no *.pkl.gz system in {systems_dir}")
        return cls(paths, msa_features_dir, uniprot_msa_features_dir, **kwargs)

    @classmethod
    def for_screening(cls, system_path: PathLike, smiles: Iterable[str], msa_features_dir: PathLike,
                      uniprot_msa_features_dir: Optional[PathLike] = None, **kwargs: Any) -> "PhysDockDataset":
        """One entry per SMILES; the system id of an entry is ``md5(smiles)``."""
        entries = [SystemEntry(str(system_path), smiles=s, system_id=convert_md5_string(s)) for s in smiles]
        return cls(entries, msa_features_dir, uniprot_msa_features_dir, **kwargs)

    @classmethod
    def from_train_val(cls, root: PathLike = _REPO_ROOT / "data" / "v1", **kwargs: Any) -> "PhysDockDataset":
        """The released training set (``scripts/download_dataset.py``): ``train_val/``, MSA features and sample weights.

        Defaults to training mode with ``on_error="resample"``.
        """
        root = Path(root)
        paths = sorted(glob.glob(str(root / "train_val" / "*.pkl.gz")))
        if not paths:
            raise FileNotFoundError(f"no system in {root / 'train_val'}")
        weights = None
        weights_file = root / "train_val_samples_weights.json"
        if weights_file.is_file():
            table = load_json(weights_file)
            entries = [SystemEntry(p) for p in paths]
            missing = [e.name for e in entries if e.name not in table]
            if missing:
                raise ValueError(f"{len(missing)} systems have no sampling weight, e.g. {missing[:3]}")
            weights = [table[e.name] for e in entries]
        kwargs.setdefault("training", True)
        kwargs.setdefault("on_error", "resample")
        return cls(paths, root / "msa_features", root / "uniprot_msa_features", sample_weights=weights, **kwargs)


class PhysDockCollator:
    """Collate :class:`Sample` objects.

    With one sample per batch (what the model expects) the sample is returned unchanged, without a batch dimension
    (``None`` stays ``None``). Several samples are padded to the largest size of every symbolic dimension and stacked
    along a new leading dimension; ``meta`` is then a list.
    """

    def __call__(self, samples: Sequence[Optional[Sample]]) -> Optional[Sample]:
        if len(samples) == 1:
            return samples[0]
        samples = [s for s in samples if s is not None]
        if not samples:
            return None
        keys = set(samples[0].features)
        if any(set(s.features) != keys for s in samples):
            raise ValueError("samples with different feature keys cannot be batched")
        sizes: Dict[str, int] = {}
        for s in samples:
            for key, value in s.features.items():
                for dim, size in zip(FEATURES[key].shape, value.shape):
                    if isinstance(dim, str):
                        sizes[dim] = max(sizes.get(dim, 0), size)
        out = {}
        for key in keys:
            dims = FEATURES[key].shape
            padded = []
            for s in samples:
                value = s.features[key]
                pad = []
                for dim, size in reversed(list(zip(dims, value.shape))):
                    pad += [0, (sizes[dim] - size) if isinstance(dim, str) else 0]
                padded.append(F.pad(value, pad) if any(pad) else value)
            out[key] = torch.stack(padded, dim=0)
        return Sample(out, [s.meta for s in samples])  # type: ignore[arg-type]


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: seed NumPy from the worker's torch seed and use one torch thread.

    Torch seeds ``random`` and ``torch`` per worker, but not NumPy; without this all workers would draw the same
    NumPy random numbers (MSA padding, key-residue masks, reference-conformer rotations).
    """
    np.random.seed(torch.initial_seed() % 2 ** 32)
    torch.set_num_threads(1)


class PhysDockDataloader(DataLoader):
    """``torch.utils.data.DataLoader`` with the PhysDock defaults.

    Args:
        dataset: The dataset.
        batch_size: Samples per batch (the model expects 1).
        shuffle: Shuffle the entries (ignored with ``weighted_sampling``).
        num_workers: Worker processes.
        weighted_sampling: Draw entries with replacement according to ``dataset.sample_weights``.
        num_samples: Samples per epoch for weighted sampling (default: ``len(dataset)``).
        seed: Seed of the sampler (and of the worker seeds), for reproducible epochs.
        **kwargs: Further ``DataLoader`` arguments (``pin_memory``, ``prefetch_factor``, ...).

    Raises:
        ValueError: If weighted sampling is requested for a dataset without weights.
    """

    def __init__(
            self,
            dataset: PhysDockDataset,
            batch_size: int = 1,
            shuffle: bool = False,
            num_workers: int = 0,
            weighted_sampling: bool = False,
            num_samples: Optional[int] = None,
            seed: Optional[int] = None,
            **kwargs: Any,
    ) -> None:
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

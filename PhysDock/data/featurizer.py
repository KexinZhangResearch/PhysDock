"""System pickle -> model features (SystemFeaturizer).

A system is a pickled ``{chain_id: chain}`` dict (``prepare_system.py``); numeric chain ids are ligand chains. Every
chain has ``ccds``, ``all_atom_positions`` / ``all_atom_mask`` (one [n_atoms_of_ccd, 3] / [n_atoms_of_ccd] array per
component, library atom order) and optional PLIP flags (``"salt bridges"`` ...). Protein chains need MSA features
``<md5("protein:" + sequence)>.pkl.gz`` in the two MSA directories.

With the same states of Python's ``random``, NumPy's and torch's global RNGs, the features are bit-identical to the
released pipeline, including its quirks marked [legacy] (MSA padded to 16384 rows before the profile, deletion
"mean" is a sum, chained MSA resampling, key-residue masking also at inference). Inputs on which the old loader
crashed are handled: empty chains are dropped, UNK components are dropped, standard residues also need their centre
and pseudo-beta atom, ligand chains sharing a receptor entity get their own entity, and a SMILES ligand is always
inside the crop (centred on the reference ligand).
"""
import logging
import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, NamedTuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

import PhysDock.data.constants as rc
from PhysDock.data.ccd import CCDConformer, CCDLibrary
from PhysDock.data.constants.PDBData import protein_letters_3to1_extended
from PhysDock.data.constants.periodic_table import PeriodicTable
from PhysDock.data.meta import SystemMeta
from PhysDock.data.tools.feature_processing_multimer import pair_and_merge
from PhysDock.data.tools.rdkit import get_features_from_smi
from PhysDock.utils.io_utils import convert_md5_string, load_pkl
from PhysDock.utils.tensor_utils import centre_random_augmentation_np_apply, dgram_from_positions

logger = logging.getLogger(__name__)

_SMILES_CCD, _SMILES_CHAIN = "XXX", "99"      # component / chain id of a SMILES ligand
_TOKEN_KEYS_1D = ("restype", "residue_index", "is_protein", "is_short_poly", "is_ligand", "asym_id", "entity_id",
                  "sym_id", "token_mask", "token_to_centre_atom", "token_to_pseudo_beta_atom", "split_sizes", "is_pocket")


class MissingMSAError(FileNotFoundError):
    """MSA features of a protein chain are missing."""


@dataclass(frozen=True)
class FeatureConfig:
    """Featurisation settings (defaults = released pipeline). train_* are only used with training=True."""
    crop_size: Optional[int] = 256              # max tokens; None = no crop
    atom_crop_size: int = 2048                  # max atoms of a crop
    msa_rows: int = 16384                       # rows of the merged MSA (padded by resampling) [legacy]
    max_msa_clusters: int = 128                 # MSA rows in msa_feat (query + random rows)
    num_msa_samples: int = 1                    # > 1: further samples in msa_feat_resampled (redocking rounds)
    key_res_random_mask_ratio: float = 0.5      # prob. that a key-residue flag is masked [legacy: also at inference]
    pocket_type: str = "atom"                   # "atom": any atom / "ca": CA within the cutoff
    pocket_cutoff: float = 6.0
    pocket_dist_type: str = "ligand"            # "ligand": any ligand atom / "ligand_centre": ligand centre
    use_pocket: bool = True
    use_key_res: bool = True
    use_x_gt_ligand_as_ref_pos: bool = False    # ablation only
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
    train_use_template_ratio: float = 0.25      # prob. that the template is used
    train_template_mask_max_ratio: float = 0.4

    def __post_init__(self) -> None:
        if self.crop_size is not None and (self.crop_size < 1 or self.atom_crop_size < 1):
            raise ValueError("crop sizes must be positive")
        if self.max_msa_clusters < 1 or self.num_msa_samples < 1:
            raise ValueError("max_msa_clusters and num_msa_samples must be >= 1")
        if self.pocket_type not in ("atom", "ca") or self.pocket_dist_type not in ("ligand", "ligand_centre"):
            raise ValueError(f"bad pocket_type {self.pocket_type!r} / pocket_dist_type {self.pocket_dist_type!r}")


@dataclass(frozen=True)
class SystemEntry:
    """One system: pickle path, optional SMILES that replaces the ligand (screening), optional id."""
    path: str
    smiles: Optional[str] = None
    system_id: Optional[str] = None
    remove_ligand: bool = False

    @property
    def name(self) -> str:
        if self.system_id is not None:
            return self.system_id
        base = os.path.basename(self.path)
        return base[:-7] if base.endswith(".pkl.gz") else base[:-4] if base.endswith(".pkl") else base


class Sample(NamedTuple):
    features: dict      # {key: tensor}, see PhysDock.data.FEATURES
    meta: SystemMeta


def msa_feat_for_round(features, round_id: int) -> torch.Tensor:
    """msa_feat for round 0, msa_feat_resampled[round_id - 1] afterwards -> [N_msa, N_token, 34]."""
    if round_id == 0:
        return features["msa_feat"]
    resampled = features.get("msa_feat_resampled")
    n = 1 if resampled is None else 1 + resampled.shape[-4]
    if not 0 <= round_id < n:
        raise IndexError(f"round {round_id} needs num_msa_samples > {round_id}, the features hold {n}")
    return resampled[..., round_id - 1, :, :, :]


@dataclass
class Chain:
    """Complete components of one chain; n = components, a = atoms."""
    chain_id: str
    chain_class: str                        # "protein" or "ligand"
    entity_key: str                         # "-".join(input ccds): chains with the same key form an entity
    ccds: List[str]                         # (n,)
    x_gt: np.ndarray                        # [a, 3]
    atom_to_conformer_atom: np.ndarray      # [a] atom index inside its library conformer
    residue_index: np.ndarray               # [n] position in the input chain
    conformer_sizes: np.ndarray             # [n] atoms per component
    restype: np.ndarray                     # [n]
    is_protein: np.ndarray                  # [n]
    is_ligand: np.ndarray                   # [n]
    is_short_poly: np.ndarray               # [n] standard residues of a peptide-like ligand chain
    is_pocket: np.ndarray                   # [n]
    key_res_feat: np.ndarray                # [n, 7]
    msa: np.ndarray                         # [rows, n]
    deletion_matrix: np.ndarray             # [rows, n]
    msa_all_seq: Optional[np.ndarray] = None                     # [rows_all, n] UniProt MSA (proteins)
    deletion_matrix_all_seq: Optional[np.ndarray] = None
    msa_species_identifiers_all_seq: Optional[np.ndarray] = None  # [rows_all]
    asym_id: Optional[np.ndarray] = None    # [n], set by SystemFeaturizer._assembly
    sym_id: Optional[np.ndarray] = None
    entity_id: Optional[np.ndarray] = None

    def select(self, keep: np.ndarray) -> None:
        """Keep the components where keep [n] is True (in place)."""
        atoms = np.repeat(keep, self.conformer_sizes)
        self.x_gt = self.x_gt[atoms]
        self.atom_to_conformer_atom = self.atom_to_conformer_atom[atoms]
        for key in ("restype", "residue_index", "conformer_sizes", "key_res_feat", "is_pocket", "is_protein",
                    "is_short_poly", "is_ligand", "asym_id", "sym_id", "entity_id"):
            setattr(self, key, getattr(self, key)[keep])
        self.ccds = [ccd for ccd, k in zip(self.ccds, keep) if k]
        for key in ("msa", "deletion_matrix", "msa_all_seq", "deletion_matrix_all_seq"):
            if getattr(self, key) is not None:
                setattr(self, key, getattr(self, key)[:, keep])


class SystemFeaturizer:
    """Featurise one SystemEntry -> Sample. Uses the global random / numpy / torch RNGs in the legacy order."""

    def __init__(self, ccd: CCDLibrary, msa_features_dir, uniprot_msa_features_dir, config: FeatureConfig = None,
                 training: bool = False, msa_cache_size: int = 8) -> None:
        self.ccd = ccd
        self.msa_features_dir = str(msa_features_dir)
        self.uniprot_msa_features_dir = str(uniprot_msa_features_dir)
        self.config = config if config is not None else FeatureConfig()
        self.training = training
        if training and self.config.crop_size is None:
            raise ValueError("training needs crop_size (features are padded to the crop sizes)")
        self.msa_cache_size = msa_cache_size
        self._msa_cache = OrderedDict()

    def __call__(self, entry: SystemEntry) -> Sample:
        cfg = self.config
        # 1. pocket / key-residue settings (random in training)
        if self.training:
            pocket_type = random.choices(["atom", "ca"], [cfg.train_pocket_type_atom_ratio, 1 - cfg.train_pocket_type_atom_ratio])[0]
            dist_type = random.choices(["ligand", "ligand_centre"],
                                       [cfg.train_pocket_dist_type_ligand_ratio, 1 - cfg.train_pocket_dist_type_ligand_ratio])[0]
            if dist_type == "ligand":
                lo, hi = cfg.train_pocket_cutoff_ligand_min, cfg.train_pocket_cutoff_ligand_max
            else:
                lo, hi = cfg.train_pocket_cutoff_ligand_centre_min, cfg.train_pocket_cutoff_ligand_centre_max
            cutoff = lo + random.random() * (hi - lo)
            use_pocket = random.random() < cfg.train_use_pocket_ratio
            use_key_res = random.random() < cfg.train_use_key_res_ratio
        else:
            pocket_type, cutoff, dist_type = cfg.pocket_type, cfg.pocket_cutoff, cfg.pocket_dist_type
            use_pocket, use_key_res = cfg.use_pocket, cfg.use_key_res

        # 2. read and check the system
        data = load_pkl(entry.path)
        _check_system(data, entry.path)
        receptor_ids = [c for c in data if not c.isdigit()]
        ligand_ids = [c for c in data if c.isdigit()]

        # 3. pocket of every receptor chain (from the reference ligand)
        ligand_atoms = [_present_atoms(data[c]) for c in ligand_ids]
        labels = OrderedDict((c, data[c]) for c in receptor_ids)
        pockets = {c: _pocket(data[c], ligand_atoms, pocket_type, dist_type, cutoff) for c in receptor_ids if ligand_ids}

        # 4. ligand chains, or the SMILES ligand
        conformers, ref_mol, crop_centre = {}, None, None
        if entry.smiles is not None:
            label, conformers[_SMILES_CCD], ref_mol = _smiles_ligand(entry.smiles)
            labels[_SMILES_CHAIN] = label
            if sum(len(a) for a in ligand_atoms):
                crop_centre = np.mean(np.concatenate(ligand_atoms, axis=0), axis=0)[None]
        elif not entry.remove_ligand:
            for c in ligand_ids:
                labels[c] = data[c]
            if len(ligand_ids) == 1 and len(data[ligand_ids[0]]["ccds"]) == 1:
                ref_mol = self.ccd.ref_mol(data[ligand_ids[0]]["ccds"][0])

        # 5. per-chain features (incomplete components dropped)
        chains = OrderedDict()
        for chain_id, label in labels.items():
            for ccd in label["ccds"]:
                if ccd not in conformers and not rc.is_unk(ccd):
                    conformers[ccd] = self.ccd.conformer(ccd)
            chain = self._chain(chain_id, label, conformers, pockets.get(chain_id), use_pocket, use_key_res, entry.path)
            if chain is None:
                logger.warning("%s: chain %s has no complete component and is skipped", entry.name, chain_id)
                continue
            chains[chain_id] = chain
        if not chains:
            raise ValueError(f"{entry.name}: no chain with a complete component")
        if ref_mol is not None and entry.smiles is None and ligand_ids[0] not in chains:
            ref_mol = None

        # 6. chain / entity / symmetry ids, 7. crop, 8. MSA merge, 9. atoms and tokens, 10. tensors
        asym_to_chain = self._assembly(chains)
        if cfg.crop_size is not None:
            self._crop(chains, conformers, crop_centre, entry.name, _SMILES_CHAIN if entry.smiles is not None else None)
        raw, order = self._pair_and_merge(chains, asym_to_chain)
        ccds = [ccd for c in order for ccd in chains[c].ccds]
        is_lig_conf = [chains[c].chain_class == "ligand" for c in order for _ in chains[c].ccds]
        feats, meta = self._tokens(raw, ccds, conformers, is_lig_conf, asym_to_chain, entry.name, ref_mol)
        return Sample(self._transform(feats), meta)

    # ------------------------------------------------------------------ chains
    def _load_msa(self, md5: str, sequence: str, chain_id: str, path: str) -> dict:
        if md5 in self._msa_cache:
            self._msa_cache.move_to_end(md5)
            return self._msa_cache[md5]
        files = [os.path.join(self.msa_features_dir, f"{md5}.pkl.gz"), os.path.join(self.uniprot_msa_features_dir, f"{md5}.pkl.gz")]
        missing = [f for f in files if not os.path.isfile(f)]
        if missing:
            raise MissingMSAError(f"{path}: MSA features of chain {chain_id!r} not found: {missing}; "
                                  f"generate them with scripts/prepare/homo_search.py for\n>{md5}\n{sequence}")
        msa = {**load_pkl(files[0]), **load_pkl(files[1])}
        for key in ("msa", "deletion_matrix", "msa_all_seq", "deletion_matrix_all_seq", "msa_species_identifiers_all_seq"):
            if key not in msa:
                raise ValueError(f"MSA features {md5} lack {key!r}")
            if key != "msa_species_identifiers_all_seq" and (msa[key].ndim != 2 or msa[key].shape[1] != len(sequence)):
                raise ValueError(f"MSA features {md5} ({key} {msa[key].shape}) do not match the {len(sequence)} "
                                 f"residues of chain {chain_id!r} in {path}")
        if self.msa_cache_size > 0:
            self._msa_cache[md5] = msa
            if len(self._msa_cache) > self.msa_cache_size:
                self._msa_cache.popitem(last=False)
        return msa

    def _chain(self, chain_id, label, conformers, pocket, use_pocket, use_key_res, path) -> Optional[Chain]:
        """Chain of the complete components of an input chain, or None if there is none."""
        ccds_in = list(label["ccds"])
        n = len(ccds_in)
        if n == 0:
            return None
        chain_class = "ligand" if chain_id.isdigit() else "protein"
        if chain_class == "protein":
            sequence = "".join(protein_letters_3to1_extended.get(ccd, "X") for ccd in ccds_in)
            msa = self._load_msa(convert_md5_string(f"protein:{sequence}"), sequence, chain_id, path)
        else:
            codes = np.array([[rc.standard_ccds.index(c) if c in rc.standard_ccds else 20 for c in ccds_in]] * 2, dtype=np.int8)
            msa = {"msa": codes, "deletion_matrix": np.zeros_like(codes)}

        if "salt bridges" in label and use_key_res:
            key_res = np.stack([label["salt bridges"], label["pi-cation interactions"], label["hydrophobic interactions"],
                                label["pi-stacking"], label["hydrogen bonds"], label["metal complexes"],
                                np.zeros_like(label["salt bridges"])], axis=-1).astype(np.float32)
        else:
            key_res = np.zeros([n, 7], dtype=np.float32)
        key_res = key_res * (np.random.random([n, 7]) > self.config.key_res_random_mask_ratio)    # [legacy] also at inference
        if pocket is None or not use_pocket:
            pocket = np.zeros([n], dtype=np.float32)

        kept = np.zeros([n], dtype=np.bool_)
        x_gt, atom_index, residue_index, sizes, restype, ccds = [], [], [], [], [], []
        for i, ccd in enumerate(ccds_in):
            if rc.is_unk(ccd):
                continue
            conf = conformers[ccd]
            mask = np.asarray(label["all_atom_mask"][i]).astype(np.bool_)
            if mask.shape[0] != conf.num_atoms:
                raise ValueError(f"{path}: chain {chain_id!r} component {i} ({ccd!r}) has {mask.shape[0]} atom slots, "
                                 f"the CCD library has {conf.num_atoms}")
            if not mask.any():
                continue
            if rc.is_standard(ccd):
                if not mask[1] or (ccd != "GLY" and not mask[4]):          # [legacy] CA and CB
                    continue
                centre = conf.index(rc.standard_ccd_to_token_centre_atom_name[ccd])
                beta = conf.index(rc.standard_ccd_to_token_pseudo_beta_atom_name[ccd])
                if centre is None or beta is None or not mask[centre] or not mask[beta]:
                    continue
            kept[i] = True
            x_gt.append(np.asarray(label["all_atom_positions"][i])[mask])
            atom_index.append(np.arange(conf.num_atoms)[mask])
            residue_index.append(i)
            sizes.append(int(mask.sum()))
            restype.append(rc.standard_ccds.index(ccd) if ccd in rc.standard_ccds else 20)
            ccds.append(ccd)
        if not kept.any():
            return None
        if msa["msa"].shape[1] != n or msa["deletion_matrix"].shape[1] != n:
            raise ValueError(f"{path}: MSA of chain {chain_id!r} has {msa['msa'].shape[1]} columns for {n} components")

        m = len(ccds)
        protein = chain_class == "protein"
        return Chain(
            chain_id=chain_id, chain_class=chain_class, entity_key="-".join(ccds_in), ccds=ccds,
            x_gt=np.concatenate(x_gt, axis=0),
            atom_to_conformer_atom=np.concatenate(atom_index, axis=0, dtype=np.int32),
            residue_index=np.array(residue_index, dtype=np.int64),
            conformer_sizes=np.array(sizes, dtype=np.int64),
            restype=np.array(restype, dtype=np.int64),
            is_protein=np.array([protein] * m).astype(np.float32),
            is_ligand=np.array([not protein] * m).astype(np.float32),
            is_short_poly=np.array([not protein and m >= 2 and rc.is_standard(c) for c in ccds]).astype(np.float32),
            is_pocket=pocket[kept],
            key_res_feat=key_res[kept],
            msa=msa["msa"][:, kept],
            deletion_matrix=msa["deletion_matrix"][:, kept],
            msa_all_seq=msa["msa_all_seq"][:, kept] if "msa_all_seq" in msa else None,
            deletion_matrix_all_seq=msa["deletion_matrix_all_seq"][:, kept] if "msa_all_seq" in msa else None,
            msa_species_identifiers_all_seq=msa.get("msa_species_identifiers_all_seq"),
        )

    def _assembly(self, chains) -> dict:
        """Set asym_id / sym_id / entity_id (entity = same entity_key); returns {asym_id: chain_id}."""
        entities = OrderedDict()
        for chain_id, chain in chains.items():
            entities.setdefault(chain.entity_key, []).append(chain_id)
        asym_to_chain = {}
        for entity_id, chain_ids in enumerate(entities.values()):
            chain_ids = list(chain_ids)
            if self.training and self.config.train_shuffle_sym_id:
                random.shuffle(chain_ids)
            for sym_id, chain_id in enumerate(chain_ids):
                asym = len(asym_to_chain)
                n = len(chains[chain_id].ccds)
                chains[chain_id].asym_id = np.full([n], asym, dtype=np.int32)
                chains[chain_id].sym_id = np.full([n], sym_id, dtype=np.int32)
                chains[chain_id].entity_id = np.full([n], entity_id, dtype=np.int32)
                asym_to_chain[asym] = chain_id
        return asym_to_chain

    def _crop(self, chains, conformers, centre_override, name: str, first_chain: Optional[str]) -> None:
        """Spatial crop (in place): take whole components by distance of their tokens to a centre until full."""
        cfg = self.config
        order = list(chains)
        x_gt = np.concatenate([chains[c].x_gt for c in order], axis=0)
        # one entry per token: centre atom, component, component size, standard?, asym id, belongs to first_chain?
        tok_atom, tok_conf, tok_size, tok_std, tok_asym, tok_first = [], [], [], [], [], []
        ligand_atoms = []
        atom = conf_id = 0
        for chain_id in order:
            chain = chains[chain_id]
            if chain_id.isdigit() and len(chain.ccds) == 1:
                ligand_atoms.append(chain.x_gt)
            offset = 0
            for ccd, size, asym in zip(chain.ccds, chain.conformer_sizes.tolist(), chain.asym_id.tolist()):
                if rc.is_standard(ccd):
                    centre = conformers[ccd].index(rc.standard_ccd_to_token_centre_atom_name[ccd])
                    tok_atom.append(atom + int(np.flatnonzero(chain.atom_to_conformer_atom[offset:offset + size] == centre)[0]))
                    n_tok = 1
                else:
                    tok_atom += range(atom, atom + size)
                    n_tok = size
                tok_conf += [conf_id] * n_tok
                tok_size += [size] * n_tok
                tok_std += [rc.is_standard(ccd)] * n_tok
                tok_asym += [asym] * n_tok
                tok_first += [chain_id == first_chain] * n_tok
                atom += size
                offset += size
                conf_id += 1
        x_tok = x_gt[tok_atom]
        asym_tok = np.array(tok_asym)

        seed = random.random()
        if centre_override is not None and not self.training:
            centre = centre_override
        elif not self.training and len(ligand_atoms) == 1:
            centre = np.mean(np.concatenate(ligand_atoms, axis=0), axis=0)[None]
        elif seed < (cfg.train_spatial_crop_ligand_ratio if self.training else 1.0) and ligand_atoms:
            centre = random.choice(np.concatenate(ligand_atoms, axis=0))[None]
        elif seed < cfg.train_spatial_crop_ligand_ratio + cfg.train_spatial_crop_interface_ratio and len(set(tok_asym)) > 1:
            same = asym_tok[None] * asym_tok[:, None]                    # [legacy] product of the chain ids
            dist = np.linalg.norm(x_tok[:, None] - x_tok[None], axis=-1) + same * 100
            interface = np.any(dist < cfg.train_spatial_crop_interface_threshold, axis=-1)
            centre = random.choice(x_tok[interface])[None] if interface.sum() > 0 else random.choice(x_tok)[None]
        else:
            centre = random.choice(x_tok)[None]
        dist = np.linalg.norm(x_tok - centre, axis=-1)
        if first_chain is not None:
            dist = np.where(np.array(tok_first, dtype=np.bool_), -1.0, dist)

        selected = np.zeros([conf_id], dtype=np.bool_)
        n_atoms = n_tokens = 0
        for t in np.argsort(dist).tolist():
            c = tok_conf[t]
            if selected[c]:
                continue
            add_tokens = 1 if tok_std[t] else tok_size[t]
            if n_atoms + tok_size[t] > cfg.atom_crop_size or n_tokens + add_tokens > cfg.crop_size:
                break
            selected[c] = True
            n_atoms += tok_size[t]
            n_tokens += add_tokens
        if not selected.any():
            raise ValueError(f"{name}: the component closest to the crop centre does not fit into the crop "
                             f"(crop_size={cfg.crop_size}, atom_crop_size={cfg.atom_crop_size})")

        start = 0
        for chain_id in order:
            keep = selected[start:start + len(chains[chain_id].ccds)]
            start += len(keep)
            if keep.any():
                chains[chain_id].select(keep)
            else:
                del chains[chain_id]

    # ------------------------------------------------------------------ MSA
    def _pair_and_merge(self, chains, asym_to_chain):
        """Pair / merge the MSAs and concatenate the component arrays in the merged chain order -> (raw, order)."""
        chain_ids = list(chains)
        keys = ("x_gt", "atom_to_conformer_atom", "residue_index", "conformer_sizes", "restype", "is_protein",
                "is_short_poly", "is_ligand", "is_pocket", "key_res_feat")
        if len(chain_ids) == 1 and chains[chain_ids[0]].chain_class == "ligand":
            chain = chains[chain_ids[0]]
            raw = {k: getattr(chain, k) for k in keys + ("asym_id", "sym_id", "entity_id")}
            raw["msa"] = np.repeat(chain.msa[:1], 256, axis=0)
            raw["deletion_matrix"] = np.repeat(chain.deletion_matrix[:1], 256, axis=0)
            order = chain_ids
        else:
            _split_mixed_entities([chains[c] for c in chain_ids])
            merged = pair_and_merge([chains[c] for c in chain_ids], self.config.msa_rows)
            order = [asym_to_chain[a] for a in OrderedDict.fromkeys(merged["asym_id"].tolist())]
            raw = {k: merged[k] for k in ("msa", "deletion_matrix", "asym_id", "sym_id", "entity_id")}
            for k in keys:
                raw[k] = np.concatenate([getattr(chains[c], k) for c in order], axis=0)
        raw["profile"] = _profile(raw["msa"])
        # [legacy] atan(sum / 3) * 2 / pi, a sum and not a mean
        raw["deletion_mean"] = (torch.atan(torch.from_numpy(np.sum(raw["deletion_matrix"], axis=0, dtype=np.int64)) / 3.0)
                                * (2.0 / torch.pi)).numpy()
        return raw, order

    # ------------------------------------------------------------------ atoms and tokens
    def _tokens(self, raw, ccds, conformers, is_lig_conf, asym_to_chain, name, ref_mol):
        """Atom/token maps, reference features, token pair features and the metadata."""
        sizes = raw["conformer_sizes"]
        atom_index = raw["atom_to_conformer_atom"]
        is_std = np.array([rc.is_standard(c) for c in ccds], dtype=np.bool_)
        n_conf = len(ccds)

        # a standard residue is one token, every atom of any other component is a token
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
        ref_feat = np.concatenate(ref_feat, axis=0).astype(np.float32)
        ref_pos = ref_feat[:, :3]                                               # view into ref_feat
        if self.config.use_x_gt_ligand_as_ref_pos:
            lig = raw["is_ligand"][atom_to_conformer].astype(np.bool_)
            ref_pos[lig] = raw["x_gt"][lig] - np.mean(raw["x_gt"][lig], axis=0, keepdims=True)

        meta = SystemMeta(system_id=name, ref_mol=ref_mol, ccds=list(ccds), conformer_sizes=sizes.copy(),
                          residue_index=raw["residue_index"].copy(), asym_id=raw["asym_id"].copy(),
                          conformer_is_ligand=list(is_lig_conf), chain_ids=dict(asym_to_chain), atom_names=atom_names,
                          atom_elements=atom_elements, atom_to_conformer_atom=atom_index.copy())

        feats = {"x_gt": raw["x_gt"]}
        for key in ("is_protein", "is_short_poly", "is_ligand", "residue_index", "restype", "asym_id", "entity_id",
                    "sym_id", "deletion_mean", "profile", "is_pocket", "key_res_feat"):
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

        if self.training and random.random() < self.config.train_charility_augmentation_ratio:
            # hide the chirality of a random half of the ligand atoms
            chirality = ref_feat[:, 158:161]
            unknown = np.zeros_like(chirality)
            unknown[:, 2] = 1
            is_lig_atom = feats["is_ligand"][atom_to_token]
            hide = (np.random.randint(0, 2, [len(is_lig_atom)]) * is_lig_atom).astype(np.bool_)
            ref_feat[:, 158:161] = np.where(hide[:, None], unknown, chirality)

        feats.update({
            "atom_to_token": atom_to_token.astype(np.int64), "atom_to_conformer": atom_to_conformer, "ref_feat": ref_feat,
            "ref_pos": ref_pos, "split_sizes": split_sizes, "token_to_centre_atom": centre.astype(np.int64),
            "token_to_pseudo_beta_atom": beta.astype(np.int64), "token_bonds": token_bonds, "ligand_pair_feat": pair_feat,
            "atom_mask": np.ones_like(raw["x_gt"][..., 0]).astype(np.float32),
            "token_mask": np.ones_like(feats["asym_id"]).astype(np.float32),
        })
        return feats, meta

    # ------------------------------------------------------------------ tensors
    def _transform(self, feats) -> dict:
        """Target and MSA features, padding (training), template, final protein/ligand flags."""
        cfg = self.config
        t = {k: torch.from_numpy(v) for k, v in feats.items()}
        t["target_feat"] = torch.cat([F.one_hot(t["restype"].long(), 32).float(), t.pop("profile").float(),
                                      t.pop("deletion_mean")[..., None].float()], dim=-1)

        msa, deletion = t.pop("msa"), t.pop("deletion_matrix")
        pi = torch.acos(torch.zeros(1)) * 2
        samples = []
        for _ in range(cfg.num_msa_samples):
            rows = [0] + torch.randperm(len(msa))[:cfg.max_msa_clusters - 1].tolist()
            msa, deletion = msa[rows], deletion[rows]                 # [legacy] each sample is drawn from the previous
            samples.append(torch.cat([F.one_hot(msa.long(), 32).float(),
                                      torch.clamp(deletion.float(), min=0., max=1.)[..., None].float(),
                                      (torch.atan(deletion / 3.) * (2. / pi))[..., None].float()], dim=-1))
        t["msa_feat"] = samples[0]
        if len(samples) > 1:
            t["msa_feat_resampled"] = torch.stack(samples[1:], dim=0)
        random.random()          # [legacy] draw of the removed inter-chain bond search, keeps the random stream aligned

        if self.training:
            self._pad(t)

        z_mask = t["token_mask"][None] * t["token_mask"][:, None]
        protein2d = t["is_protein"][None] * t["is_protein"][:, None]
        x_beta = t["x_gt"][t["token_to_pseudo_beta_atom"]]
        dgram = dgram_from_positions(x_beta, no_bins=39) * protein2d[..., None] * z_mask[..., None]
        templ_pair = z_mask * protein2d
        t["templ_mask"] = torch.tensor(1, dtype=torch.float32)
        if self.training:
            if random.random() > 1 - cfg.train_use_template_ratio:     # same draw as the released 0.75 rule
                bert = torch.rand([len(x_beta)]) > random.random() * (1 - cfg.train_template_mask_max_ratio)
                templ_pair = (bert[None] * bert[:, None]) * z_mask * protein2d
            else:
                t["templ_mask"] = torch.tensor(0, dtype=torch.float32)
        t["templ_feat"] = torch.cat([dgram * templ_pair[..., None], templ_pair[..., None]], dim=-1).float()

        is_short_poly = t.pop("is_short_poly")
        t["is_protein"] = t["is_protein"] + is_short_poly
        t["is_ligand"] = t["is_ligand"] - is_short_poly
        return t

    def _pad(self, t) -> None:
        """Pad token and atom dimensions to crop_size / atom_crop_size (training)."""
        pad_token = self.config.crop_size - len(t["residue_index"])
        pad_atom = self.config.atom_crop_size - len(t["x_gt"])
        if pad_token > 0:
            for k in _TOKEN_KEYS_1D:
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
# helpers
# ======================================================================================
def _check_system(data, path) -> None:
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path}: not a system pickle (expected a non-empty {{chain_id: chain}} dict)")
    for chain_id, chain in data.items():
        if not isinstance(chain_id, str):
            raise ValueError(f"{path}: chain ids must be strings, got {chain_id!r}")
        for key in ("ccds", "all_atom_positions", "all_atom_mask"):
            if key not in chain:
                raise ValueError(f"{path}: chain {chain_id!r} has no {key!r}")
        if not len(chain["ccds"]) == len(chain["all_atom_positions"]) == len(chain["all_atom_mask"]):
            raise ValueError(f"{path}: chain {chain_id!r} has inconsistent numbers of components")
        for i, (ccd, pos, mask) in enumerate(zip(chain["ccds"], chain["all_atom_positions"], chain["all_atom_mask"])):
            if not rc.is_unk(ccd) and (np.ndim(pos) != 2 or np.shape(pos)[-1] != 3 or np.shape(mask) != (len(pos),)):
                raise ValueError(f"{path}: chain {chain_id!r} component {i}: positions {np.shape(pos)} and mask "
                                 f"{np.shape(mask)} do not match")


def _present_atoms(chain) -> np.ndarray:
    """Coordinates of the atoms with a position [n, 3]."""
    if not len(chain["all_atom_positions"]):
        return np.zeros([0, 3], dtype=np.float32)
    pos = np.concatenate(chain["all_atom_positions"], axis=0)
    return pos[np.concatenate(chain["all_atom_mask"], axis=0).astype(np.bool_)]


def _pocket(chain, ligand_atoms, pocket_type: str, dist_type: str, cutoff: float, chunk: int = 4096) -> np.ndarray:
    """Pocket flag per component of a receptor chain [n_components]."""
    coords, owner = [], []
    for i, (ccd, pos, mask) in enumerate(zip(chain["ccds"], chain["all_atom_positions"], chain["all_atom_mask"])):
        mask = np.asarray(mask).astype(np.bool_)
        if not rc.is_standard(ccd) or rc.is_unk(ccd) or mask.shape[0] < 2 or not mask[1]:   # [legacy] atom 1 = CA
            continue
        if pocket_type == "atom":
            coords.append(pos[mask])
            owner += [i] * int(mask.sum())
        else:
            coords.append(pos[1][None])
            owner.append(i)
    pocket = np.zeros([len(chain["ccds"])], dtype=np.float32)
    if not coords:
        return pocket
    coords, owner = np.concatenate(coords, axis=0), np.asarray(owner)
    used = []
    for lig in ligand_atoms:
        if not len(lig):
            continue
        if dist_type == "ligand":
            close = np.zeros(len(coords), dtype=np.bool_)
            for s in range(0, len(coords), chunk):              # chunked to bound memory, same float32 values
                close[s:s + chunk] = np.any(np.linalg.norm(coords[s:s + chunk, None] - lig[None], axis=-1) < cutoff, axis=-1)
        else:
            close = np.linalg.norm(coords - np.mean(lig, axis=0, keepdims=True), axis=-1) < cutoff
        used.append(owner[close])
    if used:
        pocket[sorted(set(np.concatenate(used).tolist()))] = 1.
    return pocket


def _smiles_ligand(smiles: str):
    """(chain dict, CCDConformer, RDKit mol) of a SMILES ligand."""
    label, conf, mol = get_features_from_smi(smiles)
    names = [f"{PeriodicTable[e] + str(i):<4}" for i, e in enumerate(conf["ref_element"])]
    chain = {"all_atom_positions": label["x_gt"][None], "all_atom_mask": label["x_exists"][None], "ccds": [_SMILES_CCD]}
    return chain, CCDConformer(conf, conf["ref_pos"], names, with_pairs=True), mol


def _split_mixed_entities(chains: List[Chain]) -> None:
    """Ligand chains that share an entity with a receptor chain get a new entity (else the MSA merge fails)."""
    classes = {}
    for c in chains:
        classes.setdefault(int(c.entity_id[0]), set()).add(c.chain_class)
    next_entity = max(int(c.entity_id.max()) for c in chains) + 1
    for entity in sorted(e for e, cls in classes.items() if len(cls) > 1):
        ligands = sorted((c for c in chains if int(c.entity_id[0]) == entity and c.chain_class == "ligand"),
                         key=lambda c: int(c.asym_id[0]))
        for sym_id, c in enumerate(ligands):
            c.entity_id = np.full_like(c.entity_id, next_entity)
            c.sym_id = np.full_like(c.sym_id, sym_id)
        next_entity += 1


def _profile(msa: np.ndarray) -> np.ndarray:
    """Residue frequencies per column [n_cols, 32] (bincount, no one-hot of the whole MSA)."""
    n_rows, n_cols = msa.shape
    if msa.size and (msa.min() < 0 or msa.max() > 31):
        raise ValueError(f"MSA residue codes must be in [0, 31], got [{msa.min()}, {msa.max()}]")
    counts = np.zeros(n_cols * 32, dtype=np.int64)
    offsets = np.arange(n_cols, dtype=np.int64)[None] * 32
    for s in range(0, n_rows, 2048):
        counts += np.bincount((msa[s:s + 2048].astype(np.int64) + offsets).ravel(), minlength=n_cols * 32)
    return (torch.from_numpy(counts.reshape(n_cols, 32).astype(np.float32)) / n_rows).numpy()

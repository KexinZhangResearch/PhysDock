"""CCD library (ccd_id_meta_data.pkl.gz): reference conformer arrays of every chemical component."""
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem

import PhysDock.data.constants as rc
from PhysDock.utils.io_utils import load_pkl

_REPO_ROOT = Path(__file__).resolve().parents[2]


def default_ccd_path() -> Path:
    """<repo>/data/ccd_id_meta_data.pkl.gz (or <repo>/params/...)."""
    for path in (_REPO_ROOT / "data" / "ccd_id_meta_data.pkl.gz", _REPO_ROOT / "params" / "ccd_id_meta_data.pkl.gz"):
        if path.is_file():
            return path
    raise FileNotFoundError("ccd_id_meta_data.pkl.gz not found in <repo>/data or <repo>/params")


class CCDConformer:
    """Reference arrays of one component: ref_feat [n, 167], pair_feat [n, n, 42] (None for standard residues),
    bonds [n, n], atom_names (n,), elements [n]."""

    def __init__(self, feats, ref_pos: np.ndarray, atom_names, with_pairs: bool) -> None:
        f32 = lambda key: feats[key].astype(np.float32)[..., None]
        self.ref_feat = np.concatenate([
            ref_pos, feats["ref_charge"][..., None], rc.eye_128[feats["ref_element"]].astype(np.float32), f32("ref_is_aromatic"),
            rc.eye_9[feats["ref_degree"]].astype(np.float32), rc.eye_7[feats["ref_hybridization"]].astype(np.float32),
            rc.eye_9[feats["ref_implicit_valence"]].astype(np.float32), rc.eye_3[feats["ref_chirality"]].astype(np.float32),
            *[f32(f"ref_in_ring_of_{k}") for k in range(3, 9)],
        ], axis=-1).astype(np.float32)
        self.pair_feat = None
        if with_pairs:
            self.pair_feat = np.concatenate([
                rc.eye_32[feats["d_token"]].astype(np.float32), rc.eye_5[feats["bond_type"]].astype(np.float32),
                f32("token_bonds"), f32("bond_as_double"), f32("bond_in_ring"), f32("bond_is_conjugated"),
                f32("bond_is_aromatic"),
            ], axis=-1).astype(np.float32)
        self.bonds = np.asarray(feats["token_bonds"]).astype(np.float32)
        self.atom_names = tuple(atom_names)
        self.elements = np.asarray(feats["ref_element"])
        self.num_atoms = len(self.ref_feat)

    def index(self, atom_name: str) -> Optional[int]:
        return self.atom_names.index(atom_name) if atom_name in self.atom_names else None


class CCDLibrary:
    """ccd_id_meta_data.pkl.gz (path or loaded dict) with a cache of CCDConformer objects."""

    def __init__(self, source=None, cache_size: int = 256) -> None:
        if source is None:
            source = default_ccd_path()
        self.path = None if isinstance(source, dict) else str(source)
        self._entries = source if isinstance(source, dict) else load_pkl(self.path)
        self.cache_size = cache_size
        self._cache = OrderedDict()

    def __contains__(self, ccd) -> bool:
        return ccd in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def entry(self, ccd: str):
        if ccd not in self._entries:
            raise KeyError(f"component {ccd!r} is not in the CCD library {self.path or ''}")
        return self._entries[ccd]

    def conformer(self, ccd: str) -> CCDConformer:
        if ccd in self._cache:
            self._cache.move_to_end(ccd)
            return self._cache[ccd]
        e = self.entry(ccd)
        conf = CCDConformer(e, e["ref_pos"] - np.mean(e["ref_pos"], axis=0, keepdims=True), e["ref_atom_name_chars"],
                            with_pairs=not rc.is_standard(ccd))
        self._cache[ccd] = conf
        if len(self._cache) > self.cache_size + len(rc.standard_ccds):
            self._cache.popitem(last=False)
        return conf

    def ref_mol(self, ccd: str) -> Optional[Chem.Mol]:
        mol = self.entry(ccd).get("ref_mol")
        return Chem.Mol(mol) if mol is not None else None

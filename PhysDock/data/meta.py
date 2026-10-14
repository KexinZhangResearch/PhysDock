"""SystemMeta: non-tensor information of a featurised system, used to write PDB files."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from rdkit import Chem

logger = logging.getLogger(__name__)

PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


@dataclass
class SystemMeta:
    """Per-conformer lists are in feature order."""
    system_id: str
    ref_mol: Optional[Chem.Mol]         # ligand molecule (single one-component ligand chain or SMILES)
    ccds: List[str]
    conformer_sizes: np.ndarray         # atoms per conformer
    residue_index: np.ndarray           # position of every conformer in its input chain
    asym_id: np.ndarray                 # chain index of every conformer
    conformer_is_ligand: List[bool]     # written as HETATM
    chain_ids: dict                     # {asym_id: input chain id}
    atom_names: List[str]
    atom_elements: List[str]
    atom_to_conformer_atom: np.ndarray  # atom index inside its library conformer
    _pdb: Optional[tuple] = None

    @property
    def num_atoms(self) -> int:
        return len(self.atom_names)

    def _pdb_fields(self):
        """Fixed parts of every ATOM / HETATM line; serial and residue numbers wrap so the columns never overflow."""
        if len(self.chain_ids) > len(PDB_CHAIN_IDS):
            logger.warning("%s: %d chains, PDB chain letters repeat after %d", self.system_id, len(self.chain_ids),
                           len(PDB_CHAIN_IDS))
        prefix, suffix, het = [], [], []
        for conf, size in enumerate(self.conformer_sizes.tolist()):
            record = "HETATM" if self.conformer_is_ligand[conf] else "ATOM"
            chain = PDB_CHAIN_IDS[int(self.asym_id[conf]) % len(PDB_CHAIN_IDS)]
            res_name = self.ccds[conf].split()[0][-3:]
            res_id = (int(self.residue_index[conf]) + 1) % 10000
            for _ in range(size):
                i = len(prefix)
                name = self.atom_names[i] if len(self.atom_names[i]) == 4 else f" {self.atom_names[i]}"
                prefix.append(f"{record:<6}{(i + 1) % 100000:>5} {name:<4} {res_name:>3} {chain:>1}{res_id:>4}    ")
                suffix.append(f"{1.00:>6.2f}{70.:>6.2f}          {self.atom_elements[i]:>2}{0:>2}")
                het.append(record == "HETATM")
        return prefix, suffix, het

    def to_pdb(self, coords, receptor_only: bool = False, ligand_only: bool = False) -> str:
        """coords: [N_atom, 3] (padding rows ignored) -> PDB text"""
        if receptor_only and ligand_only:
            raise ValueError("receptor_only and ligand_only are mutually exclusive")
        if self._pdb is None:
            self._pdb = self._pdb_fields()
        prefix, suffix, het = self._pdb
        if isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu()
        xyz = coords[:self.num_atoms].tolist()
        if len(xyz) < self.num_atoms:
            raise ValueError(f"got {len(xyz)} coordinates for {self.num_atoms} atoms")
        lines = [f"{p}{x:>8.3f}{y:>8.3f}{z:>8.3f}{s}" for p, s, h, (x, y, z) in zip(prefix, suffix, het, xyz)
                 if not (receptor_only and h) and not (ligand_only and not h)]
        return "MODEL     1\n" + "\n".join(lines) + "\nTER\nENDMDL\nEND"

    def write_pdb(self, path, coords, receptor_only: bool = False, ligand_only: bool = False) -> None:
        Path(path).write_text(self.to_pdb(coords, receptor_only=receptor_only, ligand_only=ligand_only))

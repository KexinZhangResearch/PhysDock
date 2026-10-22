"""Real proteins for the model: a window of `crop` consecutive residues of a protein chain of the training set (data/v1/train_val), as a system dict of pomelo.data.molecule.

As in PhysDock, a standard residue is one token and one conformer: its reference conformer is the CCD one (pomelo.data.molecule.random_frames puts every residue into a frame of its
own), its atom features are those of the atoms in the CCD reference molecule, the bonds inside a residue are features, the peptide bonds are bonds of the system but no feature (they join
two conformers; the chain is in residue_index). The atoms are the heavy atoms that are resolved in the structure, in the order of the CCD library (so the coordinates, the reference
molecule of the residue and its bonds line up).

The stereo table of the evaluation: the C-alpha of every residue but glycine and the C-beta of isoleucine and threonine (RDKit's potential stereocentres of the reference molecules), each
with its three heavy neighbours and the implicit hydrogen; the label is the sign of the signed volume of the ordered neighbours in the real structure, so all L residues carry the same
label and the mirror image the opposite one.
"""
import gzip
import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem

from pomelo.data.molecule import ATOM_KEYS, BOND_TYPES, ELEMENTS, HYBRIDISATION, add_features, random_frames, signed_volume

ROOT = Path(os.environ.get("PHYSDOCK_ROOT", Path(__file__).resolve().parents[3]))          # the PhysDock repository (the data lives there)
STANDARD = "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
PEPTIDE_BOND = (1.0, 1.6)                                     # C(i)-N(i+1) in Angstrom: a window with a larger gap is not one chain
_CCD = None
_STEREO: Dict[str, List[int]] = {}


def ccd_library():
    global _CCD
    if _CCD is None:
        _CCD = pickle.load(gzip.open(ROOT / "data" / "ccd_id_meta_data.pkl.gz", "rb"))
    return _CCD


def load_chain(path: str, chain: Optional[str] = None):
    """(chain id, residue names, positions [list of [n_atoms, 3]], masks) of the longest protein chain (or `chain`) of a system file"""
    d = pickle.load(gzip.open(path, "rb"))
    prot = {c: v for c, v in d.items() if not c.isdigit() and isinstance(v, dict) and "ccds" in v}
    c = chain or max(prot, key=lambda k: len(prot[k]["ccds"]))
    v = prot[c]
    return c, list(v["ccds"]), [np.asarray(p, dtype=np.float32) for p in v["all_atom_positions"]], [np.asarray(m) > 0 for m in v["all_atom_mask"]]


def window_ok(ccds, pos, mask, start: int, crop: int, min_complete: float = 0.95) -> bool:
    """a window of consecutive residues: all standard, the backbone N, CA, C resolved in every residue, at least `min_complete` of the heavy atoms resolved (the carboxylate oxygen of
    the chain end aside), and the peptide bonds between neighbours of the length of a bond"""
    if start < 0 or start + crop > len(ccds):
        return False
    idx = range(start, start + crop)
    if any(ccds[i] not in STANDARD for i in idx) or any(not mask[i][:3].all() for i in idx):
        return False
    have = sum(int(mask[i][:-1].sum()) if mask[i].shape[0] > len(ccd_library()[ccds[i]]["ref_mol"].GetAtoms()) - 1 else int(mask[i].sum()) for i in idx)
    full = sum(ccd_library()[ccds[i]]["ref_mol"].GetNumAtoms() - 1 for i in idx)                       # without the OXT of the free amino acid
    if have < min_complete * full:
        return False
    for i in range(start, start + crop - 1):
        gap = float(np.linalg.norm(pos[i][2] - pos[i + 1][0]))
        if not PEPTIDE_BOND[0] < gap < PEPTIDE_BOND[1]:
            return False
    return True


def stereo_atoms(ccd: str) -> List[int]:
    """atom indices of the tetrahedral stereocentres of a residue (RDKit's potential stereo of its reference molecule)"""
    if ccd not in _STEREO:
        m = ccd_library()[ccd]["ref_mol"]
        _STEREO[ccd] = [i.centeredOn for i in Chem.FindPotentialStereo(m) if i.type == Chem.StereoType.Atom_Tetrahedral]
    return _STEREO[ccd]


def build_window(ccds, pos, mask, start: int, crop: int = 128) -> Dict[str, torch.Tensor]:
    lib = ccd_library()
    feats = {k: [] for k in ATOM_KEYS + ("atom_to_token",)}
    coords, refs, bonds, glob = [], [], [], {}
    centres_local = []
    for u, r in enumerate(range(start, start + crop)):
        m = lib[ccds[r]]["ref_mol"]
        keep = [a for a in range(m.GetNumAtoms()) if mask[r][a]]
        for a in keep:
            at = m.GetAtomWithIdx(a)
            glob[(u, a)] = len(coords)
            coords.append(pos[r][a])
            refs.append(np.asarray(lib[ccds[r]]["ref_pos"][a], dtype=np.float32))                    # the reference conformer of the residue (CCD), its own frame
            for k, v in (("ref_element", ELEMENTS.get(at.GetAtomicNum(), 9)), ("ref_charge", at.GetFormalCharge()), ("ref_is_aromatic", int(at.GetIsAromatic())),
                         ("ref_hybridization", HYBRIDISATION.get(at.GetHybridization(), 3)), ("ref_n_h", at.GetTotalNumHs()), ("ref_in_ring", int(at.IsInRing())),
                         ("atom_to_token", u)):
                feats[k].append(v)                                                               # the features of the atom in the reference molecule (the free amino acid)
        for b in m.GetBonds():
            i, j = (u, b.GetBeginAtomIdx()), (u, b.GetEndAtomIdx())
            if i in glob and j in glob:
                bonds.append((glob[i], glob[j], BOND_TYPES.get(b.GetBondType(), 1)))
        if u > 0 and (u - 1, 2) in glob and (u, 0) in glob:
            bonds.append((glob[(u - 1, 2)], glob[(u, 0)], 1))                                    # the peptide bond C(i-1) - N(i): a bond of the system, not a feature (two conformers)
        centres_local.append([(glob[(u, c)], [glob[(u, x.GetIdx())] for x in m.GetAtomWithIdx(c).GetNeighbors() if (u, x.GetIdx()) in glob])
                              for c in stereo_atoms(ccds[r]) if (u, c) in glob])
    x = torch.from_numpy(np.stack(coords)).float()
    x = x - x.mean(0)
    centre, nbrs = [], []
    for per_res in centres_local:
        for c, nb in per_res:
            if len(nb) in (3, 4):
                centre.append(c)
                nbrs.append(sorted(nb) + [-1] * (4 - len(nb)))
    centre_t, nbrs_t = torch.tensor(centre, dtype=torch.int64), torch.tensor(nbrs, dtype=torch.int64)
    sign = torch.sign(signed_volume(x, centre_t, nbrs_t)).float()
    out = {k: torch.tensor(v) for k, v in feats.items()}
    conformer = out["atom_to_token"].clone()                                                     # a residue is a token and a conformer: one frame per residue
    out.update({"bonds": torch.tensor(bonds), "x_gt": x, "stereo_center": centre_t, "stereo_nbrs": nbrs_t, "stereo_sign": sign, "atom_to_conformer": conformer,
                "ref_pos": random_frames(torch.from_numpy(np.stack(refs)).float(), conformer, seed=start),
                "restype": torch.tensor([STANDARD.index(ccds[r]) for r in range(start, start + crop)]),
                "residue_index": torch.arange(start, start + crop), "asym_id": torch.zeros(crop, dtype=torch.int64)})
    return add_features(out)


def window_meta(ccds, mask, start: int, crop: int) -> List[Tuple[str, str, int, str]]:
    """(residue name, atom name, residue number, element) of the atoms of build_window, in its order: the resolved heavy atoms of every residue in the order of the CCD library"""
    lib = ccd_library()
    out = []
    for u, r in enumerate(range(start, start + crop)):
        entry = lib[ccds[r]]
        for a in range(entry["ref_mol"].GetNumAtoms()):
            if mask[r][a]:
                out.append((ccds[r], "".join(entry["ref_atom_name_chars"][a]).strip(), start + u + 1, entry["ref_mol"].GetAtomWithIdx(a).GetSymbol()))
    return out


def write_pdb(path, meta, models, chain: str = "A") -> None:
    """a PDB file of one or more models of the same atoms: meta from window_meta, models [[Na, 3] arrays in Angstrom]"""
    lines = []
    for k, x in enumerate(models):
        lines.append(f"MODEL     {k + 1:4d}")
        for i, ((res, name, num, el), (px, py, pz)) in enumerate(zip(meta, x.tolist())):
            atom = f" {name:<3s}" if len(el) == 1 and len(name) < 4 else f"{name:<4s}"
            lines.append(f"ATOM  {i + 1:5d} {atom} {res:>3s} {chain}{num:4d}    {px:8.3f}{py:8.3f}{pz:8.3f}  1.00  0.00          {el:>2s}")
        lines.append("ENDMDL")
    Path(path).write_text("\n".join(lines) + "\nEND\n")


def select(n: int = 10, crop: int = 128, seed: int = 0, max_identity: float = 0.3, max_scan: int = 4000) -> List[dict]:
    """n different proteins of the training set: a random window of `crop` residues each, from distinct PDB entries, with a sequence identity (position by position) below
    max_identity between any two windows. Returns [{name, chain, start, sequence}]."""
    root = ROOT / "data" / "v1" / "train_val"
    names = sorted(os.listdir(root))
    rng = random.Random(seed)
    rng.shuffle(names)
    chosen, seen = [], set()
    for name in names[:max_scan]:
        pdb = name.split("__")[0][:4]
        if pdb in seen:
            continue
        try:
            chain, ccds, pos, mask = load_chain(str(root / name))
        except Exception:
            continue
        starts = [s for s in range(0, len(ccds) - crop + 1, 8) if window_ok(ccds, pos, mask, s, crop)]
        if not starts:
            continue
        s = rng.choice(starts)
        seq = ccds[s:s + crop]
        if any(sum(a == b for a, b in zip(seq, o["sequence"])) / crop > max_identity for o in chosen):
            continue
        chosen.append({"name": name, "chain": chain, "start": s, "sequence": seq})
        seen.add(pdb)
        if len(chosen) == n:
            return chosen
    raise RuntimeError(f"only {len(chosen)} of {n} proteins found in {max_scan} systems")


PROTEINS_JSON = Path(__file__).with_name("proteins_10.json")


def make_dataset(n: Optional[int] = None) -> Dict[str, Dict[str, torch.Tensor]]:
    """the 10 proteins of proteins_10.json (made by select(), written once: the experiment must always see the same ones)"""
    if not PROTEINS_JSON.exists():
        PROTEINS_JSON.write_text(json.dumps(select(), indent=1))
    entries = json.loads(PROTEINS_JSON.read_text())[:n]
    out = {}
    for e in entries:
        chain, ccds, pos, mask = load_chain(str(ROOT / "data" / "v1" / "train_val" / e["name"]), e["chain"])
        out[e["name"].split("__")[0] + "_" + chain] = build_window(ccds, pos, mask, e["start"], len(e["sequence"]))
    return out

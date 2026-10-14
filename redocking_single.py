#!/usr/bin/env python3
"""Redock ONE protein-ligand system with PhysDock.

Unlike ``redocking.py`` (which walks a directory of systems and expects a ``features/`` folder with fixed
sub-folder names) every input is given explicitly:

    python redocking_single.py \\
        --input_pkl            data/demo/redocking/Posebusters_subset/5SAK_ZRY_A_1.pkl.gz \\
        --msa_features         data/demo/redocking/features/msa_features \\
        --uniprot_msa_features data/demo/redocking/features/uniprot_msa_features \\
        --output_dir           out/5SAK_ZRY \\
        --max_samples 40 --max_rounds 5 --num_samples_per_round 20 --crop_size 256 --atom_crop_size 2048 \\
        --enable_physics_correction --use_pocket --use_key_res

``--msa_features`` / ``--uniprot_msa_features`` are directories holding ``<md5>.pkl.gz`` files, where md5 is
``md5("protein:" + sequence)`` of every receptor chain (this is what ``run_homo_search.py`` writes).  The script
checks up front that all of them exist and tells you which ones are missing.

Output directory (everything is written directly into ``--output_dir``)::

    ligand_gt.sdf  receptor_gt.pdb  system_gt.pdb        ground truth, from the input pkl
    ligand_pred_<k>.sdf                                  predicted ligand, <k> = 0 .. max_samples-1
    receptor_pred_<k>.pdb  system_pred_<k>.pdb           predicted receptor / complex, superposed on the GT pocket
    summary.json                                         run settings, timings, ligand RMSD of every sample

Model weights are looked up in ``data/params/params.pt`` then ``params/params.pt``; the CCD metadata in
``data/ccd_id_meta_data.pkl.gz`` then ``params/ccd_id_meta_data.pkl.gz`` (override with --params / --ccd_meta).
The sampling procedure is the one of ``redocking.py``.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from PhysDock import PhysDock
from PhysDock.data.constants.PDBData import protein_letters_3to1_extended
from PhysDock.data import CCDLibrary, FeatureConfig, PhysDockDataset, SystemEntry, msa_feat_for_round
from PhysDock.utils.import_weights import load_pretrained
from PhysDock.utils.io_utils import convert_md5_string, dump_json, dump_txt, load_pkl
from PhysDock.utils.tensor_utils import weighted_rigid_align

RDLogger.DisableLog("rdApp.*")
torch.set_float32_matmul_precision("high")      # TF32 matmuls, as the released pipeline

REPO = Path(__file__).resolve().parent
PARAMS_CANDIDATES = [REPO / "data" / "params" / "params.pt", REPO / "params" / "params.pt"]
CCD_META_CANDIDATES = [REPO / "data" / "ccd_id_meta_data.pkl.gz",
                       REPO / "params" / "ccd_id_meta_data.pkl.gz"]


class InputError(Exception):
    """Problem with the inputs: reported as a plain message, without a traceback."""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# Input checks
# --------------------------------------------------------------------------------------
def pick_file(user_value, candidates, what: str, flag: str) -> Path:
    if user_value is not None:
        p = Path(user_value).expanduser()
        if not p.is_file():
            raise InputError(f"{what} not found: {p}")
        return p
    for c in candidates:
        if c.is_file():
            return c
    raise InputError(f"{what} not found. Looked in:\n    " + "\n    ".join(map(str, candidates))
                     + f"\n  Pass it explicitly with {flag}.")


def inspect_system(input_pkl: Path, msa_dir: Path, uniprot_dir: Path):
    """Validate the system pickle and the MSA folders *before* the (slow) model is loaded."""
    data = load_pkl(str(input_pkl))
    if data is None:
        raise InputError(f"{input_pkl}: unsupported file type (expected .pkl or .pkl.gz)")
    if not isinstance(data, dict) or not data or not all(isinstance(v, dict) and "ccds" in v for v in data.values()):
        raise InputError(f"{input_pkl} is not a PhysDock system pickle (expected {{chain_id: {{'ccds': ..., ...}}}})")
    ligand_ids = [c for c in data if str(c).isdigit()]
    protein_ids = [c for c in data if not str(c).isdigit()]
    if not protein_ids:
        raise InputError("the system has no receptor chain (receptor chains have non-numeric ids)")
    if len(ligand_ids) != 1:
        raise InputError(f"redocking needs exactly one ligand chain (numeric chain id), found {len(ligand_ids)}: {ligand_ids}")
    if len(data[ligand_ids[0]]["ccds"]) != 1:
        raise InputError(f"the ligand chain must consist of a single CCD component, got {data[ligand_ids[0]]['ccds']}")

    needed = {}
    for chain_id in protein_ids:
        seq = "".join(protein_letters_3to1_extended.get(ccd, "X") for ccd in data[chain_id]["ccds"])
        needed.setdefault(convert_md5_string(f"protein:{seq}"), []).append((chain_id, len(seq)))
    missing = []
    for md5, chains in needed.items():
        for label, folder in (("--msa_features", msa_dir), ("--uniprot_msa_features", uniprot_dir)):
            f = folder / f"{md5}.pkl.gz"
            if not f.is_file():
                missing.append(f"    {f}   (chain {', '.join(c for c, _ in chains)}, {chains[0][1]} residues; {label})")
    if missing:
        raise InputError("MSA feature files missing:\n" + "\n".join(missing)
                         + "\n  Generate them with run_homo_search.py, or point the options at the right folders.")
    all_ccds = {ccd for chain in data.values() for ccd in chain["ccds"]}
    return {"protein_chains": protein_ids, "ligand_chain": ligand_ids[0], "ligand_ccd": data[ligand_ids[0]]["ccds"][0],
            "ccds": all_ccds, "msa_md5": list(needed)}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def load_model(params_path: Path, device: torch.device) -> PhysDock:
    # built on the meta device without initialisation; every parameter is assigned from the checkpoint (strict)
    return load_pretrained(params_path, device=device)


def write_ligand_sdf(ref_mol, coords, path: Path) -> None:
    mol = copy.deepcopy(ref_mol)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x, y, z = (float(v) for v in coords[i])
        conf.SetAtomPosition(i, Point3D(x, y, z))
    dump_txt(Chem.MolToMolBlock(mol, includeStereo=True), str(path))


def embed_reference_conformers(ref_mol, num_confs: int, device, seed):
    mol = copy.deepcopy(ref_mol)
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, enforceChirality=True,
                                           randomSeed=-1 if seed is None else seed, numThreads=0))
    if not cids:
        raise RuntimeError("RDKit could not embed any reference conformer of the ligand")
    coords = np.stack([mol.GetConformer(c).GetPositions() for c in cids]).astype(np.float32)
    return torch.from_numpy(coords).to(device)


def chirality_ok(expected: dict, pdb_block: str) -> bool:
    try:
        got = {i[0]: i[1] for i in Chem.FindMolChiralCenters(Chem.MolFromPDBBlock(pdb_block, sanitize=False))}
    except Exception:  # noqa: BLE001 - an unparsable pose counts as a failed pose
        return False
    return all(centre in got and got[centre] == tag for centre, tag in expected.items())


def clean_previous_outputs(out: Path) -> None:
    """Remove sample files of an earlier run in this folder so old and new samples can never be mixed."""
    for kind in ("ligand", "system", "receptor"):
        for f in out.glob(f"{kind}_pred_*"):
            if f.is_file():
                f.unlink()


# --------------------------------------------------------------------------------------
# Main routine
# --------------------------------------------------------------------------------------
def redock_single(
        input_pkl, msa_features, uniprot_msa_features, output_dir, *,
        params_path=None, ccd_meta_path=None,
        max_samples=5, physics_correction=False, max_rounds=10, num_samples_per_round=5, steps=40,
        mmff_iters=5, eta=6.0, num_confs=128, crop_size=None, atom_crop_size=None,
        use_x_gt_ligand_as_ref_pos=False, pocket_type="atom", pocket_cutoff=10.0, pocket_dist_type="ligand",
        use_pocket=False, use_key_res=False, key_res_random_mask_ratio=0.5, rho=1000,
        device=None, seed=None,
) -> dict:
    t_start = time.time()
    input_pkl, out = Path(input_pkl).expanduser(), Path(output_dir).expanduser()
    msa_dir, uni_dir = Path(msa_features).expanduser(), Path(uniprot_msa_features).expanduser()
    if not input_pkl.is_file():
        raise InputError(f"input pkl not found: {input_pkl}")
    for flag, d in (("--msa_features", msa_dir), ("--uniprot_msa_features", uni_dir)):
        if not d.is_dir():
            raise InputError(f"{flag} is not a directory: {d}")
    if max_samples < 1 or max_rounds < 1 or num_samples_per_round < 1 or steps < 2:
        raise InputError("--max_samples, --max_rounds, --num_samples_per_round must be >= 1 and --steps >= 2")
    if (crop_size is None) != (atom_crop_size is None):
        if crop_size is None:
            raise InputError("--atom_crop_size requires --crop_size")
        atom_crop_size = 8 * crop_size       # same ratio as the recommended 256 / 2048
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise InputError("CUDA is not available; use --device cpu (very slow) or run on a GPU machine")
    device = torch.device(device)
    params_path = pick_file(params_path, PARAMS_CANDIDATES, "model parameters (params.pt)", "--params")
    ccd_meta_path = pick_file(ccd_meta_path, CCD_META_CANDIDATES, "CCD metadata (ccd_id_meta_data.pkl.gz)", "--ccd_meta")

    info = inspect_system(input_pkl, msa_dir, uni_dir)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    out.mkdir(parents=True, exist_ok=True)
    clean_previous_outputs(out)

    log(f"system {input_pkl.name}: receptor chains {info['protein_chains']}, ligand {info['ligand_ccd']}")
    log(f"device {device}, params {params_path}")

    # ---- features -----------------------------------------------------------------------
    config = FeatureConfig(
        crop_size=crop_size, atom_crop_size=atom_crop_size or 2048, pocket_type=pocket_type,
        pocket_cutoff=pocket_cutoff, pocket_dist_type=pocket_dist_type, use_pocket=use_pocket, use_key_res=use_key_res,
        key_res_random_mask_ratio=key_res_random_mask_ratio, use_x_gt_ligand_as_ref_pos=use_x_gt_ligand_as_ref_pos,
        num_msa_samples=max_rounds if physics_correction else 1,
    )
    ccd = CCDLibrary(ccd_meta_path)
    unknown = sorted(c for c in info["ccds"] if c not in ccd)
    if unknown:
        raise InputError(f"components not in the CCD metadata {ccd_meta_path.name}: {unknown}")
    dataset = PhysDockDataset([SystemEntry(str(input_pkl))], msa_dir, uni_dir, ccd=ccd, config=config)
    t0 = time.time()
    tensors, meta = dataset[0]
    tensors = {k: v.to(device) for k, v in tensors.items()}
    log(f"features ready ({time.time() - t0:.1f}s): {len(tensors['restype'])} tokens, {len(tensors['x_gt'])} atoms")

    ref_mol = meta.ref_mol
    if ref_mol is None:
        raise InputError("no reference molecule for the ligand (needs exactly one ligand chain with one CCD component)")
    atom_token = tensors["atom_to_token"]
    is_ligand_atom = tensors["is_ligand"][atom_token].bool()
    n_lig = int(is_ligand_atom.sum())
    if n_lig != ref_mol.GetNumAtoms():
        raise InputError(f"the pkl has {n_lig} ligand atoms but the reference molecule of {info['ligand_ccd']} has "
                         f"{ref_mol.GetNumAtoms()}; coordinates cannot be mapped onto the molecule")
    is_lig_cpu = is_ligand_atom.cpu()

    # alignment weights: receptor atoms (pocket residues only when the pocket feature is used)
    weights = (tensors["token_mask"] * tensors["is_protein"])[atom_token]
    if use_pocket:
        weights = tensors["is_pocket"][atom_token] * weights
    if float(weights.sum()) == 0.0:
        raise InputError("no receptor atoms to align on (is the pocket empty? try a larger --pocket_cutoff)")
    x_gt = tensors["x_gt"][None]
    x_gt_cpu = tensors["x_gt"].cpu()

    # ---- ground truth outputs + chirality reference -----------------------------------------------
    meta.write_pdb(out / "system_gt.pdb", x_gt_cpu)
    meta.write_pdb(out / "receptor_gt.pdb", x_gt_cpu, receptor_only=True)
    write_ligand_sdf(ref_mol, x_gt_cpu[is_lig_cpu], out / "ligand_gt.sdf")
    chiral_init = {i[0]: i[1] for i in Chem.FindMolChiralCenters(ref_mol)}
    chiral_gt = {i[0]: i[1] for i in Chem.FindMolChiralCenters(Chem.MolFromPDBBlock(
        meta.to_pdb(tensors["ref_pos"].cpu(), ligand_only=True),
        sanitize=False))}
    chiral_centres = {k: v for k, v in chiral_gt.items() if k in chiral_init}

    # ---- model -----------------------------------------------------------------------------------
    t0 = time.time()
    model = load_model(params_path, device)
    log(f"model loaded ({time.time() - t0:.1f}s)")

    ref_mol_poses = ref_mol_poses_dist = None
    if physics_correction:
        ref_mol_poses = embed_reference_conformers(ref_mol, num_confs, device, seed)
        ref_mol_poses_dist = torch.norm(ref_mol_poses[:, :, None] - ref_mol_poses[:, None], dim=-1)
        log(f"{len(ref_mol_poses)} reference conformers embedded")

    # ---- sampling rounds (same procedure as redocking.py) -------------------------------------------
    accepted, rejected = [], deque([], maxlen=max_samples)
    ligand_templates, reference_templates = [], []
    gamma = float(eta)
    rounds_run = 0
    for recycle_id in range(max_rounds):
        if recycle_id > 0 and not physics_correction:
            break
        if recycle_id >= 1:
            tensors["msa_feat"] = msa_feat_for_round(tensors, recycle_id)
        t0 = time.time()
        with torch.no_grad():
            x_pred = model.sample_diffusion(
                tensors, num_sample=num_samples_per_round, steps=steps,
                mmff_gamma_0_factor=gamma, align_ref_pos=recycle_id > 0,
                ref_mol=ref_mol,
                ref_mol_poses=(x_gt[:, is_ligand_atom] if use_x_gt_ligand_as_ref_pos else
                               torch.stack(ligand_templates + reference_templates, dim=0) if recycle_id > 0 else None),
                use_ref_mol_poses=recycle_id != 0 and physics_correction,
                ode_step_scale_eta=1.0, mmff_iters=mmff_iters, karras_noise_schedule_power=rho,
            )
        rounds_run += 1
        x_pred_cpu = x_pred.cpu()
        pass_flags = []
        for x, x_cpu in zip(x_pred, x_pred_cpu):
            ok = True if not physics_correction else chirality_ok(
                chiral_centres, meta.to_pdb(x_cpu, ligand_only=True))
            pass_flags.append(ok)
            if ok:
                ligand_templates.append(x[is_ligand_atom])
                accepted.append(x)
            else:
                rejected.append(x)
        log(f"round {recycle_id + 1}/{max_rounds}: {sum(pass_flags)}/{len(pass_flags)} poses kept chirality, "
            f"{len(accepted)} accepted in total ({time.time() - t0:.1f}s)")
        if physics_correction:
            gamma = gamma * 1.15 if any(pass_flags) else max(gamma * 0.7, 1)
            if len(accepted) >= max_samples:
                break
            ligand_poses = x_pred[:, is_ligand_atom]
            ligand_dist = torch.norm(ligand_poses[:, :, None] - ligand_poses[:, None], dim=-1)
            delta = (ligand_dist[:, None] - ref_mol_poses_dist[None]).abs()
            eps = 0.25 * (torch.sigmoid(-0.5 + delta) + torch.sigmoid(-1 + delta)
                          + torch.sigmoid(-2 + delta) + torch.sigmoid(-4 + delta))
            eps = eps.mean(dim=[-1, -2, -4])
            used = torch.argsort(eps)[:max_samples - len(ligand_templates)]
            reference_templates = [ref_mol_poses[i] for i in used]

    n_rejected_kept = 0
    if len(accepted) < num_samples_per_round:        # same rule as redocking.py: top up with chirality-failed poses
        n_rejected_kept = len(rejected)
        accepted = accepted + list(rejected)
    kept = accepted[:max_samples]
    if not kept:
        raise RuntimeError("no sample was produced")

    # ---- write samples ------------------------------------------------------------------------------
    aligned = []
    for k, x in enumerate(kept):
        xa = weighted_rigid_align(x_gt, x, weights=weights)[0].cpu()      # sample superposed on the GT (pocket CA)
        aligned.append(xa)
        write_ligand_sdf(ref_mol, xa[is_lig_cpu], out / f"ligand_pred_{k}.sdf")
        meta.write_pdb(out / f"system_pred_{k}.pdb", xa)
        meta.write_pdb(out / f"receptor_pred_{k}.pdb", xa, receptor_only=True)

    lig_gt = x_gt_cpu[is_lig_cpu].numpy()
    lig_pred = np.stack([xa[is_lig_cpu].numpy() for xa in aligned])
    rmsd_all = np.sqrt(((lig_pred - lig_gt[None]) ** 2).sum(-1).mean(-1))
    result = {"output_dir": str(out), "num_samples": len(kept), "num_accepted": len(accepted) - n_rejected_kept,
              "rounds": rounds_run, "rmsd_all": [round(float(r), 4) for r in rmsd_all]}

    result["seconds"] = round(time.time() - t_start, 1)
    summary = {"input_pkl": str(input_pkl), "msa_features": str(msa_dir), "uniprot_msa_features": str(uni_dir),
               "params": str(params_path), "seed": seed, "device": str(device), **result,
               "settings": {"max_samples": max_samples, "physics_correction": physics_correction,
                            "max_rounds": max_rounds, "num_samples_per_round": num_samples_per_round, "steps": steps,
                            "mmff_iters": mmff_iters, "eta": eta, "num_confs": num_confs, "crop_size": crop_size,
                            "atom_crop_size": atom_crop_size, "pocket_type": pocket_type,
                            "pocket_cutoff": pocket_cutoff, "pocket_dist_type": pocket_dist_type,
                            "use_pocket": use_pocket, "use_key_res": use_key_res, "rho": rho}}
    dump_json(summary, str(out / "summary.json"))
    return result


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Redock a single protein-ligand system with PhysDock.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    g = ap.add_argument_group("inputs / outputs")
    g.add_argument("-i", "--input_pkl", required=True, help="system .pkl.gz (receptor + ligand)")
    g.add_argument("-m", "--msa_features", required=True, help="folder with <md5>.pkl.gz MSA features")
    g.add_argument("-u", "--uniprot_msa_features", required=True, help="folder with <md5>.pkl.gz UniProt MSA features")
    g.add_argument("-o", "--output_dir", required=True, help="output folder (created if needed)")
    g.add_argument("--params", default=None, help="model weights (default: data/params/params.pt, params/params.pt)")
    g.add_argument("--ccd_meta", default=None,
                   help="CCD metadata (default: data/ccd_id_meta_data.pkl.gz, params/ccd_id_meta_data.pkl.gz)")
    g = ap.add_argument_group("sampling (same names/defaults as redocking.py)")
    g.add_argument("--max_samples", type=int, default=5, help="number of samples to keep")
    g.add_argument("--enable_physics_correction", action="store_true", help="enable physics correction")
    g.add_argument("--max_rounds", type=int, default=10, help="max sampling rounds (only with physics correction)")
    g.add_argument("--num_samples_per_round", type=int, default=5, help="samples drawn per round")
    g.add_argument("--steps", type=int, default=40, help="denoising steps")
    g.add_argument("--mmff_iters", type=int, default=5, help="MMFF steps per denoising step")
    g.add_argument("--eta", type=float, default=6.0, help="adaptive factor of the physics projection")
    g.add_argument("--num_confs", type=int, default=128, help="RDKit reference conformers")
    g.add_argument("--rho", type=int, default=1000, help="Karras noise schedule power")
    g.add_argument("--use_x_gt_ligand_as_ref_pos", action="store_true",
                   help="ablation only: use the ground-truth ligand as reference conformer")
    g.add_argument("--seed", type=int, default=None, help="random seed (default: nondeterministic)")
    g = ap.add_argument_group("features")
    g.add_argument("--crop_size", type=int, default=None, help="token crop size (default: no crop)")
    g.add_argument("--atom_crop_size", type=int, default=None, help="atom crop size (default: 8 x crop_size)")
    g.add_argument("--pocket_type", choices=["atom", "ca"], default="atom")
    g.add_argument("--pocket_cutoff", type=float, default=10.0)
    g.add_argument("--pocket_dist_type", choices=["ligand", "ligand_centre"], default="ligand")
    g.add_argument("--use_pocket", action="store_true", help="use the pocket feature")
    g.add_argument("--use_key_res", action="store_true", help="use the key-residue feature")
    g.add_argument("--key_res_random_mask_ratio", type=float, default=0.5)
    g = ap.add_argument_group("runtime")
    g.add_argument("--device", default=None, help="torch device (default: cuda:0 if available, else cpu)")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        res = redock_single(
            a.input_pkl, a.msa_features, a.uniprot_msa_features, a.output_dir,
            params_path=a.params, ccd_meta_path=a.ccd_meta, max_samples=a.max_samples,
            physics_correction=a.enable_physics_correction, max_rounds=a.max_rounds,
            num_samples_per_round=a.num_samples_per_round, steps=a.steps, mmff_iters=a.mmff_iters, eta=a.eta,
            num_confs=a.num_confs, crop_size=a.crop_size, atom_crop_size=a.atom_crop_size,
            use_x_gt_ligand_as_ref_pos=a.use_x_gt_ligand_as_ref_pos, pocket_type=a.pocket_type,
            pocket_cutoff=a.pocket_cutoff, pocket_dist_type=a.pocket_dist_type, use_pocket=a.use_pocket,
            use_key_res=a.use_key_res, key_res_random_mask_ratio=a.key_res_random_mask_ratio, rho=a.rho,
            device=a.device, seed=a.seed)
    except InputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    rm = res["rmsd_all"]
    log(f"done in {res['seconds']}s: {res['num_samples']} samples ({res['num_accepted']} accepted, {res['rounds']} round(s)); "
        f"ligand RMSD to GT: best {min(rm):.2f} A, median {float(np.median(rm)):.2f} A")
    log(f"results in {res['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

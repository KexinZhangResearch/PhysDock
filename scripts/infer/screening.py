import copy
import os
import shutil
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PHYSDOCK_ROOT", str(ROOT))

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from PhysDock.configs import to_dict  # noqa: E402

import numpy as np
import torch
import tqdm
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

from PhysDock.data import FeatureConfig, PhysDockDataloader, PhysDockDataset, msa_feat_for_round
from PhysDock.models.guidance import embed_conformers
from PhysDock.utils.import_weights import load_pretrained
from PhysDock.utils.io_utils import convert_md5_string, dump_json, dump_txt, load_txt
from PhysDock.utils.tensor_utils import weighted_rigid_align

RDLogger.DisableLog('rdApp.*')


def screening(
        input_pkl_path,
        smi_file_path,
        msa_features_dir,
        features,
        output_dir=None,
        params_path=None,
        model_cfg=None,
        ccd_path=None,
        num_workers=4,
        max_samples=5,
        physics_correction=False,
        max_rounds=10,
        num_augmentation_sample=5,
        steps=40,
        mmff_iters=5,
        mmff_gamma_0_factor_start=6.0,
        num_confs=128,
        karras_noise_schedule_power=1000,
        ranking=True,
        sidechain_relaxation=False,
        align_mode="pocket_ca",  # pocket_ca and ca
        device_id=0,
        dtype=torch.float32,
):
    crop_size, atom_crop_size, pocket_type, pocket_cutoff, pocket_dist_type, use_pocket, use_key_res, \
        key_res_random_mask_ratio, use_x_gt_ligand_as_ref_pos = (features[k] for k in (
            "crop_size", "atom_crop_size", "pocket_type", "pocket_cutoff", "pocket_dist_type", "use_pocket",
            "use_key_res", "key_res_random_mask_ratio", "use_x_gt_ligand_as_ref_pos"))
    print('''****************************************************************************
*    ██████╗ ██╗  ██╗██╗   ██╗███████╗██████╗  ██████╗  ██████╗██╗  ██╗    *
*    ██╔══██╗██║  ██║╚██╗ ██╔╝██╔════╝██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝    *
*    ██████╔╝███████║ ╚████╔╝ ███████╗██║  ██║██║   ██║██║     █████╔╝     *
*    ██╔═══╝ ██╔══██║  ╚██╔╝  ╚════██║██║  ██║██║   ██║██║     ██╔═██╗     *
*    ██║     ██║  ██║   ██║   ███████║██████╔╝╚██████╔╝╚██████╗██║  ██╗    *
*    ╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═════╝  ╚═════╝  ╚═════╝╚═╝  ╚═╝    *
****************************************************************************''')
    print("Starting Initialization ...")
    # Initialization

    device = f"cuda:{device_id}"
    assert os.path.isfile(input_pkl_path)
    if output_dir is None:
        output_dir = os.path.join(os.path.split(input_pkl_path)[0],
                                  os.path.split(input_pkl_path)[1].split(".")[0] + "_screening")
    os.makedirs(output_dir, exist_ok=True)
    if use_x_gt_ligand_as_ref_pos:  # DEBUG: use GT Ligand as reference conformer
        print(
            "You are using GT ligand conformer as reference conformer, which is dangerous and is only for ablation study.")

    feature_config = FeatureConfig(**dict(features, num_msa_samples=max_rounds))

    smis = load_txt(smi_file_path).split()
    smis_data = {smi: convert_md5_string(smi) for smi in smis}
    dump_json(smis_data, os.path.join(
        output_dir, os.path.split(smi_file_path)[1].split(".")[0] + f"_smi_md5_meta_data.json"))
    print(f"Finish dumping SMILES to MD5 meta data!")

    # one entry per SMILES (system id md5(smi)); molecules that cannot be featurised are logged and skipped
    dataset = PhysDockDataset.for_screening(input_pkl_path, smis, msa_features_dir, ccd=ccd_path, config=feature_config, on_error="skip")
    dataloader = PhysDockDataloader(dataset, batch_size=1, num_workers=num_workers)

    model = load_pretrained(params_path, model_cfg, device=device, dtype=dtype)

    print(f"Finish Initialization of model and feature loader!")
    print("################### Screening Configs #####################")
    print(f"# Total Molecules: {len(smis)}")
    print(f"# Max Samples per Molecule: {max_samples}")
    print(f"# Num of Augmentation Samples per Diffusion Conditioning: {num_augmentation_sample}")
    print(f"# Max Rounds: {max_rounds}")
    print(f"# Total Denoising Steps: {steps}")
    print(f"# Sample Diffusion Karras Noise Schedule Power: {karras_noise_schedule_power}")
    print(f"# Adaptive Projection Strategy Boundary: {mmff_gamma_0_factor_start}")
    print(f"# Num of Reference Conformers: {num_confs}")
    print(f"# MMFF Steps per Denoising Step: {mmff_iters}")
    if isinstance(crop_size, int):
        print(f"# Spatial Crop: True")
        print(f"# Crop Size: {crop_size}")
        print(f"# Atom Crop Size: {atom_crop_size}")
    else:
        print(f"# Spatial Crop: False")
    if use_pocket:
        print(f"# Use Pockets Feature: True")
        print(f"# Pocket Type: {pocket_type}")
        print(f"# Pocket Cutoff: {pocket_cutoff}")
        print(f"# pocket Distance Type: {pocket_type}")
    else:
        print(f"# Use Pockets Feature: False")
    if use_key_res:
        print(f"# Use Key Residue Feature: {use_key_res}")
        print(f"# Key Res Random Mask Ratio: {key_res_random_mask_ratio}")
    else:
        print(f"# Use Key Residue Feature: {use_key_res}")
    print(f"# Align to GT mode: {align_mode}")
    print(f"# K-Means Clustering and Ranking: {ranking}")
    print(f"# Sidechain Relaxation: {sidechain_relaxation and ranking}")
    print("###########################################################")

    for batch_id, sample in tqdm.tqdm(enumerate(dataloader), total=len(smis)):
        if sample is None:
            continue
        tensors, infer_meta_data = sample
        for k, v in tensors.items():
            tensors[k] = v.to(device)
        try:
            sample_id = infer_meta_data.system_id

            accept_samples = []
            reject_samples = deque([], maxlen=max_samples)
            ligand_templates = []  # Sample Ligand
            reference_templates = []  # RDkit

            weights = None
            x_gt = None
            # infer_meta_data = None
            ref_mol_poses = None
            ref_mol_poses_dist = None
            mmff_gamma_0_factor = mmff_gamma_0_factor_start

            skip_this = False
            chiral_centers = None
            ref_mol_num_error = False
            is_ligand_atom = None
            ref_mol = None
            for recycle_id in range(max_rounds):
                if recycle_id > 0 and not physics_correction:
                    break
                if skip_this:
                    break
                # Get Model Inputs and Meta Data
                if recycle_id >= 1:
                    tensors["msa_feat"] = msa_feat_for_round(tensors, recycle_id)
                ref_mol = infer_meta_data.ref_mol
                is_ligand_atom = tensors["is_ligand"][tensors["atom_to_token"]].bool()

                if os.path.exists(f"{output_dir}/{sample_id}/tmp/receptor_pred_4.pdb"):
                    continue
                if recycle_id == 0:
                    if len(tensors["x_gt"][is_ligand_atom]) != ref_mol.GetNumAtoms():
                        ref_mol_num_error = True

                    weights = (tensors["token_mask"] * tensors["is_protein"])[tensors["atom_to_token"].long()]
                    if align_mode == "pocket_ca":
                        if use_pocket:
                            weights = tensors["is_pocket"][tensors["atom_to_token"].long()] * weights
                    else:
                        raise NotImplementedError()
                    #     if align_mode == "pocket_ca":
                    #

                    x_gt = tensors["x_gt"][None]
                    os.makedirs(f"{output_dir}/{sample_id}", exist_ok=True)
                    os.makedirs(f"{output_dir}/{sample_id}/tmp", exist_ok=True)

                    infer_meta_data.write_pdb(f"{output_dir}/{sample_id}/tmp/system_gt.pdb", tensors["x_gt"])

                    infer_meta_data.write_pdb(f"{output_dir}/{sample_id}/tmp/receptor_gt.pdb", tensors["x_gt"], receptor_only=True)
                    ref_mol_copy = copy.deepcopy(ref_mol)
                    ligand_cpu = tensors["x_gt"][is_ligand_atom]
                    conf = ref_mol_copy.GetConformer()
                    for i in range(conf.GetNumAtoms()):
                        conf.SetAtomPosition(i, Point3D(*ligand_cpu[i].tolist()))

                    ligand_block = Chem.MolToMolBlock(ref_mol_copy, includeStereo=True)
                    dump_txt(ligand_block, f"{output_dir}/{sample_id}/tmp/ligand_gt.sdf")

                    chiral_centers_init = {i[0]: i[1] for i in Chem.FindMolChiralCenters(ref_mol)}
                    chiral_centers_gt = {i[0]: i[1] for i in Chem.FindMolChiralCenters(Chem.MolFromPDBBlock(
                        infer_meta_data.to_pdb(tensors["ref_pos"], ligand_only=True), sanitize=False
                    ))}
                    chiral_centers = {k: v for k, v in chiral_centers_gt.items() if k in chiral_centers_init}


                    if physics_correction:
                        ref_mol_poses = embed_conformers(ref_mol, num_confs).to(device)[:,
                                        :len(tensors["x_gt"][is_ligand_atom])]
                        ref_mol_poses_dist = torch.norm(ref_mol_poses[:, :, None] - ref_mol_poses[:, None], dim=-1)
                    else:
                        ref_mol_poses = None
                        ref_mol_poses_dist = None

                # break
                def _check_ref_mol_chirality(chiral_centers, PDB_BLOCK):  # PDB_BLOCK
                    try:
                        new_chiral_centers = {i[0]: i[1] for i in
                                              Chem.FindMolChiralCenters(
                                                  Chem.MolFromPDBBlock(PDB_BLOCK, sanitize=False))}
                    except:
                        return False

                    equal = True
                    for centre in chiral_centers:
                        if centre in new_chiral_centers:
                            if chiral_centers[centre] != new_chiral_centers[centre]:
                                equal = False
                                break
                        else:
                            equal = False
                            break
                    return equal

                with torch.no_grad():
                    x_pred = model.sample_diffusion(
                        tensors,
                        num_sample=num_augmentation_sample,
                        steps=steps,
                        # SDE Config
                        mmff_gamma_0_factor=mmff_gamma_0_factor,
                        align_ref_pos=recycle_id > 0,
                        # ODE Config
                        ref_mol=infer_meta_data.ref_mol if not ref_mol_num_error else None,
                        ref_mol_poses=x_gt[:, is_ligand_atom] if use_x_gt_ligand_as_ref_pos else \
                            torch.stack(ligand_templates + reference_templates, dim=0) if recycle_id > 0 else None,
                        use_ref_mol_poses=recycle_id != 0 and physics_correction,
                        ode_step_scale_eta=1.0 if not ref_mol_num_error else 1.5,
                        mmff_iters=mmff_iters,
                        karras_noise_schedule_power=karras_noise_schedule_power,
                    )
                x_pred_cpu = x_pred.cpu()

                pass_flags = []
                for x_id, (x, x_cpu) in enumerate(zip(x_pred, x_pred_cpu)):

                    if physics_correction:
                        ligand_pdb_block = infer_meta_data.to_pdb(x_cpu, ligand_only=True)
                        pass_flag = _check_ref_mol_chirality(chiral_centers, ligand_pdb_block)
                    else:
                        pass_flag = True
                    pass_flags.append(pass_flag)
                    if pass_flag:
                        ligand_pred = x[is_ligand_atom]
                        ligand_templates.append(ligand_pred)
                        accept_samples.append(x)
                    else:
                        reject_samples.append(x)
                if physics_correction:
                    if any(pass_flags):
                        mmff_gamma_0_factor = mmff_gamma_0_factor * 1.15
                    else:
                        mmff_gamma_0_factor = max(mmff_gamma_0_factor * 0.7, 1)
                    if len(accept_samples) >= max_samples:
                        break

                    ligand_poses = x_pred[:, is_ligand_atom]
                    ligand_dist = torch.norm(ligand_poses[:, :, None] - ligand_poses[:, None], dim=-1)
                    delta = (ligand_dist[:, None] - ref_mol_poses_dist[None]).abs()
                    epsilon = 0.25 * (torch.sigmoid(-0.5 + delta) + torch.sigmoid(-1 + delta) + torch.sigmoid(
                        -2 + delta) + torch.sigmoid(-4 + delta))
                    epsilon = epsilon.mean(dim=[-1, -2, -4])
                    used_inds = torch.argsort(epsilon)[:max_samples - len(ligand_templates)]
                    reference_templates = []
                    for ind in used_inds:
                        reference_templates.append(ref_mol_poses[ind])

            if len(accept_samples) < num_augmentation_sample:
                accept_samples = accept_samples + [_ for _ in reject_samples]

            is_ligand_atom = is_ligand_atom.cpu()
            for accept_sample_id, x in enumerate(accept_samples[:max_samples]):
                x_aligned = weighted_rigid_align(x_gt, x, weights=weights)[0].cpu()
                block = infer_meta_data.to_pdb(x_aligned)
                receptor_block = infer_meta_data.to_pdb(x_aligned, receptor_only=True)
                ref_mol_copy = copy.deepcopy(ref_mol)
                ligand_cpu = x_aligned[is_ligand_atom]
                conf = ref_mol_copy.GetConformer()
                for i in range(conf.GetNumAtoms()):
                    conf.SetAtomPosition(i, Point3D(*ligand_cpu[i].tolist()))

                ligand_block = Chem.MolToMolBlock(ref_mol_copy, includeStereo=True)
                dump_txt(ligand_block,
                         f"{output_dir}/{sample_id}/tmp/ligand_pred_{accept_sample_id}.sdf")
                dump_txt(block, f"{output_dir}/{sample_id}/tmp/system_pred_{accept_sample_id}.pdb")
                dump_txt(receptor_block, f"{output_dir}/{sample_id}/tmp/receptor_pred_{accept_sample_id}.pdb")
            if ranking:

                def _get_coor(fname):
                    mol = Chem.MolFromMolBlock(load_txt(fname), sanitize=False, removeHs=True)
                    mol = Chem.RemoveHs(mol)
                    # return mol
                    conf = mol.GetConformer()
                    coors = []
                    for i in range(mol.GetNumAtoms()):
                        pos = conf.GetAtomPosition(i)
                        coors.append([pos.x, pos.y, pos.z])
                    return np.array(coors)

                gt = _get_coor(f"{output_dir}/{sample_id}/tmp/ligand_gt.sdf")
                rmsds = []
                preds = []
                skip = 0
                for i in range(1000):
                    try:
                        if not os.path.exists(f"{output_dir}/{sample_id}/tmp/ligand_pred_{i}.sdf"):
                            continue
                        # print(f"{dir}/{file}/{file}_aligned_{i}_ligand.pdb")

                        pred = _get_coor(f"{output_dir}/{sample_id}/tmp/ligand_pred_{i}.sdf")
                        preds.append(pred)
                        rmsd = np.sqrt(np.mean(np.linalg.norm(pred - gt, axis=-1) ** 2, axis=0))
                        rmsds.append(rmsd)
                    except Exception as e:
                        skip = 1
                        break
                if skip:
                    continue
                pred = np.stack(preds, axis=0)
                dist = np.sqrt(np.mean(np.linalg.norm(pred[:, None] - pred[None], axis=-1) ** 2, axis=-1))

                def get_representatives(distance_matrix, num_clusters=5):
                    num_elements = len(distance_matrix)
                    coordinates = np.zeros((num_elements, num_elements))
                    for i in range(num_elements):
                        coordinates[i] = distance_matrix[i]

                    from sklearn.cluster import KMeans         # only needed for ranking
                    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
                    kmeans.fit(coordinates)

                    labels = kmeans.labels_

                    representatives_indices = []
                    for cluster_id in range(num_clusters):
                        cluster_indices = np.where(labels == cluster_id)[0]
                        avg_distances = np.mean(distance_matrix[cluster_indices, :], axis=0)
                        representative_index = cluster_indices[np.argmin(avg_distances[cluster_indices])]
                        representatives_indices.append(representative_index)
                    return representatives_indices

                num_clusters = 5
                if len(dist) > num_clusters:
                    ids = get_representatives(dist, num_clusters)
                    ids_1 = get_representatives(dist, 1)[0]
                    if ids_1 in ids:
                        ids.remove(ids_1)
                        ids = [ids_1] + ids
                    else:
                        ids = [ids_1] + ids[:4]
                    rmsds = [rmsds[i] for i in ids]  # 70 42 36 31 30
                else:
                    ids = list(range(len(rmsds)))

                # Copy
                # shutil.copy(f"{output_dir}/{sample_id}/tmp/ligand_gt.sdf",
                #             f"{output_dir}/{sample_id}/ligand_gt.sdf")
                # shutil.copy(f"{output_dir}/{sample_id}/tmp/system_gt.pdb",
                #             f"{output_dir}/{sample_id}/system_gt.pdb")
                shutil.copy(f"{output_dir}/{sample_id}/tmp/receptor_gt.pdb",
                            f"{output_dir}/{sample_id}/receptor_gt.pdb")
                for rank_id, i in enumerate(ids):
                    shutil.copy(f"{output_dir}/{sample_id}/tmp/ligand_pred_{i}.sdf",
                                f"{output_dir}/{sample_id}/ligand_rank_{rank_id}.sdf")
                    shutil.copy(f"{output_dir}/{sample_id}/tmp/system_pred_{i}.pdb",
                                f"{output_dir}/{sample_id}/system_rank_{rank_id}.pdb")
                    shutil.copy(f"{output_dir}/{sample_id}/tmp/receptor_pred_{i}.pdb",
                                f"{output_dir}/{sample_id}/receptor_rank_{rank_id}.pdb")
                if sidechain_relaxation:
                    from PhysDock.utils.relaxation import relax   # needs openmm / pdbfixer
                    # print("Start Relaxation!")
                    for rank_id, i in enumerate(ids):
                        # print(rank_id)
                        relax(
                            f"{output_dir}/{sample_id}/receptor_rank_{rank_id}.pdb",
                            f"{output_dir}/{sample_id}/ligand_rank_{rank_id}.sdf"
                        )
                dump_json(rmsds, f"{output_dir}/{sample_id}/top5_rmsd.json")

        except Exception as e:
            print(e)
            continue


@hydra.main(config_path="../../configs", config_name="screening", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    sp = cfg.sampling
    screening(
        input_pkl_path=cfg.input_pkl,
        smi_file_path=cfg.smiles,
        msa_features_dir=cfg.features_dir,
        features=to_dict(cfg.features),
        output_dir=cfg.output_dir,
        params_path=cfg.params,
        model_cfg=cfg.model,
        ccd_path=cfg.ccd,
        num_workers=cfg.num_workers,
        max_samples=sp.max_samples,
        physics_correction=sp.physics_correction,
        max_rounds=sp.max_rounds,
        num_augmentation_sample=sp.num_samples_per_round,
        steps=sp.steps,
        mmff_iters=sp.mmff_iters,
        mmff_gamma_0_factor_start=sp.eta,
        num_confs=sp.num_confs,
        karras_noise_schedule_power=sp.rho,
        ranking=cfg.ranking,
        sidechain_relaxation=cfg.sidechain_relaxation,
        align_mode=cfg.align_mode,
        device_id=cfg.device_id,
    )


if __name__ == "__main__":
    main()

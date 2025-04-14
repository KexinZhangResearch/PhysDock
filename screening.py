import copy
import sys
import torch
import os
import tqdm
import pandas as pd
import argparse

from PhysDock.utils.import_weights import import_state_dict
from PhysDock.data.feature_loader import FeatureLoader
from PhysDock import PhysDock, PhysDockConfig
from PhysDock.utils.tensor_utils import weighted_rigid_align
from PhysDock.utils.io_utils import load_txt, dump_txt, chunk_lists, load_json, convert_md5_string, dump_json, \
    find_files
from collections import deque
import numpy as np
import shutil
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from PhysDock.data.relaxation import relax
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

RDLogger.DisableLog('rdApp.*')


def screening(
        input_pkl_path,
        smi_file_path,
        msa_features_dir,

        output_dir=None,
        max_samples=5,
        physics_correction=False,
        max_rounds=10,
        num_augmentation_sample=5,
        steps=40,
        mmff_iters=5,
        mmff_gamma_0_factor_start=6.0,
        num_confs=128,
        crop_size=256,
        atom_crop_size=256 * 8,

        use_x_gt_ligand_as_ref_pos=False,
        pocket_type="atom",
        pocket_cutoff=10,
        pocket_dist_type="ligand",
        use_pocket=True,
        use_key_res=False,  # VS not use Key Tes
        key_res_random_mask_ratio=0.5,  # Redock use random key res: 50%
        karras_noise_schedule_power=1000,
        ranking=True,
        sidechain_relaxation=False,
        align_mode="pocket_ca",  # pocket_ca and ca

        device_id=0,
        dtype=torch.float32,
):
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
    ccd_id_meta_data_path = os.path.join(os.path.split(__file__)[0], "params", "ccd_id_meta_data.pkl.gz")
    params_path = os.path.join(os.path.split(__file__)[0], "params", "params.pt")

    device = f"cuda:{device_id}"
    assert os.path.isfile(input_pkl_path)
    if output_dir is None:
        output_dir = os.path.join(os.path.split(input_pkl_path)[0],
                                  os.path.split(input_pkl_path)[1].split(".")[0] + "_screening")
    os.makedirs(output_dir, exist_ok=True)
    if use_x_gt_ligand_as_ref_pos:  # DEBUG: use GT Ligand as reference conformer
        print(
            "You are using GT ligand conformer as reference conformer, which is dangerous and is only for ablation study.")

    feature_loader = FeatureLoader(
        msa_features_dir=msa_features_dir,
        ccd_id_meta_data=ccd_id_meta_data_path,
        crop_size=crop_size,
        atom_crop_size=atom_crop_size,
        inference_mode=True,
        infer_pocket_type=pocket_type,
        infer_pocket_cutoff=pocket_cutoff,
        infer_pocket_dist_type=pocket_dist_type,
        infer_use_pocket=use_pocket,
        infer_use_key_res=use_key_res,
        key_res_random_mask_ratio=key_res_random_mask_ratio,
        use_x_gt_ligand_as_ref_pos=use_x_gt_ligand_as_ref_pos,
        num_recycles=max_rounds,
    )

    smis = load_txt(smi_file_path).split()
    smis_data = {smi: convert_md5_string(smi) for smi in smis}
    dump_json(smis_data, os.path.join(
        output_dir, os.path.split(smi_file_path)[1].split(".")[0] + f"_smi_md5_meta_data.json"))
    print(f"Finish dumping SMILES to MD5 meta data!")

    def _collator_fn(smi):
        smi = smi[0]
        try:
            tensors, infer_meta_data = feature_loader.load(
                system_pkl_path=input_pkl_path,
                remove_ligand=True,
                smi=smi
            )
            return smi, tensors, infer_meta_data
        except Exception as e:
            return smi, None, None

    dataloader = DataLoader(
        smis,
        collate_fn=_collator_fn,
        batch_size=1,
        num_workers=4,
    )

    config = PhysDockConfig(model_name="medium")
    model = PhysDock(config).to(device).to(dtype)
    import_state_dict(model, params_path)
    model.eval()

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

    for batch_id, (smi, tensors, infer_meta_data) in tqdm.tqdm(enumerate(dataloader), total=len(smis)):
        if infer_meta_data is None:
            continue
        for k, v in tensors.items():
            tensors[k] = v.to(device)
        try:
            sample_id = convert_md5_string(smi)
            infer_meta_data["system_id"] = sample_id

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
                    tensors["msa_feat"] = tensors["batch_msa_feat"][recycle_id]
                ref_mol = infer_meta_data["ref_mol"]
                is_ligand_atom = tensors["is_ligand"][tensors["atom_id_to_token_id"]].bool()

                if os.path.exists(f"{output_dir}/{sample_id}/tmp/receptor_pred_4.pdb"):
                    continue
                if recycle_id == 0:
                    if len(tensors["x_gt"][is_ligand_atom]) != ref_mol.GetNumAtoms():
                        ref_mol_num_error = True

                    weights = (tensors["s_mask"] * tensors["is_protein"])[tensors["atom_id_to_token_id"].long()]
                    # if use_pocket:
                    if align_mode == "pocket_ca":
                        weights = tensors["pocket_res_feat"][tensors["atom_id_to_token_id"].long()] * weights
                    # else:
                    #     if align_mode == "pocket_ca":
                    #         raise NotImplementedError()

                    x_gt = tensors["x_gt"][None]
                    os.makedirs(f"{output_dir}/{sample_id}", exist_ok=True)
                    os.makedirs(f"{output_dir}/{sample_id}/tmp", exist_ok=True)

                    feature_loader.write_pdb(
                        tensors["x_gt"],
                        f"{output_dir}/{sample_id}/tmp/system_gt.pdb",
                        infer_meta_data
                    )

                    feature_loader.write_pdb(
                        tensors["x_gt"],
                        f"{output_dir}/{sample_id}/tmp/receptor_gt.pdb",
                        infer_meta_data,
                        receptor_only=True
                    )
                    ref_mol_copy = copy.deepcopy(ref_mol)
                    ligand_cpu = tensors["x_gt"][is_ligand_atom]
                    conf = ref_mol_copy.GetConformer()
                    for i in range(conf.GetNumAtoms()):
                        conf.SetAtomPosition(i, Point3D(*ligand_cpu[i].tolist()))

                    ligand_block = Chem.MolToMolBlock(ref_mol_copy, includeStereo=True)
                    dump_txt(ligand_block, f"{output_dir}/{sample_id}/tmp/ligand_gt.sdf")

                    chiral_centers_init = {i[0]: i[1] for i in Chem.FindMolChiralCenters(ref_mol)}
                    chiral_centers_gt = {i[0]: i[1] for i in Chem.FindMolChiralCenters(Chem.MolFromPDBBlock(
                        feature_loader.write_pdb_block(
                            tensors["ref_pos"],  # Assert Ref Pos Chirality Centre is Accurate
                            infer_meta_data=infer_meta_data,
                            ligand_only=True
                        ), sanitize=False
                    ))}
                    chiral_centers = {k: v for k, v in chiral_centers_gt.items() if k in chiral_centers_init}

                    def _get_ref_mol_poses(ref_mol, num_confs=128):
                        mol = copy.deepcopy(ref_mol)
                        cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, enforceChirality=True)
                        num_atoms = mol.GetNumAtoms()
                        coordinates = torch.zeros(num_confs, num_atoms, 3)
                        for i, cid in enumerate(cids):
                            conf = mol.GetConformer(cid)
                            for j in range(num_atoms):
                                pos = conf.GetAtomPosition(j)
                                coordinates[i, j, 0] = pos.x
                                coordinates[i, j, 1] = pos.y
                                coordinates[i, j, 2] = pos.z
                        return coordinates

                    if physics_correction:
                        ref_mol_poses = _get_ref_mol_poses(ref_mol, num_confs=num_confs).to(device)[:,
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
                        ref_mol=infer_meta_data["ref_mol"] if not ref_mol_num_error else None,
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
                        ligand_pdb_block = feature_loader.write_pdb_block(x_cpu, infer_meta_data=infer_meta_data,
                                                                          ligand_only=True)
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
                block = feature_loader.write_pdb_block(x_aligned, infer_meta_data=infer_meta_data)
                receptor_block = feature_loader.write_pdb_block(x_aligned, infer_meta_data=infer_meta_data,
                                                                receptor_only=True)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PhysDock Redocking')
    parser.add_argument('-i', type=str, help='system pkl path', required=True)
    parser.add_argument('-s', type=str, help='text file that contains SMILES', required=True)
    parser.add_argument('-f', type=str, help='msa features dir', required=True)
    parser.add_argument('--output_dir', type=str, help='output folder', required=False, default=None)
    parser.add_argument('--max_samples', type=int, help='num samples for each system', default=5)
    parser.add_argument('--enable_physics_correction', help='enable physics correction', action="store_true")
    parser.add_argument('--max_rounds', type=int, help='max try', default=10)
    parser.add_argument('--num_samples_per_round', type=int, help='num augmentation samples per round', default=5)
    parser.add_argument('--steps', type=int, help='denoising steps', default=40)
    parser.add_argument('--mmff_iters', type=int, help='mmff steps of each denoising step', default=5)
    parser.add_argument('--eta',
                        type=int, help='adaptive factor to control physics projection', default=6)
    parser.add_argument('--num_confs', type=int, help='reference conformers of each ligand', default=128)
    parser.add_argument('--crop_size', type=int, help='token crop size. no crop set to none', default=None)
    parser.add_argument('--atom_crop_size', type=int, help='atom_crop_size', default=None)
    parser.add_argument('--ebable_x_gt_ligand_as_ref_pos', help='ebable_x_gt_ligand_as_ref_pos', action="store_true")
    parser.add_argument('--pocket_type', type=str, help='pocket selection scheme', default="atom")
    parser.add_argument('--pocket_cutoff', type=int, help='pocket selection cutoff', default=10)
    parser.add_argument('--pocket_dist_type', type=str, help='pocket crop centre', default="ligand")
    parser.add_argument('--use_pocket', help='whether to use pocket feature', action="store_true")
    parser.add_argument('--use_key_res', help='whether to use key res feature', action="store_true")
    parser.add_argument('--key_res_random_mask_ratio', type=float, help='whether to use key res feature', default=0.5)
    parser.add_argument('--rho', type=int, help='karras noise schedule power', default=1000)
    parser.add_argument('--enable_ranking', help='ranking samples', action="store_true")
    parser.add_argument('--align_mode', type=str, help='align mode', default="pocket_ca")
    parser.add_argument('--device_id', type=int, help='cuda device id', default=0)
    parser.add_argument('--enable_sidechain_relaxation', help='relax sidechains', action="store_true")

    args = parser.parse_args()

    screening(
        input_pkl_path=args.i,
        smi_file_path=args.s,
        msa_features_dir=args.f,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        physics_correction=args.enable_physics_correction,
        max_rounds=args.max_rounds,
        num_augmentation_sample=args.num_samples_per_round,
        steps=args.steps,
        mmff_iters=args.mmff_iters,
        mmff_gamma_0_factor_start=args.eta,
        num_confs=args.num_confs,
        crop_size=args.crop_size,
        atom_crop_size=args.atom_crop_size,
        use_x_gt_ligand_as_ref_pos=args.ebable_x_gt_ligand_as_ref_pos,
        pocket_type=args.pocket_type,
        pocket_cutoff=args.pocket_cutoff,
        pocket_dist_type=args.pocket_dist_type,
        use_pocket=args.use_pocket,
        use_key_res=args.use_key_res,  # VS not use Key Tes
        key_res_random_mask_ratio=args.key_res_random_mask_ratio,  # Redock use random key res: 50%
        karras_noise_schedule_power=args.rho,
        ranking=args.enable_ranking,
        sidechain_relaxation=args.enable_sidechain_relaxation,
        align_mode=args.align_mode,  # pocket_ca and ca
        device_id=args.device_id,
        dtype=torch.float32,
    )

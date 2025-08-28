import copy
import os
import random
from functools import reduce
from operator import add
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

from PhysDock.data.constants.PDBData import protein_letters_3to1_extended
from PhysDock.data.constants import restype_constants as rc
from PhysDock.utils.io_utils import convert_md5_string, load_json, load_pkl, dump_txt, find_files
from PhysDock.data.tools.feature_processing_multimer import pair_and_merge
from PhysDock.utils.tensor_utils import centre_random_augmentation_np_apply, dgram_from_positions, \
    centre_random_augmentation_np_batch
from PhysDock.data.constants.periodic_table import PeriodicTable
from PhysDock.data.tools.rdkit import get_features_from_smi

PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


class FeatureLoader:
    def __init__(
            self,
            # Dataset Config
            dataset_path=None,
            msa_features_dir=None,
            ccd_id_meta_data=None,

            crop_size=256,
            atom_crop_size=256 * 8,

            # Infer config
            inference_mode=False,
            infer_pocket_type="atom",  # "ca"
            infer_pocket_cutoff=6,  # 8 10 12
            infer_pocket_dist_type="ligand",  # "ligand_centre"
            infer_use_pocket=True,
            infer_use_key_res=True,

            # Train Config
            train_pocket_type_atom_ratio=0.5,
            train_pocket_cutoff_ligand_min=6,
            train_pocket_cutoff_ligand_max=12,
            train_pocket_cutoff_ligand_centre_min=10,
            train_pocket_cutoff_ligand_centre_max=16,
            train_pocket_dist_type_ligand_ratio=0.5,
            train_use_pocket_ratio=0.5,
            train_use_key_res_ratio=0.5,

            train_shuffle_sym_id=True,
            train_spatial_crop_ligand_ratio=0.2,
            train_spatial_crop_interface_ratio=0.4,
            train_spatial_crop_interface_threshold=15.,
            train_charility_augmentation_ratio=0.1,
            train_use_template_ratio=0.75,
            train_template_mask_max_ratio=0.4,

            # Other Configs
            max_msa_clusters=128,
            key_res_random_mask_ratio=0.5,

            # Abalation
            use_x_gt_ligand_as_ref_pos=False,

            # Recycle
            num_recycles=None
    ):
        # Init Dataset
        if dataset_path is not None:
            self.msa_features_path = os.path.join(dataset_path, "msa_features")
            self.uniprot_msa_features_path = os.path.join(dataset_path, "uniprot_msa_features")
            if os.path.exists(os.path.join(dataset_path, "train_val")):
                self.used_sample_ids = find_files(os.path.join(dataset_path, "train_val"))
                if os.path.exists(os.path.join(dataset_path, "train_val_weights.json")):
                    weights = load_json(os.path.join(dataset_path, "train_val_weights.json"))
                    self.weights = np.array([weights[sample_id] for sample_id in self.used_sample_ids])
                    self.probabilities = torch.from_numpy(self.weights / self.weights.sum())
        if msa_features_dir is not None:
            self.msa_features_path = os.path.join(msa_features_dir, "msa_features")
            self.uniprot_msa_features_path = os.path.join(msa_features_dir, "uniprot_msa_features")
            # self.ccd_id_meta_data = load_pkl(
            #     "/2022133002/projects/stdock/stdock_v9.5/scripts/ccd_meta_data_confs_chars.pkl.gz")
            # self.ccd_id_ref_mol = load_pkl("/2022133002/projects/stdock/stdock_v9.5/scripts/ccd_dict.pkl.gz")
            #
            # # self.ccd_id_meta_data = load_pkl(
            # #     "/2022133002/projects/stdock/stdock_v9.5/scripts/ccd_meta_data_confs_chars.pkl.gz")
            # ddd_ccd_meta = load_pkl(
            #     "/2022133002/projects/stdock/stdock_v9.5/scripts/phi_ccd_meta_data_confs_chars.pkl.gz")
            # self.ccd_id_meta_data.update(ddd_ccd_meta)
            # self.ccd_id_ref_mol = load_pkl("/2022133002/projects/stdock/stdock_v9.5/scripts/ccd_dict.pkl.gz")
            # ccd_id_ref_mol_phi = load_pkl("/2022133002/projects/stdock/stdock_v9.5/scripts/phi_ccd_dict.pkl.gz")
            # self.ccd_id_ref_mol.update(ccd_id_ref_mol_phi)
        if ccd_id_meta_data is None:
            assert os.path.exists(os.path.join(dataset_path, "ccd_id_meta_data.pkl.gz")) or ccd_id_meta_data is not None
            self.ccd_id_meta_data = load_pkl(os.path.join(dataset_path, "ccd_id_meta_data.pkl.gz"))
        else:
            self.ccd_id_meta_data = load_pkl(ccd_id_meta_data)


        # Inference Config
        self.inference_mode = inference_mode
        self.infer_use_pocket = infer_use_pocket
        self.infer_use_key_res = infer_use_key_res
        self.infer_pocket_type = infer_pocket_type
        self.infer_pocket_cutoff = infer_pocket_cutoff
        self.infer_pocket_dist_type = infer_pocket_dist_type

        # Training Config
        self.train_pocket_type_atom_ratio = train_pocket_type_atom_ratio
        self.train_pocket_cutoff_ligand_min = train_pocket_cutoff_ligand_min
        self.train_pocket_cutoff_ligand_max = train_pocket_cutoff_ligand_max
        self.train_pocket_cutoff_ligand_centre_min = train_pocket_cutoff_ligand_centre_min
        self.train_pocket_cutoff_ligand_centre_max = train_pocket_cutoff_ligand_centre_max
        self.train_pocket_dist_type_ligand_ratio = train_pocket_dist_type_ligand_ratio
        self.train_use_pocket_ratio = train_use_pocket_ratio
        self.train_use_key_res_ratio = train_use_key_res_ratio
        self.train_shuffle_sym_id = train_shuffle_sym_id
        self.train_spatial_crop_ligand_ratio = train_spatial_crop_ligand_ratio
        self.train_spatial_crop_interface_ratio = train_spatial_crop_interface_ratio
        self.train_spatial_crop_interface_threshold = train_spatial_crop_interface_threshold
        self.train_charility_augmentation_ratio = train_charility_augmentation_ratio
        self.train_use_template_ratio = train_use_template_ratio
        self.train_template_mask_max_ratio = train_template_mask_max_ratio

        # Other Configs
        self.token_bond_threshold = 2.4
        self.key_res_random_mask_ratio = key_res_random_mask_ratio
        self.crop_size = crop_size
        self.atom_crop_size = atom_crop_size
        self.max_msa_clusters = max_msa_clusters

        self.use_x_gt_ligand_as_ref_pos = use_x_gt_ligand_as_ref_pos

        self.num_recycles = num_recycles

    def _update_CONF_META_DATA(self, CONF_META_DATA, ccds):
        for ccd in ccds:
            if ccd in CONF_META_DATA:
                continue
            ccd_features = self.ccd_id_meta_data[ccd]
            ref_pos = ccd_features["ref_pos"]
            ref_pos = ref_pos - np.mean(ref_pos, axis=0, keepdims=True)
            CONF_META_DATA[ccd] = {
                "ref_feat": np.concatenate([
                    ref_pos,
                    ccd_features["ref_charge"][..., None],
                    rc.eye_128[ccd_features["ref_element"]].astype(np.float32),
                    ccd_features["ref_is_aromatic"].astype(np.float32)[..., None],
                    rc.eye_9[ccd_features["ref_degree"]].astype(np.float32),
                    rc.eye_7[ccd_features["ref_hybridization"]].astype(np.float32),
                    rc.eye_9[ccd_features["ref_implicit_valence"]].astype(np.float32),
                    rc.eye_3[ccd_features["ref_chirality"]].astype(np.float32),
                    ccd_features["ref_in_ring_of_3"].astype(np.float32)[..., None],
                    ccd_features["ref_in_ring_of_4"].astype(np.float32)[..., None],
                    ccd_features["ref_in_ring_of_5"].astype(np.float32)[..., None],
                    ccd_features["ref_in_ring_of_6"].astype(np.float32)[..., None],
                    ccd_features["ref_in_ring_of_7"].astype(np.float32)[..., None],
                    ccd_features["ref_in_ring_of_8"].astype(np.float32)[..., None],
                ], axis=-1),
                "rel_tok_feat": np.concatenate([
                    rc.eye_32[ccd_features["d_token"]].astype(np.float32),
                    rc.eye_5[ccd_features["bond_type"]].astype(np.float32),
                    ccd_features["token_bonds"].astype(np.float32)[..., None],
                    ccd_features["bond_as_double"].astype(np.float32)[..., None],
                    ccd_features["bond_in_ring"].astype(np.float32)[..., None],
                    ccd_features["bond_is_conjugated"].astype(np.float32)[..., None],
                    ccd_features["bond_is_aromatic"].astype(np.float32)[..., None],
                ], axis=-1),
                "ref_atom_name_chars": ccd_features["ref_atom_name_chars"],
                "ref_element": ccd_features["ref_element"],
                "token_bonds": ccd_features["token_bonds"],
            }

        return CONF_META_DATA

    def _update_chain_feature(self, chain_feature, CONF_META_DATA, use_pocket, use_key_res, ):
        ccds_ori = chain_feature["ccds"]
        chain_class = chain_feature["chain_class"]
        if chain_class == "protein":
            sequence = "".join([protein_letters_3to1_extended.get(ccd, "X") for ccd in ccds_ori])
            md5 = convert_md5_string(f"protein:{sequence}")
            # with open("add_msa.fasta", "a") as f:
            #     f.write(f">{md5}\n{sequence}\n")
            try:
                # import shutil
                # shutil.copy(
                #     os.path.join(self.msa_features_path, f"{md5}.pkl.gz"),
                #     os.path.join("/home/zhangkexin/research/PhysDock/examples/demo/features/msa_features",
                #                  f"{md5}.pkl.gz")
                # )
                # shutil.copy(
                #     os.path.join(self.uniprot_msa_features_path, f"{md5}.pkl.gz"),
                #     os.path.join("/home/zhangkexin/research/PhysDock/examples/demo/features/uniprot_msa_features",
                #                  f"{md5}.pkl.gz")
                # )
                chain_feature.update(
                    load_pkl(os.path.join(self.msa_features_path, f"{md5}.pkl.gz"))
                )
            except:
                print(f"Can't find msa feature!!! md5: {md5}")
                with open("add_msa.fasta", "a") as f:
                    f.write(f">{md5}\n{sequence}\n")

            chain_feature.update(
                load_pkl(os.path.join(self.uniprot_msa_features_path, f"{md5}.pkl.gz"))
            )
        else:
            chain_feature["msa"] = np.array([[rc.standard_ccds.index(ccd)
                                              if ccd in rc.standard_ccds else 20 for ccd in ccds_ori]] * 2,
                                            dtype=np.int8)
            chain_feature["deletion_matrix"] = np.zeros_like(chain_feature["msa"])

        # Merge Key Res Feat & Augmentation
        if "salt bridges" in chain_feature and use_key_res:
            key_res_feat = np.stack([
                chain_feature["salt bridges"],
                chain_feature["pi-cation interactions"],
                chain_feature["hydrophobic interactions"],
                chain_feature["pi-stacking"],
                chain_feature["hydrogen bonds"],
                chain_feature["metal complexes"],
                np.zeros_like(chain_feature["salt bridges"]),
            ], axis=-1).astype(np.float32)
        else:
            key_res_feat = np.zeros([len(ccds_ori), 7], dtype=np.float32)
        is_key_res = np.any(key_res_feat.astype(np.bool_), axis=-1).astype(np.float32)
        # Augmentation
        # if not self.inference_mode:
        key_res_feat = (key_res_feat *
                        (np.random.random([len(ccds_ori), 7]) > self.key_res_random_mask_ratio))
        if "pocket_res_feat" in chain_feature and use_pocket:
            pocket_res_feat = chain_feature["pocket_res_feat"]
        else:
            pocket_res_feat = np.zeros([len(ccds_ori)], dtype=np.float32)
        x_gt = []
        atom_id_to_conformer_atom_id = []

        # Conformer
        conformer_id_to_chunk_sizes = []
        residue_index = []
        restype = []
        ccds = []

        conformer_exists = []

        for c_id, ccd in enumerate(chain_feature["ccds"]):
            no_atom_this_conf = len(CONF_META_DATA[ccd]["ref_feat"])
            conformer_atom_ids_this_conf = np.arange(no_atom_this_conf)

            x_gt_this_conf = chain_feature["all_atom_positions"][c_id]
            x_exists_this_conf = chain_feature["all_atom_mask"][c_id].astype(np.bool_)

            # TODO DEBUG
            # conformer_exist = np.sum(x_exists_this_conf).item() > len(x_exists_this_conf) - 2
            conformer_exist = np.any(x_exists_this_conf).item()

            if rc.is_standard(ccd):
                conformer_exist = conformer_exist and x_exists_this_conf[1]
                if ccd != "GLY":
                    conformer_exist = conformer_exist and x_exists_this_conf[4]

            conformer_exists.append(conformer_exist)
            if conformer_exist:
                # Atomwise
                x_gt.append(x_gt_this_conf[x_exists_this_conf])
                atom_id_to_conformer_atom_id.append(conformer_atom_ids_this_conf[x_exists_this_conf])
                # Tokenwise
                residue_index.append(c_id)
                conformer_id_to_chunk_sizes.append(np.sum(x_exists_this_conf).item())
                restype.append(rc.standard_ccds.index(ccd) if ccd in rc.standard_ccds else 20)
                ccds.append(ccd)
        # print(ccds)
        # print("x_gt", x_gt)
        x_gt = np.concatenate(x_gt, axis=0)
        atom_id_to_conformer_atom_id = np.concatenate(atom_id_to_conformer_atom_id, axis=0, dtype=np.int32)
        residue_index = np.array(residue_index, dtype=np.int64)
        conformer_id_to_chunk_sizes = np.array(conformer_id_to_chunk_sizes, dtype=np.int64)
        restype = np.array(restype, dtype=np.int64)

        conformer_exists = np.array(conformer_exists, dtype=np.bool_)

        chain_feature_update = {
            "x_gt": x_gt,
            "atom_id_to_conformer_atom_id": atom_id_to_conformer_atom_id,
            "residue_index": residue_index,
            "conformer_id_to_chunk_sizes": conformer_id_to_chunk_sizes,
            "restype": restype,
            "ccds": ccds,
            "msa": chain_feature["msa"][:, conformer_exists],
            "deletion_matrix": chain_feature["deletion_matrix"][:, conformer_exists],
            "chain_class": chain_class,
            "key_res_feat": key_res_feat[conformer_exists],
            "is_key_res": is_key_res[conformer_exists],
            "pocket_res_feat": pocket_res_feat[conformer_exists],
        }

        chain_feature_update["is_protein"] = np.array([chain_class == "protein"] * len(ccds)).astype(np.float32)
        chain_feature_update["is_ligand"] = np.array([chain_class != "protein"] * len(ccds)).astype(np.float32)
        # Assert Short Poly Chain like peptide
        chain_feature_update["is_short_poly"] = np.array(
            [chain_class != "protein" and len(ccds) >= 2 and rc.is_standard(ccd) for ccd in ccds]
        ).astype(np.float32)

        if "msa_all_seq" in chain_feature:
            chain_feature_update["msa_all_seq"] = chain_feature["msa_all_seq"][:, conformer_exists]
            chain_feature_update["deletion_matrix_all_seq"] = \
                chain_feature["deletion_matrix_all_seq"][:, conformer_exists]
            chain_feature_update["msa_species_identifiers_all_seq"] = chain_feature["msa_species_identifiers_all_seq"]
        del chain_feature
        return chain_feature_update

    def _update_smi(self, smi, all_chain_labels, CONF_META_DATA):
        ccd = "XXX"
        chain_id = "99"
        label_feature, conf_feature, ref_mol = get_features_from_smi(smi)
        all_chain_labels[chain_id] = {
            "all_atom_positions": label_feature["x_gt"][None],
            "all_atom_mask": label_feature["x_exists"][None],
            "ccds": [ccd]
        }
        ref_atom_name_chars = []
        for id, ele in enumerate(conf_feature["ref_element"]):
            atom_name = f"{PeriodicTable[ele] + str(id):<4}"
            ref_atom_name_chars.append(atom_name)
        CONF_META_DATA[ccd] = {
            "ref_feat": np.concatenate([
                conf_feature["ref_pos"],
                conf_feature["ref_charge"][..., None],
                rc.eye_128[conf_feature["ref_element"]].astype(np.float32),
                conf_feature["ref_is_aromatic"].astype(np.float32)[..., None],
                rc.eye_9[conf_feature["ref_degree"]].astype(np.float32),
                rc.eye_7[conf_feature["ref_hybridization"]].astype(np.float32),
                rc.eye_9[conf_feature["ref_implicit_valence"]].astype(np.float32),
                rc.eye_3[conf_feature["ref_chirality"]].astype(np.float32),
                conf_feature["ref_in_ring_of_3"].astype(np.float32)[..., None],
                conf_feature["ref_in_ring_of_4"].astype(np.float32)[..., None],
                conf_feature["ref_in_ring_of_5"].astype(np.float32)[..., None],
                conf_feature["ref_in_ring_of_6"].astype(np.float32)[..., None],
                conf_feature["ref_in_ring_of_7"].astype(np.float32)[..., None],
                conf_feature["ref_in_ring_of_8"].astype(np.float32)[..., None],
            ], axis=-1),
            "rel_tok_feat": np.concatenate([
                rc.eye_32[conf_feature["d_token"]].astype(np.float32),
                rc.eye_5[conf_feature["bond_type"]].astype(np.float32),
                conf_feature["token_bonds"].astype(np.float32)[..., None],
                conf_feature["bond_as_double"].astype(np.float32)[..., None],
                conf_feature["bond_in_ring"].astype(np.float32)[..., None],
                conf_feature["bond_is_conjugated"].astype(np.float32)[..., None],
                conf_feature["bond_is_aromatic"].astype(np.float32)[..., None],
            ], axis=-1),
            "ref_atom_name_chars": ref_atom_name_chars,
            "ref_element": conf_feature["ref_element"],
            "token_bonds": conf_feature["token_bonds"],
        }

        return all_chain_labels, CONF_META_DATA, ref_mol

    def _add_assembly_feature(self, all_chain_features, SEQ3, ASYM_ID):
        entities = {}
        for chain_id, seq3 in SEQ3.items():
            if seq3 not in entities:
                entities[seq3] = [chain_id]
            else:
                entities[seq3].append(chain_id)

        asym_id = 0
        for entity_id, seq3 in enumerate(list(entities.keys())):
            chain_ids = copy.deepcopy(entities[seq3])
            if not self.inference_mode and self.train_shuffle_sym_id:
                # sym_id augmentation
                random.shuffle(chain_ids)
            for sym_id, chain_id in enumerate(chain_ids):
                num_conformers = len(all_chain_features[chain_id]["ccds"])
                all_chain_features[chain_id]["asym_id"] = \
                    np.full([num_conformers], fill_value=asym_id, dtype=np.int32)
                all_chain_features[chain_id]["sym_id"] = \
                    np.full([num_conformers], fill_value=sym_id, dtype=np.int32)
                all_chain_features[chain_id]["entity_id"] = \
                    np.full([num_conformers], fill_value=entity_id, dtype=np.int32)

                all_chain_features[chain_id]["sequence_3"] = seq3
                ASYM_ID[asym_id] = chain_id

                asym_id += 1
        return all_chain_features, ASYM_ID

    def _crop_all_chain_features(self, all_chain_features, infer_meta_data, crop_centre=None):
        CONF_META_DATA = infer_meta_data["CONF_META_DATA"]
        ordered_chain_ids = list(all_chain_features.keys())

        x_gt = np.concatenate([all_chain_features[chain_id]["x_gt"] for chain_id in ordered_chain_ids], axis=0)

        token_id_to_centre_atom_id = []
        token_id_to_conformer_id = []
        token_id_to_ccd_chunk_sizes = []
        token_id_to_ccd = []
        asym_id_ca = []
        token_id = 0
        atom_id = 0
        conf_id = 0
        x_gt_ligand = []
        for chain_id in ordered_chain_ids:
            if chain_id.isdigit() and len(all_chain_features[chain_id]["ccds"]) == 1:
                x_gt_ligand.append(all_chain_features[chain_id]["x_gt"])
            atom_offset = 0
            for ccd, chunk_size_this_ccd, asym_id in zip(
                    all_chain_features[chain_id]["ccds"],
                    all_chain_features[chain_id]["conformer_id_to_chunk_sizes"],
                    all_chain_features[chain_id]["asym_id"],
            ):
                inner_atom_idx = all_chain_features[chain_id]["atom_id_to_conformer_atom_id"][
                                 atom_offset:atom_offset + chunk_size_this_ccd]
                atom_names = [CONF_META_DATA[ccd]["ref_atom_name_chars"][i] for i in inner_atom_idx]
                if rc.is_standard(ccd):

                    for atom_id_this_ccd, atom_name in enumerate(atom_names):
                        if atom_name == rc.standard_ccd_to_token_centre_atom_name[ccd]:
                            token_id_to_centre_atom_id.append(atom_id)
                            token_id_to_conformer_id.append(conf_id)
                            token_id_to_ccd_chunk_sizes.append(chunk_size_this_ccd)
                            token_id_to_ccd.append(ccd)
                            asym_id_ca.append(asym_id)
                        atom_id += 1
                    token_id += 1

                else:
                    for atom_id_this_ccd, atom_name in enumerate(atom_names):
                        token_id_to_centre_atom_id.append(atom_id)
                        token_id_to_conformer_id.append(conf_id)
                        token_id_to_ccd_chunk_sizes.append(chunk_size_this_ccd)
                        token_id_to_ccd.append(ccd)
                        asym_id_ca.append(asym_id)
                        atom_id += 1
                        token_id += 1
                atom_offset += chunk_size_this_ccd
                conf_id += 1

        x_gt_ca = x_gt[token_id_to_centre_atom_id]
        asym_id_ca = np.array(asym_id_ca)

        crop_scheme_seed = random.random()

        # Crop Ligand Centre
        if self.inference_mode and len(x_gt_ligand) == 1:
            x_gt_ligand = np.concatenate(x_gt_ligand, axis=0)
            x_gt_sel = np.mean(x_gt_ligand, axis=0)[None]

        # Spatial Crop Ligand
        elif crop_scheme_seed < (self.train_spatial_crop_ligand_ratio if not self.inference_mode else 1.0) and len(
                x_gt_ligand) > 0:
            x_gt_ligand = np.concatenate(x_gt_ligand, axis=0)
            x_gt_sel = random.choice(x_gt_ligand)[None]
        # Spatial Crop Interface
        elif crop_scheme_seed < self.train_spatial_crop_ligand_ratio + self.train_spatial_crop_interface_ratio and len(
                set(asym_id_ca.tolist())) > 1:
            chain_same = asym_id_ca[None] * asym_id_ca[:, None]
            dist = np.linalg.norm(x_gt_ca[:, None] - x_gt_ca[None], axis=-1)

            dist = dist + chain_same * 100
            # interface_threshold
            mask = np.any(dist < self.train_spatial_crop_interface_threshold, axis=-1)
            if sum(mask) > 0:
                x_gt_sel = random.choice(x_gt_ca[mask])[None]
            else:
                x_gt_sel = random.choice(x_gt_ca)[None]
        # Spatial Crop
        else:
            x_gt_sel = random.choice(x_gt_ca)[None]
        dist = np.linalg.norm(x_gt_ca - x_gt_sel, axis=-1)
        token_idxs = np.argsort(dist)

        select_ccds_idx = []
        to_sum_atom = 0
        to_sum_token = 0
        for token_idx in token_idxs:
            ccd_idx = token_id_to_conformer_id[token_idx]
            ccd_chunk_size = token_id_to_ccd_chunk_sizes[token_idx]
            ccd_this_token = token_id_to_ccd[token_idx]
            if ccd_idx in select_ccds_idx:
                continue
            if to_sum_atom + ccd_chunk_size > self.atom_crop_size:
                break
            to_add_token = 1 if rc.is_standard(ccd_this_token) else ccd_chunk_size
            if to_sum_token + to_add_token > self.crop_size:
                break
            select_ccds_idx.append(ccd_idx)
            to_sum_atom += ccd_chunk_size
            to_sum_token += to_add_token

        ccd_all_id = 0
        crop_chains = []
        for chain_id in ordered_chain_ids:
            conformer_used_mask = []
            atom_used_mask = []
            ccds = []
            for ccd, chunk_size_this_ccd in zip(
                    all_chain_features[chain_id]["ccds"],
                    all_chain_features[chain_id]["conformer_id_to_chunk_sizes"],
            ):
                if ccd_all_id in select_ccds_idx:
                    ccds.append(ccd)
                    if chain_id not in crop_chains:
                        crop_chains.append(chain_id)
                conformer_used_mask.append(ccd_all_id in select_ccds_idx)
                atom_used_mask += [ccd_all_id in select_ccds_idx] * chunk_size_this_ccd
                ccd_all_id += 1
            conf_mask = np.array(conformer_used_mask).astype(np.bool_)
            atom_mask = np.array(atom_used_mask).astype(np.bool_)
            # Update All Chain Features
            all_chain_features[chain_id]["x_gt"] = all_chain_features[chain_id]["x_gt"][atom_mask]
            all_chain_features[chain_id]["atom_id_to_conformer_atom_id"] = \
                all_chain_features[chain_id]["atom_id_to_conformer_atom_id"][atom_mask]
            all_chain_features[chain_id]["restype"] = all_chain_features[chain_id]["restype"][conf_mask]
            all_chain_features[chain_id]["residue_index"] = all_chain_features[chain_id]["residue_index"][conf_mask]
            all_chain_features[chain_id]["conformer_id_to_chunk_sizes"] = \
                all_chain_features[chain_id]["conformer_id_to_chunk_sizes"][conf_mask]
            # BUG Fix
            all_chain_features[chain_id]["key_res_feat"] = all_chain_features[chain_id]["key_res_feat"][conf_mask]
            all_chain_features[chain_id]["pocket_res_feat"] = all_chain_features[chain_id]["pocket_res_feat"][conf_mask]
            all_chain_features[chain_id]["is_key_res"] = all_chain_features[chain_id]["is_key_res"][conf_mask]
            all_chain_features[chain_id]["is_protein"] = all_chain_features[chain_id]["is_protein"][conf_mask]
            all_chain_features[chain_id]["is_short_poly"] = all_chain_features[chain_id]["is_short_poly"][conf_mask]
            all_chain_features[chain_id]["is_ligand"] = all_chain_features[chain_id]["is_ligand"][conf_mask]
            all_chain_features[chain_id]["asym_id"] = all_chain_features[chain_id]["asym_id"][conf_mask]
            all_chain_features[chain_id]["sym_id"] = all_chain_features[chain_id]["sym_id"][conf_mask]
            all_chain_features[chain_id]["entity_id"] = all_chain_features[chain_id]["entity_id"][conf_mask]

            all_chain_features[chain_id]["ccds"] = ccds
            if "msa" in all_chain_features[chain_id]:
                all_chain_features[chain_id]["msa"] = all_chain_features[chain_id]["msa"][:, conf_mask]
                all_chain_features[chain_id]["deletion_matrix"] = \
                    all_chain_features[chain_id]["deletion_matrix"][:, conf_mask]
            if "msa_all_seq" in all_chain_features[chain_id]:
                all_chain_features[chain_id]["msa_all_seq"] = all_chain_features[chain_id]["msa_all_seq"][:, conf_mask]
                all_chain_features[chain_id]["deletion_matrix_all_seq"] = \
                    all_chain_features[chain_id]["deletion_matrix_all_seq"][:, conf_mask]
        # Remove Unused Chains
        for chain_id in list(all_chain_features.keys()):
            if chain_id not in crop_chains:
                all_chain_features.pop(chain_id, None)
        return all_chain_features, infer_meta_data

    def _make_ccd_features(self, raw_feats, infer_meta_data):
        CONF_META_DATA = infer_meta_data["CONF_META_DATA"]
        ccds = raw_feats["ccds"]
        atom_id_to_conformer_atom_id = raw_feats["atom_id_to_conformer_atom_id"]
        conformer_id_to_chunk_sizes = raw_feats["conformer_id_to_chunk_sizes"]

        # Atomwise
        atom_id_to_conformer_id = []
        atom_id_to_token_id = []
        ref_feat = []

        # Tokenwise
        s_mask = []
        token_id_to_conformer_id = []
        token_id_to_chunk_sizes = []
        token_id_to_centre_atom_id = []
        token_id_to_pseudo_beta_atom_id = []

        token_id = 0
        atom_id = 0
        for conf_id, (ccd, ccd_atoms) in enumerate(zip(ccds, conformer_id_to_chunk_sizes)):
            conf_meta_data = CONF_META_DATA[ccd]
            # UNK Conformer
            if rc.is_unk(ccd):
                s_mask.append(0)
                token_id_to_chunk_sizes.append(0)
                token_id_to_conformer_id.append(conf_id)
                token_id_to_centre_atom_id.append(-1)
                token_id_to_pseudo_beta_atom_id.append(-1)
                token_id += 1
            # Standard Residue
            elif rc.is_standard(ccd):
                inner_atom_idx = atom_id_to_conformer_atom_id[atom_id:atom_id + ccd_atoms.item()]
                atom_names = [conf_meta_data["ref_atom_name_chars"][i] for i in inner_atom_idx]
                ref_feat.append(conf_meta_data["ref_feat"][inner_atom_idx])
                token_id_to_conformer_id.append(conf_id)
                token_id_to_chunk_sizes.append(ccd_atoms.item())
                s_mask.append(1)
                for atom_id_this_ccd, atom_name in enumerate(atom_names):
                    # Update Atomwise Features
                    atom_id_to_conformer_id.append(conf_id)
                    atom_id_to_token_id.append(token_id)
                    # Update special atom ids
                    if atom_name == rc.standard_ccd_to_token_centre_atom_name[ccd]:
                        token_id_to_centre_atom_id.append(atom_id)
                    if atom_name == rc.standard_ccd_to_token_pseudo_beta_atom_name[ccd]:
                        token_id_to_pseudo_beta_atom_id.append(atom_id)
                    atom_id += 1
                token_id += 1
            # Nonestandard Residue & Ligand
            else:
                inner_atom_idx = atom_id_to_conformer_atom_id[atom_id:atom_id + ccd_atoms.item()]
                atom_names = [conf_meta_data["ref_atom_name_chars"][i] for i in inner_atom_idx]
                ref_feat.append(conf_meta_data["ref_feat"][inner_atom_idx])
                # ref_pos_new.append(conf_meta_data["ref_pos_new"][:, inner_atom_idx])
                for atom_id_this_ccd, atom_name in enumerate(atom_names):
                    # Update Atomwise Features
                    atom_id_to_conformer_id.append(conf_id)
                    atom_id_to_token_id.append(token_id)
                    # Update Tokenwise Features
                    token_id_to_chunk_sizes.append(1)
                    token_id_to_conformer_id.append(conf_id)
                    s_mask.append(1)
                    token_id_to_centre_atom_id.append(atom_id)
                    token_id_to_pseudo_beta_atom_id.append(atom_id)
                    atom_id += 1
                    token_id += 1

        if len(ref_feat) > 1:
            ref_feat = np.concatenate(ref_feat, axis=0).astype(np.float32)
        else:
            ref_feat = ref_feat[0].astype(np.float32)

        features = {
            # Atomwise
            "atom_id_to_conformer_id": np.array(atom_id_to_conformer_id, dtype=np.int64),
            "atom_id_to_token_id": np.array(atom_id_to_token_id, dtype=np.int64),
            "ref_feat": ref_feat,
            # Tokewise
            "token_id_to_conformer_id": np.array(token_id_to_conformer_id, dtype=np.int64),
            "s_mask": np.array(s_mask, dtype=np.int64),
            "token_id_to_centre_atom_id": np.array(token_id_to_centre_atom_id, dtype=np.int64),
            "token_id_to_pseudo_beta_atom_id": np.array(token_id_to_pseudo_beta_atom_id, dtype=np.int64),
            "token_id_to_chunk_sizes": np.array(token_id_to_chunk_sizes, dtype=np.int64),
        }
        features["ref_pos"] = features["ref_feat"][..., :3]
        return features

    def pair_and_merge(self, all_chain_features, infer_meta_data):
        CHAIN_CLASS = infer_meta_data["CHAIN_CLASS"]  # Dict
        CONF_META_DATA = infer_meta_data["CONF_META_DATA"]
        ASYM_ID = infer_meta_data["ASYM_ID"]
        homo_feats = {}

        all_chain_ids = list(all_chain_features.keys())
        if len(all_chain_ids) == 1 and CHAIN_CLASS[all_chain_ids[0]] == "ligand":
            ordered_chain_ids = all_chain_ids
            raw_feats = all_chain_features[all_chain_ids[0]]
            raw_feats["msa"] = np.repeat(raw_feats["msa"][:1], 256, axis=0)
            raw_feats["deletion_matrix"] = np.repeat(raw_feats["msa"][:1], 256, axis=0)
            keys = list(raw_feats.keys())

            for feature_name in keys:
                if feature_name not in ["x_gt", "atom_id_to_conformer_atom_id", "residue_index",
                                        "conformer_id_to_chunk_sizes", "restype", "is_protein", "is_short_poly",
                                        "is_ligand",
                                        "asym_id", "sym_id", "entity_id", "msa", "deletion_matrix", "ccds",
                                        "pocket_res_feat", "key_res_feat", "is_key_res"]:
                    raw_feats.pop(feature_name)

            # Update Profile and Deletion Mean
            msa_one_hot = F.one_hot(torch.from_numpy(raw_feats["msa"]).long(), 32).type(torch.float32)
            raw_feats["profile"] = torch.mean(msa_one_hot, dim=-3).numpy()
            del msa_one_hot
            raw_feats["deletion_mean"] = (torch.atan(
                torch.sum(torch.from_numpy(raw_feats["deletion_matrix"]), dim=0) / 3.0
            ) * (2.0 / torch.pi)).numpy()
        else:

            for chain_id in list(all_chain_features.keys()):
                homo_feats[chain_id] = {
                    "asym_id": copy.deepcopy(all_chain_features[chain_id]["asym_id"]),
                    "sym_id": copy.deepcopy(all_chain_features[chain_id]["sym_id"]),
                    "entity_id": copy.deepcopy(all_chain_features[chain_id]["entity_id"]),
                }
            for chain_id in list(all_chain_features.keys()):
                homo_feats[chain_id]["chain_class"] = all_chain_features[chain_id].pop("chain_class")
                homo_feats[chain_id]["sequence_3"] = all_chain_features[chain_id].pop("sequence_3")
                homo_feats[chain_id]["msa"] = all_chain_features[chain_id].pop("msa")
                homo_feats[chain_id]["deletion_matrix"] = all_chain_features[chain_id].pop("deletion_matrix")
                if "msa_all_seq" in all_chain_features[chain_id]:
                    homo_feats[chain_id]["msa_all_seq"] = all_chain_features[chain_id].pop("msa_all_seq")
                    homo_feats[chain_id]["deletion_matrix_all_seq"] = all_chain_features[chain_id].pop(
                        "deletion_matrix_all_seq")
                    homo_feats[chain_id]["msa_species_identifiers_all_seq"] = all_chain_features[chain_id].pop(
                        "msa_species_identifiers_all_seq")

            # Initial raw feats with merged homo feats
            raw_feats = pair_and_merge(homo_feats, is_homomer_or_monomer=False)

            # Update Profile and Deletion Mean
            msa_one_hot = F.one_hot(torch.from_numpy(raw_feats["msa"]).long(), 32).type(torch.float32)
            raw_feats["profile"] = torch.mean(msa_one_hot, dim=-3).numpy()
            del msa_one_hot
            raw_feats["deletion_mean"] = (torch.atan(
                torch.sum(torch.from_numpy(raw_feats["deletion_matrix"]), dim=0) / 3.0
            ) * (2.0 / torch.pi)).numpy()

            # Merge no homo feats according to asym_id
            ordered_asym_ids = []
            for i in raw_feats["asym_id"]:
                if i not in ordered_asym_ids:
                    ordered_asym_ids.append(i)
            ordered_chain_ids = [ASYM_ID[i] for i in ordered_asym_ids]
            for feature_name in ["chain_class", "sequence_3", "assembly_num_chains", "entity_mask", "seq_length",
                                 "num_alignments"]:
                raw_feats.pop(feature_name, None)

            for feature_name in ["x_gt", "atom_id_to_conformer_atom_id", "residue_index", "conformer_id_to_chunk_sizes",
                                 "restype", "is_protein", "is_short_poly", "is_ligand", "pocket_res_feat",
                                 "key_res_feat", "is_key_res"]:
                raw_feats[feature_name] = np.concatenate([
                    all_chain_features[chain_id].pop(feature_name) for chain_id in ordered_chain_ids
                ], axis=0)

        # Conformerwise Chain Class
        CHAIN_CLASS_NEW = []
        for chain_id in ordered_chain_ids:
            CHAIN_CLASS_NEW += [CHAIN_CLASS[chain_id]] * len(all_chain_features[chain_id]["ccds"])
        infer_meta_data["CHAIN_CLASS"] = CHAIN_CLASS_NEW

        raw_feats["ccds"] = reduce(add, [all_chain_features[chain_id].pop("ccds") for chain_id in ordered_chain_ids])

        # Create Atomwise and Tokenwise Features
        raw_feats.update(self._make_ccd_features(raw_feats, infer_meta_data))
        if self.use_x_gt_ligand_as_ref_pos:
            is_ligand_atom = raw_feats["is_ligand"][raw_feats["atom_id_to_conformer_id"]].astype(np.bool_)
            raw_feats["ref_pos"][is_ligand_atom] = raw_feats["x_gt"][is_ligand_atom] - np.mean(
                raw_feats["x_gt"][is_ligand_atom], axis=0, keepdims=True)

        asym_id_conformerwise = copy.deepcopy(raw_feats["asym_id"])
        residue_index_conformerwise = copy.deepcopy(raw_feats["residue_index"])

        # Conformerwise to Tokenwise
        token_id_to_conformer_id = raw_feats["token_id_to_conformer_id"]
        for key in ["is_protein", "is_short_poly", "is_ligand", "residue_index", "restype", "asym_id", "entity_id",
                    "sym_id", "deletion_mean", "profile", "pocket_res_feat", "key_res_feat", "is_key_res"]:
            raw_feats[key] = raw_feats[key][token_id_to_conformer_id]
        for key in ["msa", "deletion_matrix"]:
            if key in raw_feats:
                raw_feats[key] = raw_feats[key][:, token_id_to_conformer_id]
        ###################################################
        #       Centre Random Augmentation of ref pos     #
        ###################################################
        # atom_id_to_token_id
        # atom_id_to_conformer_id
        raw_feats["ref_pos"] = centre_random_augmentation_np_apply(
            raw_feats["ref_pos"], raw_feats["atom_id_to_conformer_id"]).astype(np.float32)
        raw_feats["ref_feat"][:, :3] = raw_feats["ref_pos"]

        ###################################################
        #            Create token pair features           #
        ###################################################
        no_token = len(raw_feats["token_id_to_conformer_id"])
        token_bonds = np.zeros([no_token, no_token], dtype=np.float32)
        rel_tok_feat = np.zeros([no_token, no_token, 42], dtype=np.float32)
        offset = 0
        atom_offset = 0
        for ccd, len_atoms in zip(
                raw_feats["ccds"],
                raw_feats["conformer_id_to_chunk_sizes"]
        ):
            if rc.is_standard(ccd) or rc.is_unk(ccd):
                offset += 1
            else:
                len_atoms = len_atoms.item()
                inner_atom_idx = raw_feats["atom_id_to_conformer_atom_id"][atom_offset:atom_offset + len_atoms]
                token_bonds[offset:offset + len_atoms, offset:offset + len_atoms] = \
                    CONF_META_DATA[ccd]["token_bonds"][inner_atom_idx][:, inner_atom_idx]
                rel_tok_feat[offset:offset + len_atoms, offset:offset + len_atoms] = \
                    CONF_META_DATA[ccd]["rel_tok_feat"][inner_atom_idx][:, inner_atom_idx]
                offset += len_atoms
            atom_offset += len_atoms
        raw_feats["token_bonds"] = token_bonds.astype(np.float32)
        raw_feats["token_bonds_feature"] = token_bonds.astype(np.float32)
        raw_feats["rel_tok_feat"] = rel_tok_feat.astype(np.float32)
        ###################################################
        #              Charility Augmentation             #
        ###################################################
        if not self.inference_mode:
            # TODO Charility probs
            charility_seed = random.random()
            if charility_seed < self.train_charility_augmentation_ratio:
                ref_chirality = raw_feats["ref_feat"][:, 158:161]
                ref_chirality_replace = np.zeros_like(ref_chirality)
                ref_chirality_replace[:, 2] = 1

                is_ligand_atom = raw_feats["is_ligand"][raw_feats["atom_id_to_token_id"]]
                remove_charility = (np.random.randint(0, 2, [len(is_ligand_atom)]) * is_ligand_atom).astype(
                    np.bool_)
                ref_chirality = np.where(remove_charility[:, None], ref_chirality_replace, ref_chirality)
                raw_feats["ref_feat"][:, 158:161] = ref_chirality

        # MASKS
        raw_feats["x_exists"] = np.ones_like(raw_feats["x_gt"][..., 0]).astype(np.float32)
        raw_feats["a_mask"] = raw_feats["x_exists"]
        raw_feats["s_mask"] = np.ones_like(raw_feats["asym_id"]).astype(np.float32)
        raw_feats["ref_space_uid"] = raw_feats["atom_id_to_conformer_id"]

        # Write Infer Meta Data
        infer_meta_data["ccds"] = raw_feats.pop("ccds")
        infer_meta_data["atom_id_to_conformer_atom_id"] = raw_feats.pop("atom_id_to_conformer_atom_id")
        infer_meta_data["residue_index"] = residue_index_conformerwise
        infer_meta_data["asym_id"] = asym_id_conformerwise
        infer_meta_data["conformer_id_to_chunk_sizes"] = raw_feats.pop("conformer_id_to_chunk_sizes")

        return raw_feats, infer_meta_data

    def make_feats(self, tensors):
        # Target Feat
        tensors["target_feat"] = torch.cat([
            F.one_hot(tensors["restype"].long(), 32).float(),
            tensors["profile"].float(),
            tensors["deletion_mean"][..., None].float()
        ], dim=-1)

        if self.num_recycles is None:
            # MSA Feat
            inds = [0] + torch.randperm(len(tensors["msa"]))[:self.max_msa_clusters - 1].tolist()

            tensors["msa"] = tensors["msa"][inds]
            tensors["deletion_matrix"] = tensors["deletion_matrix"][inds]

            has_deletion = torch.clamp(tensors["deletion_matrix"].float(), min=0., max=1.)
            pi = torch.acos(torch.zeros(1, device=tensors["deletion_matrix"].device)) * 2
            deletion_value = (torch.atan(tensors["deletion_matrix"] / 3.) * (2. / pi))
            tensors["msa_feat"] = torch.cat([
                F.one_hot(tensors["msa"].long(), 32).float(),
                has_deletion[..., None].float(),
                deletion_value[..., None].float(),
            ], dim=-1)
        else:
            batch_msa_feat = []
            for i in range(self.num_recycles):
                inds = [0] + torch.randperm(len(tensors["msa"]))[:self.max_msa_clusters - 1].tolist()

                tensors["msa"] = tensors["msa"][inds]
                tensors["deletion_matrix"] = tensors["deletion_matrix"][inds]

                has_deletion = torch.clamp(tensors["deletion_matrix"].float(), min=0., max=1.)
                pi = torch.acos(torch.zeros(1, device=tensors["deletion_matrix"].device)) * 2
                deletion_value = (torch.atan(tensors["deletion_matrix"] / 3.) * (2. / pi))
                msa_feat = torch.cat([
                    F.one_hot(tensors["msa"].long(), 32).float(),
                    has_deletion[..., None].float(),
                    deletion_value[..., None].float(),
                ], dim=-1)
                batch_msa_feat.append(msa_feat)
            tensors["msa_feat"] = batch_msa_feat[0]
            tensors["batch_msa_feat"] = torch.stack(batch_msa_feat, dim=0)

        tensors.pop("msa", None)
        tensors.pop("deletion_mean", None)
        tensors.pop("profile", None)
        tensors.pop("deletion_matrix", None)

        return tensors

    def _make_token_bonds(self, tensors):
        # Get Polymer-Ligand & Ligand-Ligand Within Conformer Token Bond
        # Atomwise asym_id
        asym_id = tensors["asym_id"][tensors["atom_id_to_token_id"]]
        is_ligand = tensors["is_ligand"][tensors["atom_id_to_token_id"]]

        x_gt = tensors["x_gt"]
        a_mask = tensors["a_mask"]

        # Get
        atom_id_to_token_id = tensors["atom_id_to_token_id"]

        num_token = len(tensors["asym_id"])
        between_conformer_token_bonds = torch.zeros([num_token, num_token])

        # create chainwise feature
        asym_id_chain = []
        asym_id_atom_offset = []
        asym_id_is_ligand = []
        for atom_offset, (a_id, i_id) in enumerate(zip(asym_id.tolist(), is_ligand.tolist())):
            if len(asym_id_chain) == 0 or asym_id_chain[-1] != a_id:
                asym_id_chain.append(a_id)
                asym_id_atom_offset.append(atom_offset)
                asym_id_is_ligand.append(i_id)

        len_asym_id_chain = len(asym_id_chain)
        if len_asym_id_chain >= 2:
            for i in range(0, len_asym_id_chain - 1):
                asym_id_i = asym_id_chain[i]
                mask_i = asym_id == asym_id_i
                x_gt_i = x_gt[mask_i]
                a_mask_i = a_mask[mask_i]
                for j in range(i + 1, len_asym_id_chain):
                    if not bool(asym_id_is_ligand[i]) and not bool(asym_id_is_ligand[j]):
                        continue
                    asym_id_j = asym_id_chain[j]
                    mask_j = asym_id == asym_id_j
                    x_gt_j = x_gt[mask_j]
                    a_mask_j = a_mask[mask_j]
                    dis_ij = torch.norm(x_gt_i[:, None, :] - x_gt_j[None, :, :], dim=-1)
                    dis_ij = dis_ij + (1 - a_mask_i[:, None] * a_mask_j[None]) * 1000
                    if torch.min(dis_ij) < self.token_bond_threshold:
                        ij = torch.argmin(dis_ij).item()
                        l_j = len(x_gt_j)
                        atom_i = int(ij // l_j)  # raw
                        atom_j = int(ij % l_j)  # col
                        global_atom_i = atom_i + asym_id_atom_offset[i]
                        global_atom_j = atom_j + asym_id_atom_offset[j]
                        token_i = atom_id_to_token_id[global_atom_i]
                        token_j = atom_id_to_token_id[global_atom_j]

                        between_conformer_token_bonds[token_i, token_j] = 1
                        between_conformer_token_bonds[token_j, token_i] = 1
        token_bond_seed = random.random()
        tensors["token_bonds"] = tensors["token_bonds"] + between_conformer_token_bonds
        # Docking Indicate Token Bond
        # if token_bond_seed >= 1:
        #     tensors["token_bonds_feature"] = tensors["token_bonds"]
        return tensors

    def _pad_to_size(self, tensors):

        to_pad_atom = self.atom_crop_size - len(tensors["x_gt"])
        to_pad_token = self.crop_size - len(tensors["residue_index"])
        if to_pad_token > 0:
            for k in ["restype", "residue_index", "is_protein", "is_short_poly", "is_ligand", "is_key_res",
                      "asym_id", "entity_id", "sym_id", "token_id_to_conformer_id", "s_mask",
                      "token_id_to_centre_atom_id", "token_id_to_pseudo_beta_atom_id", "token_id_to_chunk_sizes",
                      "pocket_res_feat"]:
                tensors[k] = torch.nn.functional.pad(tensors[k], [0, to_pad_token])
            for k in ["target_feat", "msa_feat", "key_res_feat", "batch_msa_feat"]:
                if k in tensors:
                    tensors[k] = torch.nn.functional.pad(tensors[k], [0, 0, 0, to_pad_token])
            for k in ["token_bonds", "token_bonds_feature"]:
                tensors[k] = torch.nn.functional.pad(tensors[k], [0, to_pad_token, 0, to_pad_token])
            for k in ["rel_tok_feat"]:
                tensors[k] = torch.nn.functional.pad(tensors[k], [0, 0, 0, to_pad_token, 0, to_pad_token])
        if to_pad_atom > 0:
            for k in ["a_mask", "x_exists", "atom_id_to_conformer_id", "atom_id_to_token_id", "ref_space_uid"]:
                tensors[k] = torch.nn.functional.pad(tensors[k], [0, to_pad_atom])
            for k in ["x_gt", "ref_feat", "ref_pos"]:  # , "ref_pos_new"
                tensors[k] = torch.nn.functional.pad(tensors[k], [0, 0, 0, to_pad_atom])
            # for k in ["z_mask"]:  # , "ref_pos_new"
            #     tensors[k] = torch.nn.functional.pad(tensors[k], [0, to_pad_atom, 0, to_pad_atom])
            # for k in ["conformer_mask_atom"]:
            #     tensors[k] = torch.nn.functional.pad(tensors[k], [0, to_pad_atom, 0, to_pad_atom])
            # for k in ["rel_token_feat_atom"]:
            #     tensors[k] = torch.nn.functional.pad(tensors[k], [0,0,0, to_pad_atom, 0, to_pad_atom])
            # rel_token_feat_atom
        return tensors

    def get_template_feat(self, tensors):
        x_gt = tensors["x_gt"][tensors["token_id_to_pseudo_beta_atom_id"]]
        z_mask = tensors["z_mask"]
        asym_id = tensors["asym_id"]
        is_protein = tensors["is_protein"]
        chain_same = (asym_id[None] == asym_id[:, None]).float()
        protein2d = is_protein[None] * is_protein[:, None]
        dgram = dgram_from_positions(x_gt, no_bins=39)
        dgram = dgram * protein2d[..., None] * z_mask[..., None]

        if not self.inference_mode:
            if random.random() > self.train_use_template_ratio:
                tensors["t_mask"] = torch.tensor(1, dtype=torch.float32)
                bert_mask = torch.rand([len(x_gt)]) > random.random() * (1 - self.train_template_mask_max_ratio)
                template_pseudo_beta_mask = (bert_mask[None] * bert_mask[:, None]) * z_mask * protein2d
            else:
                tensors["t_mask"] = torch.tensor(0, dtype=torch.float32)
                template_pseudo_beta_mask = z_mask * protein2d
        else:
            tensors["t_mask"] = torch.tensor(1, dtype=torch.float32)
            template_pseudo_beta_mask = z_mask * protein2d
        dgram = dgram * template_pseudo_beta_mask[..., None]
        templ_feat = torch.cat([dgram, template_pseudo_beta_mask[..., None]], dim=-1)
        tensors["templ_feat"] = templ_feat.float()
        return tensors

    def transform(self, raw_feats):
        # np to tensor
        tensors = dict()
        for key in raw_feats.keys():
            tensors[key] = torch.from_numpy(raw_feats[key])
        # Make Target & MSA Feat
        tensors = self.make_feats(tensors)

        # Make Token Bond Feat
        tensors = self._make_token_bonds(tensors)

        # Padding
        if not self.inference_mode:
            tensors = self._pad_to_size(tensors)

        # Mask
        tensors["z_mask"] = tensors["s_mask"][None] * tensors["s_mask"][:, None]
        tensors["ap_mask"] = tensors["a_mask"][None] * tensors["a_mask"][:, None]
        tensors["is_dna"] = torch.zeros_like(tensors["is_protein"])
        tensors["is_rna"] = torch.zeros_like(tensors["is_protein"])

        # Template
        tensors = self.get_template_feat(tensors)

        # Correct Type
        is_short_poly = tensors.pop("is_short_poly")
        tensors["is_protein"] = tensors["is_protein"] + is_short_poly
        tensors["is_ligand"] = tensors["is_ligand"] - is_short_poly
        return tensors

    # residue_index 0-100
    # CCDS
    # CCD<RES_ID>
    #
    def load(
            self,
            system_pkl_path,  # Receptor chains: all_atom_positions pocket_res_feat  Ligand_chains
            template_receptor_chain_ids=None,  # ["A"]
            template_ligand_chain_ids=None,  # ["1"]
            remove_receptor=False,
            remove_ligand=False,  # True, CCD_META_DATA ref_mol
            smi=None,  # "CCCCC"
    ):
        ##########################################################
        #               Initialization of Configs                #
        ##########################################################
        if self.inference_mode:
            pocket_type = self.infer_pocket_type
            pocket_cutoff = self.infer_pocket_cutoff
            pocket_dist_type = self.infer_pocket_dist_type
            use_pocket = self.infer_use_pocket
            use_key_res = self.infer_use_key_res
        else:
            pocket_type = random.choices(
                ["atom", "ca"],
                [self.train_pocket_type_atom_ratio, 1 - self.train_pocket_type_atom_ratio])

            pocket_dist_type = random.choices(
                ["ligand", "ligand_cetre"],
                [self.train_pocket_dist_type_ligand_ratio, 1 - self.train_pocket_dist_type_ligand_ratio])

            if pocket_dist_type == "ligand":
                pocket_cutoff = self.train_pocket_cutoff_ligand_min + random.random() * (
                        self.train_pocket_cutoff_ligand_max - self.train_pocket_cutoff_ligand_min)
            else:
                pocket_cutoff = self.train_pocket_cutoff_ligand_centre_min + random.random() * (
                        self.train_pocket_cutoff_ligand_centre_max - self.train_pocket_cutoff_ligand_centre_min)

            use_pocket = random.random() < self.train_use_pocket_ratio
            use_key_res = random.random() < self.train_use_key_res_ratio

        ##########################################################
        #               Initialization of features               #
        ##########################################################
        system_id = os.path.split(system_pkl_path)[1][:-7]
        all_chain_labels = {}
        all_chain_features = {}

        CONF_META_DATA = {}
        ref_mol = None
        CHAIN_CLASS = {}
        SEQ3 = {}
        ASYM_ID = {}

        ##########################################################
        #                  Load All Chain Labels                 #
        ##########################################################
        data = load_pkl(system_pkl_path)
        # print(data)
        if template_receptor_chain_ids is None:
            template_receptor_chain_ids = [chain_id for chain_id in data.keys() if not chain_id.isdigit()]

        if template_ligand_chain_ids is None:
            template_ligand_chain_ids = [chain_id for chain_id in data.keys() if chain_id.isdigit()]
        # TODO: Save Ligand Centre for cropped screening
        # Calculate Pocket Residue According to Template receptor and ligand
        if not remove_receptor and len(template_ligand_chain_ids) > 0:
            for receptor_chain_id in template_receptor_chain_ids:
                ccds_receptor = data[receptor_chain_id]["ccds"]
                x_gt_receptor = data[receptor_chain_id]["all_atom_positions"]
                x_exists_receptor = data[receptor_chain_id]["all_atom_mask"]
                x_gt_this_receptor = []
                atom_id_to_ccd_id = []
                for ccd_id, (ccd, x_gt_ccd, x_exists_ccd) in enumerate(
                        zip(ccds_receptor, x_gt_receptor, x_exists_receptor)):

                    if rc.is_standard(ccd):
                        x_exists_ccd_bool = x_exists_ccd.astype(np.bool_)
                        if x_exists_ccd_bool[1]:  # CA exsits
                            if pocket_type == "atom":
                                num_atoms = sum(x_exists_ccd_bool)
                                x_gt_this_receptor.append(x_gt_ccd[x_exists_ccd_bool])
                                atom_id_to_ccd_id += num_atoms * [ccd_id]
                            else:
                                x_gt_this_receptor.append(x_gt_ccd[1][None])
                                atom_id_to_ccd_id.append(ccd_id)
                x_gt_this_receptor = np.concatenate(x_gt_this_receptor, axis=0)
                atom_id_to_ccd_id = np.array(atom_id_to_ccd_id)
                used_ccd_ids = []
                for ligand_chain_id in template_ligand_chain_ids:
                    x_gt_ligand = data[ligand_chain_id]["all_atom_positions"]
                    x_exists_ligand = data[ligand_chain_id]["all_atom_mask"]
                    x_gt_ligand = np.concatenate(x_gt_ligand, axis=0)[
                        np.concatenate(x_exists_ligand, axis=0).astype(np.bool_)]
                    if pocket_dist_type == "ligand":
                        used_ccd_bool = np.any(
                            np.linalg.norm(x_gt_this_receptor[:, None] - x_gt_ligand[None], axis=-1) < pocket_cutoff,
                            axis=-1)
                    elif pocket_dist_type == "ligand_centre":
                        x_mean = np.min(x_gt_ligand, axis=0, keepdims=True)
                        used_ccd_bool = np.any(
                            np.linalg.norm(x_gt_this_receptor[:, None] - x_mean[None], axis=-1) < pocket_cutoff,
                            axis=-1)
                    else:
                        raise NotImplementedError()
                    used_ccd_ids.append(atom_id_to_ccd_id[used_ccd_bool])
                used_ccd_ids = list(sorted(list(set(np.concatenate(used_ccd_ids, axis=0).tolist()))))
                pocket_res_feat = np.zeros([len(ccds_receptor)], dtype=np.float32)
                pocket_res_feat[used_ccd_ids] = 1.
                all_chain_labels[receptor_chain_id] = data[receptor_chain_id]
                all_chain_labels[receptor_chain_id]["pocket_res_feat"] = pocket_res_feat

        if remove_ligand:
            if remove_receptor:
                assert smi is not None and self.inference_mode
            if smi is not None:
                all_chain_labels, CONF_META_DATA, ref_mol = self._update_smi(smi, all_chain_labels, CONF_META_DATA)
        else:
            assert smi is None
            for ligand_chain_id in template_ligand_chain_ids:
                all_chain_labels[ligand_chain_id] = data[ligand_chain_id]
            # For Benchmarking
            if len(template_ligand_chain_ids) == 1:
                ccds = all_chain_labels[template_ligand_chain_ids[0]]["ccds"]
                if len(ccds) == 1:
                    # ref_mol = self.ccd_id_ref_mol[ccds[0]]
                    ref_mol = self.ccd_id_meta_data[ccds[0]]["ref_mol"]
        ##########################################################
        #                 Init All Chain Features                #
        ##########################################################
        for chain_id, chain_feature in all_chain_labels.items():
            ccds = chain_feature["ccds"]
            CONF_META_DATA = self._update_CONF_META_DATA(CONF_META_DATA, ccds)
            SEQ3[chain_id] = "-".join(ccds)
            chain_class = "protein" if not chain_id.isdigit() else "ligand"
            chain_feature["chain_class"] = chain_class
            # print(chain_id)
            all_chain_features[chain_id] = self._update_chain_feature(
                chain_feature,
                CONF_META_DATA,
                use_pocket,
                use_key_res,
            )
            CHAIN_CLASS[chain_id] = chain_class
        ##########################################################
        #                  Add Assembly Feature                  #
        ##########################################################
        all_chain_features, ASYM_ID = self._add_assembly_feature(all_chain_features, SEQ3, ASYM_ID)

        infer_meta_data = {
            "CONF_META_DATA": CONF_META_DATA,
            "SEQ3": SEQ3,
            "ASYM_ID": ASYM_ID,
            "CHAIN_CLASS": CHAIN_CLASS,
            "ref_mol": ref_mol,
            "system_id": system_id,
        }
        ##########################################################
        #                       Cropping                         #
        ##########################################################
        # if not self.inference_mode:
        if self.crop_size is not None:
            all_chain_features, infer_meta_data = self._crop_all_chain_features(
                all_chain_features, infer_meta_data, crop_centre=None)  # TODO: Add Cropping Centre
        ##########################################################
        #                      Pair & Merge                      #
        ##########################################################
        raw_feats, infer_meta_data = self.pair_and_merge(all_chain_features, infer_meta_data)

        ##########################################################
        #                        Transform                       #
        ##########################################################
        tensors = self.transform(raw_feats)
        return tensors, infer_meta_data

    def write_pdb(self, x_pred, fname, infer_meta_data, receptor_only=False, ligand_only=False):
        ccds = infer_meta_data["ccds"]
        atom_id_to_conformer_atom_id = infer_meta_data["atom_id_to_conformer_atom_id"]
        ccd_chunk_sizes = infer_meta_data["conformer_id_to_chunk_sizes"].tolist()
        CHAIN_CLASS = infer_meta_data["CHAIN_CLASS"]
        conf_meta_data = infer_meta_data["CONF_META_DATA"]
        residue_index = infer_meta_data["residue_index"].tolist()
        asym_id = infer_meta_data["asym_id"].tolist()

        atom_lines = []
        atom_offset = 0
        for ccd_id, (ccd, chunk_size, res_id) in enumerate(zip(ccds, ccd_chunk_sizes, residue_index)):
            inner_atom_idx = atom_id_to_conformer_atom_id[atom_offset:atom_offset + chunk_size]
            atom_names = [conf_meta_data[ccd]["ref_atom_name_chars"][i] for i in inner_atom_idx]
            atom_elements = [PeriodicTable[conf_meta_data[ccd]["ref_element"][i]] for i in inner_atom_idx]
            chain_tag = PDB_CHAIN_IDS[int(asym_id[ccd_id])]
            record_type = "HETATM" if CHAIN_CLASS[ccd_id] == "ligand" else "ATOM"

            for ccd_atom_idx, atom_name in enumerate(atom_names):
                x = x_pred[atom_offset]
                name = atom_name if len(atom_name) == 4 else f" {atom_name}"
                res_name_3 = ccd
                alt_loc = ""
                insertion_code = ""
                occupancy = 1.00
                element = atom_elements[ccd_atom_idx]
                # b_factor = torch.argmax(plddt[atom_offset],dim=-1).item()*2 +1
                b_factor = 70.
                charge = 0
                pos = x.tolist()
                atom_line = (
                    f"{record_type:<6}{atom_offset + 1:>5} {name:<4}{alt_loc:>1}"
                    f"{res_name_3.split()[0][-3:]:>3} {chain_tag:>1}"
                    f"{res_id + 1:>4}{insertion_code:>1}   "
                    f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                    f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                    f"{element:>2}{charge:>2}"
                )
                if receptor_only and not ligand_only:
                    if record_type == "ATOM":
                        atom_lines.append(atom_line)
                elif not receptor_only and ligand_only:
                    if record_type == "HETATM":
                        atom_lines.append(atom_line)
                elif not receptor_only and not ligand_only:
                    atom_lines.append(atom_line)
                else:
                    raise NotImplementedError()
                atom_offset += 1
                if atom_offset == len(atom_id_to_conformer_atom_id):
                    break
        out = "\n".join(atom_lines)
        out = f"MODEL     1\n{out}\nTER\nENDMDL\nEND"
        dump_txt(out, fname)

    def write_pdb_block(self, x_pred, infer_meta_data, receptor_only=False, ligand_only=False):
        ccds = infer_meta_data["ccds"]
        atom_id_to_conformer_atom_id = infer_meta_data["atom_id_to_conformer_atom_id"]
        ccd_chunk_sizes = infer_meta_data["conformer_id_to_chunk_sizes"].tolist()
        CHAIN_CLASS = infer_meta_data["CHAIN_CLASS"]
        conf_meta_data = infer_meta_data["CONF_META_DATA"]
        residue_index = infer_meta_data["residue_index"].tolist()
        asym_id = infer_meta_data["asym_id"].tolist()

        atom_lines = []
        atom_offset = 0
        for ccd_id, (ccd, chunk_size, res_id) in enumerate(zip(ccds, ccd_chunk_sizes, residue_index)):
            inner_atom_idx = atom_id_to_conformer_atom_id[atom_offset:atom_offset + chunk_size]
            atom_names = [conf_meta_data[ccd]["ref_atom_name_chars"][i] for i in inner_atom_idx]
            atom_elements = [PeriodicTable[conf_meta_data[ccd]["ref_element"][i]] for i in inner_atom_idx]
            chain_tag = PDB_CHAIN_IDS[int(asym_id[ccd_id])]
            record_type = "HETATM" if CHAIN_CLASS[ccd_id] == "ligand" else "ATOM"

            for ccd_atom_idx, atom_name in enumerate(atom_names):
                x = x_pred[atom_offset]
                name = atom_name if len(atom_name) == 4 else f" {atom_name}"
                res_name_3 = ccd
                alt_loc = ""
                insertion_code = ""
                occupancy = 1.00
                element = atom_elements[ccd_atom_idx]
                # b_factor = torch.argmax(plddt[atom_offset],dim=-1).item()*2 +1
                b_factor = 70.
                charge = 0
                pos = x.tolist()
                atom_line = (
                    f"{record_type:<6}{atom_offset + 1:>5} {name:<4}{alt_loc:>1}"
                    f"{res_name_3.split()[0][-3:]:>3} {chain_tag:>1}"
                    f"{res_id + 1:>4}{insertion_code:>1}   "
                    f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                    f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                    f"{element:>2}{charge:>2}"
                )
                if receptor_only and not ligand_only:
                    if record_type == "ATOM":
                        atom_lines.append(atom_line)
                elif not receptor_only and ligand_only:
                    if record_type == "HETATM":
                        atom_lines.append(atom_line)
                elif not receptor_only and not ligand_only:
                    atom_lines.append(atom_line)
                else:
                    raise NotImplementedError()
                atom_offset += 1
                if atom_offset == len(atom_id_to_conformer_atom_id):
                    break
        out = "\n".join(atom_lines)
        out = f"MODEL     1\n{out}\nTER\nENDMDL\nEND"
        return out
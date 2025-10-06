############################
# 0.85 Receptor+Ligand
#  0.5 APO Template
#  0.5 HOLO Template
# 0.05 Protein (All APO or PRED)
# 0.1 Ligand
############################
import copy
import os
import random
from functools import reduce
from operator import add
import torch
import torch.nn.functional as F

# Key Res
# Dynamic Cutoff

import numpy as np

from stdock.data.constants.PDBData import protein_letters_3to1_extended
from stdock.data.constants import restype_constants as rc
from stdock.utils.io_utils import convert_md5_string, load_json, load_pkl, dump_txt
from stdock.data.tools.feature_processing_multimer import pair_and_merge
from stdock.utils.tensor_utils import centre_random_augmentation_np_apply, dgram_from_positions, \
    centre_random_augmentation_np_batch
from stdock.data.constants.periodic_table import PeriodicTable

PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


class FeatureLoader:
    def __init__(
            self,
            # config,
            token_crop_size=256,
            atom_crop_size=256 * 8,
            inference_mode=False,
    ):
        self.inference_mode = inference_mode
        self.msa_features_path = "/2022133002/data/stfold-data-v5/features/msa_features/"
        self.uniprot_msa_features_path = "/2022133002/data/stfold-data-v5/features/uniprot_msa_features/"

        self.token_crop_size = token_crop_size
        self.atom_crop_size = atom_crop_size
        self.token_bond_threshold = 2.4

        self.ccd_id_meta_data = load_pkl(
            "/2022133002/projects/stdock/stdock_v9.5/scripts/ccd_meta_data_confs_chars.pkl.gz")

        self.samples_path = "/2022133002/data/plinder/2024-06/v2/plinder_samples_raw_data_v2"
        to_remove = set(load_json("/2022133002/projects/stdock/stdock_v9.6/scripts/to_remove.json"))
        weights = load_json(
            "/2022133002/projects/stdock/stdock_v9.5/scripts/cluster_scripts/train_samples_new_weight_seq.json")
        self.splits = load_json("/2022133002/data/plinder/2024-06/v2/splits/splits.json")
        self.used_sample_ids = [sample_id for sample_id in weights
                                if sample_id in self.splits and self.splits[sample_id] != "test"
                                and sample_id not in to_remove]
        print("train samples", len(self.used_sample_ids))
        # self.weights = np.array(list(weights.values()))
        self.weights = np.array([weights[sample_id] for sample_id in self.used_sample_ids])

        self.probabilities = torch.from_numpy(self.weights / self.weights.sum())

        self.used_test_sample_ids = [sample_id for sample_id in weights
                                     if sample_id in self.splits and self.splits[sample_id] == "test"
                                     and sample_id not in to_remove]
        print("test samples", len(self.used_test_sample_ids))
        # self.weights = np.array(list(weights.values()))
        self.test_weights = np.array([weights[sample_id] for sample_id in self.used_test_sample_ids])

        self.test_probabilities = torch.from_numpy(self.test_weights / self.test_weights.sum())

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
            if not rc.is_standard(ccd):
                conformers = ccd_features["conformers"]
                if conformers is None:
                    conformers = np.repeat(ccd_features["ref_pos"][None], 32, axis=0)
                else:
                    conformers = np.stack([random.choice(ccd_features["conformers"]) for i in range(32)], axis=0)
                CONF_META_DATA[ccd]["batch_ref_pos"] = centre_random_augmentation_np_batch(conformers)

        return CONF_META_DATA

    def _update_CONF_META_DATA_ligand(self, CONF_META_DATA, sequence_3, ccd_features):
        ccds = sequence_3.split("-")
        # ccd_features = self.ccd_id_meta_data[ccd]
        for ccd in ccds:
            CONF_META_DATA[ccd] = {
                "ref_feat": np.concatenate([
                    ccd_features["ref_pos"],
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
            if not rc.is_standard(ccd):
                conformers = ccd_features["conformers"]
                if conformers is None:
                    conformers = np.repeat(ccd_features["ref_pos"][None], 32, axis=0)
                else:
                    conformers = np.stack([random.choice(ccd_features["conformers"]) for i in range(32)], axis=0)
                CONF_META_DATA[ccd]["batch_ref_pos"] = centre_random_augmentation_np_batch(conformers)

        return CONF_META_DATA

    def _update_chain_feature(self, chain_feature, CONF_META_DATA):

        ccds_ori = chain_feature["ccds"]
        chain_class = chain_feature["chain_class"]
        if chain_class == "protein":

            sequence = "".join([protein_letters_3to1_extended.get(ccd, "X") for ccd in ccds_ori])
            md5 = convert_md5_string(f"protein:{sequence}")

            chain_feature.update(
                load_pkl(os.path.join(self.msa_features_path, f"{md5}.pkl.gz"))
            )
            chain_feature.update(
                load_pkl(os.path.join(self.uniprot_msa_features_path, f"{md5}.pkl.gz"))
            )
        else:
            chain_feature["msa"] = np.array([[rc.standard_ccds.index(ccd)
                                              if ccd in rc.standard_ccds else 20 for ccd in ccds_ori]] * 2,
                                            dtype=np.int8)
            chain_feature["deletion_matrix"] = np.zeros_like(chain_feature["msa"])

        # Merge Key Res Feat & Augmentation
        if "salt bridges" in chain_feature:
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
            key_res_feat = np.zeros(
                [len(ccds_ori), 7], dtype=np.float32
            )
        is_key_res = np.any(key_res_feat.astype(np.bool_), axis=-1).astype(np.float32)
        # Augmentation
        if not self.inference_mode:
            key_res_feat = key_res_feat * (np.random.random([len(ccds_ori), 7]) < 0.5)
        # else:
        #     # TODO: No key res in inference mode
        #     key_res_feat = key_res_feat * 0
        # Atom
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
            #
            conformer_exist = np.any(x_exists_this_conf).item()
            if rc.is_standard(ccd):
                conformer_exist = np.sum(x_exists_this_conf).item() > len(x_exists_this_conf) - 2
                # conformer_exist = conformer_exist and x_exists_this_conf[1]
                # if ccd != "GLY":
                #     conformer_exist = conformer_exist and x_exists_this_conf[4]

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

    def _add_assembly_feature(self, all_chain_features, SEQ3):
        entities = {}
        for chain_id, seq3 in SEQ3.items():
            if seq3 not in entities:
                entities[seq3] = [chain_id]
            else:
                entities[seq3].append(chain_id)

        asym_id = 0
        ASYM_ID = {}
        for entity_id, seq3 in enumerate(list(entities.keys())):
            chain_ids = copy.deepcopy(entities[seq3])
            if not self.inference_mode:
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

    def load_all_chain_features(self, all_chain_labels):
        all_chain_features = {}
        CONF_META_DATA = {}
        SEQ3 = {}
        CHAIN_CLASS = {}

        for chain_id, chain_feature in all_chain_labels.items():
            ccds = chain_feature["ccds"]
            CONF_META_DATA = self._update_CONF_META_DATA(CONF_META_DATA, ccds)
            SEQ3[chain_id] = "-".join(ccds)
            chain_class = "protein" if not chain_id.isdigit() else "ligand"
            chain_feature["chain_class"] = chain_class

            all_chain_features[chain_id] = self._update_chain_feature(
                chain_feature,
                CONF_META_DATA
            )
            CHAIN_CLASS[chain_id] = chain_class

        all_chain_features, ASYM_ID = self._add_assembly_feature(all_chain_features, SEQ3)

        infer_meta_data = {
            "CONF_META_DATA": CONF_META_DATA,
            "SEQ3": SEQ3,
            "ASYM_ID": ASYM_ID,
            "CHAIN_CLASS": CHAIN_CLASS
        }

        return all_chain_features, infer_meta_data

    def _spatial_crop_v2(self, all_chain_features, infer_meta_data):
        CONF_META_DATA = infer_meta_data["CONF_META_DATA"]
        ordered_chain_ids = list(all_chain_features.keys())

        x_gt = np.concatenate([all_chain_features[chain_id]["x_gt"] for chain_id in ordered_chain_ids], axis=0)
        # asym_id = np.concatenate([all_chain_features[chain_id]["asym_id"] for chain_id in ordered_chain_ids], axis=0)

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
        # Spatial Crop Ligand

        if crop_scheme_seed < (0.6 if not self.inference_mode else 1.0) and len(x_gt_ligand) > 0:
            x_gt_ligand = np.concatenate(x_gt_ligand, axis=0)
            x_gt_sel = random.choice(x_gt_ligand)[None]
        # Spatial Crop Interface
        elif crop_scheme_seed < 0.8 and len(set(asym_id_ca.tolist())) > 1:
            chain_same = asym_id_ca[None] * asym_id_ca[:, None]
            dist = np.linalg.norm(x_gt_ca[:, None] - x_gt_ca[None], axis=-1)

            dist = dist + chain_same * 100
            # interface_threshold
            mask = np.any(dist < 15, axis=-1)
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
            if to_sum_token + to_add_token > self.token_crop_size:
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

        return all_chain_features

    def _spatial_crop(self, all_chain_features):

        ordered_chain_ids = list(all_chain_features.keys())
        atom_id_to_ccd_id = []
        atom_id_to_ccd_chunk_sizes = []
        atom_id_to_ccd = []

        ccd_all_id = 0
        for chain_id in ordered_chain_ids:
            for ccd, chunk_size_this_ccd in zip(
                    all_chain_features[chain_id]["ccds"],
                    all_chain_features[chain_id]["conformer_id_to_chunk_sizes"],
            ):
                atom_id_to_ccd_id += [ccd_all_id] * chunk_size_this_ccd
                atom_id_to_ccd_chunk_sizes += [chunk_size_this_ccd] * chunk_size_this_ccd
                atom_id_to_ccd += [ccd] * chunk_size_this_ccd
                ccd_all_id += 1

        to_sum_atom = 0
        to_sum_token = 0
        x_gt = np.concatenate([all_chain_features[chain_id]["x_gt"] for chain_id in ordered_chain_ids], axis=0)

        spatial_crop_ratio = 0.3 if not self.inference_mode else 0
        if random.random() < spatial_crop_ratio or len(ordered_chain_ids) == 1:
            x_gt_sel = random.choice(x_gt)[None]
        else:
            asym_id = np.array(reduce(add, [
                [asym_id] * len(all_chain_features[chain_id]["x_gt"])
                for asym_id, chain_id in enumerate(ordered_chain_ids)
            ]))
            chain_same = asym_id[None] * asym_id[:, None]
            dist = np.linalg.norm(x_gt[:, None] - x_gt[None], axis=-1)

            dist = dist + chain_same * 100
            mask = np.any(dist < 4, axis=-1)
            if sum(mask) > 0:
                x_gt_ = x_gt[mask]
            x_gt_sel = random.choice(x_gt)[None]
        dist = np.linalg.norm(x_gt - x_gt_sel, axis=-1)
        atom_idxs = np.argsort(dist)
        select_ccds_idx = []
        for atom_idx in atom_idxs:
            ccd_idx = atom_id_to_ccd_id[atom_idx]
            ccd_chunk_size = atom_id_to_ccd_chunk_sizes[atom_idx]
            ccd_this_atom = atom_id_to_ccd[atom_idx]
            if ccd_idx in select_ccds_idx:
                continue
            if to_sum_atom + ccd_chunk_size > self.atom_crop_size:
                break
            to_add_token = 1 if rc.is_standard(ccd_this_atom) else ccd_chunk_size
            if to_sum_token + to_add_token > self.token_crop_size:
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
        return all_chain_features

    def crop_all_chain_features(self, all_chain_features, infer_meta_data):
        # all_chain_features = self._spatial_crop(all_chain_features)
        all_chain_features = self._spatial_crop_v2(all_chain_features, infer_meta_data)
        return all_chain_features, infer_meta_data

    def _make_pocket_features(self, all_chain_features):
        # minimium distance 6-12
        all_chain_ids = list(all_chain_features.keys())

        for chain_id in all_chain_ids:
            all_chain_features[chain_id]["pocket_res_feat"] = np.zeros(
                [len(all_chain_features[chain_id]["ccds"])], dtype=np.bool_)

        ligand_chain_ids = [i for i in all_chain_ids if i.isdigit()]
        receptor_chain_ids = [i for i in all_chain_ids if not i.isdigit()]

        use_pocket = random.random() < 0.5
        # TODO: Inference mode assign
        if len(ligand_chain_ids) == 0 or len(receptor_chain_ids) == 0 or not use_pocket:
            for chain_id in all_chain_ids:
                all_chain_features[chain_id]["pocket_res_feat"] = all_chain_features[chain_id][
                    "pocket_res_feat"].astype(np.float32)
            return all_chain_features

        # Aug Part
        for ligand_chain_id in ligand_chain_ids:
            x_gt_ligand = all_chain_features[ligand_chain_id]["x_gt"]

            # x_gt_mean = np.mean(x_gt_ligand, axis=0) + np.random.randn(3)

            for receptor_chain_id in receptor_chain_ids:
                x_gt_receptor = all_chain_features[receptor_chain_id]["x_gt"]

                # dist = np.linalg.norm(x_gt_receptor - x_gt_mean[None], axis=-1)
                # is_pocket_atom = (dist < (random.random() * 6 + 8)).astype(np.bool_)
                is_pocket_atom = np.any(
                    np.linalg.norm(x_gt_receptor[:, None] - x_gt_ligand[None], axis=-1) < (random.random() * 6 + 6),
                    axis=-1
                )

                is_pocket_ccd = []
                offset = 0
                for chunk_size in all_chain_features[receptor_chain_id]["conformer_id_to_chunk_sizes"]:
                    is_pocket_ccd.append(np.any(is_pocket_atom[offset:offset + chunk_size]).item())
                    offset += chunk_size
                is_pocket_ccd = np.array(is_pocket_ccd, dtype=np.bool_)

                is_pocket_ccd = np.array([np.any(i).item() for i in is_pocket_ccd], dtype=np.bool_)
                all_chain_features[receptor_chain_id]["pocket_res_feat"] = all_chain_features[receptor_chain_id][
                                                                               "pocket_res_feat"] | is_pocket_ccd

        for chain_id in all_chain_ids:
            all_chain_features[chain_id]["pocket_res_feat"] = all_chain_features[chain_id]["pocket_res_feat"].astype(
                np.float32)

        return all_chain_features

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

        # Create Aug Pocket Feature
        all_chain_features = self._make_pocket_features(all_chain_features)

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
        raw_feats["ref_pos"] = centre_random_augmentation_np_apply(
            raw_feats["ref_pos"], raw_feats["atom_id_to_token_id"]).astype(np.float32)
        raw_feats["ref_feat"][:, :3] = raw_feats["ref_pos"]

        ###################################################
        #            Create token pair features           #
        ###################################################
        no_token = len(raw_feats["token_id_to_conformer_id"])
        token_bonds = np.zeros([no_token, no_token], dtype=np.float32)
        rel_tok_feat = np.zeros([no_token, no_token, 42], dtype=np.float32)
        batch_ref_pos = np.zeros([32, no_token, 3], dtype=np.float32)
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
                batch_ref_pos[:, offset:offset + len_atoms] = CONF_META_DATA[ccd]["batch_ref_pos"][:, inner_atom_idx]
                token_bonds[offset:offset + len_atoms, offset:offset + len_atoms] = \
                    CONF_META_DATA[ccd]["token_bonds"][inner_atom_idx][:, inner_atom_idx]
                rel_tok_feat[offset:offset + len_atoms, offset:offset + len_atoms] = \
                    CONF_META_DATA[ccd]["rel_tok_feat"][inner_atom_idx][:, inner_atom_idx]
                offset += len_atoms
            atom_offset += len_atoms
        raw_feats["token_bonds"] = token_bonds.astype(np.float32)
        raw_feats["token_bonds_feature"] = token_bonds.astype(np.float32)
        raw_feats["rel_tok_feat"] = rel_tok_feat.astype(np.float32)
        raw_feats["batch_ref_pos"] = batch_ref_pos.astype(np.float32)
        ###################################################
        #              Charility Augmentation             #
        ###################################################
        if not self.inference_mode:
            # TODO Charility probs
            charility_seed = random.random()
            if charility_seed < 0.1:
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

        # MSA Feat
        inds = [0] + torch.randperm(len(tensors["msa"]))[:127].tolist()

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
        if token_bond_seed >= 0:
            tensors["token_bonds_feature"] = tensors["token_bonds"]
        return tensors

    def _pad_to_size(self, tensors):

        to_pad_atom = self.atom_crop_size - len(tensors["x_gt"])
        to_pad_token = self.token_crop_size - len(tensors["residue_index"])
        if to_pad_token > 0:
            for k in ["restype", "residue_index", "is_protein", "is_short_poly", "is_ligand", "is_key_res",
                      "asym_id", "entity_id", "sym_id", "token_id_to_conformer_id", "s_mask",
                      "token_id_to_centre_atom_id", "token_id_to_pseudo_beta_atom_id", "token_id_to_chunk_sizes",
                      "pocket_res_feat"]:
                tensors[k] = torch.nn.functional.pad(tensors[k], [0, to_pad_token])
            for k in ["target_feat", "msa_feat", "batch_ref_pos", "key_res_feat"]:
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
        dgram = dgram_from_positions(x_gt)
        dgram = dgram * protein2d[..., None] * z_mask[..., None]

        # if not self.inference_mode:
        #     bert_mask = torch.rand([len(x_gt)]) > random.random() * 0.4
        #     asym_ids = list(set(asym_id.tolist()))
        #     used_asym_ids = []
        #     for a in asym_ids:
        #         if random.random() > 0.6:
        #             used_asym_ids.append(a)
        #     if len(used_asym_ids) > 0:
        #         used_asym_ids = torch.tensor(used_asym_ids)
        #         chain_bert_mask = torch.any(asym_id[:, None] == used_asym_ids[None], dim=-1)
        #         bert_mask = chain_bert_mask * bert_mask
        #     else:
        #         bert_mask = bert_mask * 0
        #     template_pseudo_beta_mask = (bert_mask[None] * bert_mask[:, None]) * z_mask * protein2d
        # else:
        #     template_pseudo_beta_mask = z_mask * protein2d
        template_pseudo_beta_mask = z_mask * protein2d
        # template_pseudo_beta_mask = protein2d * z_mask
        dgram = dgram * template_pseudo_beta_mask[..., None]
        templ_feat = torch.cat([dgram, template_pseudo_beta_mask[..., None]], dim=-1)
        tensors["templ_feat"] = templ_feat.float()[None]
        t_mask_seed = random.random()
        # Template Augmentation
        if self.inference_mode or t_mask_seed < 0.1:
            tensors["t_mask"] = torch.ones([len(tensors["templ_feat"])], dtype=torch.float32)
        else:
            tensors["t_mask"] = torch.zeros([len(tensors["templ_feat"])], dtype=torch.float32)

        # TODO: No Template
        # if not self.inference_mode:
        #     if random.random() < 0.5:
        #         tensors["templ_feat"] *= 0
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

        # # Make Pocket Res Feat
        #
        # tensors["pocket_res_feat"] = torch.zeros([l], dtype=torch.float32)

        # Make Key Res Feat
        # l = len(tensors["asym_id"])
        # tensors["key_res_feat"] = torch.zeros([l, 7], dtype=torch.float32)
        # tensors["key_res_feat"][:, 0] = 1.

        # Mask
        tensors["z_mask"] = tensors["s_mask"][None] * tensors["s_mask"][:, None]

        # Template
        tensors = self.get_template_feat(tensors)

        # Correct Type
        is_short_poly = tensors.pop("is_short_poly")
        tensors["is_protein"] = tensors["is_protein"] + is_short_poly
        tensors["is_ligand"] = tensors["is_ligand"] - is_short_poly
        tensors["is_dna"] = torch.zeros_like(tensors["is_protein"])
        tensors["is_rna"] = torch.zeros_like(tensors["is_protein"])
        return tensors

    def load(self, sample_id):
        all_chain_labels = load_pkl(os.path.join(self.samples_path, f"{sample_id}.pkl.gz"))

        all_chain_features, infer_meta_data = self.load_all_chain_features(all_chain_labels)
        infer_meta_data["system_id"] = sample_id
        # if not self.inference_mode:
        all_chain_features, infer_meta_data = self.crop_all_chain_features(all_chain_features, infer_meta_data)
        raw_feats, infer_meta_data = self.pair_and_merge(all_chain_features, infer_meta_data)

        tensors = self.transform(raw_feats)
        return tensors, infer_meta_data

    def random_load(self):

        sample_id = self.used_sample_ids[torch.multinomial(self.probabilities, 1).item()]
        print(sample_id)
        return self.load(sample_id)

    def random_load_test(self):

        sample_id = self.used_test_sample_ids[torch.multinomial(self.test_probabilities, 1).item()]
        print(sample_id)
        return self.load(sample_id)

    def weighted_random_load(self):
        weight_seed = random.random()
        if weight_seed < 0.95:
            sample_id = self.used_sample_ids[torch.multinomial(self.probabilities, 1).item()]
        else:
            return self.random_load_mol_chunks()
        return self.load(sample_id)

    def load_ligand(self, sample_id, chain_features):
        all_chain_features = {}
        CHAIN_META_DATA = {
            "spatial_crop_chain_ids": None,
            "chain_class": {},
            "chain_sequence_3s": {},
            "fake_ccds": [],
        }
        CONF_META_DATA = {}
        num_prev_fake_ccds = len(CHAIN_META_DATA["fake_ccds"])
        fake_ccd = f"{num_prev_fake_ccds:#>3}"

        chain_features["msa"] = chain_features["restype"][None]
        chain_features["deletion_matrix"] = np.ones_like(chain_features["msa"])
        chain_features["ccds"] = [fake_ccd]
        chain_features["chain_class"] = "ligand"
        chain_features["all_atom_positions"] = chain_features["all_atom_positions"][None]
        chain_features["all_atom_mask"] = chain_features["all_atom_mask"][None]
        # Update Chain and Conf
        CHAIN_META_DATA["fake_ccds"].append(fake_ccd)
        sequence_3 = fake_ccd
        chain_id = f"SDFM_{sample_id}"
        CHAIN_META_DATA["chain_sequence_3s"][chain_id] = sequence_3
        CHAIN_META_DATA["chain_class"][chain_id] = "ligand"
        CONF_META_DATA = self._update_CONF_META_DATA_ligand(
            CONF_META_DATA, sequence_3, chain_features)
        all_chain_features[chain_id] = chain_features
        all_chain_features[chain_id] = self._update_chain_feature(
            chain_features,
            CONF_META_DATA
        )
        SEQ3 = {}
        CHAIN_CLASS = {}
        SEQ3[chain_id] = "-".join([fake_ccd])
        all_chain_features, ASYM_ID = self._add_assembly_feature(all_chain_features, SEQ3)
        all_chain_features[chain_id]["conformer_id_to_chunk_sizes"] = np.array(
            [len(chain_features["ref_atom_name_chars"])], dtype=np.int64)

        all_chain_features[chain_id]["x_gt"] = chain_features["all_atom_positions"][0]
        all_chain_features[chain_id]["x_exists"] = chain_features["all_atom_mask"][0]
        CHAIN_CLASS[chain_id] = "ligand"
        infer_meta_data = {
            "CONF_META_DATA": CONF_META_DATA,
            "SEQ3": SEQ3,
            "ASYM_ID": ASYM_ID,
            "CHAIN_CLASS": CHAIN_CLASS
        }

        # if not self.inference_mode:
        #     all_chain_features, infer_meta_data = self.crop_all_chain_features(all_chain_features, infer_meta_data)

        raw_feats, infer_meta_data = self.pair_and_merge(all_chain_features, infer_meta_data)

        tensors = self.transform(raw_feats)
        infer_meta_data["system_id"] = sample_id
        return tensors, infer_meta_data

    def random_load_mol_chunks(self, ligand_db_name="1"):
        if ligand_db_name == "0":
            ligand_db = load_pkl("/2022133002/projects/stdock/stdock_v9.5/scripts/try_new.pkl.gz")
        elif ligand_db_name == "1":
            id = random.randint(1, 374)
            ligand_db = load_pkl(f"/2022133002/data/ligand_samples/samples_{id}.pkl.gz")
        elif ligand_db_name == "2":
            ligand_db = load_pkl("/2022133002/projects/stdock/stdock_v9.5/scripts/try_400k_2_new.pkl.gz")
        else:
            raise ValueError("MOL DB Name is Wrong!")
        sample_id = random.choice(list(ligand_db.keys()))
        sample_feature = ligand_db[sample_id]
        tensors, infer_meta_data = self.load_ligand(sample_id, sample_feature)
        return tensors, infer_meta_data

    def write_pdb(self, x_pred, fname, infer_meta_data):
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
                    f"{res_name_3.split()[0]:>3} {chain_tag:>1}"
                    f"{res_id + 1:>4}{insertion_code:>1}   "
                    f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                    f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                    f"{element:>2}{charge:>2}"
                )
                atom_lines.append(atom_line)
                atom_offset += 1
                if atom_offset == len(atom_id_to_conformer_atom_id):
                    break
        out = "\n".join(atom_lines)
        out = f"MODEL     1\n{out}\nTER\nENDMDL\nEND"
        dump_txt(out, fname)

from typing import Dict, Union, Any
import torch
import numpy as np
from scipy.sparse.coo import coo_matrix

# TODO: Keep Only ref_mask eq 1 for all atomwise feature


'''
    Notation:
    batch, 
    token dimension: i, j, k
    flat atom dimension: l, m | WARNING: We should flatten due to local atom attention mask
    sequence dimension: s(msa) t(time)
    head dimension: h

    ####################
    z_ij: pair repr
    {z_ij}: all pair repr 
    x: atom position
    {x_l}: flat atom list, full atomic structure
    exist a mapping: flat atom index -> token index and within token atom index: l -> i, a


    ####################
    a: atom representation
        have the same shape as s
        exist a transform: flat atom representation -> atom represenation
    s: token representation
    z: pair representation

'''

FeatureDict = Dict[str, Union[np.ndarray, coo_matrix, None, Any]]
TensorDict = Dict[str, Union[torch.Tensor, Any]]

PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

NUM_CONFORMER = "num conformer placeholder"
NUM_TOKEN = "num tokens placeholder"
NUM_ATOM = "num atoms placeholder"

NUM_SEQ = "num MSAs placeholder"
NUM_TEMPL = "num templates placeholder"

NUM_RECYCLING = "num recycling placeholder"
NUM_SAMPLE = "num sample placeholder"

SHAPE_SCHIME = {
    ################################################################
    # Conformerwise Feature

    # Tokenwise Feature
    "residue_index": [NUM_TOKEN],
    "restype": [NUM_TOKEN],
    "token_index": [NUM_TOKEN],
    "s_mask": [NUM_TOKEN],
    "is_protein": [NUM_TOKEN],
    "is_rna": [NUM_TOKEN],
    "is_dna": [NUM_TOKEN],
    "is_ligand": [NUM_TOKEN],
    "token_id_to_centre_atom_id": [NUM_TOKEN],
    "token_id_to_pseudo_beta_atom_id": [NUM_TOKEN],
    "token_id_to_chunk_sizes": [NUM_TOKEN],
    "token_id_to_conformer_id": [NUM_TOKEN],
    "asym_id": [NUM_TOKEN],
    "entity_id": [NUM_TOKEN],
    "sym_id": [NUM_TOKEN],
    "token_bonds": [NUM_TOKEN, NUM_TOKEN],
    "target_feat": [NUM_TOKEN],
    "token_exists": [NUM_TOKEN],
    "spatial_crop_target_res_mask": [NUM_TOKEN],

    # Atomwise features
    "ref_space_uid": [NUM_ATOM],
    "atom_index": [NUM_ATOM],
    "ref_feat": [NUM_ATOM, 389],
    "ref_pos": [NUM_ATOM, 3],
    "a_mask": [NUM_ATOM],
    "atom_id_to_token_id": [NUM_ATOM],
    "x_gt": [NUM_ATOM, 3],
    "x_exists": [NUM_ATOM],
    "rec_mask": [NUM_ATOM, NUM_ATOM],

    "msa": [NUM_SEQ, NUM_TOKEN],
    "deletion_matrix": [NUM_SEQ, NUM_TOKEN],
    "msa_feat": [NUM_SEQ, NUM_TOKEN, None],
    "crop_idx": [None],
    "crop_idx_atom": [None],

    #
    "x_centre": [None],
    # # Template features
    # "template_restype": [NUM_TEMPLATES, NUM_TOKEN],
    # "template_pseudo_beta_mask": [NUM_TEMPLATES, NUM_TOKEN],
    # "template_backbone_frame_mask": [NUM_TEMPLATES, NUM_TOKEN],
    # "template_distogram": [NUM_TEMPLATES, NUM_TOKEN, NUM_TOKEN, 39],
    # "template_unit_vector": [NUM_TEMPLATES, NUM_TOKEN, NUM_TOKEN, 3],

    ###########################################################
}

SUPERVISED_FEATURES = [

]

UNSUPERVISED_FEATURES = [

]

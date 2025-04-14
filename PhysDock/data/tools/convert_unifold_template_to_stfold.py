import sys
import os

import numpy as np
import torch

sys.path.append("../")

from PhysDock.utils.io_utils import load_pkl, dump_pkl
from PhysDock.data.tools.residue_constants import \
    hhblits_id_to_standard_residue_id_np, af3_if_to_residue_id

def dgram_from_positions(
        pos: torch.Tensor,
        min_bin: float = 3.25,
        max_bin: float = 50.75,
        no_bins: float = 39,
        inf: float = 1e8,
):
    dgram = torch.sum(
        (pos[..., None, :] - pos[..., None, :, :]) ** 2, dim=-1, keepdim=True
    )
    lower = torch.linspace(min_bin, max_bin, no_bins, device=pos.device) ** 2
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)
    return dgram


def convert_unifold_template_feature_to_stfold_unifold_feature(unifold_template_feature):
    try:
        print(unifold_template_feature)
        md5_string = os.path.basename(unifold_template_feature)[:-6]
        out_path = os.path.dirname(unifold_template_feature)
        out_path = os.path.dirname(out_path)
        unifold_template_feature = os.path.join(out_path, "msas", md5_string)
        out_path_final = os.path.join(out_path, "msas", "template_features")
        final = os.path.join(out_path_final, f"{md5_string}.pkl.gz")
        if os.path.exists(final):
            return dict()

        unifold_template_feature = os.path.join(unifold_template_feature, "templates", "pdb_hits.hhr.pkl.gz")
        if isinstance(unifold_template_feature, str):
            data = load_pkl(unifold_template_feature)
        else:
            data = unifold_template_feature
        template_restype = af3_if_to_residue_id[
            hhblits_id_to_standard_residue_id_np[np.argmax(data["template_aatype"], axis=-1)]]
        assert np.all(template_restype != -1)
        assert len(template_restype) >= 1
        assert len(template_restype[0, :]) >= 4
        # shape = template_restype.shape
        # template_restype=template_restype.view([-1])[].view(shape)

        bb_x_gt = torch.from_numpy(data["template_all_atom_positions"][..., :3, :])
        bb_x_mask = torch.from_numpy(data["template_all_atom_masks"][..., :3])

        bb_x_gt_beta1 = data["template_all_atom_positions"][..., 3, :]
        bb_x_gt_beta_mask1 = data["template_all_atom_masks"][..., 3]
        bb_x_gt_beta2 = data["template_all_atom_positions"][..., 1, :]
        bb_x_gt_beta_mask2 = data["template_all_atom_masks"][..., 1]

        is_gly = template_restype == 7
        template_pseudo_beta = np.where(is_gly[..., None], bb_x_gt_beta2, bb_x_gt_beta1)
        template_pseudo_beta_mask = np.where(is_gly, bb_x_gt_beta_mask2, bb_x_gt_beta_mask1)
        template_backbone_frame_mask = bb_x_mask[..., 0] * bb_x_mask[..., 1] * bb_x_mask[..., 2]
        out = {
            "template_restype": template_restype.astype(np.int8),
            "template_backbone_frame_mask": template_backbone_frame_mask.numpy().astype(np.int8),
            "template_backbone_frame": bb_x_gt.numpy().astype(np.float32),
            "template_pseudo_beta": template_pseudo_beta.astype(np.float32),
            "template_pseudo_beta_mask": template_pseudo_beta_mask.astype(np.int8),
            # "template_backbone_mask": template_mask.numpy().astype(np.int8),
        }
        # for k,v in out.items():
        #     print(k,v.shape)
        dump_pkl(out, os.path.join(out_path_final, f"{md5_string}.pkl.gz"), compress=True)
    except:
        pass
        out = dict()
    print(f"dump templ feats to {md5_string}.pkl.gz")
    return out


# HHBLITS_ID_TO_AA = {
#     0: "ALA",
#     1: "CYS",  # Also U.
#     2: "ASP",  # Also B.
#     3: "GLU",  # Also Z.
#     4: "PHE",
#     5: "GLY",
#     6: "HIS",
#     7: "ILE",
#     8: "LYS",
#     9: "LEU",
#     10: "MET",
#     11: "ASN",
#     12: "PRO",
#     13: "GLN",
#     14: "ARG",
#     15: "SER",
#     16: "THR",
#     17: "VAL",
#     18: "TRP",
#     19: "TYR",
#     20: "UNK",  # Includes J and O.
#     21: "GAP",
# }
#
# # Usage: Convert hhblits msa to af3 aatype
# #        msa = hhblits_id_to_standard_residue_id_np[hhblits_msa.astype(np.int64)]
# hhblits_id_to_standard_residue_id_np = np.array(
#     [standard_ccds.index(ccd) for id, ccd in HHBLITS_ID_TO_AA.items()]
# )
#
# of_restypes = [
#     "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
#     "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "X", "-"
# ]
#
# af3_restypes = [amino_acid_3to1[ccd] if ccd in amino_acid_3to1 else "-" if ccd == "GAP" else "None" for ccd in
#                 standard_ccds
#                 ]
#
# af3_if_to_residue_id = np.array(
#     [af3_restypes.index(restype) if restype in of_restypes else -1 for restype in af3_restypes])



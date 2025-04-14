# Attention: This Module is not used in released version!!!

import torch
import torch.nn as nn
from typing import Optional, Tuple

from PhysDock.models.primitives import Linear
from PhysDock.models.layers.transformers import AtomTransformer, Pairformer
from PhysDock.utils.tensor_utils import one_hot_with_nearest_bin
from PhysDock.data import TensorDict


class ConfidenceModule(nn.Module):
    def __init__(
            self,
            c_a: int,
            c_ap: int,
            c_s: int,
            c_z: int,
            inf: float,
            eps: float,
            no_blocks_heads: int,
            no_blocks_atom: int = 3,
            c_pae: int = 64,
            c_pde: int = 64,
            c_plddt: int = 50,
    ):
        super().__init__()
        self.linear_s_i = Linear(c_s, c_z)
        self.linear_s_j = Linear(c_s, c_z)
        self.linear_d = Linear(13, c_z, bias=False)  # no_bins

        self.pairformer = Pairformer(
            c_s,
            c_z,
            inf,
            eps,
            no_blocks=no_blocks_heads,
        )

        self.linear_pae = Linear(c_z, c_pae)
        self.linear_pde = Linear(c_z, c_pde)

        self.linear_s_a = Linear(c_s, c_a)
        self.linear_z_a = Linear(1, c_ap)
        self.atom_transformer = AtomTransformer(
            c_a,
            c_ap,
            inf,
            eps,
            no_blocks_atom,
        )
        self.linear_plddt = Linear(c_a, c_plddt)

    def forward(
            self,
            batch: TensorDict,
            s: torch.Tensor,
            z: torch.Tensor,
            x_pred: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        token_id_to_centre_atom_id = batch["token_id_to_centre_atom_id"]
        atom_id_to_token_id = batch["atom_id_to_token_id"]
        ap_mask = batch["ap_mask"]
        z_mask = batch["z_mask"]
        x_pred_token_centre = x_pred[0, token_id_to_centre_atom_id, :]

        z = z + self.linear_s_i(s)[..., None, :] + self.linear_s_j(s)[..., None, :, :]
        d = torch.norm(x_pred_token_centre[..., None, :] - x_pred_token_centre[..., None, :, :], dim=-1,
                       keepdim=True)
        v_bins = torch.linspace(3.375, 24.375, 13, device=z.device).type(z.dtype)
        z = z + self.linear_d(one_hot_with_nearest_bin(d, v_bins).to(z.device).type(z.dtype))

        s, z = self.pairformer(s, z, z_mask)
        z = z + z.transpose(-2, -3)
        p_pae = self.linear_pae(z)
        p_pde = self.linear_pde(z)

        a = self.linear_s_a(s)[atom_id_to_token_id]
        ap = self.linear_z_a(torch.norm(x_pred[0][None] - x_pred[0][:, None], dim=-1)[..., None])

        a = a + self.atom_transformer(
            a, ap, ap_mask
        )

        p_plddt = self.linear_plddt(a)

        return p_pae, p_pde, p_plddt

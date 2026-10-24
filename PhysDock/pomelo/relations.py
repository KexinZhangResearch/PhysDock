"""Pair relations as single features, for the models without a pair representation (Pomelo, SingleTrack, SingleTrackFold):

  the bond graph of the ligands (and of every other component tokenised per atom: token_bonds, ligand_pair_feat; the topological distance d_token is not used)
    bond_graph          the adjacency of every edge feature: bond type (single, double, triple, aromatic, other), bond order, in ring, conjugated, aromatic
    graph_features      per token: RWSE (the return probabilities of the random walk after 1 .. steps steps), the number of bonds of every edge feature, in the graph
    BondMessagePassing  a few rounds of messages along the bonds, one linear map per edge feature (R-GCN); a residual update of the tokens in the graph only
  the chains and entities (an equivalence relation: in the same chain / entity or not)
    relation_codes      a random orthonormal code per chain (entity), drawn anew every call: c_i . c_j = 1 in the same chain, 0 otherwise, exactly (up to code_dim groups;
                        more: random unit vectors, ~1 / sqrt(code_dim)). The codes say nothing of the labels (the order or the number of a chain), only their dot
                        products are meaningful: they enter the attentions as columns of q and k (w_h c_i . c_j, a learned weight per head; code_columns)

All of them are scalars (0e): Pomelo stays O(3)-equivariant. bond_graph, graph_features and relation_codes (no parameters, once per structure; torch.unique, QR)
run eagerly under torch.compile (and avoid an Inductor lowering failure of torch 2.6 on a cat of transposed views).
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ligand_pair_feat [N_token, N_token, 42] (PhysDock.data.ccd): d_token one-hot (0:32), bond type one-hot (32:37), token_bonds (37), bond_as_double (38), bond_in_ring (39),
# bond_is_conjugated (40), bond_is_aromatic (41). The bond type of a pair without a bond is 0 (single): every edge feature is taken on the bonds only.
EDGE_CHANNELS = [32, 33, 34, 35, 36, 38, 39, 40, 41]
N_EDGE = len(EDGE_CHANNELS)


@torch.compiler.disable
def bond_graph(b, tm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """-> A [Nt, Nt] (the bonds between real tokens, no self loops), E [N_EDGE, Nt, Nt] (A times every edge feature), in float32 (float64 kept)"""
    dt = torch.promote_types(tm.dtype, torch.float32)
    nt = tm.shape[0]
    real = (tm > 0)[:, None] & (tm > 0)[None] & ~torch.eye(nt, dtype=torch.bool, device=tm.device)
    a = ((b["token_bonds"] > 0) & real).to(dt)
    e = b["ligand_pair_feat"][..., EDGE_CHANNELS].to(dt).permute(2, 0, 1) * a[None]
    return a, e


@torch.compiler.disable
def graph_features(a: torch.Tensor, e: torch.Tensor, steps: int) -> torch.Tensor:
    """A [Nt, Nt], E [K, Nt, Nt] -> [Nt, steps + K + 1]: diag(P^k) for k = 1 .. steps (P = D^-1 A; RWSE), the bonds of every edge feature (counts; the sum of the bond
    orders), in the graph (1 if the token has a bond); 0 for a token without bonds"""
    deg = a.sum(1)
    p = a / deg.clamp(min=1)[:, None]
    walk, rw = p, []
    for _ in range(steps):
        rw.append(torch.diagonal(walk))
        walk = walk @ p
    return torch.cat([torch.stack(rw, -1), e.sum(-1).T, (deg > 0).to(a.dtype)[:, None]], -1)


def n_graph_features(steps: int) -> int:
    return steps + N_EDGE + 1


class BondMessagePassing(nn.Module):
    """rounds of h_i += in_graph_i * MLP(norm(sum_k sum_j E_k,ij W_k h_j)): one linear map per edge feature (R-GCN), only along the bonds (no pair tensor of the
    channels); the last layer of every MLP starts at 0 (a no-op at the start)"""

    def __init__(self, dim: int, rounds: int) -> None:
        super().__init__()
        self.rounds = nn.ModuleList([nn.ModuleDict({"edge": nn.Linear(dim, N_EDGE * dim, bias=False), "norm": nn.LayerNorm(dim),
                                                    "up": nn.Linear(dim, 2 * dim, bias=False), "down": nn.Linear(dim, dim, bias=False)})
                                     for _ in range(rounds)])
        for r in self.rounds:
            nn.init.zeros_(r["down"].weight)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """h [Nt, dim], E [K, Nt, Nt] -> h [Nt, dim] (changed at the tokens with a bond only)"""
        in_graph = (e.sum((0, 2)) > 0).to(h.dtype)[:, None]
        e = e.to(h.dtype)
        for r in self.rounds:
            msg = torch.einsum("kij,jkd->id", e, r["edge"](h).unflatten(-1, (N_EDGE, -1)))
            u, g = r["up"](r["norm"](msg)).chunk(2, -1)
            h = h + in_graph * r["down"](F.silu(u) * g)
        return h


@torch.compiler.disable
def relation_codes(ids: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """ids [N] (asym_id or entity_id), mask [N] -> [N, dim]: a random orthonormal code per id (random unit vectors if there are more than dim ids), 0 where mask is 0"""
    uniq, inv = torch.unique(ids, return_inverse=True)
    n = uniq.shape[0]
    dt = torch.promote_types(mask.dtype, torch.float32)
    g = torch.randn(max(n, dim), dim, dtype=dt, device=ids.device)
    table = torch.linalg.qr(g.T)[0].T[:n] if n <= dim else F.normalize(g[:n], dim=-1)
    return table[inv] * mask.to(dt)[:, None]


def chain_entity_codes(b, tm: torch.Tensor, dim: int) -> torch.Tensor:
    """the codes of the tokens [Nt, 2 dim]: chain (asym_id), entity (entity_id)"""
    return torch.cat([relation_codes(b["asym_id"], tm, dim), relation_codes(b["entity_id"], tm, dim)], -1)


def code_columns(codes: torch.Tensor, weight: torch.Tensor, scale: float, b: int, dtype):
    """the columns of q and k for w_h,chain [same chain] + w_h,entity [same entity] in a call with the given scale: codes [N, 2 dim], weight [H, 2] ->
    q [b, H, N, 2 dim], k [b, H, N, 2 dim] (the logit term: q . k * scale)"""
    h, (n, d2) = weight.shape[0], codes.shape
    w = weight.repeat_interleave(d2 // 2, dim=1)                                                     # [H, 2 dim]
    q = codes.to(dtype)[None, None].expand(b, h, n, d2)
    return q, q * (w.to(dtype) / scale)[None, :, None, :]

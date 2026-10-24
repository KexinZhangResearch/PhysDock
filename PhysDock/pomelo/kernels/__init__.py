"""Triton kernels of Pomelo (PhysDock.pomelo), separate from PhysDock's own (PhysDock.models.kernels, not used or changed here).

  attention.py   pomelo_attention: the typed attention of the DiT blocks (flash attention with the distance term and the same-chain / same-entity terms computed
                 in the kernel from the positions and the chain / entity ids, instead of extra columns of q and k), forward and backward

Switched on by the model config (backend: triton, as PhysDock's model.backend=triton); the torch path stays the reference (tests/test_pomelo_kernels.py).
"""
from PhysDock.pomelo.kernels.attention import pomelo_attention

__all__ = ["pomelo_attention"]

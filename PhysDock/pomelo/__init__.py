"""Pomelo: an alternative network for PhysDock, O(3)-equivariant by construction (rotations, translations and reflections), with typed features (0e, 0o, 1o, 1e), one DiT
trunk without a pair representation. It replaces the encoder and decoder of PhysDock.models.model.PhysDock
with the same interface (forward(batch) -> {x_gt, x_denoised, x_hat, t_hat, stream_rms}, sample_diffusion); data, sigma_data, diffusion, loss and training are PhysDock's.

  configs.py     the size of the network (presets mini, tiny)
  embedders.py   ConditionEmbedder: PhysDock's features -> the frame-free conditions of the DiT, once per structure
  dit.py         PomeloDiT: atom DiT -> pool -> token DiT -> unpool -> atom DiT, every stream all four kinds
  model.py       Pomelo: the model with PhysDock's interface
  primitives/    the typed layers (geometry, irreps, attentions, alignment, transformers)
"""
from PhysDock.pomelo.configs import Config, preset
from PhysDock.pomelo.model import Pomelo

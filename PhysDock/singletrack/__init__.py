"""SingleTrack: Pomelo's inputs and layout with one non-equivariant feature track per atom and per token (SimpleFold-like DiT). It replaces the encoder and decoder of
PhysDock.models.model.PhysDock with the same interface; data, sigma_data, diffusion, loss and training are PhysDock's.

  configs.py   the size of the network (presets mini, tiny)
  dit.py       SingleTrackDiT: the typed inputs of Pomelo flattened into the track, atom DiT -> pool -> token DiT -> unpool -> atom DiT
  model.py     SingleTrack: the model with PhysDock's interface
  simple.py    SingleTrackSimple: the same DiT with SimpleFold's plain inputs (ref_feat, noisy and template xyz, RoPE, t)
  fold.py      SingleTrackFold: SimpleFold's inputs and layout (Fourier xyz, axial RoPE on ref_pos, length condition), no ESM
"""
from PhysDock.singletrack.configs import Config, preset
from PhysDock.singletrack.model import SingleTrack
from PhysDock.singletrack.simple import SingleTrackSimple
from PhysDock.singletrack.fold import SingleTrackFold

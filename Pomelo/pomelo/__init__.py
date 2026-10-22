"""Pomelo: an all-atom co-folding model whose layers are equivariant under O(3) (rotations, translations and reflections), with typed features (0e, 0o, 1o, 1e) so that the
handedness of a structure is handled by the architecture and not learned from data. One DiT trunk without a pair representation, trained with PhysDock's data, diffusion,
loss and training loop (see README.md)."""
from pomelo.configs import Config, preset
from pomelo.models.model import Pomelo

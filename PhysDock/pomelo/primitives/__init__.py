"""The typed layers of Pomelo, bottom up (torch only).

geometry.py      the kinds of features ("0e", "0o", "1o", "1e"), their products (dot, cross, signed volume, scale) and the O(3) checker
irreps.py        feature sets {kind: tensor}: TypedLinear, TypedNorm, NormLinear, Gate, GeometricProduct (the exchange between the kinds), and the helpers add / mask_rows
attentions.py    GlobalAttention: the typed attention of a batch of noisy samples (one SDPA call), RoPE, the distance term and the direction messages
alignment.py     proper Kabsch rotations of the reference conformers and of the template into the frame of a noisy sample
transformers.py  TypedTransition
"""

"""The typed layers of Pomelo, bottom up (torch only; a Triton backend would go to pomelo.models.kernels, as in PhysDock).

geometry.py      the kinds of features ("0e", "0o", "1o", "1e"), their products (dot, cross, signed volume, scale) and the O(3) checker
irreps.py        feature sets {kind: tensor}: TypedLinear, TypedNorm, NormLinear, Gate, GeometricProduct (the exchange between the kinds), and the helpers add / mask_rows
attentions.py    SharedBiasAttention: the typed attention of a batch of noisy samples with a static logit bias, the distance term and the direction messages
transformers.py  TypedTransition
"""

"""The typed layers of Pomelo, bottom up (torch only; a Triton backend would go to pomelo.models.kernels, as in PhysDock).

geometry.py      the kinds of features ("0e", "0o", "1o", "1e"), their products (dot, cross, signed volume, scale) and the O(3) checker
irreps.py        feature sets {kind: tensor}: TypedLinear, TypedRMSNorm, Gate, GeometricProduct, and the helpers add / mask_rows / mask_pairs
attentions.py    TypedAttention (single representation, with pair messages), TypedTriangleAttention
triangle.py      directions between units, PairFromSingle, TypedTriangleMultiplication
transformers.py  TypedTransition, SingleBlock, PairBlock, TrunkBlock
"""

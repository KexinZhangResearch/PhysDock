# Pomelo

An all-atom co-folding model that is **O(3)-equivariant by construction** (rotations, translations and reflections) with **typed features**
(kinds named as in e3nn: scalar `0e`, pseudoscalar `0o`, polar vector `1o`, axial vector `1e`), so that handedness is handled by the architecture and not learned from data.
Mirroring the input (and flipping its stereo labels) mirrors the output; a pseudoscalar (`0o`) can only enter an attention logit as a product of two. The kinds say how a quantity transforms, not where it lives: single, pair and atom representations can hold any of them.

Layout follows PhysDock: `pomelo/` (package), `configs/`, `scripts/`, `tests/`.

## Steps (each with tests before the next)
1. `pomelo/geometry.py`: kinds, products (dot, cross, triple, scale, norm), the O(3) checker. **done**
2. `pomelo/irreps.py`: feature sets `{kind: tensor}`, `TypedLinear`, `TypedRMSNorm`, `Gate`, `GeometricProduct`. **done**
3. `pomelo/attention.py`: the typed attention of the single representation with pair messages (one softmax, three weight groups, direction messages). **done**
4. `pomelo/blocks.py`: `TypedTransition` and `SingleBlock` (products, attention, transition); `pomelo/checks.py`: the O(3) checks. **done**
5. `pomelo/pair.py`: the pair representation (`z_0e`, `z_0o`): `PairFromSingle`, `TypedTriangleMultiplication` (out / in), `TypedTriangleAttention` (start / end), `PairBlock`; `pomelo/trunk.py`: `TrunkBlock` (pair block, then single block). **done**
6. data: units (residues, ligand fragments) and the stereo table
7. typed geometric features (local frames, signed volumes), the input embedding of s and z, recycling
8. the trunk (stack of blocks), the decoder

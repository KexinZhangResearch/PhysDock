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
6. `pomelo/data.py` (synthetic molecules with stereo labels and units), `pomelo/embedder.py` (the six stored features), `pomelo/dit.py` (the multi-scale DiT, atoms -> units -> atoms, no augmentation),
   `pomelo/diffusion.py` (EDM noise, x_mse + smooth lDDT, sampler, metrics), `pomelo/baseline.py` (PhysDock-like models without symmetry, the same inputs and loss, random rotations as augmentation). **done**
7. `pomelo/protein.py`: 10 real proteins (128-residue windows, `pomelo/proteins_10.json`), residue relative positions in the embedder. **running** (see below)
8. `pomelo/encoder_trunk.py`: the **encoder trunk** with the reference positions as input: an atom stage per conformer (typed atom attention inside a conformer, the chirality from the
   signed volumes of the reference conformers, no stereo label read), typed pooling, then `TrunkBlock`s (single -> pair, triangle multiplication out / in, triangle attention
   start / end, transition; single block). Every conformer is in a frame of its own, so the vector terms of the attention, the single -> pair products and the triangles act between units of the
   same conformer only (`same`); a residue is a conformer, a ligand is one conformer. `TrunkDenoiser` = encoder trunk + the multi-scale DiT. **done, tested; not yet compared in a run**
   (smoke: it trains, 0.26 s/step against 0.15 for the DiT without the trunk on the small molecules)
9. the full-size model, fragments as the conformers, the decoder side of the trunk (a structure in one frame)

## Overfitting experiments
`scripts/overfit_synth.py --model {equivariant,baseline,baseline_geom,basebaseline} --dataset {synthetic,protein}`; queues of runs on both GPUs: `scripts/run_queue.sh scripts/queue_*.txt 2`;
the table of all runs: `python scripts/summarize.py logs/overfit_synth_*.log`. `pytest tests -q`: 150 tests (equivariance under rotations, reflections and translations, an independent frame per conformer, parity rules, padding, data).

# Pomelo

An all-atom co-folding model that is **O(3)-equivariant by construction** (rotations, translations and reflections) with **typed features** (e3nn kinds: scalar `0e`,
pseudoscalar `0o`, polar vector `1o`, axial vector `1e`), so that handedness is handled by the architecture and not learned from data. The mirror image of the input (the
noisy structure and every reference conformer) gives the mirror image of the output; the hand of a molecule is the hand of its reference conformer (no stereo label is read).

**One DiT trunk without a pair representation** (SimpleFold's idea: the pair stack costs most of the FLOPs; the capacity goes into a larger token trunk). Everything outside
the network is PhysDock's, unchanged: its features and data module, sigma_data, the EDM diffusion and sampler, the 48 noisy samples, `PhysDockLoss`, the Lightning module
(AdamW, the AlphaFold 3 schedule, EMA, checkpoints). The previous two-branch design (encoder trunk with a typed Pairformer + DiT) is in `old/two_branch/`.

## The network

- **Condition embedder** (`pomelo/models/embedders.py`), once per structure, frame-free (every conformer's reference positions are in a random frame of their own):
  atoms `0e` from `ref_feat` (without the positions and the RDKit chirality tag) and the distance to the conformer centre; atoms `0o` = V_ref x linear(0e), V_ref the
  signed volume of the first three bonded neighbours in the reference conformer; tokens `0e` from target / key-residue / pocket features and the pooled atoms, `0o` from
  the pooled atom `0o`; the pair features (relative position, bonds, ligand pair features, template distogram; reference distances inside a conformer for the atom pairs)
  only as **static attention biases** of every block, shared by all noisy samples.
- **DiT** (`pomelo/models/dit.py`): atom DiT (3) -> pool to tokens -> token trunk (12) -> unpool -> atom DiT (3), every stream all four kinds; the vectors come from the
  noisy sample (relative vectors, cross products, the signed volume V_x and V_ref x V_x). Every block: GeometricProduct (the exchange between the kinds: all l <= 1
  products and the signed volume), SharedBiasAttention (q.k over all kinds, the static bias, a distance term - w_h(t)|x_i - x_j|^2, direction messages; one fused kernel
  call), TypedTransition, each behind a typed AdaLN-Zero of PhysDock's timestep embedding.
- `pomelo/configs.py`: presets `mini` (the scale of PhysDock's mini_p2: c_a 128, c_s 512 even channels, 69.5M parameters) and `tiny`.

## Running

```shell
N=100 STEPS=3000 LR=3e-4 sh scripts/train/scale.sh                     # PhysDock's overfitting smoke test (scripts/overfit/scale.sh) with Pomelo: logs/train/scale_100_lr3e-4_mini
python -I scripts/train/train.py model=mini_p2 +pomelo=mini ...          # any override of PhysDock's configs/train.yaml; +pomelo={preset:mini,n_token_blocks:8}
pytest tests -q                                                         # O(3), frame-free conditions, the hand, padding, the PhysDock training step
```

`PHYSDOCK_ROOT` (default: the parent of this directory, else /2024133006/zkx_temp/PhysDock) is the PhysDock checkout whose code, configs and data are used.

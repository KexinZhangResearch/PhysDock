# Pomelo

An all-atom co-folding model that is **O(3)-equivariant by construction** (rotations, translations and reflections) with **typed features**
(kinds named as in e3nn: scalar `0e`, pseudoscalar `0o`, polar vector `1o`, axial vector `1e`), so that handedness is handled by the architecture and not learned from data.
Mirroring the input (the noisy structure and the reference conformers) mirrors the output; a pseudoscalar (`0o`) can only enter an attention logit
as a product of two. The kinds say how a quantity transforms, not where it lives: single, pair and atom representations can hold any of them.

Torch only for now (no Triton kernels yet). The layout follows PhysDock.

## Repository layout

```
pomelo/
  configs.py          Config: the size and settings of the model
  models/
    model.py          Denoiser (encoder trunk + DiT), the EDM preconditioning
    encoders.py       InputEmbedder (the conformer features), EncoderTrunk (atom stage inside every conformer + Pairformer on the tokens)
    decoders.py       PomeloDiT: the multi-scale decoder, atoms -> tokens -> atoms, conditioned on the 12 outputs of the encoder, no coordinate augmentation
    diffusion.py      EDM noise, the Karras sampler          loss.py   x_mse and smooth lDDT
    baselines.py      PhysDock-like models without any symmetry (for comparison; trained with random rotations)
    primitives/       the typed layers, bottom up:
        geometry.py     the kinds, their products (dot, cross, signed volume, scale) and the O(3) checker
        irreps.py       feature sets {kind: tensor}: TypedLinear, TypedRMSNorm, Gate, GeometricProduct, TypedPool, add / mask_rows / mask_pairs
        attentions.py   TypedAttention (single representation, pair messages), TypedTriangleAttention
        frames.py       the transport of the vectors of a conformer from its reference frame into the frame of the noisy structure
        triangle.py     directions between tokens, PairFromSingle, TypedTriangleMultiplication
        transformers.py TypedTransition, SingleBlock, PairBlock, TrunkBlock
    kernels/          (empty: where Triton kernels would go)
  data/
    molecule.py       systems as dicts of tensors with PhysDock's feature names (ref_*, atom_to_token, atom_to_conformer, restype, ...), the stereo table of the evaluation;
                      synthetic molecules (RDKit; ligands: one token per atom, one conformer); combine()
    protein.py        windows of real proteins of the PhysDock training set (a token and a conformer per residue), PDB output
    proteins_10.json  the ten proteins of the overfitting experiments (always the same ones)
  training/
    overfit.py        the loop of the overfitting experiments        metrics.py   RMSD, hand agreement of sampled structures
  utils/
    tensor_utils.py   rotations, superposition, radial basis functions        checks.py   the O(3) checks every layer has to pass
scripts/
  overfit/            overfit_synth.py, run_queue.sh + queue_*.txt (runs on both GPUs), summarize.py (one table of all runs)
  infer/              sample_pdb.py: sample structures of a trained model, write PDB files
  assets/             draw_architecture.py -> assets/architecture.png
tests/                150 tests: equivariance under rotations, reflections and translations, an independent frame per conformer, parity rules, padding, data
assets/               architecture.png
logs/ outputs/        run logs and weights, PDB files (not tracked)
```

## What is in the model

As in PhysDock: an encoder from the conformer features, a multi-scale DiT decoder. Tokens as in PhysDock (a standard residue is one token and one conformer; every atom of a ligand is one
token, the ligand one conformer). No fragments, and **no stereo label is an input**: the hand of a molecule is the hand of its reference conformer (the stereo table is for the evaluation).

- **InputEmbedder**: the static even features: atoms (element, charge, aromatic, hybridisation, degree, hydrogens, ring), atom pairs (graph distance and bond type inside a conformer),
  tokens (pooled atoms, restype), token pairs (graph distance inside a conformer, AF3's relative position).
- **EncoderTrunk**: the reference positions as vectors (`ref - conformer centre`, `ref - mean of the bonded neighbours`, their cross product), the reference distances inside a conformer,
  the signed volumes of the pairs (`0o`); atom blocks inside a conformer, atom pairs from the atoms, typed pooling to the tokens, then `TrunkBlock`s (single -> pair, triangle
  multiplication out / in, triangle attention start / end, transition, single block). Every conformer is in a frame of its own, so the vector terms act inside a conformer only.
  Output: the 12 conditions `a 0e 0o 1o 1e`, `ap 0e 0o` (atoms), `s 0e 0o 1o 1e`, `z 0e 0o` (tokens).
- **PomeloDiT**: the conditions (vectors moved into the frame of the noisy structure by the transport of their conformer; its determinant, +-1, says whether the noisy conformer has the hand
  of its reference) + the noisy coordinates (relative vectors, the RBF of the atom and the token distances); atom blocks, pooling to the tokens, token blocks, unpooling, atom blocks; an output
  head of polar vectors. Parity-equivariant, so no augmentation.
- Loss `4 * x_mse + 4 * smooth lDDT` (no superposition: the output is in the frame of the input), EDM noise, Karras sampler.

## Experiments

```shell
python scripts/overfit/overfit_synth.py --model {equivariant,baseline,baseline_geom,basebaseline} --dataset {synthetic,protein} --steps 10000
sh scripts/overfit/run_queue.sh scripts/overfit/queue_node0.txt 2          # a queue of runs, 2 at a time on this machine's GPU
python scripts/overfit/summarize.py logs/overfit_synth_*.log                # the table of all runs
python scripts/infer/sample_pdb.py --weights logs/overfit_synth_<tag>.pt --protein 3ibd_A --n 4 --out outputs
pytest tests -q
```

`PHYSDOCK_ROOT` (default: the parent of this directory) is where `data/v1` of PhysDock is read from. Pass the same `--seed` and the run is reproducible across processes
(the initialisation no longer depends on the hash seed of the process: the order in which the weights are created is fixed).

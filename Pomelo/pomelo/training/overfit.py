"""The loop of the overfitting experiments (scripts/overfit/overfit_synth.py): can a model memorise a handful of structures? Loss 4 * x_mse + 4 * smooth lDDT (as PhysDock2), Adam with
a warm-up; the models without a symmetry are trained with a random rotation of every noisy sample (the equivariant one is not). Afterwards: one denoising step from the true structure at
six noise levels (the error of D, the hand of its stereocentres, the effect of giving the enantiomer's reference conformers) and sampled structures (rotation-aligned RMSD, RMSD to the
mirror image, the hand). The stereo labels are used by the evaluation only."""
import argparse
import time
from pathlib import Path

import torch

from pomelo.configs import Config
from pomelo.data import molecule as D
from pomelo.data import protein as PR
from pomelo.models import diffusion as F
from pomelo.models.baselines import BasebaselineDenoiser, BaselineDenoiser
from pomelo.models.loss import smooth_lddt, x_mse
from pomelo.models.model import Denoiser
from pomelo.training.metrics import chirality_agreement, rmsd
from pomelo.utils.tensor_utils import random_rotations

ROOT = Path(__file__).resolve().parents[2]                 # Pomelo/ (logs/ is below it)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--blocks", type=int, nargs=2, default=[1, 2], help="decoder: atom blocks (before and after), token blocks")
    ap.add_argument("--sample_steps", type=int, default=40)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--dataset", choices=["synthetic", "protein"], default="synthetic", help="synthetic: 11 small molecules; protein: the 10 real proteins of pomelo/data/proteins_10.json (128 residues, ~1000 atoms)")
    ap.add_argument("--model", choices=["equivariant", "baseline", "baseline_geom", "basebaseline"], default="equivariant",
                    help="equivariant: Pomelo (encoder trunk + DiT); baseline: PhysDock-like DiT, not equivariant, trained with a random rotation of every noisy sample; baseline_geom: "
                         "also the RBF of the noisy distances in the pair bias; basebaseline: a PhysDock-like Pairformer trunk as well (2 standard blocks), also not equivariant")
    ap.add_argument("--c_atom", type=int, default=None, help="widths of the baselines, chosen for the parameter count of the equivariant model: 192 / 200, with the trunk 168 / 208")
    ap.add_argument("--c_token", type=int, default=None)
    return ap.parse_args(argv)


def build_model(args, cfg: Config) -> torch.nn.Module:
    if args.model == "equivariant":
        return Denoiser(cfg)
    if args.model == "basebaseline":
        return BasebaselineDenoiser(cfg, args.c_atom or 168, args.c_token or 208, c_s=64, c_z=32, n_trunk=2)
    return BaselineDenoiser(cfg, args.c_atom or 192, args.c_token or 200, geometry=args.model == "baseline_geom")


def train(den, mols, names, args, dev, augment: bool):
    opt = torch.optim.Adam(den.parameters(), lr=args.lr, betas=(0.9, 0.95))
    hist, t0 = [], time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (step + 1) / args.warmup)
        mol = mols[names[torch.randint(len(names), ()).item()]]
        x0 = mol["x_gt"] / F.SIGMA_DATA
        x0 = x0 - x0.mean(0)
        sigma = F.training_sigma(args.batch, dev)
        x0s = x0[None] @ random_rotations(args.batch, device=dev).transpose(-1, -2) if augment else x0                  # the augmentation of the model without a symmetry
        x_t = x0s + sigma[:, None, None] * F.centred_noise((args.batch, *x0.shape), device=dev)
        pred = den(mol, x_t, sigma)
        lx, ls = x_mse(pred, x0s, sigma), smooth_lddt(pred, x0s)
        loss = 4 * lx + 4 * ls
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(den.parameters(), 10.0)
        opt.step()
        hist.append([float(lx), float(ls), float(gn)])
        if (step + 1) % 100 == 0:
            h = torch.tensor(hist[-100:]).mean(0)
            print(f"step {step + 1:5d}  x_mse {h[0]:.4f}  slddt {h[1]:.4f}  gnorm {h[2]:.1f}  {(time.time() - t0) / (step + 1):.3f} s/step", flush=True)



def evaluate(den, mols, names, args, dev) -> None:
    den.eval()
    print("\none denoising step from the true structure + noise of the level sigma (normalised units; 1 = 16 A), all molecules:")
    print(f"{'sigma':>6s} {'rmsd of D (A)':>14s} {'x_mse':>8s} {'hand ok':>8s} {'enantiomer refs change D by (A)':>32s} {'hand ok (enantiomer refs)':>26s}")
    with torch.no_grad():
        for sg in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
            r, e, h, fl, hf = [], [], [], [], []
            for name in names:
                mol = mols[name]
                flipped = dict(mol)
                flipped["ref_pos"] = mol["ref_pos"] * torch.tensor([-1.0, 1.0, 1.0], device=dev)          # the reference conformers of the enantiomer
                x0 = mol["x_gt"] / F.SIGMA_DATA
                x0 = x0 - x0.mean(0)
                sig = torch.full((8,), sg, device=dev)
                x_t = x0[None] + sig[:, None, None] * F.centred_noise((8, *x0.shape), device=dev)
                d, df = den(mol, x_t, sig), den(flipped, x_t, sig)
                r.append(float((d - x0).square().sum(-1).mean().sqrt()) * F.SIGMA_DATA)
                e.append(float(x_mse(d, x0, sig)))
                fl.append(float((d - df).square().sum(-1).mean().sqrt()) * F.SIGMA_DATA)
                h.append(sum(chirality_agreement(x * F.SIGMA_DATA, mol["stereo_center"], mol["stereo_nbrs"], mol["stereo_sign"]) for x in d) / 8)
                hf.append(sum(chirality_agreement(x * F.SIGMA_DATA, mol["stereo_center"], mol["stereo_nbrs"], mol["stereo_sign"]) for x in df) / 8)
            m = lambda v: sum(v) / len(v)
            print(f"{sg:6.2f} {m(r):14.3f} {m(e):8.4f} {m(h):8.2f} {m(fl):32.3f} {m(hf):26.2f}")
    print("\nsampling (no augmentation), 4 structures per molecule, rotation-aligned RMSD in Angstrom")
    print(f"{'molecule':15s} {'atoms':>5s} {'rmsd':>6s} {'best':>6s} {'mirror':>7s} {'hand ok':>8s}")
    den.eval()
    tot = []
    for name in names:
        mol = mols[name]
        xs = F.sample(den, mol, n=4, steps=args.sample_steps)
        r = [rmsd(x, mol["x_gt"]) for x in xs]
        rm = [rmsd(x, mol["x_gt"] * torch.tensor([-1.0, 1.0, 1.0], device=dev)) for x in xs]
        ok = [chirality_agreement(x, mol["stereo_center"], mol["stereo_nbrs"], mol["stereo_sign"]) for x in xs]
        tot.append((sum(r) / 4, sum(ok) / 4))
        print(f"{name:15s} {mol['x_gt'].shape[0]:5d} {sum(r) / 4:6.2f} {min(r):6.2f} {sum(rm) / 4:7.2f} {sum(ok) / 4:8.2f}")
    print(f"mean rmsd {sum(t[0] for t in tot) / len(tot):.2f}, hand agreement {sum(t[1] for t in tot) / len(tot):.2f}")



def main(argv=None) -> None:
    args = parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    dataset = D.make_dataset(args.names) if args.dataset == "synthetic" else PR.make_dataset()
    mols = {k: {a: b.to(dev) for a, b in v.items()} for k, v in dataset.items()}
    names = list(mols)
    cfg = Config(n_atom_blocks=args.blocks[0], n_token_blocks=args.blocks[1])
    den = build_model(args, cfg).to(dev)
    augment = not args.model.startswith("equivariant")
    print(f"{args.dataset}: {len(names)} molecules, {sum(p.numel() for p in den.parameters()) / 1e6:.2f}M parameters, device {dev}", flush=True)
    train(den, mols, names, args, dev, augment)
    (ROOT / "logs").mkdir(exist_ok=True)
    torch.save(den.state_dict(), ROOT / "logs" / f"overfit_synth_{args.tag}.pt")
    evaluate(den, mols, names, args, dev)

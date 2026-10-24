"""The model of Pomelo with the interface of PhysDock's model (PhysDock/models/model.py), so that PhysDock's training (PhysDockLitModule, the data module, PhysDockLoss,
the EMA, the optimiser and its schedule) and its sampler run it unchanged: forward(batch) -> {x_gt, x_denoised, x_hat, t_hat, stream_rms}, sample_diffusion(batch, ...).

What is PhysDock's: the diffuser (PhysDock.models.diffusion, built from the `diffusion` section of the model config: EDM, the training noise ln t ~ N(-1.2, 1.5^2), the
noising of num_augmentation_sample samples with PhysDock's random rotation and translation, the preconditioning, the AF3 sampler), sigma_data. What is Pomelo's: the
condition embedder (PhysDock.pomelo.embedders) and the DiT (PhysDock.pomelo.dit) in place of PhysDock's encoder and decoder. The random rotation of the noising is kept: for an
equivariant network it changes nothing, and it keeps the training process the same as PhysDock's.
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from PhysDock.pomelo.configs import preset
from PhysDock.pomelo.dit import PomeloDiT
from PhysDock.pomelo.embedders import ConditionEmbedder


class Pomelo(nn.Module):
    """Pomelo(**model_config): model_config is PhysDock's (configs/model/mini_p2.yaml: num_augmentation_sample, diffusion, sigma_data; its encoder / decoder sections are not
    used) plus `pomelo`: a preset name of PhysDock.pomelo.configs ("mini", "tiny") or a dict of Config fields with an optional "preset"."""

    def __init__(self, num_augmentation_sample: int, diffusion: dict, sigma_data: float, pomelo="mini", backend: str = "torch", **physdock_only) -> None:
        super().__init__()
        from PhysDock.models.diffusion import build_diffuser
        self.cfg = preset(pomelo)
        self.num_augmentation_sample, self.sigma_data = num_augmentation_sample, sigma_data
        self.diffuser = build_diffuser(diffusion)
        self.diffusion_conditioning = ConditionEmbedder(self.cfg)
        self.dit = PomeloDiT(self.cfg)
        if backend not in ("torch", "triton"):
            raise ValueError(f"backend must be torch or triton, got {backend!r}")
        self.kernel_backend = backend                       # triton: PhysDock.pomelo.kernels (the model config's backend, as PhysDock's model.backend=triton)
        if backend == "triton":
            self.set_attention_backend("triton", "triton")

    def forward(self, batch):
        """Training step on one (padded) structure. Returns (normalised by sigma_data): x_gt [N_atom, 3], x_denoised [S, N_atom, 3], x_hat [S, N_atom, 3], t_hat [S],
        stream_rms [3] (the RMS of the atom and token conditions, 0: no pair representation)"""
        x_gt = batch["x_gt"] / self.sigma_data
        conds = self.dit.condition(batch, self.diffusion_conditioning(batch))
        z, t = self.diffuser.noising(x_gt, batch["atom_mask"], self.num_augmentation_sample)
        x_denoised = self.diffuser.denoise(self.dit, batch, z, t, conds)
        return {"x_gt": x_gt, "x_denoised": x_denoised, "x_hat": z, "t_hat": self.diffuser.noise_level(t), "stream_rms": self.stream_rms(conds)}

    @torch.no_grad()
    def stream_rms(self, c) -> torch.Tensor:
        am, tm = c["atom_mask"][:, None], c["token_mask"][:, None]
        rms = lambda x, m: ((x.float() ** 2 * m).sum() / (m.sum() * x.shape[-1])).sqrt()
        return torch.stack([rms(c["a"]["0e"], am), rms(c["s"]["0e"], tm), torch.zeros((), device=am.device)])

    @torch.no_grad()
    def sample_diffusion(self, batch, num_sample: int = 5, steps: int = 200, **guidance) -> torch.Tensor:
        """PhysDock's sampler (AF3 Alg. 18 with the settings of the diffusion config) -> [num_sample, N_atom, 3] Angstrom. The ligand guidance of PhysDock (MMFF relaxation,
        reference poses) is not wired in: its arguments are accepted and ignored."""
        conds = self.dit.condition(batch, self.diffusion_conditioning(batch))
        x = self.diffuser.sample(self.dit, batch, conds, batch["atom_mask"], num_sample, steps)
        return x * self.sigma_data

    def set_attention_backend(self, atom: str = "math", token: str = "math") -> None:
        """the backend of the atom and of the token attentions: an SDPA backend of PhysDock.pomelo.primitives.attentions.BACKENDS or "triton" """
        for blk in [*self.dit.atom_enc, *self.dit.atom_dec]:
            blk.attention.backend = atom
        for blk in self.dit.token_blocks:
            blk.attention.backend = token

    def prepare_training(self, compile: bool = True, checkpoint_blocks: bool = False) -> None:
        """every DiT block: torch.compile (static shapes: PhysDock pads the crop), optionally inside activation checkpointing (less memory, one more forward), used while
        gradients are on (validation and sampling, other numbers of samples, stay eager); the condition embedder compiled as one region"""
        def wrap(module, compiled_forward):
            eager = module.forward

            def forward(*args):
                if not torch.is_grad_enabled():
                    return eager(*args)
                return checkpoint(compiled_forward, *args, use_reentrant=False) if checkpoint_blocks else compiled_forward(*args)
            module.forward = forward
        if compile:
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        for blk in [*self.dit.atom_enc, *self.dit.token_blocks, *self.dit.atom_dec]:
            wrap(blk, torch.compile(blk.forward, dynamic=False) if compile else blk.forward)
        if compile:
            emb = self.diffusion_conditioning
            eager, compiled = emb.forward, torch.compile(emb.forward, dynamic=False)
            emb.forward = lambda b: compiled(b) if torch.is_grad_enabled() else eager(b)

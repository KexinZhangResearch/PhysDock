"""EDM training noise, the loss (x_mse and smooth lDDT, as PhysDock2 / AF3), a sampler, and the metrics of a sampled structure. Coordinates are normalised by sigma_data = 16 Angstrom.

The structure and the noise are centred (the translation is not part of the problem); there is no superposition (Kabsch) of the target on the prediction in the loss: the network is
equivariant, the prediction is in the frame of the input.
"""
import torch


SIGMA_DATA = 16.0


def centred_noise(shape, generator=None, device=None, dtype=torch.float32) -> torch.Tensor:
    e = torch.randn(*shape, generator=generator, device=device, dtype=dtype)
    return e - e.mean(-2, keepdim=True)


def training_sigma(b: int, device=None, mean: float = -1.2, std: float = 1.5, generator=None) -> torch.Tensor:
    return torch.exp(torch.randn(b, generator=generator, device=device) * std + mean)


@torch.no_grad()
def sample(denoiser, mol, n: int = 4, steps: int = 40, sigma_max: float = 20.0, sigma_min: float = 0.02, rho: float = 7.0, generator=None) -> torch.Tensor:
    """Euler steps of the probability-flow ODE along the Karras schedule, from centred noise; no augmentation (the network is equivariant). Returns [n, N, 3] in Angstrom"""
    dev = next(denoiser.parameters()).device
    na = mol["z"].shape[0]
    i = torch.arange(steps, device=dev, dtype=torch.float32)
    sig = (sigma_max ** (1 / rho) + i / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sig = torch.cat([sig, torch.zeros(1, device=dev)])
    x = sig[0] * centred_noise((n, na, 3), generator, dev)
    for k in range(steps):
        d = denoiser(mol, x, sig[k].expand(n))
        x = x + (sig[k + 1] - sig[k]) * (x - d) / sig[k]
    return x * SIGMA_DATA

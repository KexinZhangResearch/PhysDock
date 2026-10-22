"""EDM training noise and the sampler, as PhysDock's (PhysDock.models.diffusion, edm_ve): coordinates normalised by sigma_data = 16 Angstrom, ln(sigma) ~ N(-1.2, 1.5^2) in training,
the sampler of AF3 Alg. 18 with PhysDock's settings.

The training noise is centred: PhysDock adds plain noise and centres the noisy structure, which is the same (the target is centred); there is no superposition (Kabsch) of the target on
the prediction in the loss: the network is equivariant, the prediction is in the frame of the input. The random rotation of the sampler changes nothing for an equivariant network
(it is there for the baselines, and to be the same sampler).
"""
import torch

from pomelo.utils.tensor_utils import random_rotations


SIGMA_DATA = 16.0


def centred_noise(shape, generator=None, device=None, dtype=torch.float32) -> torch.Tensor:
    e = torch.randn(*shape, generator=generator, device=device, dtype=dtype)
    return e - e.mean(-2, keepdim=True)


def training_sigma(b: int, device=None, mean: float = -1.2, std: float = 1.5, generator=None) -> torch.Tensor:
    return torch.exp(torch.randn(b, generator=generator, device=device) * std + mean)


@torch.no_grad()
def sample(denoiser, mol, n: int = 4, steps: int = 40, sigma_max: float = 160.0, sigma_min: float = 4e-3, rho: float = 7.0, gamma_0: float = 0.8,
           gamma_min: float = 1.0 / SIGMA_DATA, noise_scale: float = 1.003, step_scale: float = 1.5, ode_step_scale: float = 1.0, trans_scale: float = 1.0 / SIGMA_DATA,
           generator=None) -> torch.Tensor:
    """PhysDock's sampler (AF3 Alg. 18, the settings of PhysDock's configs/model/medium.yaml and infer.yaml): the Karras schedule sigma_max -> sigma_min (normalised units: 160 is
    2560 A, 4e-3 is 0.064 A), before every step the centre, a random rotation and a random translation of std trans_scale (1 A); churn gamma_0 while the noise level is above gamma_min
    (1 A), the noise of the churn times noise_scale, the Euler step times step_scale (times ode_step_scale without churn). The prior is plain Gaussian noise. Returns [n, N, 3] in
    Angstrom"""
    dev = next(denoiser.parameters()).device
    na = mol["x_gt"].shape[0]
    i = torch.arange(steps, device=dev, dtype=torch.float32)
    sig = (sigma_max ** (1 / rho) + i / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sig = torch.cat([sig, torch.zeros(1, device=dev)])
    x = sig[0] * torch.randn(n, na, 3, generator=generator, device=dev)
    for k in range(steps):
        x = x - x.mean(-2, keepdim=True)
        x = x @ random_rotations(n, generator, dev).transpose(-1, -2) + trans_scale * torch.randn(n, 1, 3, generator=generator, device=dev)
        t = sig[k]
        if t > gamma_min:
            t_hat, scale = t * (1 + gamma_0), step_scale
            x = x + noise_scale * torch.sqrt(t_hat ** 2 - t ** 2) * torch.randn(x.shape, generator=generator, device=dev)
        else:
            t_hat, scale = t, ode_step_scale
        d = denoiser(mol, x, t_hat.expand(n))
        x = x + scale * (sig[k + 1] - t_hat) * (x - d) / t_hat
    return x * SIGMA_DATA

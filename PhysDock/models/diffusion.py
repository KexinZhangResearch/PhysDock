"""Diffusion processes (noising, preconditioning, sampler) around the decoder network.

Two processes (``config.diffusion.type``); sigma = sigma_data, x clean, e ~ N(0, I), coordinates in Angstrom:

* ``"edm_ve"`` (released parameters): z = x + t * e, t = noise level in Angstrom,
  x_pred = c_skip * z + c_out * net(z / sqrt(t^2 + sigma^2), noise_input(t)).
* ``"linear"`` (PhysDock2 flow matching): z = (1 - t) * x + t * sigma * e, t in (0, 1),
  x_pred = z - t * sigma * net(z / sigma, t).

Both share the sampler of AF3 Alg. 18 (augmentation, churn, Euler step along (z - x_pred) / t). Schedules and the
training noise distribution are defined on the unit-free noise level s (EDM t = sigma * s, linear t = s / (1 + s)).
"""
import dataclasses
from dataclasses import dataclass

import torch

from PhysDock.utils.tensor_utils import centre_random_augmentation


@dataclass(frozen=True)
class KarrasNoiseSchedule:
    """Noise levels (units of sigma_data) (s_max^(1/rho) + i/(N-1) (s_min^(1/rho) - s_max^(1/rho)))^rho, then 0."""
    s_min: float = 4e-4
    s_max: float = 160.0
    rho: float = 7.0

    def __call__(self, num_steps: int) -> torch.Tensor:
        """num_steps (>= 2) -> [num_steps + 1] float32 CPU tensor"""
        if num_steps < 2:
            raise ValueError(f"num_steps must be at least 2, got {num_steps}")
        i = torch.arange(num_steps, dtype=torch.float32)
        levels = (self.s_max ** (1 / self.rho) + i / (num_steps - 1) * (
                self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))) ** self.rho
        return torch.cat([levels, torch.zeros_like(levels[:1])])


@dataclass(frozen=True)
class LogNormalTimeSampler:
    """Training noise levels (units of sigma_data): ln(s) ~ N(mean, std^2)."""
    mean: float = -1.2
    std: float = 1.5

    def __call__(self, shape, dtype, device) -> torch.Tensor:
        return torch.exp(torch.normal(0, 1, shape, device=device, dtype=dtype) * self.std + self.mean)


@dataclass(frozen=True)
class SamplerSettings:
    """AF3 sampler settings: churn gamma_0 above gamma_min (Angstrom), noise scale, step scales, translation std."""
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    ode_step_scale: float = 1.0
    trans_scale: float = 1.0


class Diffuser:
    """Shared training noising, denoising and sampling; subclasses define the process."""

    def __init__(self, sigma_data: float = 16.0, time_sampler: LogNormalTimeSampler = LogNormalTimeSampler(),
                 noise_schedule: KarrasNoiseSchedule = KarrasNoiseSchedule(),
                 sampler: SamplerSettings = SamplerSettings(), trans_scale: float = 1.0) -> None:
        self.sigma_data = sigma_data
        self.time_sampler = time_sampler
        self.noise_schedule = noise_schedule
        self.sampler = sampler
        self.trans_scale = trans_scale

    # process-specific, see the subclasses
    def to_time(self, s): raise NotImplementedError
    def noise_level(self, t): raise NotImplementedError
    def diffuse(self, x, e, t): raise NotImplementedError
    def prior(self, shape, t, dtype, device): raise NotImplementedError
    def churn(self, z, t, gamma_0, noise_scale): raise NotImplementedError
    def scale_input(self, z, t): raise NotImplementedError
    def noise_input(self, t): raise NotImplementedError
    def pred_x(self, pred, z, t): raise NotImplementedError

    def denoise(self, decoder, batch, z: torch.Tensor, t: torch.Tensor, conds) -> torch.Tensor:
        """z: [S, N_atom, 3] noisy, t: [S] native time -> x_pred [S, N_atom, 3]"""
        return self.pred_x(decoder(batch, self.scale_input(z, t), self.noise_input(t), conds), z, t)

    @torch.no_grad()
    def noising(self, x_gt: torch.Tensor, atom_mask: torch.Tensor, num_samples: int):
        """x_gt: [N_atom, 3], atom_mask: [N_atom] -> (z [S, N_atom, 3] noised + randomly rotated/shifted, t [S])"""
        t = self.to_time(self.time_sampler((num_samples,), x_gt.dtype, x_gt.device))
        x = x_gt[None].repeat([num_samples, 1, 1])
        e = torch.normal(0, 1, size=x.shape, dtype=x.dtype, device=x.device)
        return centre_random_augmentation(self.diffuse(x, e, t), atom_mask, s=self.trans_scale), t

    @torch.no_grad()
    def sample(self, decoder, batch, conds, atom_mask: torch.Tensor, num_samples: int, num_steps: int,
               sampler: SamplerSettings = None, noise_schedule: KarrasNoiseSchedule = None, guidance=None) -> torch.Tensor:
        """Reverse diffusion (AF3 Alg. 18).

        Args:
            decoder: network called as decoder(batch, x_in, t_in, conds)
            batch: features; x_gt [N_atom, 3] gives shape, dtype and device
            conds: encoder output
            atom_mask: [N_atom]
            num_samples, num_steps: number of structures, number of steps (>= 2)
            sampler, noise_schedule: override the configured ones
            guidance: optional guidance(x_hat, x_denoised, noise_level) -> new x_denoised or None

        Returns:
            [num_samples, N_atom, 3]
        """
        x_gt = batch["x_gt"]
        dtype, device = x_gt.dtype, x_gt.device
        cfg = sampler if sampler is not None else self.sampler
        schedule = noise_schedule if noise_schedule is not None else self.noise_schedule
        ts = self.to_time(schedule(num_steps)).to(device).to(dtype)

        x = self.prior((num_samples, *x_gt.shape[-2:]), ts[0], dtype, device)
        for t_cur, t_next in zip(ts[:-1], ts[1:]):
            x = centre_random_augmentation(x, atom_mask, s=cfg.trans_scale)
            level = self.noise_level(t_cur)
            if level > cfg.gamma_min:
                scale = cfg.step_scale
                x_hat, t_hat = self.churn(x, t_cur, cfg.gamma_0, cfg.noise_scale)
            else:
                scale = cfg.ode_step_scale
                x_hat, t_hat = x, torch.full([num_samples], fill_value=t_cur, device=device, dtype=dtype)
            x_denoised = self.denoise(decoder, batch, x_hat, t_hat, conds)
            if guidance is not None:
                guided = guidance(x_hat, x_denoised, level)
                if guided is not None:
                    x_denoised = guided
            drift = (x_hat - x_denoised) / t_hat[..., None, None]
            x = x_hat + scale * (t_next - t_hat)[..., None, None] * drift
        return x


class EDMVEDiffuser(Diffuser):
    """EDM variance exploding: z = x + t * e. noise_conditioning "legacy" = t * ln(t / sigma) / 4 (released
    parameters), "af3" = ln(t / sigma) / 4."""

    def __init__(self, *args, noise_conditioning: str = "legacy", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if noise_conditioning not in ("af3", "legacy"):
            raise ValueError(f"noise_conditioning must be 'af3' or 'legacy', got {noise_conditioning!r}")
        self.noise_conditioning = noise_conditioning

    def to_time(self, s):
        return self.sigma_data * s

    def noise_level(self, t):
        return t

    def diffuse(self, x, e, t):
        return x + e * t[..., None, None]

    def prior(self, shape, t, dtype, device):
        return t * torch.normal(0, 1, size=shape, dtype=dtype, device=device)

    def churn(self, z, t, gamma_0, noise_scale):
        """Raise the noise level from t to t * (1 + gamma_0); returns (z_hat, t_hat [S])."""
        t_hat = torch.full([z.shape[0]], fill_value=t * (gamma_0 + 1), device=z.device, dtype=z.dtype)
        noise = torch.normal(0, 1, size=z.shape, dtype=z.dtype, device=z.device)
        return z + noise_scale * noise * torch.sqrt(t_hat ** 2 - t ** 2)[..., None, None], t_hat

    def scale_input(self, z, t):
        return z * (1 / (torch.sqrt(t[:, None, None] ** 2 + self.sigma_data ** 2)))

    def noise_input(self, t):
        c_noise = torch.log(t / self.sigma_data) / 4.0
        return c_noise if self.noise_conditioning == "af3" else t * c_noise

    def pred_x(self, pred, z, t):
        c_skip = (self.sigma_data ** 2 / (self.sigma_data ** 2 + t ** 2))[:, None, None]
        c_out = (self.sigma_data * t / torch.sqrt(self.sigma_data ** 2 + t ** 2))[:, None, None]
        return c_skip * z + c_out * pred


class LinearDiffuser(Diffuser):
    """Flow matching: z = (1 - t) * x + t * sigma * e; the network predicts the velocity e - x / sigma."""

    def to_time(self, s):
        return s / (1 + s)

    def noise_level(self, t):
        return self.sigma_data * (t / (1 - t))

    def diffuse(self, x, e, t):
        return (1 - t)[..., None, None] * x + (self.sigma_data * t)[..., None, None] * e

    def prior(self, shape, t, dtype, device):
        return (self.sigma_data * t) * torch.normal(0, 1, size=shape, dtype=dtype, device=device)

    def churn(self, z, t, gamma_0, noise_scale):
        """Raise the noise level s = t / (1 - t) to s * (1 + gamma_0) (no-op for gamma_0 = 0)."""
        if gamma_0 == 0:
            return z, torch.full([z.shape[0]], fill_value=t, device=z.device, dtype=z.dtype)
        s = t / (1 - t)
        s_hat = s * (1 + gamma_0)
        t_hat = s_hat / (s_hat + 1)
        noise = torch.normal(0, 1, size=z.shape, dtype=z.dtype, device=z.device)
        z_hat = (z / (1 - t) + noise_scale * self.sigma_data * torch.sqrt(s_hat ** 2 - s ** 2) * noise) * (1 - t_hat)
        return z_hat, torch.full([z.shape[0]], fill_value=t_hat, device=z.device, dtype=z.dtype)

    def scale_input(self, z, t):
        return z / self.sigma_data

    def noise_input(self, t):
        return t

    def pred_x(self, pred, z, t):
        return z - (self.sigma_data * t)[:, None, None] * pred


def build_diffuser(config) -> Diffuser:
    """Diffuser from the ``diffusion`` section of PhysDockConfig (type "edm_ve" or "linear")."""
    kind = config.get("type", "edm_ve")
    kwargs = dict(sigma_data=config["sigma_data"], time_sampler=LogNormalTimeSampler(**config["time_sampler"]),
                  noise_schedule=KarrasNoiseSchedule(**config["noise_schedule"]),
                  sampler=SamplerSettings(**config["sampler"]), trans_scale=config["trans_scale"])
    if kind == "edm_ve":
        return EDMVEDiffuser(**kwargs, noise_conditioning=config["noise_conditioning"])
    if kind == "linear":
        return LinearDiffuser(**kwargs)
    raise ValueError(f"unknown diffusion type {kind!r}, expected 'edm_ve' or 'linear'")


def replace(settings, **changes):
    """Copy of a frozen settings dataclass with the non-None fields of ``changes`` replaced."""
    return dataclasses.replace(settings, **{k: v for k, v in changes.items() if v is not None})

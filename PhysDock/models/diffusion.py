"""Diffusion strategies of PhysDock and the AlphaFold 3 sampler.

The network (:class:`~PhysDock.models.decoders.AllAtomDecoder`) is a plain function of *preconditioned* inputs.
Everything around it lives in a :class:`Diffuser`, written after the diffusers of PhysDock2:

* ``noising``: ground truth -> noisy training sample.
* ``denoise``: noisy sample -> predicted clean structure ``x_pred``. This is where the preconditioning of the input,
  the noise conditioning of the network and the conversion of the network output into ``x_pred`` happen.
* ``sample``: reverse diffusion (AlphaFold 3 Algorithm 18: stochastic churn + Euler steps), with an optional guidance
  hook that may replace the predicted clean structure (used for the ligand physics guidance).

Two diffusion processes are available, selected with ``config.diffusion.type`` (``build_diffuser``). Both use the same
network, sampler and loss. ``x`` is the clean structure, ``e ~ N(0, I)``, ``sigma`` is the data scale ``sigma_data``
(16 Angstrom as in AlphaFold 3) and all coordinates ``z`` are in Angstrom:

``"edm_ve"`` (default; the released PhysDock parameters)
    EDM variance-exploding diffusion (Karras et al. 2022). Native time ``t`` is the noise level in Angstrom::

        z_t   = x + t * e
        net   = network(c_in * z_t, noise input),   c_in   = 1 / sqrt(t^2 + sigma^2)
        x     = c_skip * z_t + c_out * net,         c_skip = sigma^2 / (sigma^2 + t^2),  c_out = sigma t / sqrt(sigma^2 + t^2)

``"linear"`` (PhysDock2 ``LinearDiffuser``, flow matching / rectified flow)
    Straight-line interpolation between data and noise. Native time ``t`` is in (0, 1)::

        z_t   = (1 - t) * x + t * sigma * e
        net   = network(z_t / sigma, t)             # the network predicts the velocity e - x / sigma
        x     = z_t - t * sigma * net

Common to both: with the native time ``t`` the clean-structure estimate gives the ODE direction ``(z - x) / t`` and
an Euler step ``z_r = z_t + eta * (r - t) * (z - x) / t``. The unit-free *noise level* ``sqrt(NSR)`` (noise-to-signal
ratio) ties the two processes together: ``noise_level(t)`` is ``sigma`` times it, in Angstrom (``t`` itself for
EDM-VE, ``sigma * t / (1 - t)`` for linear). Schedule, training noise distribution, churn threshold, guidance
threshold and the loss weighting are all defined on this noise level, so they work unchanged for both processes;
``to_time`` maps a unit-free noise level to the native time of a diffuser.

Noise conditioning of the EDM-VE network (``noise_conditioning``):

* ``"af3"``: ``ln(t / sigma) / 4`` as in AlphaFold 3 and PhysDock2.
* ``"legacy"``: ``t * ln(t / sigma) / 4``. The released PhysDock parameters were trained with this extra factor
  ``t``; it must be kept to use them, so it is the default of the PhysDock configuration.
"""
from __future__ import annotations

import abc
import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Protocol, Tuple, Type

import torch

from PhysDock.data import TensorDict
from PhysDock.utils.tensor_utils import centre_random_augmentation

__all__ = [
    "KarrasNoiseSchedule", "LogNormalTimeSampler", "SamplerSettings", "DenoisingNetwork", "Guidance", "Diffuser",
    "EDMVEDiffuser", "LinearDiffuser", "DIFFUSERS", "build_diffuser",
]


# ======================================================================================
# Noise schedule, training noise distribution, sampler settings
# ======================================================================================
@dataclass(frozen=True)
class KarrasNoiseSchedule:
    """Noise levels of the sampler (Karras et al. 2022, eq. 5), in units of ``sigma_data``.

    ``t_i = (s_max^(1/rho) + i / (N - 1) * (s_min^(1/rho) - s_max^(1/rho)))^rho`` for ``i = 0 .. N - 1``,
    followed by a final ``0``. The values are computed in float32 on the CPU, so the schedule is identical on
    every device.

    Args:
        s_min: Smallest noise level (in units of ``sigma_data``). AlphaFold 3 uses ``4e-4``.
        s_max: Largest noise level (in units of ``sigma_data``).
        rho: Exponent controlling how the steps are spread; larger values concentrate steps at low noise.

    Raises:
        ValueError: If ``0 < s_min < s_max`` or ``rho > 0`` does not hold.
    """

    s_min: float = 4e-4
    s_max: float = 160.0
    rho: float = 7.0

    def __post_init__(self) -> None:
        if not 0 < self.s_min < self.s_max:
            raise ValueError(f"need 0 < s_min < s_max, got s_min={self.s_min}, s_max={self.s_max}")
        if self.rho <= 0:
            raise ValueError(f"rho must be positive, got {self.rho}")

    def __call__(self, num_steps: int) -> torch.Tensor:
        """Return the ``num_steps + 1`` noise levels, from ``s_max`` down to ``s_min`` and a final ``0``.

        Args:
            num_steps: Number of denoising steps, at least 2.

        Returns:
            Float32 CPU tensor of shape ``[num_steps + 1]``.

        Raises:
            ValueError: If ``num_steps < 2``.
        """
        if num_steps < 2:
            raise ValueError(f"num_steps must be at least 2, got {num_steps}")
        step_indices = torch.arange(num_steps, dtype=torch.float32)
        levels = (self.s_max ** (1 / self.rho) + step_indices / (num_steps - 1) * (
                self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))) ** self.rho
        return torch.cat([levels, torch.zeros_like(levels[:1])])


@dataclass(frozen=True)
class LogNormalTimeSampler:
    """Training noise distribution of EDM / AlphaFold 3: ``ln(noise level / sigma_data) ~ N(mean, std^2)``.

    PhysDock2 calls this the "time weighting scheme"; it only draws noise levels.

    Args:
        mean: Mean of ``ln(noise level / sigma_data)``.
        std: Standard deviation of ``ln(noise level / sigma_data)``.
    """

    mean: float = -1.2
    std: float = 1.5

    def __post_init__(self) -> None:
        if self.std < 0:
            raise ValueError(f"std must be non-negative, got {self.std}")

    def __call__(self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Draw noise levels in units of ``sigma_data``.

        Args:
            shape: Shape of the result.
            dtype: Floating point dtype.
            device: Device of the result.

        Returns:
            Positive tensor of the given shape.
        """
        return torch.exp(torch.normal(0, 1, shape, device=device, dtype=dtype) * self.std + self.mean)


@dataclass(frozen=True)
class SamplerSettings:
    """Hyper-parameters of the stochastic sampler (AlphaFold 3 Algorithm 18); defaults are the AlphaFold 3 values.

    Setting ``gamma_0 = 0``, ``step_scale = ode_step_scale = 1`` and ``trans_scale = 0`` turns it into the plain
    probability-flow ODE Euler sampler of PhysDock2.

    Args:
        gamma_0: While the noise level exceeds ``gamma_min`` it is raised by the factor ``1 + gamma_0`` (churn);
            ``0`` switches the churn off.
        gamma_min: Churn is switched off at and below this noise level (Angstrom).
        noise_scale: Factor ``lambda`` on the noise that is added by the churn.
        step_scale: Factor ``eta`` on the Euler step while the noise level exceeds ``gamma_min``.
        ode_step_scale: Factor ``eta`` on the Euler step once the noise level is at or below ``gamma_min``.
        trans_scale: Standard deviation (Angstrom) of the random translation of the augmentation that is applied to
            the samples before every step (AlphaFold 3 Algorithm 19; PhysDock2 uses 0 here).
    """

    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    ode_step_scale: float = 1.0
    trans_scale: float = 1.0


# ======================================================================================
# Interfaces
# ======================================================================================
class DenoisingNetwork(Protocol):
    """Callable interface of the network that a :class:`Diffuser` wraps (e.g. ``AllAtomDecoder``)."""

    def __call__(self, batch: TensorDict, x_in: torch.Tensor, t_in: torch.Tensor, conds: Any) -> torch.Tensor:
        """Map preconditioned inputs to the raw network output.

        Args:
            batch: Input features.
            x_in: Preconditioned noisy coordinates, shape ``[S, N_atom, 3]``.
            t_in: Noise-conditioning input, shape ``[S]``.
            conds: Conditioning from the encoder.

        Returns:
            Raw network output, shape ``[S, N_atom, 3]``.
        """
        ...


Guidance = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Optional[torch.Tensor]]
"""``guidance(x_hat, x_denoised, sigma_cur) -> x_guided | None``: may replace the clean-structure estimate of a step.

``x_hat`` / ``x_denoised`` are the noisy coordinates and the clean-structure estimate ``[S, N_atom, 3]`` and
``sigma_cur`` is the noise level (Angstrom, 0-d) of the step *before* the churn. Returning ``None`` keeps
``x_denoised``. The ODE direction is computed from the returned structure, so guidance does not depend on the
diffusion process."""


class Diffuser(abc.ABC):
    """Noise schedule, training noising, preconditioned denoising and sampling of a diffusion model.

    Subclasses define the forward process and the parameterisation (the abstract methods); noising, denoising, the
    ODE direction and the sampling loop are shared.

    Args:
        sigma_data: Data scale in Angstrom. Unit-free noise levels (the schedule and the time sampler) are in
            units of ``sigma_data``.
        time_sampler: Distribution of the training noise levels (in units of ``sigma_data``).
        noise_schedule: Sampling noise levels (in units of ``sigma_data``).
        sampler: Default sampler hyper-parameters.
        trans_scale: Standard deviation (Angstrom) of the random translation added by the centre/rotation
            augmentation of the training samples (AlphaFold 3 Algorithm 19).

    Raises:
        ValueError: If ``sigma_data <= 0`` or ``trans_scale < 0``.
    """

    def __init__(
            self,
            sigma_data: float = 16.0,
            time_sampler: Optional[LogNormalTimeSampler] = None,
            noise_schedule: Optional[KarrasNoiseSchedule] = None,
            sampler: Optional[SamplerSettings] = None,
            trans_scale: float = 1.0,
    ) -> None:
        if sigma_data <= 0:
            raise ValueError(f"sigma_data must be positive, got {sigma_data}")
        if trans_scale < 0:
            raise ValueError(f"trans_scale must be non-negative, got {trans_scale}")
        self.sigma_data = sigma_data
        self.time_sampler = time_sampler if time_sampler is not None else LogNormalTimeSampler()
        self.noise_schedule = noise_schedule if noise_schedule is not None else KarrasNoiseSchedule()
        self.sampler = sampler if sampler is not None else SamplerSettings()
        self.trans_scale = trans_scale

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Diffuser":
        """Build a diffuser from the ``diffusion`` section of :func:`~PhysDock.configs.PhysDockConfig`.

        Args:
            config: Mapping with ``sigma_data``, ``trans_scale`` and the sub-sections ``time_sampler``
                (:class:`LogNormalTimeSampler`), ``noise_schedule`` (:class:`KarrasNoiseSchedule`) and ``sampler``
                (:class:`SamplerSettings`), plus the keys that a subclass needs.

        Returns:
            The configured diffuser.
        """
        return cls(
            sigma_data=config["sigma_data"],
            time_sampler=LogNormalTimeSampler(**config["time_sampler"]),
            noise_schedule=KarrasNoiseSchedule(**config["noise_schedule"]),
            sampler=SamplerSettings(**config["sampler"]),
            trans_scale=config["trans_scale"],
            **cls._extra_kwargs(config),
        )

    @classmethod
    def _extra_kwargs(cls, config: Mapping[str, Any]) -> Dict[str, Any]:
        """Constructor arguments of a subclass that come from the configuration (none by default)."""
        return {}

    def with_noise_schedule(self, **changes: float) -> KarrasNoiseSchedule:
        """The configured noise schedule with some fields replaced, e.g. ``with_noise_schedule(rho=1000)``.

        Args:
            **changes: Fields of :class:`KarrasNoiseSchedule` (``s_min``, ``s_max``, ``rho``).

        Returns:
            A new schedule; the configured one is not modified.
        """
        return dataclasses.replace(self.noise_schedule, **changes)

    # ----------------------------------------------------------------- the diffusion process (per subclass)
    @abc.abstractmethod
    def to_time(self, level: torch.Tensor) -> torch.Tensor:
        """Map unit-free noise levels (in units of ``sigma_data``) to the native time of this diffuser.

        Args:
            level: Noise levels ``sqrt(NSR)``, any shape.

        Returns:
            Native times, same shape.
        """

    @abc.abstractmethod
    def noise_level(self, t: torch.Tensor) -> torch.Tensor:
        """Noise level in Angstrom (``sigma_data * sqrt(NSR)``) of native times; inverse of :meth:`to_time`.

        Args:
            t: Native times, any shape.

        Returns:
            Noise levels in Angstrom, same shape.
        """

    @abc.abstractmethod
    def diffuse(self, x: torch.Tensor, e: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Noisy sample of the forward process.

        Args:
            x: Clean coordinates, shape ``[S, N_atom, 3]``.
            e: Standard normal noise, same shape.
            t: Native times, shape ``[S]``.

        Returns:
            Noisy coordinates ``z_t`` (Angstrom), shape ``[S, N_atom, 3]``.
        """

    @abc.abstractmethod
    def prior(self, shape: Tuple[int, ...], t: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Draw the starting point of the sampler: pure noise at the first (largest) native time.

        Args:
            shape: Shape of the result, ``[S, N_atom, 3]``.
            t: First native time, 0-d tensor.
            dtype: Floating point dtype.
            device: Device of the result.

        Returns:
            Noisy coordinates in Angstrom.
        """

    @abc.abstractmethod
    def churn(self, z: torch.Tensor, t: torch.Tensor, gamma_0: float, noise_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Raise the noise level of ``z`` by the factor ``1 + gamma_0`` by adding fresh noise.

        Args:
            z: Noisy coordinates at native time ``t``, shape ``[S, N_atom, 3]``.
            t: Current native time, 0-d tensor.
            gamma_0: Relative increase of the noise level.
            noise_scale: Factor on the added noise.

        Returns:
            Tuple ``(z_hat, t_hat)``: the noisier coordinates ``[S, N_atom, 3]`` and their native time ``[S]``.
        """

    @abc.abstractmethod
    def scale_input(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Precondition the network input.

        Args:
            z: Noisy coordinates, shape ``[S, N_atom, 3]``.
            t: Native times, shape ``[S]``.

        Returns:
            Tensor of shape ``[S, N_atom, 3]``.
        """

    @abc.abstractmethod
    def noise_input(self, t: torch.Tensor) -> torch.Tensor:
        """Noise-conditioning scalar for the network's timestep embedder.

        Args:
            t: Native times, shape ``[S]``.

        Returns:
            Tensor of shape ``[S]``.
        """

    @abc.abstractmethod
    def pred_x(self, pred: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convert the raw network output into the clean-structure estimate.

        Args:
            pred: Raw network output, shape ``[S, N_atom, 3]``.
            z: Noisy coordinates, same shape.
            t: Native times, shape ``[S]``.

        Returns:
            Estimated clean coordinates in Angstrom, shape ``[S, N_atom, 3]``.
        """

    @abc.abstractmethod
    def pred_e(self, pred: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convert the raw network output into the estimate of the unit-variance noise ``e``.

        Args:
            pred: Raw network output, shape ``[S, N_atom, 3]``.
            z: Noisy coordinates, same shape.
            t: Native times, shape ``[S]``.

        Returns:
            Dimensionless noise estimate, shape ``[S, N_atom, 3]``.
        """

    @abc.abstractmethod
    def pred_v(self, pred: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Velocity of the normalised sample ``z / sigma_data`` per unit of native time (PhysDock2's ``pred_v``).

        Args:
            pred: Raw network output, shape ``[S, N_atom, 3]``.
            z: Noisy coordinates, same shape.
            t: Native times, shape ``[S]``.

        Returns:
            Dimensionless velocity, shape ``[S, N_atom, 3]``.
        """

    # ----------------------------------------------------------------- shared machinery
    def denoise(
            self,
            decoder: DenoisingNetwork,
            batch: TensorDict,
            z: torch.Tensor,
            t: torch.Tensor,
            conds: Any,
    ) -> torch.Tensor:
        """Predict the clean structure: precondition the inputs, run the network, convert its output.

        Args:
            decoder: The network.
            batch: Input features, passed on to the network.
            z: Noisy coordinates, shape ``[S, N_atom, 3]``.
            t: Native times, shape ``[S]``.
            conds: Conditioning from the encoder.

        Returns:
            Estimated clean coordinates ``x_pred`` in Angstrom, shape ``[S, N_atom, 3]``.
        """
        pred = decoder(batch, self.scale_input(z, t), self.noise_input(t), conds)
        return self.pred_x(pred, z, t)

    @staticmethod
    def drift(z: torch.Tensor, x_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Direction of the probability-flow ODE ``dz/dt`` for a clean-structure estimate: ``(z - x_pred) / t``.

        This holds for both diffusion processes because ``z_t = x + t * (velocity)`` in the native time.

        Args:
            z: Noisy coordinates, shape ``[S, N_atom, 3]``.
            x_pred: Estimated clean coordinates, same shape.
            t: Native times, shape ``[S]``.

        Returns:
            Drift, shape ``[S, N_atom, 3]``.
        """
        return (z - x_pred) / t[..., None, None]

    @staticmethod
    def step(z: torch.Tensor, drift: torch.Tensor, t: torch.Tensor, r: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Euler step of the probability-flow ODE from native time ``t`` to ``r``: ``z + scale * (r - t) * drift``.

        Args:
            z: Noisy coordinates at native time ``t``, shape ``[S, N_atom, 3]``.
            drift: Update direction, same shape.
            t: Native times of ``z``, shape ``[S]``.
            r: Target native time, 0-d tensor.
            scale: Step scale ``eta``.

        Returns:
            Coordinates at (about) native time ``r``, shape ``[S, N_atom, 3]``.
        """
        dt = (r - t)[..., None, None]
        return z + scale * dt * drift

    def sample_t(self, num_samples: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Draw training times: log-normal noise levels mapped to the native time.

        Args:
            num_samples: Number of times.
            dtype: Floating point dtype.
            device: Device of the result.

        Returns:
            Native times, shape ``[num_samples]``.
        """
        return self.to_time(self.time_sampler((num_samples,), dtype, device))

    def sample_ts(
            self,
            num_steps: int,
            dtype: torch.dtype,
            device: torch.device,
            noise_schedule: Optional[KarrasNoiseSchedule] = None,
    ) -> torch.Tensor:
        """Native times of the sampler, ending with 0.

        Args:
            num_steps: Number of denoising steps (at least 2).
            dtype: Floating point dtype.
            device: Device of the result.
            noise_schedule: Use this schedule instead of the configured one.

        Returns:
            Tensor of shape ``[num_steps + 1]``.
        """
        schedule = noise_schedule if noise_schedule is not None else self.noise_schedule
        return self.to_time(schedule(num_steps)).to(device).to(dtype)

    @torch.no_grad()
    def noising(
            self,
            x_gt: torch.Tensor,
            x_exists: torch.Tensor,
            num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create training samples: ``num_samples`` randomly rotated, translated and noised copies of ``x_gt``.

        Args:
            x_gt: Ground-truth coordinates, shape ``[N_atom, 3]``.
            x_exists: 1 for atoms with a ground-truth position, shape ``[N_atom]``.
            num_samples: Number of noisy copies.

        Returns:
            Tuple ``(z_t, t)``: noisy coordinates ``[num_samples, N_atom, 3]`` (Angstrom) and their native times
            ``[num_samples]``.
        """
        t = self.sample_t(num_samples, x_gt.dtype, x_gt.device)
        x = x_gt[None].repeat([num_samples, 1, 1])
        e = torch.normal(0, 1, size=x.shape, dtype=x.dtype, device=x.device, requires_grad=False)
        z = centre_random_augmentation(self.diffuse(x, e, t), x_exists, s=self.trans_scale)
        return z, t

    @torch.no_grad()
    def sample(
            self,
            decoder: DenoisingNetwork,
            batch: TensorDict,
            conds: Any,
            x_exists: torch.Tensor,
            num_samples: int,
            num_steps: int,
            sampler: Optional[SamplerSettings] = None,
            noise_schedule: Optional[KarrasNoiseSchedule] = None,
            guidance: Optional[Guidance] = None,
    ) -> torch.Tensor:
        """Draw structures by reverse diffusion (AlphaFold 3 Algorithm 18).

        Starting from pure noise at the largest noise level, every step (1) randomly rotates and translates the
        samples, (2) while the noise level exceeds ``sampler.gamma_min`` raises it by churn (which does nothing for
        ``sampler.gamma_0 = 0``), (3) predicts the clean structure, (4) lets ``guidance`` modify that prediction,
        (5) computes the ODE direction and (6) takes an Euler step to the next noise level.

        Args:
            decoder: The network.
            batch: Input features; ``x_gt`` only provides the shape, dtype and device of the output.
            conds: Conditioning from the encoder, computed once.
            x_exists: 1 for atoms that exist, shape ``[N_atom]`` (centre of the augmentation).
            num_samples: Number of structures to draw.
            num_steps: Number of denoising steps (at least 2).
            sampler: Use these hyper-parameters instead of the configured ones.
            noise_schedule: Use this schedule instead of the configured one.
            guidance: Optional hook that may replace the clean-structure estimate, see :data:`Guidance`.

        Returns:
            Sampled coordinates in Angstrom, shape ``[num_samples, N_atom, 3]``.
        """
        x_gt = batch["x_gt"]
        dtype, device = x_gt.dtype, x_gt.device
        settings = sampler if sampler is not None else self.sampler
        ts = self.sample_ts(num_steps, dtype, device, noise_schedule)

        x_next = self.prior((num_samples, *x_gt.shape[-2:]), ts[0], dtype, device)
        for t_cur, t_next in zip(ts[:-1], ts[1:]):
            x_cur = centre_random_augmentation(x_next, x_exists, s=settings.trans_scale)
            sigma_cur = self.noise_level(t_cur)
            if sigma_cur > settings.gamma_min:
                scale = settings.step_scale
                x_hat, t_hat = self.churn(x_cur, t_cur, settings.gamma_0, settings.noise_scale)
            else:
                scale = settings.ode_step_scale
                x_hat, t_hat = x_cur, torch.full([num_samples], fill_value=t_cur, device=device, dtype=dtype)
            x_denoised = self.denoise(decoder, batch, x_hat, t_hat, conds)
            if guidance is not None:
                guided = guidance(x_hat, x_denoised, sigma_cur)
                if guided is not None:
                    x_denoised = guided
            x_next = self.step(x_hat, self.drift(x_hat, x_denoised, t_hat), t_hat, t_next, scale)
        return x_next


# ======================================================================================
# EDM variance-exploding diffuser
# ======================================================================================
class EDMVEDiffuser(Diffuser):
    """EDM variance-exploding diffusion ``z_t = x + t * e`` with the AlphaFold 3 preconditioning.

    See the module docstring for the formulas. The native time is the noise level in Angstrom.
    ``noise_conditioning`` selects what is fed to the network's timestep embedder, see there.

    Args:
        sigma_data: Data scale in Angstrom.
        time_sampler: Distribution of the training noise levels.
        noise_schedule: Sampling noise levels.
        sampler: Default sampler hyper-parameters.
        trans_scale: Standard deviation (Angstrom) of the augmentation translation of training samples.
        noise_conditioning: ``"af3"`` or ``"legacy"``.

    Raises:
        ValueError: For an unknown ``noise_conditioning``.
    """

    def __init__(
            self,
            sigma_data: float = 16.0,
            time_sampler: Optional[LogNormalTimeSampler] = None,
            noise_schedule: Optional[KarrasNoiseSchedule] = None,
            sampler: Optional[SamplerSettings] = None,
            trans_scale: float = 1.0,
            noise_conditioning: Literal["af3", "legacy"] = "legacy",
    ) -> None:
        super().__init__(sigma_data, time_sampler, noise_schedule, sampler, trans_scale)
        if noise_conditioning not in ("af3", "legacy"):
            raise ValueError(f"noise_conditioning must be 'af3' or 'legacy', got {noise_conditioning!r}")
        self.noise_conditioning = noise_conditioning

    @classmethod
    def _extra_kwargs(cls, config: Mapping[str, Any]) -> Dict[str, Any]:
        return {"noise_conditioning": config["noise_conditioning"]}

    def to_time(self, level: torch.Tensor) -> torch.Tensor:
        """``sigma_data * level``: the native time is the noise level in Angstrom."""
        return self.sigma_data * level

    def noise_level(self, t: torch.Tensor) -> torch.Tensor:
        """The native time itself."""
        return t

    def diffuse(self, x: torch.Tensor, e: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``z_t = x + t * e``, see :meth:`Diffuser.diffuse`."""
        return x + e * t[..., None, None]

    def prior(self, shape: Tuple[int, ...], t: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """``t * N(0, I)``, see :meth:`Diffuser.prior`."""
        return t * torch.normal(0, 1, size=shape, dtype=dtype, device=device, requires_grad=False)

    def churn(self, z: torch.Tensor, t: torch.Tensor, gamma_0: float, noise_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise so that the noise level grows from ``t`` to ``t_hat = t * (1 + gamma_0)``.

        The added noise has standard deviation ``noise_scale * sqrt(t_hat^2 - t^2)``. See :meth:`Diffuser.churn`.
        """
        t_hat = torch.full([z.shape[0]], fill_value=t * (gamma_0 + 1), device=z.device, dtype=z.dtype)
        noise = torch.normal(0, 1, size=z.shape, dtype=z.dtype, device=z.device, requires_grad=False)
        ksi = noise_scale * noise * torch.sqrt(t_hat ** 2 - t ** 2)[..., None, None]
        return z + ksi, t_hat      # for gamma_0 = 0 the values are unchanged; the noise is still drawn (same random stream as before)

    def scale_input(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``z / sqrt(t^2 + sigma^2)``, see :meth:`Diffuser.scale_input`."""
        c_in = 1 / (torch.sqrt(t[:, None, None] ** 2 + self.sigma_data ** 2))
        return z * c_in

    def noise_input(self, t: torch.Tensor) -> torch.Tensor:
        """``ln(t / sigma) / 4`` (``"af3"``) or ``t * ln(t / sigma) / 4`` (``"legacy"``), see the module docstring."""
        c_noise = torch.log(t / self.sigma_data) / 4.0
        return c_noise if self.noise_conditioning == "af3" else t * c_noise

    def pred_x(self, pred: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``c_skip * z + c_out * pred``, see :meth:`Diffuser.pred_x`."""
        c_skip = (self.sigma_data ** 2 / (self.sigma_data ** 2 + t ** 2))[:, None, None]
        c_out = (self.sigma_data * t / torch.sqrt(self.sigma_data ** 2 + t ** 2))[:, None, None]
        return c_skip * z + c_out * pred

    def pred_e(self, pred: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``t z / (sigma^2 + t^2) - sigma pred / sqrt(sigma^2 + t^2)``, see :meth:`Diffuser.pred_e`."""
        denom = self.sigma_data ** 2 + t ** 2
        return (t / denom)[:, None, None] * z - (self.sigma_data / torch.sqrt(denom))[:, None, None] * pred

    def pred_v(self, pred: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """For EDM-VE the normalised velocity is the noise estimate, see :meth:`Diffuser.pred_v`."""
        return self.pred_e(pred, z, t)


# ======================================================================================
# Linear (flow matching) diffuser
# ======================================================================================
class LinearDiffuser(Diffuser):
    """Linear interpolation between data and noise: ``z_t = (1 - t) * x + t * sigma * e`` with ``t`` in (0, 1).

    The flow-matching diffusion of PhysDock2 (``LinearDiffuser``). The network sees ``z / sigma`` and ``t`` and
    predicts the velocity ``e - x / sigma``; the clean structure is ``z - t * sigma * pred``. The noise level
    ``sqrt(NSR)`` of time ``t`` is ``t / (1 - t)`` (in units of ``sigma_data``), so the Karras noise schedule and the
    log-normal training distribution of the EDM diffuser are used unchanged and mapped with ``t = s / (1 + s)``.
    Time 1 is pure noise and time 0 the clean structure; the noise level diverges for ``t -> 1``.

    Args:
        sigma_data: Data scale in Angstrom.
        time_sampler: Distribution of the training noise levels.
        noise_schedule: Sampling noise levels.
        sampler: Default sampler hyper-parameters.
        trans_scale: Standard deviation (Angstrom) of the augmentation translation of training samples.
    """

    def to_time(self, level: torch.Tensor) -> torch.Tensor:
        """``level / (1 + level)``."""
        return level / (1 + level)

    def noise_level(self, t: torch.Tensor) -> torch.Tensor:
        """``sigma_data * t / (1 - t)``."""
        return self.sigma_data * (t / (1 - t))

    def diffuse(self, x: torch.Tensor, e: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``z_t = (1 - t) * x + t * sigma * e``, see :meth:`Diffuser.diffuse`."""
        return (1 - t)[..., None, None] * x + (self.sigma_data * t)[..., None, None] * e

    def prior(self, shape: Tuple[int, ...], t: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """``sigma * t * N(0, I)``: the data term is negligible at the first time, see :meth:`Diffuser.prior`."""
        return (self.sigma_data * t) * torch.normal(0, 1, size=shape, dtype=dtype, device=device, requires_grad=False)

    def churn(self, z: torch.Tensor, t: torch.Tensor, gamma_0: float, noise_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Raise the noise level ``s = t / (1 - t)`` to ``s * (1 + gamma_0)`` (PhysDock2 ``langevin_like_churn``).

        ``z / (1 - t) = x + s * sigma * e`` has the form of an EDM-VE sample; noise of standard deviation
        ``noise_scale * sigma * sqrt(s_hat^2 - s^2)`` is added to it and the result is mapped back with
        ``1 - t_hat``. See :meth:`Diffuser.churn`.
        """
        if gamma_0 == 0:                            # nothing to add (PhysDock2 has no churn at all)
            return z, torch.full([z.shape[0]], fill_value=t, device=z.device, dtype=z.dtype)
        level = t / (1 - t)
        level_hat = level * (1 + gamma_0)
        t_hat = level_hat / (level_hat + 1)
        noise = torch.normal(0, 1, size=z.shape, dtype=z.dtype, device=z.device, requires_grad=False)
        ksi = noise_scale * self.sigma_data * torch.sqrt(level_hat ** 2 - level ** 2) * noise
        z_hat = (z / (1 - t) + ksi) * (1 - t_hat)
        return z_hat, torch.full([z.shape[0]], fill_value=t_hat, device=z.device, dtype=z.dtype)

    def scale_input(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``z / sigma``: the network sees normalised coordinates, see :meth:`Diffuser.scale_input`."""
        return z / self.sigma_data

    def noise_input(self, t: torch.Tensor) -> torch.Tensor:
        """The time ``t`` itself, see :meth:`Diffuser.noise_input`."""
        return t

    def pred_x(self, pred: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``z - t * sigma * pred``, see :meth:`Diffuser.pred_x`."""
        return z - (self.sigma_data * t)[:, None, None] * pred

    def pred_e(self, pred: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``z / sigma + (1 - t) * pred``, see :meth:`Diffuser.pred_e`."""
        return z / self.sigma_data + (1 - t)[:, None, None] * pred

    def pred_v(self, pred: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The network output is the velocity itself, see :meth:`Diffuser.pred_v`."""
        return pred


# ======================================================================================
# Factory
# ======================================================================================
DIFFUSERS: Dict[str, Type[Diffuser]] = {"edm_ve": EDMVEDiffuser, "linear": LinearDiffuser}
"""Available diffusion processes by the value of ``config.diffusion.type``."""


def build_diffuser(config: Mapping[str, Any]) -> Diffuser:
    """Build the diffuser selected by ``config["type"]`` (``"edm_ve"`` if the key is missing).

    Args:
        config: The ``diffusion`` section of :func:`~PhysDock.configs.PhysDockConfig`.

    Returns:
        The configured :class:`Diffuser`.

    Raises:
        ValueError: For an unknown ``type``.
    """
    kind = config.get("type", "edm_ve")
    if kind not in DIFFUSERS:
        raise ValueError(f"unknown diffusion type {kind!r}, expected one of {sorted(DIFFUSERS)}")
    return DIFFUSERS[kind].from_config(config)

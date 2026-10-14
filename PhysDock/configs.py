"""Model configuration (sizes, diffusion, loss). Featurisation settings are in PhysDock.data.FeatureConfig."""
import ml_collections as mlc

_BLOCKS = {  # model_name: (atom, evoformer, pairformer, dit)
    "toy": (2, 2, 2, 2),
    "tiny": (2, 2, 8, 4),
    "small": (2, 3, 16, 8),
    "medium": (3, 4, 24, 12),
    "full": (3, 4, 48, 24),
}


def PhysDockConfig(model_name: str = "medium", num_augmentation_sample: int = 48, alpha_diffusion: float = 4,
                   alpha_distogram: float = 3e-2, diffusion_type: str = "edm_ve", inf: float = 1e9, eps: float = 1e-8,
                   sigma_data: float = 16.) -> mlc.ConfigDict:
    """Config of the released model ("medium") or a smaller / larger variant.

    Args:
        model_name: "toy", "tiny", "small", "medium" or "full" (number of blocks)
        num_augmentation_sample: noisy samples per structure in training
        alpha_diffusion, alpha_distogram: loss weights
        diffusion_type: "edm_ve" (released parameters) or "linear" (PhysDock2 flow matching)
        inf, eps: masking value and norm epsilon
        sigma_data: data scale (Angstrom)
    """
    if model_name not in _BLOCKS:
        raise ValueError(f"unknown model name {model_name!r}, expected one of {sorted(_BLOCKS)}")
    no_blocks_atom, no_blocks_evoformer, no_blocks_pairformer, no_blocks_dit = _BLOCKS[model_name]
    c_m, c_s, c_z, c_a, c_ap = 256, 512, 128, 128, 16

    sampler_af3 = {"gamma_0": 0.8, "gamma_min": 1.0, "noise_scale": 1.003, "step_scale": 1.5, "ode_step_scale": 1.0,
                   "trans_scale": 1.0}
    if diffusion_type == "edm_ve":
        diffusion = {
            "type": "edm_ve", "sigma_data": sigma_data, "trans_scale": 1.0,
            # released parameters: noise input t * ln(t / sigma) / 4 ("legacy"); "af3" only for new models
            "noise_conditioning": "legacy",
            "time_sampler": {"mean": -1.2, "std": 1.5},
            # s_min 4e-3 as in the released code (AF3 / PhysDock2 use 4e-4); predictions depend on it
            "noise_schedule": {"s_min": 4e-3, "s_max": 160., "rho": 7.},
            "sampler": sampler_af3,
        }
    elif diffusion_type == "linear":
        diffusion = {
            "type": "linear", "sigma_data": sigma_data, "trans_scale": 1.0,
            "time_sampler": {"mean": -1.2, "std": 1.5},
            "noise_schedule": {"s_min": 4e-4, "s_max": 160., "rho": 1000.},
            # PhysDock2: plain probability-flow ODE (no churn, unit steps, no random translation)
            "sampler": {**sampler_af3, "gamma_0": 0.0, "step_scale": 1.0, "ode_step_scale": 1.0, "trans_scale": 0.0},
        }
    else:
        raise ValueError(f"unknown diffusion type {diffusion_type!r}, expected 'edm_ve' or 'linear'")

    return mlc.ConfigDict({
        "model": {
            "c_z": c_z,
            "num_augmentation_sample": num_augmentation_sample,
            "encoder": {"ref_dim": 167, "target_dim": 65, "msa_dim": 34, "c_a": c_a, "c_ap": c_ap, "c_s": c_s,
                        "c_m": c_m, "c_z": c_z, "inf": inf, "eps": eps, "no_blocks_atom": no_blocks_atom,
                        "no_blocks_evoformer": no_blocks_evoformer, "no_blocks_pairformer": no_blocks_pairformer},
            "decoder": {"c_a": c_a, "c_ap": c_ap, "c_s": c_s, "c_z": c_z, "inf": inf, "eps": eps,
                        "no_blocks_atom": no_blocks_atom, "no_blocks_dit": no_blocks_dit},
        },
        "diffusion": diffusion,
        "loss": {
            "weighted_mse_loss": {"weight": alpha_diffusion, "sigma_data": sigma_data, "alpha_ligand": 10.0},
            "smooth_lddt_loss": {"weight": alpha_diffusion, "max_clamp_distance": 15.},
            "distogram_loss": {"weight": alpha_distogram, "min_bin": 3.25, "max_bin": 50.75, "no_bins": 39},
        },
    })

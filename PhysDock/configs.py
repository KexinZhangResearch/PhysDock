import ml_collections as mlc


def PhysDockConfig(
        inference_mode=True,
        model_name="medium",
        num_augmentation_sample=48,

        crop_size=256,
        atom_crop_size=256 * 8,

        alpha_diffusion=4,
        alpha_distogram=3e-2,
        diffusion_type="edm_ve",  # "edm_ve" (released parameters) or "linear" (flow matching, PhysDock2)
        inf=1e9,
        eps=1e-8,


        # Inference Config
        infer_pocket_type="atom",  # "ca"
        infer_pocket_cutoff=6,  # 8 10 12
        infer_pocket_dist_type="ligand",  # "ligand_centre"
        infer_use_pocket=True,
        infer_use_key_res=True,

        # Training Config
        train_pocket_type_atom_ratio=0.5,
        train_pocket_cutoff_ligand_min=6,
        train_pocket_cutoff_ligand_max=12,
        train_pocket_cutoff_ligand_centre_min=10,
        train_pocket_cutoff_ligand_centre_max=16,
        train_pocket_dist_type_ligand_ratio=0.5,
        train_use_pocket_ratio=0.5,
        train_use_key_res_ratio=0.5,

        train_shuffle_sym_id=True,
        train_spatial_crop_ligand_ratio=0.2,
        train_spatial_crop_interface_ratio=0.4,
        train_spatial_crop_interface_threshold=15.,
        train_charility_augmentation_ratio=0.1,
        train_use_template_ratio=0.75,
        train_template_mask_max_ratio=0.4,

        # Other Configs
        max_msa_clusters=128,
        key_res_random_mask_ratio=0.5,
        token_bond_threshold=2.4,
        sigma_data=16.,
):
    ref_dim = 167
    target_dim = 65
    msa_dim = 34

    inf = inf
    eps = eps

    c_m = 256  # 256
    c_s = 512  # 1024
    c_z = 128  # 64 | 128
    c_a = 128  # 128
    c_ap = 16  # 16 | 32

    if model_name == "toy":
        no_blocks_atom = 2
        no_blocks_evoformer = 2
        no_blocks_pairformer = 2
        no_blocks_dit = 2
    elif model_name == "tiny":
        no_blocks_atom = 2
        no_blocks_evoformer = 2
        no_blocks_pairformer = 8
        no_blocks_dit = 4
    elif model_name == "small":
        no_blocks_atom = 2
        no_blocks_evoformer = 3
        no_blocks_pairformer = 16
        no_blocks_dit = 8
    elif model_name == "medium":
        no_blocks_atom = 3
        no_blocks_evoformer = 4
        no_blocks_pairformer = 24
        no_blocks_dit = 12
    elif model_name == "full":
        no_blocks_atom = 3
        no_blocks_evoformer = 4
        no_blocks_pairformer = 48
        no_blocks_dit = 24
    else:
        raise ValueError("Unknown model name")

    sampler_af3 = {"gamma_0": 0.8, "gamma_min": 1.0, "noise_scale": 1.003, "step_scale": 1.5, "ode_step_scale": 1.0,
                   "trans_scale": 1.0}
    if diffusion_type == "edm_ve":
        diffusion = {
            "type": "edm_ve",
            "sigma_data": sigma_data,
            "trans_scale": 1.0,
            # The released parameters were trained with t * ln(t / sigma_data) / 4 as noise input ("legacy");
            # "af3" (ln(t / sigma_data) / 4) is only for models trained from scratch.
            "noise_conditioning": "legacy",
            "time_sampler": {"mean": -1.2, "std": 1.5},
            # s_min is 4 * 10e-4 = 0.004 in the released code (AlphaFold 3 and PhysDock2 use 4e-4). It is kept
            # because predictions of the released parameters depend on it.
            "noise_schedule": {"s_min": 4e-3, "s_max": 160., "rho": 7.},
            "sampler": sampler_af3,
        }
    elif diffusion_type == "linear":
        diffusion = {
            "type": "linear",
            "sigma_data": sigma_data,
            "trans_scale": 1.0,
            "time_sampler": {"mean": -1.2, "std": 1.5},
            # PhysDock2: Karras schedule with s_min = 4e-4 and rho = 1000.
            "noise_schedule": {"s_min": 4e-4, "s_max": 160., "rho": 1000.},
            # PhysDock2 samples with the plain probability-flow ODE: no churn, unit step, no random translation.
            "sampler": {**sampler_af3, "gamma_0": 0.0, "step_scale": 1.0, "ode_step_scale": 1.0, "trans_scale": 0.0},
        }
    else:
        raise ValueError(f"Unknown diffusion type {diffusion_type!r}, expected 'edm_ve' or 'linear'")

    config = {
        "inference_mode": inference_mode,
        "sigma_data": sigma_data,
        "data": {
            "crop_size": crop_size,
            "atom_crop_size": atom_crop_size,
            "max_msa_seqs": 16384,
            "max_uniprot_msa_seqs": 8192,
            "interface_threshold": 15,
            "token_bond_threshold": token_bond_threshold,
            "covalent_bond_threshold": 1.8,
            "max_msa_clusters": max_msa_clusters,
            "resample_msa_in_recycling": True,
        },
        "model": {
            "c_z": c_z,
            "num_augmentation_sample": num_augmentation_sample,
            "encoder": {
                "ref_dim": ref_dim,
                "target_dim": target_dim,
                "msa_dim": msa_dim,
                "c_a": c_a,
                "c_ap": c_ap,
                "c_s": c_s,
                "c_m": c_m,
                "c_z": c_z,
                "inf": inf,
                "eps": eps,
                "no_blocks_atom": no_blocks_atom,
                "no_blocks_evoformer": no_blocks_evoformer,
                "no_blocks_pairformer": no_blocks_pairformer
            },
            "decoder": {
                "c_a": c_a,
                "c_ap": c_ap,
                "c_s": c_s,
                "c_z": c_z,
                "inf": inf,
                "eps": eps,
                "no_blocks_atom": no_blocks_atom,
                "no_blocks_dit": no_blocks_dit,
            },
        },
        "diffusion": diffusion,
        "loss": {
            "weighted_mse_loss": {
                "weight": alpha_diffusion,
                "sigma_data": sigma_data,
                "alpha_dna": 5.0,
                "alpha_rna": 5.0,
                "alpha_ligand": 10.0,
            },
            "smooth_lddt_loss": {
                "weight": alpha_diffusion,
                "max_clamp_distance": 15.,
            },
            "distogram_loss": {
                "weight": alpha_distogram,
                "min_bin": 3.25,
                "max_bin": 50.75,
                "no_bins": 39,
            },
        }
    }
    return mlc.ConfigDict(config)

import ml_collections as mlc


def model_config(
        model_name="full",
        max_recycling_iters=1,  # 0
        max_msa_clusters=128,  # 32
        crop_size=256,  #
        num_augmentation_sample=48,  # 128
        alpha_confifdence=1e-4,
        alpha_diffusion=4,
        alpha_bond=0,
        alpha_distogram=3e-2,
        alpha_pae=0,
        use_template=True,  # False
        use_mini_rollout=True,  # False
        use_flash_attn=False,  # False
        custom_rel_token=-1,  # 42
        ref_dim=1 + 2 + 2 + 128 + 256,  # 167
        mini_rollout_steps=20,
        atom_attention_type="full",
        templ_dim=108,
        interaction_aware=True,
):
    sigma_data = 16
    # ref_dim = 1 + 2 + 2 + 128 + 256
    msa_dim = 34
    templ_dim = templ_dim

    inf = 1e9
    eps = 1e-8

    pair_dropout = 0.25
    msa_dropout = 0.15

    c_m = 256  # 256
    c_s = 768  # 1024
    c_z = 128  # 64 | 128
    c_tz = 64
    c_a = 128  # 128
    c_ap = 16  # 16 | 32

    no_blocks_templ = 2
    no_blocks_evo = 48
    no_blocks_atom = 3
    no_blocks_dit = 24
    no_blocks_heads = 4
    if model_name == "small_toy":
        no_blocks_templ = 1
        no_blocks_evo = 1
        no_blocks_atom = 1
        no_blocks_dit = 1
        no_blocks_heads = 1
    elif model_name == "toy":
        no_blocks_templ = 2
        no_blocks_evo = 2
        no_blocks_atom = 2
        no_blocks_dit = 2
        no_blocks_heads = 2

    elif model_name == "small":
        no_blocks_templ = 2
        no_blocks_evo = 4
        no_blocks_atom = 2
        no_blocks_dit = 2
        no_blocks_heads = 2
    elif model_name == "docking":
        no_blocks_templ = 2
        no_blocks_evo = 8
        no_blocks_atom = 2
        no_blocks_dit = 4
        no_blocks_heads = 2
    elif model_name == "medium":
        no_blocks_templ = 2
        no_blocks_evo = 16
        no_blocks_atom = 3
        no_blocks_dit = 8
        no_blocks_heads = 2
    elif model_name == "large":
        no_blocks_templ = 2
        no_blocks_evo = 24
        no_blocks_atom = 3
        no_blocks_dit = 12
        no_blocks_heads = 4
    elif model_name == "full":
        no_blocks_templ = 2
        no_blocks_evo = 48
        no_blocks_atom = 3
        no_blocks_dit = 24
        no_blocks_heads = 4

    return mlc.ConfigDict({
        "use_template": use_template,
        "use_mini_rollout": use_mini_rollout,
        "mini_rollout_steps": mini_rollout_steps,

        "data": {
            "crop_size": crop_size,
            "atom_crop_factor": 10,
            "max_msa_seqs": 16384,
            "max_uniprot_msa_seqs": 8192,
            "interface_threshold": 15,
            "token_bond_threshold": 2.4,
            "covalent_bond_threshold": 1.8,
            "max_msa_clusters": max_msa_clusters,
            "resample_msa_in_recycling": True,
            "max_recycling_iters": max_recycling_iters,  # TODO 3
            "sample_msa": {
                "max_msa_clusters": 128,
                "resample_msa_in_recycling": True,
            },
            "make_crop_ids": {
                "crop_size": 384
            }
        },
        "model": {
            "input_feature_embedder": {
                "msa_dim": msa_dim,
                "ref_dim": ref_dim,
                "c_s": c_s,
                "c_m": c_m,
                "c_z": c_z,
                "c_ap": c_ap,
                "c_a": c_a,
                "no_heads": 4,
                "c_hidden": 16,
                "inf": inf,
                "eps": eps,
                "no_blocks": 3,
                "interaction_aware": interaction_aware,
                "custom_rel_token": custom_rel_token,
            },
            "template_pair_embedder": {
                "templ_dim": templ_dim,
                "c_z": c_z,
                "c_tz": c_tz,
                "c_hidden_tz": 16,
                "no_heads_tz": 4,
                "inf": inf,
                "eps": eps,
                "no_blocks": no_blocks_templ,
            },
            "recycling_embedder": {
                "c_m": c_m,
                "c_z": c_z,
            },
            "evoformer_stack": {
                "c_m": c_m,
                "c_z": c_z,
                "c_hidden_m": 32,
                "no_heads_m": 8,
                "c_hidden_z": 32,
                "no_heads_z": 4,
                "c_hidden_opm": 32,
                "inf": inf,
                "eps": eps,
                "no_blocks": no_blocks_evo,
                "single_mode": False,
            },
            "diffusion_module": {
                "ref_dim": ref_dim,
                "c_m": c_m,
                "c_s": c_s,
                "c_z": c_z,
                "c_a": c_a,
                "c_ap": c_ap,
                "no_heads_atom": 4,
                "c_hidden_atom": 16,
                "no_heads": c_ap,
                "c_hidden": 32,
                "inf": inf,
                "eps": eps,
                "no_blocks": no_blocks_dit,
                "no_blocks_atom": no_blocks_atom,
                "num_augmentation_sample": num_augmentation_sample,
                "custom_rel_token": custom_rel_token,
                "use_flash_attn": use_flash_attn,
                "atom_attention_type": atom_attention_type
            },
            "confidence_module": {
                "c_a": c_a,
                "c_ap": c_ap,
                "c_s": c_s,
                "c_m": c_m,
                "c_z": c_z,
                "no_heads_a": 4,
                "c_hidden_a": 16,
                "c_hidden_m": 32,
                "no_heads_m": 8,
                "c_hidden_z": 32,
                "no_heads_z": 4,
                "c_hidden_opm": 32,
                "inf": inf,
                "eps": eps,
                "no_blocks": no_blocks_heads,
                "no_blocks_atom": no_blocks_atom,
                "c_pae": 64,
                "c_pde": 64,
                "c_plddt": 50,
                "c_distogram": 39,
            },
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
                "clamp_distance_loss": {
                    "weight": alpha_diffusion * 0.2,
                    "max_clamp_distance": 10,
                },

                "bond_loss": {
                    "weight": alpha_diffusion * alpha_bond,
                    "sigma_data": sigma_data,
                },
                "distogram_loss": {
                    "weight": alpha_distogram,
                    "min_bin": 3.25,
                    "max_bin": 50.75,
                    "no_bins": 39,
                    "eps": 1e-9,
                },
                "plddt_loss": {
                    "weight": alpha_confifdence,
                    "no_bins": 50,
                },
                "pae_loss": {
                    "weight": alpha_confifdence * alpha_pae,
                },
                "pde_loss": {
                    "weight": alpha_confifdence,
                    "min_bin": 0,
                    "max_bin": 32,
                    "no_bins": 64,
                },
            }
        }
    })

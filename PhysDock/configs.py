"""Access to the Hydra configs in <repo>/configs from Python (outside of a @hydra.main script)."""
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
os.environ.setdefault("PHYSDOCK_ROOT", str(REPO_ROOT))          # used by configs/paths/default.yaml


def compose(config_name: str, overrides=()) -> DictConfig:
    """Compose a config of configs/ (e.g. "train", "model/medium") with Hydra override strings."""
    from hydra import compose as hydra_compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():                 # inside a Hydra app: reuse its search path
        return hydra_compose(config_name=config_name, overrides=list(overrides))
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return hydra_compose(config_name=config_name, overrides=list(overrides))


def model_config(name: str = "medium", overrides=()) -> DictConfig:
    """configs/model/<name>.yaml ("medium" = released model); overrides e.g. ["model/diffusion=linear"]."""
    return compose(f"model/{name}", overrides).model


def to_dict(cfg) -> dict:
    """Plain dict of a (resolved) config."""
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)

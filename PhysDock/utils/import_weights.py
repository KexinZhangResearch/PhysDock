from typing import Union
import os
import torch


def import_unicore_ckpt(
        model: torch.nn.Module,
        ckpt_path: Union[str, os.PathLike],
        load_ema_state: bool = True,
        remove_compile_prefix: bool = False
):
    params = torch.load(ckpt_path, weights_only=False)
    if load_ema_state:
        try:
            params = params["ema"]["params"]
        except:
            params = params["model"]
    else:
        params = params["model"]
    keys = list(params.keys())
    for k in keys:
        if remove_compile_prefix:
            params[k[16:]] = params.pop(k)
        else:
            params[k[6:]] = params.pop(k)

    model.load_state_dict(params)
    print("Weights loaded")


def import_state_dict(
        model: torch.nn.Module,
        ckpt_path: Union[str, os.PathLike],
):
    params = torch.load(ckpt_path, weights_only=False)
    keys = list(params.keys())
    for k in keys:
        params[k[6:]] = params.pop(k)

    model.load_state_dict(params)
    print("Weights loaded")

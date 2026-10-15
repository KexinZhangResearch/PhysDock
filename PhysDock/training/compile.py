"""torch.compile of the training step, region by region (scripts/train/train.py compile=true).

Compiled: the activation-checkpointed call of every Stack block (Evoformer, Pairformer, atom / token DiT, ...) and the
computations between the stacks (atom / token input embeddings, token -> atom broadcast, decoder pair biases, loss
terms). Every region is its own graph with static shapes (training crops are padded to fixed sizes), so the backward
recomputes one checkpointed block at a time; with the fused kernels saving fewer tensors the memory is below eager
(medium, bf16-mixed, crop 256: 0.96 s / 12 GiB per step, eager 1.54 s / 19 GiB). Compiling the whole forward as one
graph lets the backward keep the recomputed activations of several blocks (27 GiB).

Only calls with gradients use the compiled code: validation and sampling (no grad, other shapes) stay eager.
"""
import torch
import torch.nn as nn

from PhysDock.models.primitives.transformers import Stack, call, checkpointed


def _compile_method(module: nn.Module, name: str) -> None:
    """Replace module.<name> by a compiled version used while grads are on."""
    eager = getattr(module, name)
    compiled = torch.compile(getattr(type(module), name), dynamic=False)

    def region(*args, **kwargs):
        if torch.is_grad_enabled():
            return compiled(module, *args, **kwargs)
        return eager(*args, **kwargs)
    setattr(module, name, region)


def compile_training(model: nn.Module, loss_fn: nn.Module = None) -> None:
    """Compile the training computation of a PhysDock model (and PhysDockLoss) in place.

    Parameter names, state dicts and checkpoints are unchanged.
    """
    torch._dynamo.config.inline_inbuilt_nn_modules = True     # parameters as graph inputs: blocks share graphs
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    # DDP: dynamo's bucket-wise graph splitting does not support the checkpoint operator (torch 2.4)
    torch._dynamo.config.optimize_ddp = False
    block_checkpointed, block = torch.compile(checkpointed, dynamic=False), torch.compile(call, dynamic=False)
    for stack in model.modules():
        if isinstance(stack, Stack):
            stack.checkpointed, stack.call = block_checkpointed, block
    encoder = model.diffusion_conditioning
    for module, name in [(encoder.atom_embedder, "embed"), (encoder.token_embedder, "embed"), (encoder, "broadcast"),
                         (model.dit, "condition")] + ([(loss_fn, "terms")] if loss_fn is not None else []):
        _compile_method(module, name)

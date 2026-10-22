"""Triton kernels of Pomelo (none yet: every layer is plain torch). As in PhysDock, a kernel would be a function with a fused forward and backward and a condition under which it
applies, wrapped by a torch fallback in pomelo.models.primitives."""

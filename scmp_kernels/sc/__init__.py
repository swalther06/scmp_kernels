"""Stochastic-computing kernels.

Two parallel public APIs:

1. **Granularity dispatcher** — ``sc_matmul(a, b, granularity=..., ...)``
   from ``matmul.py``. Recommended for new callers.

2. **Flat-API compatibility surface** — explicit ``sc_matmul_*`` names
   matching the historical ``sc_triton`` module. Used by application code
   (Q-DiT and similar) that imports specific specialized functions.

All inputs/outputs are float32; quantization happens inside the Triton kernels.
"""

from .matmul import sc_matmul

from .kernels import (
    det_kernel_tuning,
    clear_rng_cache,
    sc_matmul_per_tensor,
    sc_matmul_mlp,
    sc_matmul_grouped,
    sc_matmul_enable_triton,
    sc_matmul_enable_triton_mlp,
    sc_matmul_grouped_enable_triton,
    sc_matmul_enable_batched_bipolar,
)

__all__ = [
    "sc_matmul",
    "det_kernel_tuning",
    "clear_rng_cache",
    "sc_matmul_per_tensor",
    "sc_matmul_mlp",
    "sc_matmul_grouped",
    "sc_matmul_enable_triton",
    "sc_matmul_enable_triton_mlp",
    "sc_matmul_grouped_enable_triton",
    "sc_matmul_enable_batched_bipolar",
]

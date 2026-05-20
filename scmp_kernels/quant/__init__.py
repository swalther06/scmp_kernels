"""FP → int quantization for SC kernels.

Split out of :mod:`scmp_kernels.sc.kernels` so different quant strategies can
be explored independently of the SC matmul implementation. The output is the
SC-domain integer representation (``boundary``) consumed by the enable-signal
/ table-lookup matmul kernels:

* Bipolar  → ``(boundary int16, sign int8, scale)``
* Unipolar → ``(boundary int32, scale, zp[, row_sum])``

Two flavors are provided:

* :mod:`.fused` — Triton-fused per-tensor / per-row quant (one launch).
* :mod:`.grouped` — pure-PyTorch row-group quant for the per-row matmul path.
"""

from .fused import (
    fused_quant_kernel,
    fused_quant_bipolar_batched_kernel,
    fused_quantize_bipolar,
    fused_quantize_bipolar_perrow,
    fused_quantize_unipolar,
    _quant_dummy,
)
from .grouped import (
    _grouped_symmetric_quant,
    _grouped_asymmetric_quant,
    _grouped_symmetric_quant_batched,
)

__all__ = [
    "fused_quant_kernel",
    "fused_quant_bipolar_batched_kernel",
    "fused_quantize_bipolar",
    "fused_quantize_bipolar_perrow",
    "fused_quantize_unipolar",
    "_quant_dummy",
    "_grouped_symmetric_quant",
    "_grouped_asymmetric_quant",
    "_grouped_symmetric_quant_batched",
]

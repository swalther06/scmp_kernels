"""Fused FP → int quantization kernels and host wrappers.

Produces the SC-domain integer representation consumed by enable-signal /
table-lookup matmul kernels in :mod:`scmp_kernels.sc.kernels`:

* Bipolar  → (boundary int16, sign int8, scale float)
* Unipolar → (boundary int32, scale, zp, optional row_sum)

`boundary` is the rounded value of ``|x_int| * max_rng_val / q_max`` (bipolar)
or ``x_int * max_rng_val / q_max`` (unipolar) — i.e. the integer quantization
grid remapped to the RNG grid so the downstream lookup tables don't have to.

These kernels are extracted from the original monolithic ``sc/kernels.py`` so
that quant strategies can be explored independently of the SC matmul; the SC
side keeps consuming the same (boundary, sign, scale[, zp, row_sum]) tuples.
"""
from __future__ import annotations

import functools
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


# Local import to avoid a circular dependency at module load.  `_resolve_rng_levels`
# is about SC's RNG grid size, not quantization — it stays in sc.kernels.
def _resolve_rng_levels(sc_prec: int, rng_levels: Optional[int]) -> int:
    from ..sc.kernels import _resolve_rng_levels as _impl
    return _impl(sc_prec, rng_levels)


@triton.jit
def fused_quant_kernel(
    fp_ptr,            # (rows, cols) float32 input
    boundary_ptr,      # (rows, cols) int16 (bipolar) or int32 (unipolar) output
    sign_ptr,          # (rows, cols) int8 — only written when IS_BIPOLAR
    scale_ptr,         # (rows,) float32 — only written when PER_ROW and IS_BIPOLAR
    row_sum_ptr,       # (rows,) float32 — only written when PER_ROW and not IS_BIPOLAR
    inv_scale,         # float scalar — ignored when PER_ROW and IS_BIPOLAR (computed on-device)
    zp,                # float scalar — only used when not IS_BIPOLAR
    q_max,             # int: symmetric bound (bipolar) or asymmetric upper (unipolar)
    max_rng_val,       # int: 2^sc_prec
    rows, cols,
    BLOCK: tl.constexpr,   # = BLOCK (flat path) or COLS_PAD (per-row path)
    IS_BIPOLAR: tl.constexpr,
    PER_ROW: tl.constexpr,
):
    """Unified fused quant kernel covering 4 (mode × layout) variants.

    Compile-time variants (caller picks via constexpr flags + grid shape):
      • flat   + bipolar  → host inv_scale, writes (boundary int16, sign int8)
      • per_row + bipolar → on-device scale, writes (boundary int16, sign int8, scale)
      • flat   + unipolar → host inv_scale + zp, writes boundary int32
      • per_row + unipolar → host inv_scale + zp, writes (boundary int32, row_sum)

    Grid:
      PER_ROW=True  → (rows,)                  BLOCK = COLS_PAD = next_power_of_2(cols)
      PER_ROW=False → (cdiv(rows*cols, BLOCK),) BLOCK = 1024 (or chosen)
    """
    q_max_f = q_max.to(tl.float32)

    if PER_ROW:
        row = tl.program_id(0)
        if row >= rows:
            return
        col_offsets = tl.arange(0, BLOCK)
        mask = col_offsets < cols
        base = row * cols
        x = tl.load(fp_ptr + base + col_offsets, mask=mask, other=0.0)

        if IS_BIPOLAR:
            # Per-row symmetric: compute scale on-device, write it.
            abs_max = tl.max(tl.abs(x))
            abs_max = tl.maximum(abs_max, 1e-5)
            scale = abs_max / q_max
            inv_scale_local = 1.0 / scale
            tl.store(scale_ptr + row, scale)
            x_scaled = x * inv_scale_local
        else:
            # Per-row unipolar: host-provided scale + zp, write row_sum for zp correction.
            x_scaled = x * inv_scale + zp
    else:
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        total = rows * cols
        mask = offsets < total
        base = 0  # offsets already absolute
        col_offsets = offsets
        x = tl.load(fp_ptr + offsets, mask=mask, other=0.0)
        if IS_BIPOLAR:
            x_scaled = x * inv_scale
        else:
            x_scaled = x * inv_scale + zp

    x_rounded = libdevice.nearbyint(x_scaled)

    if IS_BIPOLAR:
        x_clamped = tl.minimum(tl.maximum(x_rounded, -q_max_f), q_max_f)
        sign_val = tl.where(x_clamped > 0.0, tl.full(x_clamped.shape, 1, dtype=tl.int8),
                            tl.where(x_clamped < 0.0, tl.full(x_clamped.shape, -1, dtype=tl.int8),
                                     tl.full(x_clamped.shape, 0, dtype=tl.int8)))
        mag = tl.abs(x_clamped)
        boundary = libdevice.nearbyint(mag * (max_rng_val / q_max)).to(tl.int16)
        if PER_ROW:
            tl.store(boundary_ptr + base + col_offsets, boundary, mask=mask)
            tl.store(sign_ptr + base + col_offsets, sign_val, mask=mask)
        else:
            tl.store(boundary_ptr + col_offsets, boundary, mask=mask)
            tl.store(sign_ptr + col_offsets, sign_val, mask=mask)
    else:
        x_clamped = tl.minimum(tl.maximum(x_rounded, 0.0), q_max_f)
        boundary = libdevice.nearbyint(x_clamped * (max_rng_val / q_max)).to(tl.int32)
        if PER_ROW:
            tl.store(boundary_ptr + base + col_offsets, boundary, mask=mask)
            # Mask out padded lanes: with other=0.0 load, padded x_scaled = zp,
            # so unmasked x_clamped lanes would each add zp to row_sum.
            row_sum = tl.sum(tl.where(mask, x_clamped, 0.0), axis=0)
            tl.store(row_sum_ptr + row, row_sum)
        else:
            tl.store(boundary_ptr + col_offsets, boundary, mask=mask)


# A single 1-element dummy tensor we can pass for unused output pointers.
# Cached per-device so we don't pay an allocator hit on every kernel launch.
# maxsize=8 covers up to 8 GPUs + CPU; cache lives for the process lifetime.
@functools.lru_cache(maxsize=8)
def _quant_dummy(device: torch.device) -> torch.Tensor:
    return torch.empty(1, dtype=torch.int8, device=device)


def fused_quantize_bipolar_perrow(
    fp_tensor: torch.Tensor,
    sc_prec: int,
    rng_levels: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused per-row bipolar quantization.

    Returns (boundary int16, sign int8, scale_row float32).
    """
    rows, cols = fp_tensor.shape
    q_max = 2 ** (sc_prec - 1) - 1
    max_rng_val = _resolve_rng_levels(sc_prec, rng_levels)

    boundary = torch.empty(rows, cols, dtype=torch.int16, device=fp_tensor.device)
    sign = torch.empty(rows, cols, dtype=torch.int8, device=fp_tensor.device)
    scale_row = torch.empty(rows, dtype=torch.float32, device=fp_tensor.device)
    dummy = _quant_dummy(fp_tensor.device)

    COLS_PAD = triton.next_power_of_2(cols)
    fused_quant_kernel[(rows,)](
        fp_tensor, boundary, sign, scale_row, dummy,
        0.0, 0.0, q_max, max_rng_val,
        rows, cols, COLS_PAD,
        IS_BIPOLAR=True, PER_ROW=True,
    )
    return boundary, sign, scale_row


def fused_quantize_bipolar(
    fp_tensor: torch.Tensor,
    abs_max: float,
    sc_prec: int,
    rng_levels: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Fused per-tensor bipolar quantization. Returns (boundary int16, sign int8, scale)."""
    rows, cols = fp_tensor.shape
    q_max = 2 ** (sc_prec - 1) - 1
    max_rng_val = _resolve_rng_levels(sc_prec, rng_levels)
    abs_max = max(abs_max, 1e-5)
    scale = abs_max / q_max
    inv_scale = 1.0 / scale

    boundary = torch.empty(rows, cols, dtype=torch.int16, device=fp_tensor.device)
    sign = torch.empty(rows, cols, dtype=torch.int8, device=fp_tensor.device)
    dummy = _quant_dummy(fp_tensor.device)

    total = rows * cols
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    fused_quant_kernel[grid](
        fp_tensor, boundary, sign, dummy, dummy,
        inv_scale, 0.0, q_max, max_rng_val,
        rows, cols, BLOCK,
        IS_BIPOLAR=True, PER_ROW=False,
    )
    return boundary, sign, scale


def fused_quantize_unipolar(
    fp_tensor: torch.Tensor,
    fp_max: float,
    fp_min: float,
    sc_prec: int,
    compute_sum: bool = False,
    rng_levels: Optional[int] = None,
) -> tuple[torch.Tensor, float, float, torch.Tensor | None]:
    """Fused unipolar quantization. Returns (boundary int32, scale, zp, row_sum-or-None)."""
    rows, cols = fp_tensor.shape
    q_max = 2 ** sc_prec - 1
    max_rng_val = _resolve_rng_levels(sc_prec, rng_levels)
    range_fp = max(fp_max - fp_min, 1e-5)
    scale = range_fp / q_max
    inv_scale = 1.0 / scale
    zp = round(-fp_min / scale)
    zp = max(0, min(q_max, zp))
    zp_f = float(zp)

    boundary = torch.empty(rows, cols, dtype=torch.int32, device=fp_tensor.device)
    dummy = _quant_dummy(fp_tensor.device)

    if compute_sum:
        row_sum = torch.empty(rows, dtype=torch.float32, device=fp_tensor.device)
        COLS_BLOCK = triton.next_power_of_2(cols)
        fused_quant_kernel[(rows,)](
            fp_tensor, boundary, dummy, dummy, row_sum,
            inv_scale, zp_f, q_max, max_rng_val,
            rows, cols, COLS_BLOCK,
            IS_BIPOLAR=False, PER_ROW=True,
        )
        return boundary, scale, zp_f, row_sum
    else:
        total = rows * cols
        BLOCK = 1024
        grid = (triton.cdiv(total, BLOCK),)
        fused_quant_kernel[grid](
            fp_tensor, boundary, dummy, dummy, dummy,
            inv_scale, zp_f, q_max, max_rng_val,
            rows, cols, BLOCK,
            IS_BIPOLAR=False, PER_ROW=False,
        )
        return boundary, scale, zp_f, None


@triton.jit
def fused_quant_bipolar_batched_kernel(
    fp_ptr,            # (BH, rows, cols) float32 input
    boundary_ptr,      # (BH, cols, rows) int16 output — transposed layout
    sign_ptr,          # (BH, cols, rows) int8 output — transposed layout
    inv_scale_ptr,     # (BH,) float32 — per-head inv_scale
    q_max,             # int: 2^(sc_prec-1) - 1
    q_min,             # int: -(2^(sc_prec-1))
    max_rng_val,       # int: 2^sc_prec - 1
    slice_size,        # int: rows * cols (elements per head)
    rows,              # int: number of rows (N for q, M for k)
    cols,              # int: number of cols (D)
    BLOCK: tl.constexpr,
):
    """Batched bipolar quantization with fused transpose.

    Reads from (BH, rows, cols) and writes to (BH, cols, rows) layout,
    eliminating the 4 separate .transpose().contiguous() calls."""
    batch_id = tl.program_id(1)
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < slice_size

    # Load from (BH, rows, cols) layout — linear access
    base_in = batch_id * slice_size
    x = tl.load(fp_ptr + base_in + offsets, mask=mask, other=0.0)
    inv_scale = tl.load(inv_scale_ptr + batch_id)

    x_scaled = x * inv_scale
    x_rounded = libdevice.nearbyint(x_scaled)
    x_clamped = tl.minimum(tl.maximum(x_rounded, q_min.to(tl.float32)), q_max.to(tl.float32))

    sign_val = tl.where(x_clamped > 0.0, tl.full(x_clamped.shape, 1, dtype=tl.int8),
                        tl.where(x_clamped < 0.0, tl.full(x_clamped.shape, -1, dtype=tl.int8),
                                 tl.full(x_clamped.shape, 0, dtype=tl.int8)))
    mag = tl.abs(x_clamped)
    boundary = libdevice.nearbyint(mag * (max_rng_val / q_max)).to(tl.int16)

    # Compute transposed store offsets: (row, col) → (col, row)
    # linear offset → (row_idx, col_idx) → store at col_idx * rows + row_idx
    row_idx = offsets // cols
    col_idx = offsets % cols
    base_out = batch_id * slice_size
    store_offsets = base_out + col_idx * rows + row_idx

    tl.store(boundary_ptr + store_offsets, boundary, mask=mask)
    tl.store(sign_ptr + store_offsets, sign_val, mask=mask)


__all__ = [
    "fused_quant_kernel",
    "fused_quant_bipolar_batched_kernel",
    "fused_quantize_bipolar",
    "fused_quantize_bipolar_perrow",
    "fused_quantize_unipolar",
    "_quant_dummy",
]

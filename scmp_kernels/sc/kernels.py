"""
Triton GPU-accelerated stochastic computing matrix multiplication.

This module provides a drop-in replacement for matmul_sc() using Triton kernels
with bit-packed representation for maximum performance.

Supports:
- Bipolar mode (XNOR gate): for symmetric quantization, values in [-max, max]
- Unipolar mode (AND gate): for asymmetric quantization, values in [0, max]

Supports all config types:
- LFSR with per-element scrambling
- Fully independent LFSRs
- Sobol sequences (simple and DSE)
"""
from __future__ import annotations

import json
import math
import os
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
import numpy as np
from typing import Optional

# Import from new architecture
from .sng import RNGPool, SNGBank
from .constants import FP8_E4M3_MAX, FP8_E5M2_MAX, INT8_MAX


# =============================================================================
# RNG Sequence Cache
# =============================================================================

_rng_seq_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _config_cache_key(config: dict, sc_prec: int, device: torch.device) -> str:
    """Create a hashable cache key from config, precision, and device."""
    return json.dumps(config, sort_keys=True) + f"|{sc_prec}|{device}"


def _get_cached_sequences(
    config: dict, sc_prec: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get RNG sequences from cache, or generate and cache on miss.

    Returns:
        (rand_seqs_a_t, rand_seqs_b_t): int32 tensors on the given device.
    """
    key = _config_cache_key(config, sc_prec, device)
    if key not in _rng_seq_cache:
        stoc_len = 2 ** sc_prec
        rng_pool = RNGPool(config["rng_pool"], sc_prec)
        sng_a = SNGBank(rng_pool, config["sng"]["q"])
        sng_b = SNGBank(rng_pool, config["sng"]["k"])
        rand_seqs_a_t = torch.tensor(
            sng_a.get_all_sequences(stoc_len), dtype=torch.int32, device=device)
        rand_seqs_b_t = torch.tensor(
            sng_b.get_all_sequences(stoc_len), dtype=torch.int32, device=device)
        _rng_seq_cache[key] = (rand_seqs_a_t, rand_seqs_b_t)
    return _rng_seq_cache[key]


def clear_rng_cache():
    """Clear the cached RNG sequences and enable tables to free GPU memory."""
    _rng_seq_cache.clear()
    _enable_table_cache.clear()
    _k_table_cache.clear()



# =============================================================================
# Enable-Signal Kernels (Table-Lookup Matmul)
# =============================================================================

@triton.jit
def build_cum_indicator_kernel(
    rng_b_ptr,       # (D, stoc_len) int32 — per-dimension RNG sequences for B
    cum_ptr,         # (D, stoc_len+1, V) int16 — output cumulative indicator table
    D: tl.constexpr,
    stoc_len: tl.constexpr,
    V: tl.constexpr,          # max_rng_val + 1 = 2^sc_prec
):
    """
    Build cumulative indicator table for B's RNG sequence.

    cum[d, k, v] = |{i < k : rng_b[d, i] <= v}|

    One program per dimension d.
    """
    d = tl.program_id(0)
    if d >= D:
        return

    v_range = tl.arange(0, V)
    cum_stride_d = (stoc_len + 1) * V  # stride for d dimension

    # Initialize cum[d, 0, :] = 0
    tl.store(cum_ptr + d * cum_stride_d + v_range,
             tl.zeros([V], dtype=tl.int16))

    # Build prefix sums
    running = tl.zeros([V], dtype=tl.int16)
    for k in range(stoc_len):
        r = tl.load(rng_b_ptr + d * stoc_len + k)
        delta = tl.where(v_range > r,
                         tl.full([V], 1, dtype=tl.int16),
                         tl.zeros([V], dtype=tl.int16))
        running = running + delta
        offset = d * cum_stride_d + (k + 1) * V
        tl.store(cum_ptr + offset + v_range, running)


@triton.jit
def compute_k_table_kernel(
    rng_a_ptr,       # (D, stoc_len) int32 — per-dimension RNG sequences for A
    k_table_ptr,     # (D, V) int16 — output k-table
    D: tl.constexpr,
    stoc_len: tl.constexpr,
    V: tl.constexpr,          # max_rng_val + 1
):
    """
    Compute popcount table for A's RNG sequence.

    k_table[d, v] = |{t : rng_a[d, t] <= v}|

    One program per dimension d.
    """
    d = tl.program_id(0)
    if d >= D:
        return

    v_range = tl.arange(0, V)

    # Count how many rng_a values are <= each v
    counts = tl.zeros([V], dtype=tl.int16)
    for t in range(stoc_len):
        r = tl.load(rng_a_ptr + d * stoc_len + t)
        counts += tl.where(v_range > r,
                           tl.full([V], 1, dtype=tl.int16),
                           tl.zeros([V], dtype=tl.int16))

    tl.store(k_table_ptr + d * V + v_range, counts)


# =============================================================================
# Enable-Signal Tiled Kernels
# =============================================================================

@triton.jit
def enable_matmul_tiled_kernel(
    cum_ptr,           # (D, stoc_len+1, V) int16
    k_table_ptr,       # (D, V) int16
    boundary_a_ptr,    # (D, N) int16 — transposed for coalesced access
    boundary_b_ptr,    # (D, M) int16 — transposed for coalesced access
    sign_a_ptr,        # (D, N) int8 — only read when IS_BIPOLAR
    sign_b_ptr,        # (D, M) int8 — only read when IS_BIPOLAR
    output_ptr,        # (N, M) float32
    N, M, D,
    stoc_len: tl.constexpr,
    V: tl.constexpr,
    q_max_sq,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_BIPOLAR: tl.constexpr,
):
    """Tiled enable-signal matmul.

    BLOCK_K tiles the D dimension with static_range for compiler unrolling.
    Boundary/sign tensors use (D, N) layout for coalesced thread access.
    ``IS_BIPOLAR`` selects sign-magnitude (loads sa/sb, with all-zero skip)
    vs asymmetric (no sign loads). Sign pointers are only read when
    ``IS_BIPOLAR`` is True; pass any valid dummy tensor in the unipolar path.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < N
    n_mask = n_offsets < M
    gather_mask = m_mask[:, None] & n_mask[None, :]

    cum_stride_d = (stoc_len + 1) * V
    scale = q_max_sq / stoc_len
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    num_k_blocks = (D + BLOCK_K - 1) // BLOCK_K
    for k_block in range(num_k_blocks):
        k_start = k_block * BLOCK_K
        for ki in tl.static_range(BLOCK_K):
            d = k_start + ki
            if d < D:
                if IS_BIPOLAR:
                    # Bipolar fast path: sign load + early skip on all-zero rows/cols.
                    sa_i8 = tl.load(sign_a_ptr + d * N + m_offsets, mask=m_mask, other=0)
                    if tl.sum(tl.abs(sa_i8).to(tl.int32)) > 0:
                        sb_i8 = tl.load(sign_b_ptr + d * M + n_offsets, mask=n_mask, other=0)
                        if tl.sum(tl.abs(sb_i8).to(tl.int32)) > 0:
                            sa = sa_i8.to(tl.float32)
                            sb = sb_i8.to(tl.float32)
                            ba = tl.load(boundary_a_ptr + d * N + m_offsets, mask=m_mask, other=0).to(tl.int32)
                            bb = tl.load(boundary_b_ptr + d * M + n_offsets, mask=n_mask, other=0).to(tl.int32)
                            k_vals = tl.load(k_table_ptr + d * V + ba, mask=m_mask, other=0).to(tl.int32)
                            cum_offsets = (d * cum_stride_d
                                           + k_vals[:, None].to(tl.int64) * V
                                           + bb[None, :].to(tl.int64))
                            counts = tl.load(cum_ptr + cum_offsets, mask=gather_mask, other=0).to(tl.float32)
                            acc += counts * sa[:, None] * sb[None, :]
                else:
                    ba = tl.load(boundary_a_ptr + d * N + m_offsets, mask=m_mask, other=0).to(tl.int32)
                    bb = tl.load(boundary_b_ptr + d * M + n_offsets, mask=n_mask, other=0).to(tl.int32)
                    k_vals = tl.load(k_table_ptr + d * V + ba, mask=m_mask, other=0).to(tl.int32)
                    cum_offsets = (d * cum_stride_d
                                   + k_vals[:, None].to(tl.int64) * V
                                   + bb[None, :].to(tl.int64))
                    counts = tl.load(cum_ptr + cum_offsets, mask=gather_mask, other=0).to(tl.float32)
                    acc += counts

    # Apply loop-invariant scale once (enables FMA fusion in the inner loop)
    acc *= scale

    out_offsets = m_offsets[:, None] * M + n_offsets[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + out_offsets, acc, mask=out_mask)


# =============================================================================
# Enable-Signal Compact Kernels (no cum_indicator table, O(D*V) memory)
# =============================================================================

# =============================================================================
# Enable-Signal Compact Kernels for MLP
# Two versions:
#   - "dot" kernel: tl.dot vectorized, single program per (m,n) tile.
#     Used by enable_matmul_compact() for attention (small D).
#   - "splitd" kernel: tl.dot + split-D parallelism + transposed inputs.
#     Used by enable_matmul_compact_mlp() for MLP (large D).
#     Inputs are (D,N)/(D,M) layout for coalesced memory access.
#     Grid z-axis splits D into BLOCK_D chunks with atomic accumulation.
# =============================================================================

@triton.jit
def enable_matmul_compact_dot_kernel(
    rng_b_ptr,         # (D, stoc_len) int32
    k_table_ptr,       # (D, V) int16
    boundary_a_ptr,    # (N, D) int16 (bipolar) or int32 (unipolar)
    boundary_b_ptr,    # (M, D) int16 (bipolar) or int32 (unipolar)
    sign_a_ptr,        # (N, D) int8 — only read when IS_BIPOLAR
    sign_b_ptr,        # (M, D) int8 — only read when IS_BIPOLAR
    output_ptr,        # (N, M) float32
    N, M, D,
    stoc_len: tl.constexpr,
    V: tl.constexpr,
    q_max_sq,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BATCH_T: tl.constexpr,
    IS_BIPOLAR: tl.constexpr,
):
    """Compact enable-signal matmul. tl.dot vectorized, no split-D.

    For small D (attention). Inputs in (N, D) / (M, D) row-major layout.
    ``IS_BIPOLAR`` selects sign-magnitude (bipolar, multiplies by sa*sb) vs
    asymmetric (unipolar, no sign multiplication). Sign pointers are loaded
    only when ``IS_BIPOLAR`` is True; pass any valid dummy tensor for them
    in the unipolar call path.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < N
    n_mask = n_offsets < M

    scale = q_max_sq / stoc_len
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    num_batches: tl.constexpr = stoc_len // BATCH_T

    for d in range(D):
        ba = tl.load(boundary_a_ptr + m_offsets * D + d, mask=m_mask, other=0).to(tl.int32)
        bb = tl.load(boundary_b_ptr + n_offsets * D + d, mask=n_mask, other=0).to(tl.int32)
        k_vals = tl.load(k_table_ptr + d * V + ba, mask=m_mask, other=0).to(tl.int32)

        counts = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
        for tb in range(num_batches):
            t_base = tb * BATCH_T
            t_indices = t_base + tl.arange(0, BATCH_T)
            rng_vals = tl.load(rng_b_ptr + d * stoc_len + t_indices)
            t_ok = (t_indices[None, :] < k_vals[:, None]).to(tl.int8)
            r_ok = (bb[None, :] > rng_vals[:, None]).to(tl.int8)
            counts += tl.dot(t_ok, r_ok, out_dtype=tl.int32)

        if IS_BIPOLAR:
            sa = tl.load(sign_a_ptr + m_offsets * D + d, mask=m_mask, other=0).to(tl.float32)
            sb = tl.load(sign_b_ptr + n_offsets * D + d, mask=n_mask, other=0).to(tl.float32)
            acc += counts.to(tl.float32) * scale * sa[:, None] * sb[None, :]
        else:
            acc += counts.to(tl.float32) * scale

    out_offsets = m_offsets[:, None] * M + n_offsets[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + out_offsets, acc, mask=out_mask)


# --- Split-D kernels for MLP (transposed inputs, atomic accumulation) ---

# =============================================================================
# Fused Quantization Kernels
# Replaces scattered PyTorch elementwise ops (div, round, clamp, sign, abs,
# boundary computation) with a single kernel pass per operand.
# =============================================================================

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
            row_sum = tl.sum(x_clamped, axis=0)
            tl.store(row_sum_ptr + row, row_sum)
        else:
            tl.store(boundary_ptr + col_offsets, boundary, mask=mask)


# A single 1-element dummy tensor we can pass for unused output pointers.
def _quant_dummy(device):
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
) -> tuple[torch.Tensor, float, float, float, torch.Tensor | None]:
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


# Threshold: use compact path when cum_indicator would exceed this many bytes.
# On Blackwell (RTX PRO 6000 measured) the table kernel's gather beats the
# compact kernel's tl.dot inner loop — table ~25 s/it vs compact ~28 s/it e2e.
# On older cards with smaller L2 (e.g. 4080), compact was ~6% faster because
# the 18 MB cum_indicator polluted L2 (commit d60e442). Default now prefers
# table; set SC_FORCE_COMPACT=1 to force compact (recover 4080-era behaviour).
_COMPACT_ENABLE_THRESHOLD_BYTES = 1 << 40  # default: always table
if os.environ.get("SC_FORCE_COMPACT", "0") == "1":
    _COMPACT_ENABLE_THRESHOLD_BYTES = 0
if os.environ.get("SC_FORCE_TABLE", "0") == "1":
    _COMPACT_ENABLE_THRESHOLD_BYTES = 1 << 40  # no-op: already default

# =============================================================================
# Batched Kernels — one launch for all B*H heads
# =============================================================================

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


@triton.jit
def enable_matmul_bipolar_batched_kernel(
    cum_ptr,           # (D, stoc_len+1, V) int16 — shared across heads
    k_table_ptr,       # (D, V) int16 — shared across heads
    boundary_a_ptr,    # (BH, D, N) int16 — transposed for coalesced access
    boundary_b_ptr,    # (BH, D, M) int16 — transposed for coalesced access
    sign_a_ptr,        # (BH, D, N) int8 — transposed for coalesced access
    sign_b_ptr,        # (BH, D, M) int8 — transposed for coalesced access
    output_ptr,        # (BH, N, M) float32
    scale_ptr,         # (BH,) float32 — per-head (scale_a * scale_b)
    N, M, D,
    stoc_len: tl.constexpr,
    V: tl.constexpr,
    q_max_sq,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Batched tiled enable-signal matmul (bipolar). One launch for all heads.
    Boundary/sign tensors use (BH, D, N/M) layout for coalesced thread access."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    batch_id = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < N
    n_mask = n_offsets < M
    gather_mask = m_mask[:, None] & n_mask[None, :]

    # Per-head base offsets (D*N = N*D, same total stride)
    ba_base = batch_id * D * N
    bb_base = batch_id * D * M

    cum_stride_d = (stoc_len + 1) * V
    inner_scale = q_max_sq / stoc_len
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    num_k_blocks = (D + BLOCK_K - 1) // BLOCK_K
    for k_block in range(num_k_blocks):
        k_start = k_block * BLOCK_K
        for ki in tl.static_range(BLOCK_K):
            d = k_start + ki
            if d < D:
                # Load signs as int8 for early-exit check
                sa_i8 = tl.load(sign_a_ptr + ba_base + d * N + m_offsets, mask=m_mask, other=0)
                # Skip if all sign_a are zero — no contribution to acc
                if tl.sum(tl.abs(sa_i8).to(tl.int32)) > 0:
                    sb_i8 = tl.load(sign_b_ptr + bb_base + d * M + n_offsets, mask=n_mask, other=0)
                    # Skip if all sign_b are zero
                    if tl.sum(tl.abs(sb_i8).to(tl.int32)) > 0:
                        sa = sa_i8.to(tl.float32)
                        sb = sb_i8.to(tl.float32)
                        # Coalesced loads — (D, N/M) layout
                        ba = tl.load(boundary_a_ptr + ba_base + d * N + m_offsets,
                                     mask=m_mask, other=0).to(tl.int32)
                        bb = tl.load(boundary_b_ptr + bb_base + d * M + n_offsets,
                                     mask=n_mask, other=0).to(tl.int32)

                        k_vals = tl.load(k_table_ptr + d * V + ba, mask=m_mask, other=0).to(tl.int32)

                        cum_offsets = (d * cum_stride_d
                                       + k_vals[:, None].to(tl.int64) * V
                                       + bb[None, :].to(tl.int64))
                        counts = tl.load(cum_ptr + cum_offsets, mask=gather_mask, other=0).to(tl.float32)

                        acc += counts * sa[:, None] * sb[None, :]

    # Apply combined inner_scale × head_scale in one multiply
    head_scale = tl.load(scale_ptr + batch_id)
    acc *= inner_scale * head_scale

    out_base = batch_id * N * M
    out_offsets = m_offsets[:, None] * M + n_offsets[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + out_base + out_offsets, acc, mask=out_mask)


def _sc_matmul_per_head_bipolar(
    q_flat: torch.Tensor,       # (BH, N, D) float32
    k_flat: torch.Tensor,       # (BH, N, D) float32
    q_maxs: torch.Tensor,       # (BH,) float32 — per-head max
    q_mins: torch.Tensor,       # (BH,) float32 — per-head min
    k_maxs: torch.Tensor,       # (BH,) float32
    k_mins: torch.Tensor,       # (BH,) float32
    sc_prec: int,
    config: dict,
    stoc_len: Optional[int] = None,
    rng_levels: Optional[int] = None,
) -> torch.Tensor:
    """
    Batched bipolar enable-signal SC matmul for all heads in one launch.

    Replaces the per-head loop with two kernel launches:
    1. fused_quant_bipolar_batched_kernel (for q and k)
    2. enable_matmul_bipolar_batched_kernel

    Args:
        q_flat, k_flat: (BH, N, D) float32 tensors
        q_maxs, q_mins, k_maxs, k_mins: (BH,) per-head ranges
        sc_prec: SC precision
        config: SC RNG config dict
        stoc_len: Stochastic stream length. If None, uses 2^sc_prec.

    Returns:
        output: (BH, N, N) float32
    """
    if stoc_len is None:
        stoc_len = 2 ** sc_prec

    q_flat = q_flat.contiguous()
    k_flat = k_flat.contiguous()
    BH, N, D = q_flat.shape
    M = N  # QK matmul: N x N
    device = q_flat.device

    q_max = 2 ** (sc_prec - 1) - 1
    q_min = -(2 ** (sc_prec - 1))
    max_rng_val = _resolve_rng_levels(sc_prec, rng_levels)
    q_max_sq = float(q_max * q_max)

    # --- Per-head inv_scale on GPU ---
    abs_max_q = torch.maximum(q_maxs.abs(), q_mins.abs()).clamp(min=1e-5)
    abs_max_k = torch.maximum(k_maxs.abs(), k_mins.abs()).clamp(min=1e-5)
    scale_q = abs_max_q / q_max
    scale_k = abs_max_k / q_max
    inv_scale_q = 1.0 / scale_q  # (BH,)
    inv_scale_k = 1.0 / scale_k  # (BH,)

    # --- Batched fused quantization (writes directly in transposed (BH, D, N/M) layout) ---
    slice_size = N * D
    boundary_q = torch.empty(BH, D, N, dtype=torch.int16, device=device)
    sign_q = torch.empty(BH, D, N, dtype=torch.int8, device=device)
    boundary_k = torch.empty(BH, D, M, dtype=torch.int16, device=device)
    sign_k = torch.empty(BH, D, M, dtype=torch.int8, device=device)

    BLOCK = 1024
    grid_quant = (triton.cdiv(slice_size, BLOCK), BH)
    fused_quant_bipolar_batched_kernel[grid_quant](
        q_flat, boundary_q, sign_q, inv_scale_q,
        q_max, q_min, max_rng_val, slice_size, N, D, BLOCK,
    )
    fused_quant_bipolar_batched_kernel[grid_quant](
        k_flat, boundary_k, sign_k, inv_scale_k,
        q_max, q_min, max_rng_val, slice_size, M, D, BLOCK,
    )

    # --- Get cached tables (shared across all heads) ---
    rand_seqs_a_t, rand_seqs_b_t = _get_cached_sequences(config, sc_prec, device)
    V = max_rng_val + 1
    cum_table_bytes = D * (stoc_len + 1) * V * 2
    use_compact = cum_table_bytes > _COMPACT_ENABLE_THRESHOLD_BYTES

    if use_compact:
        # Compact path not batched yet — fall back to per-head loop
        # Compact kernel expects (N, D) layout; transpose from (D, N)
        k_table = _get_cached_k_table(
            config, sc_prec, device, rand_seqs_a_t, stoc_len, rng_levels=max_rng_val
        )
        rng_b_prefix = _prepare_rng_prefix(rand_seqs_b_t, sc_prec, stoc_len, max_rng_val)
        output = torch.empty(BH, N, M, dtype=torch.float32, device=device)
        for i in range(BH):
            sc_raw = enable_matmul_compact(
                rng_b_prefix, k_table,
                boundary_q[i].t().contiguous(), boundary_k[i].t().contiguous(),
                sign_q[i].t().contiguous(), sign_k[i].t().contiguous(),
                N, M, D, stoc_len, q_max_sq, is_bipolar=True,
            )
            output[i] = sc_raw * (scale_q[i] * scale_k[i]).item()
        return output

    cum_indicator, k_table = _get_cached_enable_tables(
        config, sc_prec, device, rand_seqs_a_t, rand_seqs_b_t,
        stoc_len, rng_levels=max_rng_val)

    # V from actual table layout (may be padded to power-of-2)
    V_actual = cum_indicator.shape[2]

    # Per-head output scale on GPU
    out_scale = scale_q * scale_k  # (BH,)

    # --- Batched matmul kernel ---
    # boundary/sign already in (BH, D, N/M) layout from fused-transpose quant kernel
    output = torch.empty(BH, N, M, dtype=torch.float32, device=device)

    # Adaptive tile size: smaller tiles for small N/M to increase GPU occupancy
    if N <= 64 or M <= 64:
        BLOCK_M = 16
        BLOCK_N = 16
    else:
        BLOCK_M = 32
        BLOCK_N = 32
    # BLOCK_K=4 reduces register pressure vs 8, improving occupancy
    if D >= 4 and D % 4 == 0:
        BLOCK_K = 4
    elif D % 2 == 0:
        BLOCK_K = 2
    else:
        BLOCK_K = 1

    # Warp count tuning: more warps for larger tiles, fewer for smaller
    nw = 8 if BLOCK_M == 32 else 2
    grid_mm = (triton.cdiv(N, BLOCK_M), triton.cdiv(M, BLOCK_N), BH)
    enable_matmul_bipolar_batched_kernel[grid_mm](
        cum_indicator, k_table,
        boundary_q, boundary_k,
        sign_q, sign_k,
        output, out_scale,
        N, M, D,
        stoc_len, V_actual, q_max_sq,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=nw,
    )

    return output


# =============================================================================
# Enable-Signal Host Functions
# =============================================================================

_enable_table_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
_k_table_cache: dict[str, torch.Tensor] = {}


def _resolve_rng_levels(sc_prec: int, rng_levels: Optional[int]) -> int:
    """Resolve the RNG/grid size used by enable-signal lookup tables.

    Legacy behavior ties the enable grid to ``2**sc_prec``. The fixed-level
    runtime mode keeps quantization at a fixed ``sc_prec`` while varying the
    effective stochastic stream length, in which case callers can override the
    enable grid with the real ``stoc_len``.
    """
    if rng_levels is None:
        return 2 ** sc_prec
    return int(rng_levels)


# Owen-style per-dimension XOR scramble. Used by fixed-level SC mode to break
# the Sobol-prefix stratification artifact: when stoc_len < 2**sc_prec, the
# first `stoc_len` values of a Sobol-sc_prec sequence fall on a coarse lattice
# (multiples of 2**(sc_prec - ceil(log2(stoc_len)))), which biases
# small-magnitude boundaries. A deterministic per-dim XOR mask shifts each
# dimension's stratum origin independently, removing the systematic bias
# without sacrificing low-discrepancy inside each stratum.
#
# Mask source is selected via env var SC_OWEN_MODE:
#   - "counter" (default): m[d] = d mod base_levels. Round-robin "clock"
#     mask. Strictly equipartitioned across all 2-power moduli (mod 2, 4,
#     8, ...), so a single mask works correctly for every stoc_len value
#     and every per-row mixed-precision schedule with no recalibration.
#     Hardware-friendly: implementable as a wire tap on the row counter,
#     no ROM needed.
#   - "bitrev": m[d] = bit_reverse(d mod base_levels). Same equipartition
#     property as "counter" but breaks "low bits run consecutively" so
#     adjacent-D correlations don't resonate with the mask period.
#   - "random": legacy behavior. m[d] ~ Uniform[0, base_levels) drawn from
#     a fixed seed (deterministic but with sampling fluctuations).
#   - "off": disable scrambling (same as SC_DISABLE_OWEN=1; biased path).
#
# Cached enable tables are keyed by (config, sc_prec, stoc_len, rng_levels)
# but NOT by the scramble mode/seed; switching modes mid-process requires
# clear_rng_cache().
_OWEN_SCRAMBLE_SEED = 0x5A5A5A5A


def _bit_reverse(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Bit-reverse the lower ``n_bits`` of each integer in ``x``."""
    y = torch.zeros_like(x)
    for i in range(n_bits):
        y = y | (((x >> i) & 1) << (n_bits - 1 - i))
    return y


def _owen_scramble(prefix: torch.Tensor, base_levels: int) -> torch.Tensor:
    """Deterministic per-dimension XOR mask on ``prefix``."""
    if os.environ.get("SC_DISABLE_OWEN", "0") == "1":
        return prefix.contiguous()

    mode = os.environ.get("SC_OWEN_MODE", "counter").lower()
    if mode == "off":
        return prefix.contiguous()

    D = prefix.shape[0]

    if mode == "counter":
        idx = torch.arange(D, device=prefix.device, dtype=torch.int64) % base_levels
        masks = idx.to(prefix.dtype).unsqueeze(1)
    elif mode == "bitrev":
        n_bits = int(round(math.log2(base_levels)))
        idx = torch.arange(D, device=prefix.device, dtype=torch.int64) % base_levels
        masks = _bit_reverse(idx, n_bits).to(prefix.dtype).unsqueeze(1)
    else:  # "random" — legacy fixed-seed PRNG
        g = torch.Generator(device=prefix.device).manual_seed(_OWEN_SCRAMBLE_SEED)
        masks = torch.randint(
            0, base_levels, (D, 1), generator=g, device=prefix.device
        ).to(prefix.dtype)

    return (prefix ^ masks).contiguous()


def _prepare_rng_prefix(
    rng: torch.Tensor,
    sc_prec: int,
    stoc_len: int,
    rng_levels: Optional[int],
) -> torch.Tensor:
    """Slice and, if needed, rescale RNG integers onto a smaller enable grid."""
    grid_levels = _resolve_rng_levels(sc_prec, rng_levels)
    base_levels = 2 ** sc_prec
    is_prefix = stoc_len < rng.shape[1]
    prefix = rng[:, :stoc_len].contiguous() if is_prefix else rng
    if grid_levels == base_levels:
        # Fixed-level path: if we're truncating a longer Sobol sequence, apply
        # Owen scramble to break the prefix stratification artifact. When the
        # sequence is used in full (non-truncated), no scramble is needed.
        if is_prefix:
            return _owen_scramble(prefix, base_levels)
        return prefix

    prefix_i64 = prefix.to(torch.int64)
    scaled = torch.div(prefix_i64 * grid_levels, base_levels, rounding_mode="floor")
    return scaled.to(prefix.dtype).contiguous()


def _enable_table_cache_key(config: dict, sc_prec: int, device: torch.device) -> str:
    """Cache key for enable tables (same as RNG cache key)."""
    return json.dumps(config, sort_keys=True) + f"|{sc_prec}|{device}|enable"


def build_enable_tables(
    rng_a: torch.Tensor,
    rng_b: torch.Tensor,
    sc_prec: int,
    stoc_len: Optional[int] = None,
    rng_levels: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build lookup tables for enable-signal multiplication.

    Args:
        rng_a: (D, max_stoc_len) int32 tensor — per-dimension RNG sequences for A
        rng_b: (D, max_stoc_len) int32 tensor — per-dimension RNG sequences for B
        sc_prec: SC precision (controls quantization grid: V = 2^sc_prec + 1)
        stoc_len: Stochastic stream length. If None, uses 2^sc_prec.
                  When < 2^sc_prec, uses a prefix of the RNG sequences.

    Returns:
        cum_indicator: (D, stoc_len+1, V) int16 tensor
        k_table: (D, V) int16 tensor
    """
    if stoc_len is None:
        stoc_len = 2 ** sc_prec
    D = rng_a.shape[0]
    grid_levels = _resolve_rng_levels(sc_prec, rng_levels)
    V = grid_levels + 1
    # Triton tl.arange requires power-of-2 sizes; pad V for the kernel
    V_PADDED = triton.next_power_of_2(V)
    device = rng_a.device

    # Use prefix of RNG sequences for shorter stoc_len
    rng_a_prefix = _prepare_rng_prefix(rng_a, sc_prec, stoc_len, grid_levels)
    rng_b_prefix = _prepare_rng_prefix(rng_b, sc_prec, stoc_len, grid_levels)

    cum_indicator = torch.zeros(D, stoc_len + 1, V_PADDED, dtype=torch.int16, device=device)
    k_table = torch.zeros(D, V_PADDED, dtype=torch.int16, device=device)

    # Launch table-build kernels
    build_cum_indicator_kernel[(D,)](
        rng_b_prefix, cum_indicator,
        D, stoc_len, V_PADDED,
    )
    compute_k_table_kernel[(D,)](
        rng_a_prefix, k_table,
        D, stoc_len, V_PADDED,
    )

    return cum_indicator, k_table


def _get_cached_enable_tables(
    config: dict, sc_prec: int, device: torch.device,
    rng_a: torch.Tensor, rng_b: torch.Tensor,
    stoc_len: Optional[int] = None,
    rng_levels: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get or build+cache enable tables (cum_indicator + k_table)."""
    if stoc_len is None:
        stoc_len = 2 ** sc_prec
    grid_levels = _resolve_rng_levels(sc_prec, rng_levels)
    key = _enable_table_cache_key(config, sc_prec, device) + f"|sl={stoc_len}|rng={grid_levels}"
    if key not in _enable_table_cache:
        _enable_table_cache[key] = build_enable_tables(
            rng_a, rng_b, sc_prec, stoc_len, rng_levels=grid_levels
        )
    return _enable_table_cache[key]


def _get_cached_k_table(
    config: dict, sc_prec: int, device: torch.device,
    rng_a: torch.Tensor,
    stoc_len: Optional[int] = None,
    rng_levels: Optional[int] = None,
) -> torch.Tensor:
    """Get or build+cache k_table only (for compact path)."""
    if stoc_len is None:
        stoc_len = 2 ** sc_prec
    grid_levels = _resolve_rng_levels(sc_prec, rng_levels)
    key = _enable_table_cache_key(config, sc_prec, device) + f"|k_only|sl={stoc_len}|rng={grid_levels}"
    if key not in _k_table_cache:
        _k_table_cache[key] = build_k_table_only(
            rng_a, sc_prec, stoc_len, rng_levels=grid_levels
        )
    return _k_table_cache[key]


def enable_matmul_triton(
    cum_indicator: torch.Tensor,
    k_table: torch.Tensor,
    boundary_a: torch.Tensor,
    boundary_b: torch.Tensor,
    sign_a: torch.Tensor,
    sign_b: torch.Tensor,
    N: int, M: int, D: int,
    stoc_len: int,
    q_max_sq: float,
    is_bipolar: bool,
) -> torch.Tensor:
    """
    Launch enable-signal matmul kernel.

    Args:
        cum_indicator: (D, stoc_len+1, V) int16
        k_table: (D, V) int16
        boundary_a: (N, D) int16
        boundary_b: (M, D) int16
        sign_a: (N, D) int8 (bipolar only)
        sign_b: (M, D) int8 (bipolar only)
        N, M, D: matrix dimensions
        stoc_len: stochastic stream length
        q_max_sq: q_max^2 for decoding
        is_bipolar: True for bipolar mode

    Returns:
        output: (N, M) float32
    """
    V = cum_indicator.shape[2]
    output = torch.empty(N, M, dtype=torch.float32, device=boundary_a.device)

    # Transpose boundary/sign to (D, N/M) for coalesced kernel access
    boundary_a = boundary_a.t().contiguous()
    boundary_b = boundary_b.t().contiguous()
    if is_bipolar:
        sign_a = sign_a.t().contiguous()
        sign_b = sign_b.t().contiguous()

    # Adaptive tile size: smaller tiles for small N/M to increase GPU occupancy
    if N <= 64 or M <= 64:
        BLOCK_M = 16
        BLOCK_N = 16
    else:
        BLOCK_M = 32
        BLOCK_N = 32
    # BLOCK_K=4 reduces register pressure vs 8, improving occupancy
    if D >= 4 and D % 4 == 0:
        BLOCK_K = 4
    elif D % 2 == 0:
        BLOCK_K = 2
    else:
        BLOCK_K = 1
    # Warp count tuning: more warps for larger tiles, fewer for smaller
    nw = 8 if BLOCK_M == 32 else 2
    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(M, BLOCK_N))
    if not is_bipolar:
        _dummy_sign = torch.empty(1, dtype=torch.int8, device=output.device)
        sign_a = sign_b = _dummy_sign
    enable_matmul_tiled_kernel[grid](
        cum_indicator, k_table,
        boundary_a, boundary_b,
        sign_a, sign_b,
        output,
        N, M, D, stoc_len, V,
        q_max_sq, BLOCK_M, BLOCK_N, BLOCK_K,
        IS_BIPOLAR=is_bipolar,
        num_warps=nw,
    )

    return output


def build_k_table_only(
    rng_a: torch.Tensor,
    sc_prec: int,
    stoc_len: Optional[int] = None,
    rng_levels: Optional[int] = None,
) -> torch.Tensor:
    """Build only the k_table (not cum_indicator). Used by compact enable path.

    Args:
        rng_a: (D, max_stoc_len) int32 tensor.
        sc_prec: SC precision (controls V = 2^sc_prec + 1).
        stoc_len: Stream length. If None, uses 2^sc_prec.
                  When < 2^sc_prec, uses prefix of rng_a.
    """
    if stoc_len is None:
        stoc_len = 2 ** sc_prec
    D = rng_a.shape[0]
    grid_levels = _resolve_rng_levels(sc_prec, rng_levels)
    V = grid_levels + 1
    V_PADDED = triton.next_power_of_2(V)
    rng_a_prefix = _prepare_rng_prefix(rng_a, sc_prec, stoc_len, grid_levels)
    k_table = torch.zeros(D, V_PADDED, dtype=torch.int16, device=rng_a.device)
    compute_k_table_kernel[(D,)](rng_a_prefix, k_table, D, stoc_len, V_PADDED)
    return k_table


def enable_matmul_compact(
    rng_b: torch.Tensor,
    k_table: torch.Tensor,
    boundary_a: torch.Tensor,
    boundary_b: torch.Tensor,
    sign_a: torch.Tensor,
    sign_b: torch.Tensor,
    N: int, M: int, D: int,
    stoc_len: int,
    q_max_sq: float,
    is_bipolar: bool,
) -> torch.Tensor:
    """
    Compact enable-signal matmul for attention (small D). No split-D.

    Uses rng_b (D, stoc_len) int32 directly, computing counts on-the-fly.
    Inputs in (N, D) / (M, D) row-major layout.
    """
    V = k_table.shape[1]
    output = torch.empty(N, M, dtype=torch.float32, device=boundary_a.device)

    # tl.dot requires K >= 32 on Blackwell (>= 16 on older archs), so the compact
    # dot kernel needs BATCH_T=32. When stoc_len < 32 the inner loop would iterate
    # zero times and silently zero the output. For those tiny streams, build a
    # small cum_indicator on the fly (memory is trivial: D*(stoc_len+1)*V*2 bytes,
    # e.g. ~41 KB for D=72, sl=16, V=257) and dispatch through the table kernel.
    if stoc_len < 32:
        cum_indicator = torch.empty(
            D, stoc_len + 1, V, dtype=torch.int16, device=rng_b.device
        )
        build_cum_indicator_kernel[(D,)](rng_b, cum_indicator, D, stoc_len, V)
        return enable_matmul_triton(
            cum_indicator, k_table, boundary_a, boundary_b, sign_a, sign_b,
            N, M, D, stoc_len, q_max_sq, is_bipolar,
        )

    BLOCK_M = 32
    BLOCK_N = 32
    BATCH_T = 32
    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(M, BLOCK_N))
    # Unipolar path doesn't read sign pointers (constexpr-branched out).
    # Pass dummy 1-element int8 tensors; they're never dereferenced inside.
    if not is_bipolar:
        _dummy_sign = torch.empty(1, dtype=torch.int8, device=output.device)
        sign_a = sign_b = _dummy_sign
    enable_matmul_compact_dot_kernel[grid](
        rng_b, k_table,
        boundary_a, boundary_b,
        sign_a, sign_b,
        output,
        N, M, D, stoc_len, V,
        q_max_sq, BLOCK_M, BLOCK_N, BATCH_T,
        IS_BIPOLAR=is_bipolar,
    )

    return output


def enable_matmul_compact_mlp(
    rng_b: torch.Tensor,
    k_table: torch.Tensor,
    boundary_a: torch.Tensor,
    boundary_b: torch.Tensor,
    sign_a: torch.Tensor,
    sign_b: torch.Tensor,
    N: int, M: int, D: int,
    stoc_len: int,
    q_max_sq: float,
    is_bipolar: bool,
) -> torch.Tensor:
    """
    Chunked cum_indicator matmul for MLP layers (large D).

    Instead of computing counts on-the-fly in O(stoc_len) per element,
    builds cum_indicator in D-chunks and uses O(1) table lookup per element.
    Each chunk's cum_indicator fits in ~34MB (vs ~608MB for full D).

    Algorithm:
      For each D-chunk [d_start, d_end):
        1. Build cum_indicator for rng_b[d_start:d_end] — O(D_CHUNK * stoc_len * V)
        2. Run fast tiled matmul with O(1) lookup — O(N * M * D_CHUNK)
        3. Accumulate partial result into output
    """
    V = k_table.shape[1]
    device = boundary_a.device
    output = torch.zeros(N, M, dtype=torch.float32, device=device)

    # D_CHUNK chosen to keep cum_indicator under ~34MB:
    # D_CHUNK * (stoc_len+1) * V * 2 bytes
    D_CHUNK = 128

    # Reusable buffer for cum_indicator (allocated once, reused per chunk)
    cum_buf = torch.zeros(D_CHUNK, stoc_len + 1, V, dtype=torch.int16, device=device)
    # Reusable buffer for partial output
    partial = torch.empty(N, M, dtype=torch.float32, device=device)

    # Adaptive tile size: smaller tiles for small N/M to increase GPU occupancy
    if N <= 64 or M <= 64:
        BLOCK_M = 16
        BLOCK_N = 16
    else:
        BLOCK_M = 32
        BLOCK_N = 32
    if D_CHUNK >= 4 and D_CHUNK % 4 == 0:
        BLOCK_K = 4
    elif D_CHUNK % 2 == 0:
        BLOCK_K = 2
    else:
        BLOCK_K = 1
    # Warp count tuning: more warps for larger tiles, fewer for smaller
    nw = 8 if BLOCK_M == 32 else 2
    grid_mm = (triton.cdiv(N, BLOCK_M), triton.cdiv(M, BLOCK_N))

    for d_start in range(0, D, D_CHUNK):
        d_end = min(d_start + D_CHUNK, D)
        d_len = d_end - d_start

        # Build cum_indicator for this chunk of D dimensions
        rng_b_chunk = rng_b[d_start:d_end].contiguous()
        if d_len < D_CHUNK:
            # Last chunk: allocate smaller buffer
            cum_chunk = torch.zeros(d_len, stoc_len + 1, V, dtype=torch.int16, device=device)
        else:
            cum_chunk = cum_buf
            cum_chunk.zero_()
        build_cum_indicator_kernel[(d_len,)](
            rng_b_chunk, cum_chunk,
            d_len, stoc_len, V,
        )

        # Slice boundaries/signs for this D-chunk, transpose to (D, N/M) for coalesced access
        ba_chunk = boundary_a[:, d_start:d_end].t().contiguous()
        bb_chunk = boundary_b[:, d_start:d_end].t().contiguous()
        k_tab_chunk = k_table[d_start:d_end].contiguous()

        # Run fast tiled matmul with O(1) cum_indicator lookup
        if is_bipolar:
            sa_chunk = sign_a[:, d_start:d_end].t().contiguous()
            sb_chunk = sign_b[:, d_start:d_end].t().contiguous()
        else:
            sa_chunk = sb_chunk = torch.empty(1, dtype=torch.int8, device=partial.device)
        enable_matmul_tiled_kernel[grid_mm](
            cum_chunk, k_tab_chunk,
            ba_chunk, bb_chunk,
            sa_chunk, sb_chunk,
            partial,
            N, M, d_len, stoc_len, V,
            q_max_sq, BLOCK_M, BLOCK_N, BLOCK_K,
            IS_BIPOLAR=is_bipolar,
            num_warps=nw,
        )

        output += partial

    return output


@torch.no_grad()
def _sc_matmul_per_tensor(
    a: torch.Tensor,
    b: torch.Tensor,
    max_fp_a: float,
    min_fp_a: float,
    max_fp_b: float = None,
    min_fp_b: float = None,
    mode: str = "bipolar",
    sc_prec: int = 8,
    config: Optional[dict] = None,
    stoc_len: Optional[int] = None,
    rng_levels: Optional[int] = None,
) -> torch.Tensor:
    """
    Enable-signal SC matmul on GPU using Triton table-lookup kernels.

    Same interface as sc_matmul_enable() but uses Triton for acceleration.
    Pipeline: quantize → build tables (GPU) → matmul (GPU) → dequantize.

    Args:
        a: Left operand, shape (N, D) or (B, N, D). FP values.
        b: Right operand, shape (M, D) or (B, M, D). FP values.
        max_fp_a: Max FP value for operand a.
        min_fp_a: Min FP value for operand a.
        max_fp_b: Max FP value for operand b. If None, uses max_fp_a.
        min_fp_b: Min FP value for operand b. If None, uses min_fp_a.
        mode: "bipolar" (symmetric sign-magnitude) or "unipolar" (asymmetric AND).
        sc_prec: SC precision. Controls quantization grid (q_max, V, max_rng_val).
        config: Optional SC RNG/SNG config dict. If None, uses sobol_simple.
        stoc_len: Stochastic stream length. If None, uses 2^sc_prec.
                  Shorter stoc_len = fewer iterations = proportional speedup.

    Returns:
        Result tensor in FP, shape (N, M) or (B, N, M).
    """
    if stoc_len is None:
        stoc_len = 2 ** sc_prec

    if max_fp_b is None:
        max_fp_b = max_fp_a
    if min_fp_b is None:
        min_fp_b = min_fp_a

    if a.dim() == 3:
        return _sc_matmul_enable_triton_batched(
            a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b,
            mode, sc_prec, config, stoc_len=stoc_len, rng_levels=rng_levels
        )

    assert a.dim() == 2 and b.dim() == 2, f"Expected 2D, got a:{a.dim()}D b:{b.dim()}D"
    assert a.shape[1] == b.shape[1], f"Dim mismatch: a={a.shape[1]}, b={b.shape[1]}"

    N, D = a.shape
    M = b.shape[0]
    max_rng_val = _resolve_rng_levels(sc_prec, rng_levels)

    if config is None:
        from .config_helpers import make_sobol_simple_config
        config = make_sobol_simple_config(D, D, sc_prec)

    device = a.device
    if device.type != 'cuda':
        a = a.cuda()
        b = b.cuda()
    a = a.float()
    b = b.float()

    # Get cached RNG sequences
    rand_seqs_a_t, rand_seqs_b_t = _get_cached_sequences(config, sc_prec, a.device)

    # Choose compact vs table-based path based on memory
    V = max_rng_val + 1
    cum_table_bytes = D * (stoc_len + 1) * V * 2  # int16
    use_compact = cum_table_bytes > _COMPACT_ENABLE_THRESHOLD_BYTES

    # Use cached enable tables to avoid rebuilding every call
    if use_compact:
        k_table = _get_cached_k_table(
            config, sc_prec, a.device, rand_seqs_a_t, stoc_len, rng_levels=max_rng_val
        )
        rng_b_for_compact = _prepare_rng_prefix(rand_seqs_b_t, sc_prec, stoc_len, max_rng_val)
    else:
        cum_indicator, k_table = _get_cached_enable_tables(
            config, sc_prec, a.device, rand_seqs_a_t, rand_seqs_b_t,
            stoc_len, rng_levels=max_rng_val)
        rng_b_for_compact = None

    if mode not in ("bipolar", "unipolar"):
        raise ValueError(f"Unknown mode: {mode}")
    result = _sc_matmul_enable_triton(
        a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
        cum_indicator if not use_compact else None,
        k_table, max_rng_val, N, D, M, stoc_len,
        mode=mode, rng_b=rng_b_for_compact,
    )

    if device.type != 'cuda':
        result = result.to(device)

    return result


def _sc_matmul_enable_triton(
    a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
    cum_indicator, k_table, max_rng_val, N, D, M, stoc_len,
    *,
    mode: str,
    rng_b=None,
):
    """Enable-signal SC matmul via Triton.

    ``mode="bipolar"`` uses sign-magnitude (quantize to ±q_max, multiply by
    ±1 signs inside the kernel). ``mode="unipolar"`` uses asymmetric +
    zero-point (quantize to [0, q_max], correct for zp on host).
    """
    is_bipolar = (mode == "bipolar")
    if is_bipolar:
        q_max = 2 ** (sc_prec - 1) - 1
        q_max_sq = float(q_max * q_max)
        abs_max_a = max(abs(max_fp_a), abs(min_fp_a), 1e-5)
        abs_max_b = max(abs(max_fp_b), abs(min_fp_b), 1e-5)
        boundary_a, sign_a, scale_a = fused_quantize_bipolar(
            a, abs_max_a, sc_prec, rng_levels=max_rng_val)
        boundary_b, sign_b, scale_b = fused_quantize_bipolar(
            b, abs_max_b, sc_prec, rng_levels=max_rng_val)
    else:
        q_max_sq = float((2 ** sc_prec - 1) ** 2)
        boundary_a, scale_a, zp_a_f, a_sum = fused_quantize_unipolar(
            a, max_fp_a, min_fp_a, sc_prec,
            compute_sum=True, rng_levels=max_rng_val)
        boundary_b, scale_b, zp_b_f, b_sum = fused_quantize_unipolar(
            b, max_fp_b, min_fp_b, sc_prec,
            compute_sum=True, rng_levels=max_rng_val)
        sign_a = sign_b = None

    if rng_b is not None:
        sc_raw = enable_matmul_compact(
            rng_b, k_table, boundary_a, boundary_b,
            sign_a, sign_b, N, M, D, stoc_len, q_max_sq, is_bipolar=is_bipolar)
    else:
        sc_raw = enable_matmul_triton(
            cum_indicator, k_table, boundary_a, boundary_b,
            sign_a, sign_b, N, M, D, stoc_len, q_max_sq, is_bipolar=is_bipolar)

    if not is_bipolar:
        # Zero-point correction (a_sum/b_sum already computed by fused kernel)
        sc_raw = sc_raw + (-zp_b_f * a_sum[:, None]
                          - zp_a_f * b_sum[None, :]
                          + D * zp_a_f * zp_b_f)

    return sc_raw * (scale_a * scale_b)


def _sc_matmul_enable_triton_batched(
    a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, mode, sc_prec, config,
    stoc_len=None, rng_levels=None,
):
    """Batched enable-signal SC matmul via Triton with CUDA streams."""
    B, N, D = a.shape
    M = b.shape[1]
    output = torch.empty(B, N, M, dtype=torch.float32, device=a.device)

    streams = [torch.cuda.Stream() for _ in range(B)]
    for i in range(B):
        with torch.cuda.stream(streams[i]):
            output[i] = _sc_matmul_per_tensor(
                a[i], b[i], max_fp_a, min_fp_a, max_fp_b, min_fp_b,
                mode, sc_prec, config,
                stoc_len=stoc_len, rng_levels=rng_levels,
            )
    for s in streams:
        s.synchronize()
    return output


# =============================================================================
# Enable-Signal SC Matmul for MLP Layers (large D, table default / compact opt-in)
# =============================================================================


def _sc_matmul_enable_triton_bipolar_mlp(
    a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
    k_table, rng_b, N, D, M, stoc_len,
    group_a=0, group_b=0,
    cum_indicator=None,
    rng_levels: Optional[int] = None,
):
    """Bipolar enable-signal SC matmul for MLP with per-row-group quantization."""
    q_max = 2 ** (sc_prec - 1) - 1    # 127: for scale, boundary norm, and dequant
    q_max_sq = float(q_max * q_max)

    # Default: per-row for activation, per-channel for weight
    if group_a <= 0:
        group_a = 1
    if group_b <= 0:
        group_b = 1

    scale_a_row, a_int, sign_a = _grouped_symmetric_quant(a, group_a, q_max)
    scale_b_row, b_int, sign_b = _grouped_symmetric_quant(b, group_b, q_max)

    # Convert quantized integers to boundaries for enable-signal lookup
    max_rng_val = _resolve_rng_levels(sc_prec, rng_levels)
    boundary_a = (a_int.abs() * max_rng_val / q_max).round().short()
    boundary_b = (b_int.abs() * max_rng_val / q_max).round().short()

    if cum_indicator is not None:
        sc_raw = enable_matmul_triton(
            cum_indicator, k_table, boundary_a, boundary_b,
            sign_a, sign_b, N, M, D, stoc_len, q_max_sq, is_bipolar=True,
        )
    else:
        sc_raw = enable_matmul_compact_mlp(
            rng_b, k_table, boundary_a, boundary_b,
            sign_a, sign_b, N, M, D, stoc_len, q_max_sq, is_bipolar=True,
        )

    # Per-element dequantization with row-group scales
    return sc_raw * (scale_a_row[:, None] * scale_b_row[None, :])


def _sc_matmul_bipolar_mlp_chunked(
    a, b, sc_prec, k_table, rng_b, chunk_d,
    stoc_len=None,
    rng_levels: Optional[int] = None,
):
    """
    Bipolar SC matmul for MLP with internal chunk_d loop.

    Handles the entire D-chunking internally, replacing the Python loop in
    sc_mlp.py. Key optimizations vs calling _sc_matmul_per_row_mlp in a loop:
    - Build cum_indicator ONCE (all chunks share same config/RNG)
    - Use fused per-row quant kernel (1 launch vs ~12 PyTorch ops per chunk)
    - No .item() GPU sync calls (bipolar doesn't need max/min)
    - Minimal Python overhead per chunk

    Total kernel launches: 1 (build) + num_chunks * 3 (quant_a + quant_b + matmul)
    vs old: num_chunks * ~38 launches + 4 syncs each
    """
    N, D = a.shape
    M = b.shape[0]
    if stoc_len is None:
        stoc_len = 2 ** sc_prec
    q_max = 2 ** (sc_prec - 1) - 1
    q_max_sq = float(q_max * q_max)
    max_rng_val = _resolve_rng_levels(sc_prec, rng_levels)

    V = k_table.shape[1]
    device = a.device
    output = torch.zeros(N, M, dtype=torch.float32, device=device)

    # Build cum_indicator ONCE — all chunks share the same RNG sequences
    cum_indicator = torch.zeros(chunk_d, stoc_len + 1, V, dtype=torch.int16, device=device)
    build_cum_indicator_kernel[(chunk_d,)](
        rng_b, cum_indicator,
        chunk_d, stoc_len, V,
    )

    # Tiled matmul params — adaptive tile size for small N/M
    if N <= 64 or M <= 64:
        BLOCK_M = 16
        BLOCK_N = 16
    else:
        BLOCK_M = 32
        BLOCK_N = 32
    if chunk_d >= 4 and chunk_d % 4 == 0:
        BLOCK_K = 4
    elif chunk_d % 2 == 0:
        BLOCK_K = 2
    else:
        BLOCK_K = 1
    # Warp count tuning: more warps for larger tiles, fewer for smaller
    nw = 8 if BLOCK_M == 32 else 2
    grid_mm = (triton.cdiv(N, BLOCK_M), triton.cdiv(M, BLOCK_N))

    # Reusable buffer for partial matmul output
    partial = torch.empty(N, M, dtype=torch.float32, device=device)

    # Per-row scale accumulators for dequantization
    # Each chunk has its own per-row scales; we accumulate via outer product
    for d_start in range(0, D, chunk_d):
        d_end = min(d_start + chunk_d, D)
        d_len = d_end - d_start

        # Slice input chunks (make contiguous for kernel addressing)
        a_chunk = a[:, d_start:d_end].contiguous()
        b_chunk = b[:, d_start:d_end].contiguous()

        # Fused per-row quant: 1 kernel launch each (vs ~12 PyTorch ops each)
        boundary_a, sign_a, scale_a = fused_quantize_bipolar_perrow(
            a_chunk, sc_prec, rng_levels=max_rng_val
        )
        boundary_b, sign_b, scale_b = fused_quantize_bipolar_perrow(
            b_chunk, sc_prec, rng_levels=max_rng_val
        )

        # Handle last chunk if smaller than chunk_d
        if d_len < chunk_d:
            cum_chunk = torch.zeros(d_len, stoc_len + 1, V, dtype=torch.int16, device=device)
            rng_b_chunk = rng_b[:d_len].contiguous()
            build_cum_indicator_kernel[(d_len,)](
                rng_b_chunk, cum_chunk,
                d_len, stoc_len, V,
            )
            k_tab_chunk = k_table[:d_len].contiguous()
        else:
            cum_chunk = cum_indicator
            k_tab_chunk = k_table

        # Transpose boundary/sign to (D, N/M) for coalesced kernel access
        boundary_a_t = boundary_a.t().contiguous()
        boundary_b_t = boundary_b.t().contiguous()
        sign_a_t = sign_a.t().contiguous()
        sign_b_t = sign_b.t().contiguous()

        # Fast tiled matmul with O(1) cum_indicator lookup
        enable_matmul_tiled_kernel[grid_mm](
            cum_chunk, k_tab_chunk,
            boundary_a_t, boundary_b_t,
            sign_a_t, sign_b_t,
            partial,
            N, M, d_len, stoc_len, V,
            q_max_sq, BLOCK_M, BLOCK_N, BLOCK_K,
            IS_BIPOLAR=True,
            num_warps=nw,
        )

        # Accumulate with per-chunk dequantization scales
        output += partial * (scale_a[:, None] * scale_b[None, :])

    return output


def _sc_matmul_enable_triton_unipolar_mlp(
    a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
    k_table, rng_b, N, D, M, stoc_len,
    cum_indicator=None,
    rng_levels: Optional[int] = None,
):
    """Unipolar enable-signal SC matmul for MLP (table default, compact opt-in)."""
    q_max_sq = float((2 ** sc_prec - 1) ** 2)

    boundary_a, scale_a, zp_a_f, a_sum = fused_quantize_unipolar(
        a, max_fp_a, min_fp_a, sc_prec,
        compute_sum=True, rng_levels=rng_levels)
    boundary_b, scale_b, zp_b_f, b_sum = fused_quantize_unipolar(
        b, max_fp_b, min_fp_b, sc_prec,
        compute_sum=True, rng_levels=rng_levels)

    if cum_indicator is not None:
        sc_raw = enable_matmul_triton(
            cum_indicator, k_table, boundary_a, boundary_b,
            None, None, N, M, D, stoc_len, q_max_sq, is_bipolar=False,
        )
    else:
        sc_raw = enable_matmul_compact_mlp(
            rng_b, k_table, boundary_a, boundary_b,
            None, None, N, M, D, stoc_len, q_max_sq, is_bipolar=False,
        )

    correction = (-zp_b_f * a_sum[:, None]
                  - zp_a_f * b_sum[None, :]
                  + D * zp_a_f * zp_b_f)
    corrected = sc_raw + correction

    return corrected * (scale_a * scale_b)


def _sc_matmul_enable_triton_mlp_batched(
    a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, mode, sc_prec, config,
    stoc_len=None, rng_levels=None,
):
    """Batched enable-signal SC matmul for MLP via CUDA streams."""
    B, N, D = a.shape
    M = b.shape[1]
    output = torch.empty(B, N, M, dtype=torch.float32, device=a.device)

    streams = [torch.cuda.Stream() for _ in range(B)]
    for i in range(B):
        with torch.cuda.stream(streams[i]):
            output[i] = _sc_matmul_per_row_mlp(
                a[i], b[i], max_fp_a, min_fp_a, max_fp_b, min_fp_b,
                mode, sc_prec, config,
                stoc_len=stoc_len, rng_levels=rng_levels,
            )
    for s in streams:
        s.synchronize()
    return output


@torch.no_grad()
def _sc_matmul_per_row_mlp(
    a: torch.Tensor,
    b: torch.Tensor,
    max_fp_a: float = 0.0,
    min_fp_a: float = 0.0,
    max_fp_b: float = None,
    min_fp_b: float = None,
    mode: str = "bipolar",
    sc_prec: int = 8,
    config: Optional[dict] = None,
    group_a: int = 1,
    group_b: int = 1,
    chunk_d: int = 0,
    stoc_len: Optional[int] = None,
    rng_levels: Optional[int] = None,
) -> torch.Tensor:
    """
    Enable-signal SC matmul for MLP layers.

    Args:
        chunk_d: If > 0, split D into chunks of this size for reduced SC error.
                 When chunk_d > 0 and mode == "bipolar", uses optimized internal
                 chunking that builds cum_indicator once and uses fused per-row
                 quantization (~10x fewer kernel launches vs external loop).
        group_a: rows per quantization group for a (1 = per-row, default).
        group_b: rows per quantization group for b (1 = per-row/per-channel, default).
        stoc_len: Stochastic stream length. If None, uses 2^sc_prec.
    """
    if stoc_len is None:
        stoc_len = 2 ** sc_prec

    if max_fp_b is None:
        max_fp_b = max_fp_a
    if min_fp_b is None:
        min_fp_b = min_fp_a

    if a.dim() == 3:
        return _sc_matmul_enable_triton_mlp_batched(
            a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b,
            mode, sc_prec, config, stoc_len=stoc_len, rng_levels=rng_levels
        )

    assert a.dim() == 2 and b.dim() == 2, f"Expected 2D, got a:{a.dim()}D b:{b.dim()}D"
    assert a.shape[1] == b.shape[1], f"Dim mismatch: a={a.shape[1]}, b={b.shape[1]}"

    N, D = a.shape
    M = b.shape[0]

    device = a.device
    if device.type != 'cuda':
        a = a.cuda()
        b = b.cuda()
    a = a.float()
    b = b.float()

    # Fast path: bipolar + chunk_d → optimized internal chunking
    if mode == "bipolar" and chunk_d > 0 and D > chunk_d:
        # Config is for chunk_d dimensions (not full D)
        if config is None:
            from .config_helpers import make_sobol_simple_config
            config = make_sobol_simple_config(chunk_d, chunk_d, sc_prec)

        rand_seqs_a_t, rand_seqs_b_t = _get_cached_sequences(config, sc_prec, a.device)
        grid_levels = _resolve_rng_levels(sc_prec, rng_levels)
        k_table = _get_cached_k_table(
            config, sc_prec, a.device, rand_seqs_a_t, stoc_len, rng_levels=grid_levels
        )
        rng_b = _prepare_rng_prefix(rand_seqs_b_t, sc_prec, stoc_len, grid_levels)

        result = _sc_matmul_bipolar_mlp_chunked(
            a, b, sc_prec, k_table, rng_b, chunk_d,
            stoc_len=stoc_len, rng_levels=grid_levels,
        )

        if device.type != 'cuda':
            result = result.to(device)
        return result

    # Standard path (no chunk_d, or unipolar, or D <= chunk_d)
    if config is None:
        from .config_helpers import make_sobol_simple_config
        config = make_sobol_simple_config(D, D, sc_prec)

    rand_seqs_a_t, rand_seqs_b_t = _get_cached_sequences(config, sc_prec, a.device)

    # Choose compact vs table-based path based on memory (default: table)
    grid_levels = _resolve_rng_levels(sc_prec, rng_levels)
    V = grid_levels + 1
    cum_table_bytes = D * (stoc_len + 1) * V * 2
    use_compact = cum_table_bytes > _COMPACT_ENABLE_THRESHOLD_BYTES

    if use_compact:
        k_table = _get_cached_k_table(
            config, sc_prec, a.device, rand_seqs_a_t, stoc_len, rng_levels=grid_levels
        )
        rng_b = _prepare_rng_prefix(rand_seqs_b_t, sc_prec, stoc_len, grid_levels)
        cum_indicator = None
    else:
        cum_indicator, k_table = _get_cached_enable_tables(
            config, sc_prec, a.device, rand_seqs_a_t, rand_seqs_b_t,
            stoc_len, rng_levels=grid_levels)
        rng_b = None

    if mode == "bipolar":
        result = _sc_matmul_enable_triton_bipolar_mlp(
            a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
            k_table, rng_b, N, D, M, stoc_len,
            group_a=group_a, group_b=group_b,
            cum_indicator=cum_indicator, rng_levels=grid_levels,
        )
    elif mode == "unipolar":
        result = _sc_matmul_enable_triton_unipolar_mlp(
            a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
            k_table, rng_b, N, D, M, stoc_len,
            cum_indicator=cum_indicator, rng_levels=grid_levels,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if device.type != 'cuda':
        result = result.to(device)

    return result


# =============================================================================
# Enable-Signal Grouped Quantization
# =============================================================================

@torch.no_grad()
def _sc_matmul_per_row(
    a: torch.Tensor,
    b: torch.Tensor,
    group_a: int = 1,
    group_b: int = 1,
    mode: str = "unipolar",
    sc_prec: int = 8,
    config: Optional[dict] = None,
    stoc_len: Optional[int] = None,
    rng_levels: Optional[int] = None,
) -> torch.Tensor:
    """
    Enable-signal SC matmul with per-row-group quantization: a @ b^T.

    Same as sc_matmul_grouped but uses enable-signal table-lookup instead
    of packed AND matmul.

    Args:
        a: (N, D) left operand.
        b: (M, D) right operand.
        group_a: rows per quantization group for a (1 = per-row, N = per-tensor).
        group_b: rows per quantization group for b (1 = per-row, M = per-tensor).
        mode: "bipolar" or "unipolar".
        sc_prec: SC precision (controls quantization grid).
        config: RNG/SNG config dict.
        stoc_len: Stochastic stream length. If None, uses 2^sc_prec.

    Returns:
        (N, M) result tensor in FP.
    """
    if stoc_len is None:
        stoc_len = 2 ** sc_prec

    assert a.dim() == 2 and b.dim() == 2, f"Expected 2D, got a:{a.dim()}D b:{b.dim()}D"
    assert a.shape[1] == b.shape[1], f"Inner dim mismatch: {a.shape[1]} vs {b.shape[1]}"

    N, D = a.shape
    M = b.shape[0]
    max_rng_val = _resolve_rng_levels(sc_prec, rng_levels)

    if config is None:
        from .config_helpers import make_sobol_simple_config
        config = make_sobol_simple_config(D, D, sc_prec)

    device = a.device
    if device.type != 'cuda':
        a = a.cuda()
        b = b.cuda()
    a = a.float()
    b = b.float()

    # Get cached RNG sequences
    rand_seqs_a_t, rand_seqs_b_t = _get_cached_sequences(config, sc_prec, a.device)

    # Choose compact vs table-based path based on memory, use cached tables
    V = max_rng_val + 1
    cum_table_bytes = D * (stoc_len + 1) * V * 2
    use_compact = cum_table_bytes > _COMPACT_ENABLE_THRESHOLD_BYTES

    if mode not in ("bipolar", "unipolar"):
        raise ValueError(f"Unknown mode: {mode}")
    if use_compact:
        k_table = _get_cached_k_table(
            config, sc_prec, a.device, rand_seqs_a_t, stoc_len, rng_levels=max_rng_val
        )
        rng_b = _prepare_rng_prefix(rand_seqs_b_t, sc_prec, stoc_len, max_rng_val)
        result = _sc_matmul_grouped_enable(
            a, b, group_a, group_b, sc_prec,
            None, k_table, max_rng_val, N, D, M, stoc_len,
            mode=mode, rng_b=rng_b,
        )
    else:
        cum_indicator, k_table = _get_cached_enable_tables(
            config, sc_prec, a.device, rand_seqs_a_t, rand_seqs_b_t,
            stoc_len, rng_levels=max_rng_val)
        result = _sc_matmul_grouped_enable(
            a, b, group_a, group_b, sc_prec,
            cum_indicator, k_table, max_rng_val, N, D, M, stoc_len,
            mode=mode,
        )

    if device.type != 'cuda':
        result = result.to(device)

    return result


def _sc_matmul_grouped_enable(
    a, b, group_a, group_b, sc_prec,
    cum_indicator, k_table, max_rng_val, N, D, M, stoc_len,
    *,
    mode: str,
    rng_b=None,
):
    """Enable-signal SC matmul with per-row-group quantization.

    Bipolar: symmetric quant via _grouped_symmetric_quant → sign-magnitude SC.
    Unipolar: asymmetric quant via _grouped_asymmetric_quant → zp correction.
    Both share the same enable-signal matmul kernel suite.
    """
    is_bipolar = (mode == "bipolar")
    if is_bipolar:
        q_max = 2 ** (sc_prec - 1) - 1
        scale_a_row, a_int, sign_a = _grouped_symmetric_quant(a, group_a, q_max)
        scale_b_row, b_int, sign_b = _grouped_symmetric_quant(b, group_b, q_max)
        boundary_a = (a_int.abs() * max_rng_val / q_max).round().short()
        boundary_b = (b_int.abs() * max_rng_val / q_max).round().short()
    else:
        q_max = 2 ** sc_prec - 1
        scale_a_row, zp_a_row, a_int = _grouped_asymmetric_quant(a, group_a, q_max)
        scale_b_row, zp_b_row, b_int = _grouped_asymmetric_quant(b, group_b, q_max)
        boundary_a = (a_int * max_rng_val / q_max).round().short()
        boundary_b = (b_int * max_rng_val / q_max).round().short()
        sign_a = sign_b = None

    q_max_sq = float(q_max * q_max)
    if rng_b is not None:
        sc_raw = enable_matmul_compact(
            rng_b, k_table, boundary_a, boundary_b,
            sign_a, sign_b, N, M, D, stoc_len, q_max_sq, is_bipolar=is_bipolar)
    else:
        sc_raw = enable_matmul_triton(
            cum_indicator, k_table, boundary_a, boundary_b,
            sign_a, sign_b, N, M, D, stoc_len, q_max_sq, is_bipolar=is_bipolar)

    if not is_bipolar:
        # Per-element zero-point correction
        a_sum = a_int.sum(dim=1)
        b_sum = b_int.sum(dim=1)
        sc_raw = sc_raw + (-zp_b_row[None, :] * a_sum[:, None]
                          - zp_a_row[:, None] * b_sum[None, :]
                          + D * zp_a_row[:, None] * zp_b_row[None, :])

    return sc_raw * (scale_a_row[:, None] * scale_b_row[None, :])


def _grouped_symmetric_quant(x, G, q_max):
    """Per-row-group symmetric quantization for bipolar mode.

    Args:
        x: (rows, cols) float tensor
        G: number of rows per quantization group
        q_max: max quantized value (e.g. 127 for 8-bit bipolar)

    Returns:
        scale_row: (rows,) per-row scale
        x_int:     (rows, cols) quantized values in [-q_max, q_max]
        sign:      (rows, cols) sign bits (+1/-1)
    """
    rows, cols = x.shape

    if G >= rows:
        # Single group (per-tensor) — fast path
        abs_max = x.abs().max().clamp(min=1e-5)
        scale = abs_max / q_max
        x_int = (x / scale).round().clamp(-q_max, q_max)
        sign = torch.sign(x_int).to(torch.int8)
        sign[sign == 0] = 1  # Handle zeros as positive
        return scale.expand(rows), x_int, sign

    num_full = rows // G
    rem = rows % G

    parts_scale = []
    parts_int = []
    parts_sign = []

    if num_full > 0:
        x_full = x[:num_full * G].reshape(num_full, G, cols)
        gabs_max = x_full.abs().amax(dim=(1, 2)).clamp(min=1e-5)  # (num_full,)
        gscale = gabs_max / q_max

        # Expand scales for broadcasting
        gscale_exp = gscale[:, None, None].expand(num_full, G, cols)
        x_full_quant = (x_full / gscale_exp).round().clamp(-q_max, q_max)
        x_full_sign = torch.sign(x_full_quant).to(torch.int8)
        x_full_sign[x_full_sign == 0] = 1

        parts_scale.append(gscale.repeat_interleave(G))
        parts_int.append(x_full_quant.reshape(num_full * G, cols))
        parts_sign.append(x_full_sign.reshape(num_full * G, cols))

    if rem > 0:
        x_rem = x[num_full * G:]
        rabs_max = x_rem.abs().max().clamp(min=1e-5)
        rscale = rabs_max / q_max
        x_rem_quant = (x_rem / rscale).round().clamp(-q_max, q_max)
        x_rem_sign = torch.sign(x_rem_quant).to(torch.int8)
        x_rem_sign[x_rem_sign == 0] = 1
        
        parts_scale.append(rscale.expand(rem))
        parts_int.append(x_rem_quant)
        parts_sign.append(x_rem_sign)

    scale_row = torch.cat(parts_scale)  # (rows,)
    x_int = torch.cat(parts_int, dim=0) # (rows, cols)
    sign = torch.cat(parts_sign, dim=0) # (rows, cols)

    return scale_row, x_int, sign


def _grouped_asymmetric_quant(x, G, q_max):
    """Per-row-group asymmetric quantization.

    Args:
        x: (rows, cols) float tensor
        G: number of rows per quantization group
        q_max: max quantized value (e.g. 255 for 8-bit)

    Returns:
        scale_row: (rows,) per-row scale
        zp_row:    (rows,) per-row zero-point
        x_int:     (rows, cols) quantized values in [0, q_max]
    """
    rows, cols = x.shape

    if G >= rows:
        # Single group (per-tensor) — fast path
        x_max = x.max()
        x_min = x.min()
        range_x = (x_max - x_min).clamp(min=1e-5)
        scale = range_x / q_max
        zp = (-x_min / scale).round().clamp(0, q_max)
        x_int = (x / scale + zp).round().clamp(0, q_max)
        return scale.expand(rows), zp.expand(rows), x_int

    num_full = rows // G
    rem = rows % G

    parts_scale = []
    parts_zp = []

    if num_full > 0:
        x_full = x[:num_full * G].reshape(num_full, G, cols)
        gmax = x_full.amax(dim=(1, 2))      # (num_full,)
        gmin = x_full.amin(dim=(1, 2))
        grange = (gmax - gmin).clamp(min=1e-5)
        gscale = grange / q_max              # (num_full,)
        gzp = (-gmin / gscale).round().clamp(0, q_max)
        parts_scale.append(gscale.repeat_interleave(G))
        parts_zp.append(gzp.repeat_interleave(G))

    if rem > 0:
        x_rem = x[num_full * G:]
        rmax = x_rem.max()
        rmin = x_rem.min()
        rrange = (rmax - rmin).clamp(min=1e-5)
        rscale = rrange / q_max
        rzp = (-rmin / rscale).round().clamp(0, q_max)
        parts_scale.append(rscale.expand(rem))
        parts_zp.append(rzp.expand(rem))

    scale_row = torch.cat(parts_scale)       # (rows,)
    zp_row = torch.cat(parts_zp)             # (rows,)
    x_int = (x / scale_row[:, None] + zp_row[:, None]).round().clamp(0, q_max)

    return scale_row, zp_row, x_int


# =============================================================================
# det kernel tuning — opt-in tile-size heuristic for SC kernels on EVA-ViTDet
# shapes. Ported from vit_sc/sc/sc_triton.py.
#
# Default (cls / scmp_llm): the bit-stable cls heuristic — (N≤64 or M≤64 → 16,16;
# else 32,32) — see the in-place pickers inside enable_matmul_triton and
# _sc_matmul_per_head_bipolar. det runs that enter ``det_kernel_tuning()``
# switch to ``_pick_enable_block_sizes`` tuned on RTX PRO 6000 Blackwell across
# det's shape spectrum (SC attn at K∈{88, 256, 6400}; SC linear at M=6400,
# K∈{1408, 6144}).
# =============================================================================

_DET_KERNEL_TUNING_DEPTH = 0


class det_kernel_tuning:
    """Re-entrant context manager opting in to det-tuned tile selection."""

    def __enter__(self):
        global _DET_KERNEL_TUNING_DEPTH
        _DET_KERNEL_TUNING_DEPTH += 1
        return self

    def __exit__(self, *_):
        global _DET_KERNEL_TUNING_DEPTH
        _DET_KERNEL_TUNING_DEPTH -= 1


def _det_kernel_tuning_active() -> bool:
    return _DET_KERNEL_TUNING_DEPTH > 0


def _pick_block_k(D: int) -> int:
    if D >= 4 and D % 4 == 0:
        return 4
    if D % 2 == 0:
        return 2
    return 1


def _pick_enable_block_sizes(rows_a: int, rows_b: int, D: int) -> tuple:
    if rows_a <= 64 or rows_b <= 64:
        return 16, 16, _pick_block_k(D), 2
    if D <= 256:
        if rows_a * rows_b >= 1 << 20:
            return 16, 64, _pick_block_k(D), 8     # global-attn-like
        return 16, 32, _pick_block_k(D), 4         # window-attn-like
    if rows_b >= 256:
        return 32, 64, _pick_block_k(D), 8         # SCLinear (proj/fc1/fc2)
    return 32, 32, _pick_block_k(D), 8             # global-av (large K, small N)


# =============================================================================
# Batched per-row-group symmetric quant — used by _sc_matmul_per_row_batched.
# Ported from vit_sc/sc/sc_triton.py.
# =============================================================================

def _grouped_symmetric_quant_batched(x, G, q_max):
    """Batched per-row-group symmetric quant. ``x`` is (BH, rows, cols).

    Currently only ``G == 1`` (per-row) and ``G >= rows`` (per-batch-tensor)
    are implemented. Returns (scale_row, x_int, sign) shaped (BH, rows[, cols]).
    """
    assert x.dim() == 3, f"expected (BH, rows, cols), got {tuple(x.shape)}"
    BH, rows, cols = x.shape

    if G >= rows:
        abs_max = x.abs().amax(dim=(1, 2)).clamp(min=1e-5)        # (BH,)
        scale = abs_max / q_max                                    # (BH,)
        x_int = (x / scale[:, None, None]).round().clamp(-q_max, q_max)
        sign = torch.sign(x_int).to(torch.int8)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return scale[:, None].expand(BH, rows).contiguous(), x_int, sign

    if G == 1:
        abs_max = x.abs().amax(dim=2).clamp(min=1e-5)               # (BH, rows)
        scale = abs_max / q_max                                    # (BH, rows)
        x_int = (x / scale[:, :, None]).round().clamp(-q_max, q_max)
        sign = torch.sign(x_int).to(torch.int8)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return scale, x_int, sign

    raise NotImplementedError(
        f"_grouped_symmetric_quant_batched: G={G} not supported "
        f"(use G=1 or G>=rows)")


# =============================================================================
# Batched 3D version of _sc_matmul_per_row. Ported from vit_sc.
# =============================================================================

@torch.no_grad()
def _sc_matmul_per_row_batched(
    a: torch.Tensor,
    b: torch.Tensor,
    group_a: int = 1,
    group_b: int = 1,
    mode: str = "bipolar",
    sc_prec: int = 8,
    config: Optional[dict] = None,
    stoc_len: Optional[int] = None,
    rng_levels: Optional[int] = None,
) -> torch.Tensor:
    """Batched 3D version of ``_sc_matmul_per_row``: ``a @ b^T``
    over a leading batch dim, in one kernel launch.

    Args:
        a: (BH, M, D) left operand.
        b: (BH, N, D) right operand (already row-major).
        group_a, group_b: rows per quantization group inside each batch.
            Currently supports 1 (per-row) or >=rows (per-batch-tensor).
        mode: "bipolar" only — unipolar falls back to per-batch loop over
            the 2D entry.
        sc_prec, config, stoc_len: same semantics as the 2D entry.

    Returns:
        (BH, M, N) float32.
    """
    assert a.dim() == 3 and b.dim() == 3, \
        f"Expected 3D, got a:{a.dim()}D b:{b.dim()}D"
    BH, M, D = a.shape
    BH_b, N, D_b = b.shape
    assert BH == BH_b, f"batch mismatch: a BH={BH} vs b BH={BH_b}"
    assert D == D_b, f"inner-dim mismatch: a D={D} vs b D={D_b}"

    if stoc_len is None:
        stoc_len = 2 ** sc_prec

    if config is None:
        from .config_helpers import make_sobol_simple_config
        config = make_sobol_simple_config(D, D, sc_prec)

    device = a.device

    # Unipolar: no batched kernel yet. Per-batch loop fallback.
    if mode == "unipolar":
        out = torch.empty(BH, M, N, dtype=torch.float32, device=device)
        for i in range(BH):
            out[i] = _sc_matmul_per_row(
                a[i].contiguous(), b[i].contiguous(),
                group_a=group_a, group_b=group_b,
                mode=mode, sc_prec=sc_prec, config=config, stoc_len=stoc_len,
                rng_levels=rng_levels,
            )
        return out
    if mode != "bipolar":
        raise ValueError(f"Unknown mode: {mode}")

    a = a.float().contiguous()
    b = b.float().contiguous()

    rand_seqs_a_t, rand_seqs_b_t = _get_cached_sequences(config, sc_prec, device)
    max_rng_val = _resolve_rng_levels(sc_prec, rng_levels)
    V = max_rng_val + 1
    cum_table_bytes = D * (stoc_len + 1) * V * 2
    use_compact = cum_table_bytes > _COMPACT_ENABLE_THRESHOLD_BYTES

    # Compact path (forced via env var) not batched yet — per-batch loop.
    if use_compact:
        out = torch.empty(BH, M, N, dtype=torch.float32, device=device)
        for i in range(BH):
            out[i] = _sc_matmul_per_row(
                a[i], b[i],
                group_a=group_a, group_b=group_b,
                mode=mode, sc_prec=sc_prec, config=config, stoc_len=stoc_len,
                rng_levels=rng_levels,
            )
        return out

    q_max = 2 ** (sc_prec - 1) - 1
    q_max_sq = float(q_max * q_max)

    scale_a_row, a_int, sign_a = _grouped_symmetric_quant_batched(
        a, group_a, q_max)                         # scale (BH, M); a_int/sign (BH, M, D)
    scale_b_row, b_int, sign_b = _grouped_symmetric_quant_batched(
        b, group_b, q_max)                         # scale (BH, N); b_int/sign (BH, N, D)

    boundary_a = (a_int.abs() * (max_rng_val / q_max)).round().short()  # (BH, M, D)
    boundary_b = (b_int.abs() * (max_rng_val / q_max)).round().short()  # (BH, N, D)

    boundary_a_t = boundary_a.transpose(1, 2).contiguous()  # (BH, D, M)
    sign_a_t = sign_a.transpose(1, 2).contiguous()
    boundary_b_t = boundary_b.transpose(1, 2).contiguous()  # (BH, D, N)
    sign_b_t = sign_b.transpose(1, 2).contiguous()

    cum_indicator, k_table = _get_cached_enable_tables(
        config, sc_prec, device, rand_seqs_a_t, rand_seqs_b_t, stoc_len,
        rng_levels=rng_levels)
    V_actual = cum_indicator.shape[2]

    head_scale_ones = torch.ones(BH, dtype=torch.float32, device=device)
    output = torch.empty(BH, M, N, dtype=torch.float32, device=device)

    BLOCK_M, BLOCK_N, BLOCK_K, nw = _pick_enable_block_sizes(M, N, D)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), BH)
    # Kernel's (N, M) = (rows of A, rows of B) = our (M, N).
    enable_matmul_bipolar_batched_kernel[grid](
        cum_indicator, k_table,
        boundary_a_t, boundary_b_t,
        sign_a_t, sign_b_t,
        output, head_scale_ones,
        M, N, D,
        stoc_len, V_actual, q_max_sq,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=nw,
    )

    output.mul_(scale_a_row.unsqueeze(2) * scale_b_row.unsqueeze(1))
    return output

"""Unified public SC matmul entry point.

A single ``sc_matmul`` function dispatches to the 5 specialized Triton-backed
kernels in ``kernels.py`` based on:

  * ``granularity`` — quantization scope ("per_tensor" / "per_row" / "per_head")
  * input dimensionality (2D vs 3D — auto-detected)
  * ``chunk_d``      — MLP-fast-path inner-dim chunking

Activation ranges (max / min) are computed inside the kernels — callers do
not pass them. Outlier-robust calibrated ranges would be a future addition
behind an explicit ``a_observer=`` kwarg if ever needed.
"""

from __future__ import annotations

from typing import Optional

import torch

from .kernels import (
    _sc_matmul_per_tensor,
    _sc_matmul_per_row,
    _sc_matmul_per_row_mlp,
    _sc_matmul_per_row_batched,
    _sc_matmul_per_head_bipolar,
)


_VALID_GRANULARITIES = ("per_tensor", "per_row", "per_head")
_VALID_MODES = ("bipolar", "unipolar")


@torch.no_grad()
def sc_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    granularity: str = "per_row",
    *,
    mode: str = "bipolar",
    sc_prec: int = 8,
    stoc_len: Optional[int] = None,
    chunk_d: int = 0,
    group_a: int = 1,
    group_b: int = 1,
    rng_levels: Optional[int] = None,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """Stochastic-computing matmul ``a @ b.T``.

    Args:
        a, b: input tensors. 2D ``(N, D)`` / ``(M, D)`` or 3D
            ``(BH, N, D)`` / ``(BH, M, D)`` — input dimensionality is
            auto-detected. Must match ``D``.
        granularity: quantization scope.

            * ``"per_tensor"`` — one ``(max, min)`` for the whole matrix.
              Computed via ``a.max() / a.min()``.
            * ``"per_row"`` (default) — per-row-group quantization. With
              ``group_a=group_b=1`` (default) this is true per-row.
              Use ``group_a=N`` to fall back to per-tensor on operand ``a``.
            * ``"per_head"`` — one ``(max, min)`` per leading-dim slice
              (e.g. per attention head). Requires 3D input. Currently
              ``mode="bipolar"`` only.
        mode: SC quantization scheme.

            * ``"bipolar"`` (default) — symmetric sign-magnitude,
              ``q_max = 2 ** (sc_prec - 1) - 1``.
            * ``"unipolar"`` — asymmetric zero-point,
              ``q_max = 2 ** sc_prec - 1``.
        sc_prec: SC precision. Controls the quantization grid.
        stoc_len: stochastic stream length. Defaults to ``2 ** sc_prec``.
        chunk_d: inner-dim chunk size. Pass ``0`` (default) to disable.
            **Only valid for ``granularity="per_row"`` + ``mode="bipolar"``.**
            Splits ``D`` into chunks of ``chunk_d`` so the ``cum_indicator``
            table fits in L2 cache — required for wide MLP layers
            (e.g. ``D >= 1024``).
        group_a, group_b: per-row quantization group sizes for ``a`` and ``b``
            (used only with ``granularity="per_row"``). ``1`` = per-row,
            ``N``/``M`` = per-tensor.
        rng_levels: enable-signal RNG grid size for fixed-level precision.
            When ``None`` (default) the grid follows ``sc_prec``. Pass a
            specific integer to keep an int8 quant grid while varying
            ``stoc_len``.
        config: optional Sobol RNG/SNG config dict. Auto-built when ``None``.

    Returns:
        Output tensor. 2D inputs → ``(N, M)`` float32. 3D inputs → ``(BH, N, M)``
        float32 (same shape for all granularities, including ``per_head``).

    Raises:
        ValueError: if ``chunk_d > 0`` is combined with anything other than
            ``granularity="per_row"`` + ``mode="bipolar"``.
        ValueError: if ``granularity="per_head"`` is requested with a non-3D
            input, or with ``mode != "bipolar"``.
        ValueError: for unknown ``granularity`` or ``mode`` values.
    """
    if granularity not in _VALID_GRANULARITIES:
        raise ValueError(
            f"sc_matmul: unknown granularity '{granularity}'. "
            f"Expected one of {_VALID_GRANULARITIES}.")
    if mode not in _VALID_MODES:
        raise ValueError(
            f"sc_matmul: unknown mode '{mode}'. "
            f"Expected one of {_VALID_MODES}.")

    # ---- chunk_d compatibility gate -----------------------------------------
    # Currently chunk_d is only implemented in the per-row + bipolar MLP
    # fast path. Other granularities and unipolar quantization will silently
    # ignore chunk_d in the underlying kernels, which is a footgun for
    # callers expecting the chunking to take effect. Raise here instead.
    if chunk_d > 0:
        if granularity != "per_row":
            raise ValueError(
                f"sc_matmul: chunk_d > 0 requires granularity='per_row', "
                f"got '{granularity}'. D-chunking is only implemented in the "
                f"per-row MLP fast path.")
        if mode != "bipolar":
            raise ValueError(
                f"sc_matmul: chunk_d > 0 requires mode='bipolar', got '{mode}'. "
                f"D-chunking is not implemented for unipolar quantization.")
        if a.dim() != 2 or b.dim() != 2:
            raise ValueError(
                f"sc_matmul: chunk_d > 0 requires 2D inputs (the per-row MLP "
                f"fast path is 2D only), got a.dim()={a.dim()}, b.dim()={b.dim()}.")

    # ---- per_head shape gate -------------------------------------------------
    if granularity == "per_head":
        if a.dim() != 3 or b.dim() != 3:
            raise ValueError(
                f"sc_matmul: granularity='per_head' requires 3D inputs (BH, *, D), "
                f"got a.dim()={a.dim()}, b.dim()={b.dim()}.")
        if mode != "bipolar":
            raise ValueError(
                f"sc_matmul: granularity='per_head' currently only supports "
                f"mode='bipolar', got '{mode}'.")

    # ---- dispatch ------------------------------------------------------------
    if granularity == "per_tensor":
        # Compute the per-tensor range on host (one .max / .min sync).
        # Same as what every caller did before — see sc_attention.py:327 etc.
        a_max = a.max().item()
        a_min = a.min().item()
        b_max = b.max().item()
        b_min = b.min().item()
        return _sc_matmul_per_tensor(
            a, b,
            max_fp_a=a_max, min_fp_a=a_min,
            max_fp_b=b_max, min_fp_b=b_min,
            mode=mode, sc_prec=sc_prec,
            stoc_len=stoc_len, config=config,
        )

    if granularity == "per_row":
        if chunk_d > 0:
            return _sc_matmul_per_row_mlp(
                a, b,
                mode=mode, sc_prec=sc_prec, config=config,
                group_a=group_a, group_b=group_b, chunk_d=chunk_d,
                stoc_len=stoc_len, rng_levels=rng_levels,
            )
        if a.dim() == 3:
            return _sc_matmul_per_row_batched(
                a, b,
                group_a=group_a, group_b=group_b,
                mode=mode, sc_prec=sc_prec, config=config,
                stoc_len=stoc_len, rng_levels=rng_levels,
            )
        return _sc_matmul_per_row(
            a, b,
            group_a=group_a, group_b=group_b,
            mode=mode, sc_prec=sc_prec, config=config,
            stoc_len=stoc_len, rng_levels=rng_levels,
        )

    # granularity == "per_head" — bipolar, 3D, already validated above.
    # per_head_bipolar requires a non-None config; the other specialized
    # entry points have a lazy fallback. Build the default here.
    if config is None:
        from .config_helpers import make_sobol_simple_config
        D = a.shape[-1]
        config = make_sobol_simple_config(D, D, sc_prec)

    a_maxs = a.amax(dim=(1, 2))
    a_mins = a.amin(dim=(1, 2))
    b_maxs = b.amax(dim=(1, 2))
    b_mins = b.amin(dim=(1, 2))
    return _sc_matmul_per_head_bipolar(
        a, b,
        q_maxs=a_maxs, q_mins=a_mins,
        k_maxs=b_maxs, k_mins=b_mins,
        sc_prec=sc_prec, config=config,
        stoc_len=stoc_len, rng_levels=rng_levels,
    )

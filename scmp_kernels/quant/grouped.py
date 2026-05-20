"""PyTorch grouped-symmetric / grouped-asymmetric quantization.

Used by the per-row enable-signal SC matmul paths when the per-row scope is
expressed as row groups (e.g. group_a=G means G consecutive rows share a
scale). Pure PyTorch — no Triton kernels — because the grouped path is bound
by the .amax / .reshape ops anyway and Triton offers no win at this stage.
"""
from __future__ import annotations

import torch


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


__all__ = [
    "_grouped_symmetric_quant",
    "_grouped_asymmetric_quant",
    "_grouped_symmetric_quant_batched",
]

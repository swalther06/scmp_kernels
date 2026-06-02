"""Smoke tests for scmp_kernels.sc — verify imports succeed and the unified
sc_matmul dispatches correctly for each granularity.

The numerical correctness is checked by the in-repo SC benchmarks
(scmp_llm/SC/bench_table_vs_compact.py, vit_sc cls/det evaluations).
Here we only verify:

  1. The package imports cleanly.
  2. ``sc_matmul`` runs on CUDA for each granularity and returns the
     expected output shape.
  3. ``chunk_d`` validation raises on unsupported combinations.
  4. ``granularity="per_head"`` shape gating works.

Requires a CUDA-capable Triton install.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SC kernels require CUDA + Triton.")


def test_import_only():
    from scmp_kernels import sc_matmul                 # noqa: F401
    from scmp_kernels.sc import sc_matmul, det_kernel_tuning  # noqa: F401


def test_per_tensor_2d():
    from scmp_kernels import sc_matmul
    a = torch.randn(8, 64, device="cuda")
    b = torch.randn(16, 64, device="cuda")
    y = sc_matmul(a, b, granularity="per_tensor", sc_prec=8)
    assert y.shape == (8, 16)
    assert y.dtype == torch.float32


def test_per_row_2d():
    from scmp_kernels import sc_matmul
    a = torch.randn(8, 64, device="cuda")
    b = torch.randn(16, 64, device="cuda")
    y = sc_matmul(a, b, granularity="per_row", sc_prec=8)
    assert y.shape == (8, 16)


def test_per_row_3d_batched():
    from scmp_kernels import sc_matmul
    a = torch.randn(4, 8, 64, device="cuda")
    b = torch.randn(4, 16, 64, device="cuda")
    y = sc_matmul(a, b, granularity="per_row", sc_prec=8)
    assert y.shape == (4, 8, 16)


def test_per_head_bipolar():
    from scmp_kernels import sc_matmul
    q = torch.randn(16, 197, 64, device="cuda")
    k = torch.randn(16, 197, 64, device="cuda")
    y = sc_matmul(q, k, granularity="per_head", sc_prec=8)
    assert y.shape == (16, 197, 197)


def test_per_row_mlp_chunk_d():
    from scmp_kernels import sc_matmul
    # chunk_d > 0 requires per_row + bipolar + 2D.
    a = torch.randn(64, 1024, device="cuda")
    b = torch.randn(128, 1024, device="cuda")
    y = sc_matmul(a, b, granularity="per_row", chunk_d=72, sc_prec=8)
    assert y.shape == (64, 128)


# ---------------------------------------------------------------------------
# Validation gates — must raise ValueError on unsupported combinations.
# ---------------------------------------------------------------------------

def test_chunk_d_rejects_per_tensor():
    from scmp_kernels import sc_matmul
    a = torch.randn(64, 1024, device="cuda")
    b = torch.randn(128, 1024, device="cuda")
    with pytest.raises(ValueError, match="chunk_d"):
        sc_matmul(a, b, granularity="per_tensor", chunk_d=72)


def test_chunk_d_rejects_unipolar():
    from scmp_kernels import sc_matmul
    a = torch.randn(64, 1024, device="cuda")
    b = torch.randn(128, 1024, device="cuda")
    with pytest.raises(ValueError, match="chunk_d"):
        sc_matmul(a, b, granularity="per_row", mode="unipolar", chunk_d=72)


def test_chunk_d_rejects_3d():
    from scmp_kernels import sc_matmul
    a = torch.randn(4, 64, 1024, device="cuda")
    b = torch.randn(4, 128, 1024, device="cuda")
    with pytest.raises(ValueError, match="chunk_d"):
        sc_matmul(a, b, granularity="per_row", chunk_d=72)


def test_per_head_rejects_2d():
    from scmp_kernels import sc_matmul
    a = torch.randn(64, 1024, device="cuda")
    b = torch.randn(128, 1024, device="cuda")
    with pytest.raises(ValueError, match="per_head"):
        sc_matmul(a, b, granularity="per_head")


def test_per_head_rejects_unipolar():
    from scmp_kernels import sc_matmul
    q = torch.randn(4, 8, 64, device="cuda")
    k = torch.randn(4, 8, 64, device="cuda")
    with pytest.raises(ValueError, match="per_head"):
        sc_matmul(q, k, granularity="per_head", mode="unipolar")


def test_unknown_granularity():
    from scmp_kernels import sc_matmul
    a = torch.randn(8, 64, device="cuda")
    b = torch.randn(16, 64, device="cuda")
    with pytest.raises(ValueError, match="granularity"):
        sc_matmul(a, b, granularity="per_block")


def test_unknown_mode():
    from scmp_kernels import sc_matmul
    a = torch.randn(8, 64, device="cuda")
    b = torch.randn(16, 64, device="cuda")
    with pytest.raises(ValueError, match="mode"):
        sc_matmul(a, b, mode="ternary")


# ---------------------------------------------------------------------------
# Multi-GPU device guard — regression for the cuda:1 illegal-memory-access bug.
# Triton launches on the current device (cuda:0); without the device guard in
# sc_matmul, tensors on cuda:1 (e.g. layers sharded by device_map="auto" on a
# 70B) crash with "CUDA error: an illegal memory access was encountered".
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="multi-GPU device-guard test needs >= 2 CUDA devices.")
@pytest.mark.parametrize("granularity,chunk_d,shape", [
    ("per_tensor", 0, "2d"),
    ("per_row", 0, "2d"),
    ("per_row", 128, "2d"),     # MLP chunked path — where the 70B crash hit
    ("per_row", 0, "3d"),
    ("per_head", 0, "3d"),
])
def test_runs_on_second_gpu(granularity, chunk_d, shape):
    """sc_matmul on cuda:1 inputs must run and match the cuda:0 result.

    The current device stays cuda:0 (default); the guard inside sc_matmul is
    what makes the kernels launch on cuda:1 instead of faulting.
    """
    from scmp_kernels import sc_matmul
    torch.manual_seed(0)
    if shape == "2d":
        a0 = torch.randn(8, 256, device="cuda:0")
        b0 = torch.randn(16, 256, device="cuda:0")
    else:
        a0 = torch.randn(4, 8, 64, device="cuda:0")
        b0 = torch.randn(4, 16, 64, device="cuda:0")

    assert torch.cuda.current_device() == 0
    y0 = sc_matmul(a0, b0, granularity=granularity, chunk_d=chunk_d, sc_prec=8)
    y1 = sc_matmul(a0.to("cuda:1"), b0.to("cuda:1"),
                   granularity=granularity, chunk_d=chunk_d, sc_prec=8)

    assert y1.device.index == 1                 # output stays on the input GPU
    assert torch.cuda.current_device() == 0     # guard restored the device
    # Same RNG config + inputs -> identical SC result on either GPU.
    assert torch.equal(y0.cpu(), y1.cpu())

"""Kernel-level SC MSE profiler.

Compares ``sc_matmul(a, b)`` against the exact ``a @ b.T`` on captured real
operand pairs, sweeping SC configurations. Unit-level (no model forward).

Fixed:   sc_prec=8, mode=bipolar, granularity=per_row, chunk_d=128.
Swept:   stoc_len (+ the uSystolic halve variant) x SC_OWEN_MODE scramble.

The RNG enable-table cache is NOT keyed by scramble mode, so we call
``clear_rng_cache()`` before every measured call.

    python bench/mse_profile.py --tensors captured.pt --out mse_profile.csv
"""
from __future__ import annotations
import argparse, csv, math, os, torch
from scmp_kernels import sc_matmul
from scmp_kernels.sc import clear_rng_cache

SC_PREC = 8
CHUNK_D = 128


def metrics(sc: torch.Tensor, ref: torch.Tensor) -> dict:
    d = (sc - ref).float()
    mse = d.pow(2).mean().item()
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "max_abs": d.abs().max().item(),
        "rel_fro": (d.norm() / ref.float().norm().clamp_min(1e-12)).item(),
        "cos": torch.nn.functional.cosine_similarity(
            sc.float().flatten(), ref.float().flatten(), dim=0).item(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensors", default="captured.pt")
    ap.add_argument("--out", default="mse_profile.csv")
    ap.add_argument("--stoc-lens", default="256,128,64,32,16")
    ap.add_argument("--owen-modes", default="off,counter,bitrev,random")
    args = ap.parse_args()

    stoc_lens = [int(x) for x in args.stoc_lens.split(",")]
    owen_modes = [x.strip() for x in args.owen_modes.split(",")]
    # (stoc_len, halve_flag): explicit lengths + the halve (uSystolic) variant
    sl_configs = [(sl, False) for sl in stoc_lens] + [(None, True)]

    data = torch.load(args.tensors)
    dev = "cuda"
    rows = []

    for name, (a_cpu, b_cpu) in sorted(data.items()):
        a = a_cpu.to(dev).float()
        b = b_cpu.to(dev).float()
        ref = a @ b.t()                      # exact reference
        ref_rms = ref.float().pow(2).mean().sqrt().item()
        for owen in owen_modes:
            os.environ["SC_OWEN_MODE"] = owen
            os.environ.pop("SC_DISABLE_OWEN", None)
            for sl, halve in sl_configs:
                clear_rng_cache()
                sc = sc_matmul(
                    a, b, granularity="per_row", mode="bipolar",
                    sc_prec=SC_PREC, stoc_len=sl, chunk_d=CHUNK_D,
                    halve_bipolar_stoc_len=halve,
                )
                m = metrics(sc, ref)
                eff_sl = sl if sl is not None else 2 ** (SC_PREC - 1)
                kind = "halve" if halve else "full"
                rows.append({
                    "tensor": name, "owen": owen, "stream": kind,
                    "stoc_len": eff_sl, "ref_rms": ref_rms, **m,
                })

    # CSV
    fields = ["tensor", "owen", "stream", "stoc_len", "ref_rms",
              "mse", "rmse", "max_abs", "rel_fro", "cos"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # table
    print(f"\n{'tensor':18s} {'owen':8s} {'stream':6s} {'sL':>4s} "
          f"{'mse':>11s} {'rmse':>9s} {'rel_fro':>9s} {'cos':>8s}")
    print("-" * 84)
    for r in rows:
        print(f"{r['tensor']:18s} {r['owen']:8s} {r['stream']:6s} {r['stoc_len']:>4d} "
              f"{r['mse']:11.3e} {r['rmse']:9.3e} {r['rel_fro']:9.3e} {r['cos']:8.5f}")
    print(f"\nwrote {len(rows)} rows -> {args.out}")

    # quick wins: best scramble per (tensor, stoc_len) by rel_fro
    print("\n=== best SC_OWEN_MODE per (tensor, stoc_len) by rel_fro ===")
    from collections import defaultdict
    best = {}
    for r in rows:
        k = (r["tensor"], r["stream"], r["stoc_len"])
        if k not in best or r["rel_fro"] < best[k]["rel_fro"]:
            best[k] = r
    for k, r in sorted(best.items()):
        print(f"  {r['tensor']:18s} {r['stream']:6s} sL={r['stoc_len']:>4d} "
              f"-> {r['owen']:8s} (rel_fro {r['rel_fro']:.3e})")


if __name__ == "__main__":
    main()

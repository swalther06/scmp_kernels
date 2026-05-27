"""Ablation / sweep: SC matmul accuracy across the full design grid, at the
kernel level on captured real operands.

Axes (all crossed):
  * stoc_len   in {256,192,128,96,64,48,32,16}   (stream length)
  * halve      in {off, on}    (uSystolic: on -> RNG enable-grid = 2^(prec-1))
  * scramble   in {off, counter, bitrev}  (Owen de-bias mode; off = ablation baseline)

Fixed: per_row, bipolar, sc_prec=8, chunk_d=128. SC_SCRAMBLE_RESCALE follows
the scramble setting (1 unless scramble=off). Metric: rel_fro vs exact a @ b.T,
meaned over the captured tensors. RNG cache cleared before every call.

    python bench/ablation.py --tensors bench/captured_llama8b.pt
"""
from __future__ import annotations
import argparse, csv, os, torch
from collections import defaultdict
from scmp_kernels import sc_matmul
from scmp_kernels.sc import clear_rng_cache

SC_PREC, CHUNK_D = 8, 128
STOC_LENS = [256, 192, 128, 96, 64, 48, 32, 16]
HALVES = [("full", False), ("halve", True)]
SCRAMBLES = ["off", "counter", "bitrev"]   # off = ablation baseline


def mse(sc, ref):
    return (sc - ref).float().pow(2).mean().item()


def rel_fro(sc, ref):
    return ((sc - ref).float().norm() / ref.float().norm().clamp_min(1e-12)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensors", default="bench/captured_llama8b.pt")
    ap.add_argument("--out", default="bench/ablation.csv")
    ap.add_argument("--stoc-lens", default=",".join(map(str, STOC_LENS)))
    ap.add_argument("--scrambles", default=",".join(SCRAMBLES))
    args = ap.parse_args()
    stoc_lens = [int(x) for x in args.stoc_lens.split(",")]
    scrambles = [s.strip() for s in args.scrambles.split(",")]

    data = torch.load(args.tensors)
    rows = []
    for name, (a_cpu, b_cpu) in sorted(data.items()):
        a, b = a_cpu.cuda().float(), b_cpu.cuda().float()
        ref = a @ b.t()
        for sl in stoc_lens:
            for hlab, halve in HALVES:
                for scr in scrambles:
                    os.environ["SC_OWEN_MODE"] = scr
                    os.environ["SC_SCRAMBLE_RESCALE"] = "0" if scr == "off" else "1"
                    os.environ.pop("SC_DISABLE_OWEN", None)
                    clear_rng_cache()
                    sc = sc_matmul(a, b, granularity="per_row", mode="bipolar",
                                   sc_prec=SC_PREC, stoc_len=sl, chunk_d=CHUNK_D,
                                   halve_bipolar_stoc_len=halve)
                    rows.append({"tensor": name, "proj": name.split(".")[1],
                                 "stoc_len": sl, "halve": hlab, "scramble": scr,
                                 "mse": mse(sc, ref), "rel_fro": rel_fro(sc, ref)})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tensor", "proj", "stoc_len", "halve", "scramble", "mse", "rel_fro"])
        w.writeheader(); w.writerows(rows)

    # mean MSE over tensors per (stoc_len, halve, scramble)
    agg = defaultdict(list)
    for r in rows:
        agg[(r["stoc_len"], r["halve"], r["scramble"])].append(r["mse"])
    mean = {k: sum(v) / len(v) for k, v in agg.items()}

    cols = [(h, s) for h, _ in HALVES for s in scrambles]
    print("\n=== mean MSE over 21 tensors (lower is better) ===")
    hdr = "stoc_len | " + " ".join(f"{h[:4]}/{s[:4]:<8s}" for h, s in cols)
    print(hdr); print("-" * len(hdr))
    for sl in stoc_lens:
        cells = " ".join(f"{mean[(sl, h, s)]:<13.4e}" for h, s in cols)
        print(f"{sl:>8d} | {cells}")

    # scramble contribution: bitrev vs off, per stoc_len (full stream), MSE
    print("\n=== scramble effect on MSE (bitrev vs off), full stream ===")
    for sl in stoc_lens:
        on, off = mean[(sl, "full", "bitrev")], mean[(sl, "full", "off")]
        print(f"  stoc_len={sl:>3d}: off={off:.4e} bitrev={on:.4e}  ({(off-on)/on*100:+5.0f}% if removed)")

    # halve cost: full vs halve at bitrev
    print("\n=== halve cost on MSE (full vs halve, bitrev) ===")
    for sl in stoc_lens:
        full, halve = mean[(sl, "full", "bitrev")], mean[(sl, "halve", "bitrev")]
        print(f"  stoc_len={sl:>3d}: full={full:.4e} halve={halve:.4e}  ({(halve-full)/full*100:+5.0f}%)")

    print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()

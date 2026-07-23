#!/usr/bin/env python3
"""Step 1 of the mixed-precision energy flow: trace -> per-(shape, stream-length) BUNDLES.

SCMP splits each SC matmul PER ROW by importance into stream-length tiers
(rows -> L in {32,64,128,256,...}). A DiT layer's matmul GEMM(M, K, N) therefore
becomes a bundle of disjoint row-tiers:  GEMM(rows_L, K, N) @ L, summed.

This aggregates the per-call trace into, for each distinct matmul shape
(K=d_in, N=d_out) and stream length L (stoc_len), the total work over the whole
run:  rows, MACs, calls.  Each distinct (shape, L) is ONE Timeloop run in step 3;
the rows/MACs here are the weights you multiply that run's per-MAC energy by.

Reads the *_percall.jsonl (streamed line-by-line, so the 100+ MB files are fine).

Usage:
    python3 trace_to_bundles.py sc_traces/trace_mp_cosine_avg32_percall.jsonl
    python3 trace_to_bundles.py <trace.jsonl> -o bundles.csv
"""
import argparse
import csv
import json
import sys
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="path to a *_percall.jsonl SC-matmul trace")
    ap.add_argument("-o", "--out", help="write the bundle table to this CSV")
    args = ap.parse_args()

    # (K=d_in, N=d_out, L=stoc_len) -> {rows, macs, calls}
    agg: dict[tuple, dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "macs": 0, "calls": 0})
    n_records = 0
    with open(args.trace) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "d_in" not in rec:      # first line is the header
                continue
            key = (rec["d_in"], rec["d_out"], rec["stoc_len"])
            a = agg[key]
            a["rows"] += rec.get("rows", 0)
            a["macs"] += rec.get("macs", 0)
            a["calls"] += 1
            n_records += 1

    # Per-shape rollups (for the % breakdown and totals).
    shape_rows: dict[tuple, int] = defaultdict(int)
    shape_macs: dict[tuple, int] = defaultdict(int)
    for (k, n, L), a in agg.items():
        shape_rows[(k, n)] += a["rows"]
        shape_macs[(k, n)] += a["macs"]
    total_macs = sum(a["macs"] for a in agg.values())

    rows_out = []
    for (k, n, L), a in sorted(agg.items(), key=lambda kv: (-shape_macs[(kv[0][0], kv[0][1])],
                                                            kv[0][0], kv[0][1], kv[0][2])):
        rows_out.append({
            "K": k, "N": n, "L": L,
            "rows": a["rows"], "macs": a["macs"], "calls": a["calls"],
            "frac_rows_in_shape": a["rows"] / shape_rows[(k, n)],
            "frac_macs_of_model": a["macs"] / total_macs,
        })

    # ---- human-readable per-shape breakdown -------------------------------
    print(f"scanned {n_records:,} call records | {len(agg)} distinct (shape,L) "
          f"= Timeloop runs needed | total MACs {total_macs:,}\n")
    shapes = sorted(shape_macs, key=lambda s: -shape_macs[s])
    for (k, n) in shapes:
        tiers = sorted([r for r in rows_out if r["K"] == k and r["N"] == n],
                       key=lambda r: r["L"])
        share = shape_macs[(k, n)] / total_macs
        print(f"GEMM K={k} N={n}   ({share*100:4.1f}% of model MACs, "
              f"{shape_rows[(k,n)]:,} rows total)")
        for r in tiers:
            print(f"    L={r['L']:>4}  rows={r['rows']:>12,}  "
                  f"({r['frac_rows_in_shape']*100:5.1f}% of this layer's rows)  "
                  f"macs={r['macs']:>16,}")
        print()

    # ---- CSV (the machine-readable bundle table for step 3) ---------------
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print(f"wrote bundle table -> {args.out}")


if __name__ == "__main__":
    main()

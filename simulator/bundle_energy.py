#!/usr/bin/env python3
"""Step 3: energy of a mixed-precision DiT from the per-(shape, L) bundle table.

Ties the whole flow together. `make trace_breakdown` produces a bundle CSV
(trace_to_bundles.py) with one row per distinct (shape=K,N, stream-length L)
tier and its real `rows`/`macs` counts. For EACH tier this script:

    1. runs one Timeloop mapping via `make mapping PROB_M=M_rep PROB_K=K
       PROB_N=N STREAM_LEN=L` (which regenerates configs and runs the container
       flow with the k8m16n8 array + GLB_SIZE_KB defaults from the Makefile),
    2. reads back the energy/MAC (and per-component breakdown) at that L, and
    3. multiplies by the tier's real MAC count to get the tier's energy.

Because mixed precision is assigned PER ROW (the M dimension), the tiers of a
shape are DISJOINT row-sets, so summing over tiers is concatenation -- no
double counting. The per-MAC energy is measured at a small representative M
(M_rep) and scaled by

    s = macs(tier) / (M_rep*K*N) = rows(tier) / M_rep

which carries the tier's row proportion (per-MAC energy is ~M-invariant once
reuse saturates -- that's the approximation here).

Energy note: the SC energy is currently LUMPED into the compute action and
scales linearly with L (see accelergy-sc-plugin/sc_mac.py). There is no
dynamic/leak split yet, so "static" is reported as 0 -- when leakage is
characterized it will separate out of the MACC bucket automatically.

Usage:
    make trace_breakdown                      # produce the bundle CSV first
    python3 bundle_energy.py sc_traces/trace_mp_cosine_avg32_bundles.csv
    python3 bundle_energy.py <csv> --m-rep 512 --glb-size-kb 512
    python3 bundle_energy.py <csv> --dry-run  # preview the tiers/weights only
"""
import argparse
import csv
import os
import re
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
STATS = os.path.join(HERE, "timeloop_container_run/out/timeloop-mapper.stats.txt")
# Components Timeloop reports as fJ/Compute; grouped into SC-compute vs memory.
COMPUTE_COMPONENTS = {"MACC", "Peripheral", "SobolBank"}   # the SC datapath (our plug-in)
MEMORY_COMPONENTS = {"GlobalBuffer", "DRAM"}               # CACTI storage


def run_mapping(M: int, K: int, N: int, L: int, glb_size_kb: float) -> dict:
    """`make mapping` for one GEMM(M,K,N) at stream length L; parse the stats.

    Returns total energy (uJ), cycles, and a per-component fJ/MAC dict.
    """
    cmd = ["make", "mapping",
           f"PROB_M={M}", f"PROB_K={K}", f"PROB_N={N}", f"STREAM_LEN={L}",
           f"GLB_SIZE_KB={glb_size_kb:g}"]
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if not os.path.exists(STATS):
        raise RuntimeError(
            f"no stats for {K}x{N}@L{L} (M={M})\n--- make tail ---\n{r.stdout[-800:]}"
        )
    txt = open(STATS).read()
    m_e = re.search(r"Energy:\s*([\d.eE+-]+)\s*uJ", txt)
    m_c = re.search(r"Cycles:\s*(\d+)", txt)
    if not (m_e and m_c):
        raise RuntimeError(f"could not parse Energy/Cycles for {K}x{N}@L{L}")
    # Per-component "Name = <fJ/Compute>" lines (already per-MAC).
    comp = {n: float(v) for n, v in
            re.findall(r"^\s*([A-Za-z_]+)\s*=\s*([\d.]+)\s*$", txt, re.M)}
    return {"energy_uj": float(m_e.group(1)),
            "cycles": int(m_c.group(1)),
            "fj_per_mac": comp}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundles", help="bundle CSV from trace_to_bundles.py / make trace_breakdown")
    ap.add_argument("--m-rep", type=int, default=512,
                    help="representative per-call M each mapping runs at (default 512)")
    ap.add_argument("--glb-size-kb", type=float, default=256.0,
                    help="on-chip SRAM size passed to each mapping (default 256)")
    ap.add_argument("--out", default=None,
                    help="summary CSV path (default: <bundles>_energy.csv)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the (shape,L) runs + weights without invoking make")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.bundles)))
    out_csv = args.out or re.sub(r"(_bundles)?\.csv$", "_energy.csv", args.bundles)

    per_shape = defaultdict(lambda: {"E_pj": 0.0, "macs": 0, "cyc": 0.0})
    comp_pj = defaultdict(float)               # model component -> pJ
    total = {"E_pj": 0.0, "macs": 0, "cyc": 0.0}
    tier_rows, failures = [], []

    hdr = f"{'K':>6} {'N':>6} {'L':>4} {'rows':>13} {'scale s':>10} {'pJ/MAC':>8} {'E_tier(uJ)':>11}"
    print(f"{len(rows)} (shape,L) tiers | M_rep={args.m_rep} | GLB={args.glb_size_kb:g} KB\n")
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        K, N, L = int(r["K"]), int(r["N"]), int(r["L"])
        macs, rows_L = int(r["macs"]), int(r["rows"])
        s = macs / (args.m_rep * K * N)        # = rows_L / M_rep

        if args.dry_run:
            print(f"{K:>6} {N:>6} {L:>4} {rows_L:>13,} {s:>10.1f} {'(dry)':>8} {'(dry)':>11}")
            continue

        try:
            run = run_mapping(args.m_rep, K, N, L, args.glb_size_kb)
        except RuntimeError as e:
            print(f"{K:>6} {N:>6} {L:>4} {rows_L:>13,} {s:>10.1f}   FAILED", file=sys.stderr)
            failures.append((K, N, L, str(e).splitlines()[0]))
            continue

        pj_per_mac = sum(run["fj_per_mac"].get(c, 0.0)
                         for c in COMPUTE_COMPONENTS | MEMORY_COMPONENTS) / 1e3
        e_tier_pj = pj_per_mac * macs           # = energy/MAC * #MACs for this L
        c_tier = run["cycles"] * s
        per_shape[(K, N)]["E_pj"] += e_tier_pj
        per_shape[(K, N)]["macs"] += macs
        per_shape[(K, N)]["cyc"] += c_tier
        total["E_pj"] += e_tier_pj
        total["macs"] += macs
        total["cyc"] += c_tier
        for name, fj in run["fj_per_mac"].items():
            if name in COMPUTE_COMPONENTS or name in MEMORY_COMPONENTS:
                comp_pj[name] += fj / 1e3 * macs
        tier_rows.append({"K": K, "N": N, "L": L, "rows": rows_L, "macs": macs,
                          "scale_s": f"{s:.3f}", "pJ_per_mac": f"{pj_per_mac:.4f}",
                          "E_tier_uJ": f"{e_tier_pj/1e6:.4f}", "cycles": int(c_tier)})
        print(f"{K:>6} {N:>6} {L:>4} {rows_L:>13,} {s:>10.1f} {pj_per_mac:>8.4f} {e_tier_pj/1e6:>11.4f}")

    if args.dry_run:
        return

    # ---- per-shape (layer) rollup ----
    print("\n=== per-shape (layer) ===")
    for (K, N), d in sorted(per_shape.items(), key=lambda kv: -kv[1]["E_pj"]):
        print(f"  GEMM {K:>5}x{N:<5}  {d['E_pj']/1e6:9.3f} uJ   "
              f"{d['E_pj']/d['macs']:.4f} pJ/MAC (effective)   {d['macs']:>16,} MACs")

    # ---- component breakdown (SC compute vs memory) ----
    E = total["E_pj"] or 1.0
    sc = sum(comp_pj[c] for c in COMPUTE_COMPONENTS)
    mem = sum(comp_pj[c] for c in MEMORY_COMPONENTS)
    print("\n=== component breakdown (model) ===")
    for name in sorted(comp_pj, key=lambda k: -comp_pj[k]):
        print(f"  {name:<14} {comp_pj[name]/1e6:9.3f} uJ  "
              f"{comp_pj[name]/total['macs']:.4f} pJ/MAC  {100*comp_pj[name]/E:5.1f}%")
    print(f"  {'-- SC compute':<14} {sc/1e6:9.3f} uJ  {100*sc/E:5.1f}%")
    print(f"  {'-- memory':<14} {mem/1e6:9.3f} uJ  {100*mem/E:5.1f}%")

    # ---- model total ----
    print("\n=== MODEL TOTAL (SC region) ===")
    print(f"  dynamic energy : {total['E_pj']/1e6:.3f} uJ  ({total['E_pj']/1e9:.4f} mJ)")
    print(f"  static energy  : 0.000 uJ   (no dyn/leak split yet -- lumped into compute)")
    print(f"  total MACs     : {total['macs']:,}")
    print(f"  energy/MAC     : {total['E_pj']/total['macs']:.4f} pJ/MAC (effective, mixed precision)")
    print(f"  latency        : {total['cyc']:,.0f} Timeloop cycles (logical; sequential tiers)")
    if failures:
        print(f"\n!! {len(failures)} tier(s) FAILED to map:")
        for K, N, L, msg in failures:
            print(f"   {K}x{N}@L{L}: {msg}")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["K", "N", "L", "rows", "macs",
                                          "scale_s", "pJ_per_mac", "E_tier_uJ", "cycles"])
        w.writeheader()
        w.writerows(tier_rows)
    print(f"\nwrote per-tier summary -> {out_csv}")


if __name__ == "__main__":
    main()

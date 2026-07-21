"""Search for an optimal GEMM mapping onto the SC systolic array using timeloop-mapper.

Maps a GEMM problem (M, K, N) onto the array modeled in
gemm_cycle_accurate_sim.cpp: Inputs (M x K), Weights (K x N), Outputs (M x N),
with spatial dims P_ROWS x P_COLS (tiles) and N_H x N_W (PEs/tile), and a
K-deep dot product per PE.

Drives the timeloop-mapper C++ binary (vendored at simulator/timeloop)
directly via subprocess, rather than through pytimeloop -- pytimeloop has no
PyPI package and needs a from-source build (Timeloop's C++ core + islpy w/
Barvinok support) just to get an in-process Python API that, per its own
backend_calls.py, generates the same YAML and shells out to timeloop-mapper
anyway. Building just the plain timeloop-mapper binary (ordinary Timeloop
deps only, no islpy/Barvinok) is a strictly smaller lift, and we get the
whole config surface for free since it's just YAML we already generate.

This is still scaffolding for the arch/energy model and operand generation
(see TODOs) -- but the problem-spec schema and the invocation/output-parsing
below are validated against real examples vendored in simulator/timeloop
(problem-shapes/gemm_ABZ.yaml, orojenesis/src/utils.py's RunMapper) and in
the sibling timeloop-accelergy-exercises repo's reference outputs.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

THIS_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Problem / architecture description
# ---------------------------------------------------------------------------

@dataclass
class GemmProblem:
    """Logical GEMM dimensions: Inputs (M x K) @ Weights (K x N) -> Outputs (M x N)."""
    M: int
    K: int
    N: int


@dataclass
class ArchConfig:
    """Mirrors the CFG_* knobs in gemm_cycle_accurate_sim.cpp.

    These are the file's own 4 DSE axes (see its "All four DSE axes are
    implemented" doc comment): P (p_rows x p_cols, tile grid), N (n_h x n_w,
    PEs/tile), K (k_depth, dot-product lane width), M (m_parallel, bitstream
    samples/lane/cycle). m_parallel doesn't tile any GEMM data-reuse
    dimension (M/K/N) -- it's a bit-serial SC throughput knob, not something
    Timeloop's mapper searches over -- see stream_length()/cycles_per_mac_window()
    and apply_hardware_timing() below for how it folds into real cycle counts.
    """
    p_rows: int = 4      # systolic tile rows  (spatial, output-M side)
    p_cols: int = 4      # systolic tile cols  (spatial, output-N side)
    n_h: int = 1         # PEs per tile, row direction (spatial, output-M side)
    n_w: int = 1         # PEs per tile, col direction (spatial, output-N side)
    k_depth: int = 4     # dot-product lane width per PE (reduction dim, temporal/spatial-K)
    m_parallel: int = 1  # bitstream samples processed per lane per cycle (CFG_M)
    mag_bits: int = 7    # magnitude precision (sets T = 2^mag_bits by default)


def stream_length(arch: ArchConfig) -> int:
    """Bitstream length T (cycles to stream one full window at m_parallel=1).

    Matches gemm_cycle_accurate_sim.cpp's default CFG_STOC_LEN = 1 << MAG_BITS
    -- not independently overridable via this project's Makefile, so mag_bits
    determines T. This is the uSystolic/HUB sign-magnitude cycle-halving
    convention (wu-hpca2022): bipolar magnitudes only carry mag_bits of
    information, so a stream/RNG grid of 2^mag_bits is sufficient -- matches
    scmp_kernels' halve_bipolar_stoc_len=True and SCArch's
    power_array_3d.sv (T=128 for int8), not the 2^(mag_bits+1) legacy default.
    """
    return 1 << arch.mag_bits


def cycles_per_mac_window(arch: ArchConfig) -> int:
    """Real hardware cycles for one K-deep dot-product accumulate window.

    T/m_parallel ticks to stream all bits through (m_parallel samples/lane/
    cycle, inversely proportional to T per the .cpp file), + 1 pipeline cycle.
    """
    return stream_length(arch) // arch.m_parallel + 1


# ---------------------------------------------------------------------------
# Timeloop problem / architecture / mapper spec builders
# ---------------------------------------------------------------------------

class Tagged(dict):
    """A dict that dumps with a YAML tag, e.g. `!Container`/`!Component`."""

    def __init__(self, tag: str, data: dict[str, Any]):
        super().__init__(data)
        self.tag = tag


class FlowList(list):
    """A list that dumps in flow style, e.g. `[ M, K, N ]` instead of block."""


class QuotedStr(str):
    """A string dumped double-quoted. timeloopfe *evaluates* attribute/variable
    scalars as expressions, so a bare string like 45nm parses as `45` then `nm`
    (a syntax error); quoting makes it a string literal instead."""


class TimeloopDumper(yaml.SafeDumper):
    pass


TimeloopDumper.add_representer(
    Tagged, lambda dumper, data: dumper.represent_mapping(data.tag, dict(data))
)
TimeloopDumper.add_representer(
    FlowList, lambda dumper, data: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", list(data), flow_style=True
    )
)
TimeloopDumper.add_representer(
    QuotedStr, lambda dumper, data: dumper.represent_scalar(
        "tag:yaml.org,2002:str", str(data), style='"'
    )
)


def _flow_seq(dim: str) -> FlowList:
    """A single-dim projection term, e.g. `[ [M] ]`."""
    return FlowList([FlowList([dim])])


def build_problem_spec(problem: GemmProblem) -> dict[str, Any]:
    """Build a timeloop problem YAML dict for the GEMM shape.

    Dataspace projections follow the array's operand layout:
      Inputs  -> (M, K)
      Weights -> (K, N)
      Outputs -> (M, N), read-write (accumulates over K)
    """
    # v0.4 schema: explicit `version`, `data_spaces`/`read_write` spelled with
    # underscores (see build_arch_spec's docstring for why this project has
    # moved to the v0.4/timeloopfe-flavored schema throughout).
    return {
        "problem": {
            "version": 0.4,
            "shape": {
                "name": "GEMM",
                "dimensions": FlowList(["M", "K", "N"]),
                "data_spaces": [
                    {
                        "name": "Inputs",
                        "projection": [_flow_seq("M"), _flow_seq("K")],
                    },
                    {
                        "name": "Weights",
                        "projection": [_flow_seq("K"), _flow_seq("N")],
                    },
                    {
                        "name": "Outputs",
                        "projection": [_flow_seq("M"), _flow_seq("N")],
                        "read_write": True,
                    },
                ],
            },
            "instance": {"M": problem.M, "K": problem.K, "N": problem.N},
        }
    }


def build_globals_spec() -> dict[str, Any]:
    """Top-level `globals` section (timeloopfe v4 requires it)."""
    return {"globals": {"version": 0.4}}


def build_variables_spec() -> dict[str, Any]:
    """Top-level `variables` section (the global clock).

    NB: timeloopfe *evaluates* variable values as expressions, so only put
    numbers here -- `technology` is a string ("45nm"), which would be parsed as
    an expression and fail, so it lives on the `chip` container attribute
    instead (component attributes are not expression-evaluated).
    """
    return {
        "variables": {
            "version": 0.4,
            "global_cycle_seconds": 1e-9,
        }
    }


def build_arch_spec(arch: ArchConfig, with_glb: bool = True) -> dict[str, Any]:
    """Build a timeloopfe-v4 architecture for the SC systolic array.

    with_glb=False drops the on-chip SRAM (operands streamed straight from
    DRAM) -- the "simple tiling" baseline for comparison against the
    reuse-blocked optimal.

    Hierarchy: DRAM -> on-chip SRAM (GlobalBuffer) -> P_H x P_W PE mesh ->
    MACC. MACC's compute energy comes from the accelergy-sc-plugin's measured
    `sc_mac_inner` estimator; DRAM/SRAM from Accelergy's built-in CACTI models.

    Consumed by the container flow (`make mapping`) via
    pytimeloop.timeloopfe.v4 -- NOT the native timeloop-mapper binary. The two
    have different schemas; this file targets the one that actually runs here.

    Three SC energy components (PaYN structure), each with its own action driver:
      - MACC (sc_mac_inner): INNER compute, per MAC.
      - Peripheral (peripheral): the binary->stochastic comparators, per operand
        READ (a level in the operand path -> mapping-dependent via reuse).
      - SobolBank (sobol_bank): the shared RNG, per CYCLE (leak). Not in the
        operand dataflow, so it bypasses all dataspaces and only contributes
        leak; instances=1 (broadcast to the whole array).
    (`sc_mac_inner` is a *compute* class -- accepted once the container registers
    it in COMPUTE_CLASSES; peripheral/sobol_bank parse as storage levels.)
    """
    mesh_y = arch.p_rows * arch.n_h        # spatial-M extent (mesh rows)
    mesh_x = arch.p_cols * arch.n_w        # spatial-N extent (mesh cols)
    # `technology` sits on the top container so every component (incl. DRAM)
    # inherits it -- Accelergy requires it on all of them. QuotedStr keeps
    # timeloopfe from evaluating "45nm" as an expression.
    nodes = [
        Tagged("!Container", {
            "name": "system",
            "attributes": {"technology": QuotedStr("45nm")},
        }),
        Tagged("!Component", {
            "name": "DRAM",
            "class": "DRAM",
            # depth just needs to exceed the backing working set;
            # 2**28 words covers up to ~16k x 16k tensors.
            "attributes": {"width": 256, "datawidth": 8, "depth": 1 << 28},
        }),
    ]
    if with_glb:
        nodes.append(Tagged("!Component", {
            "name": "GlobalBuffer",
            "class": "SRAM",
            # depth/width/datawidth drive the CACTI SRAM energy model.
            # Placeholder GLB sizing -- tune to the real on-chip buffer.
            "attributes": {"depth": 16384, "width": 256, "datawidth": 8},
        }))
    nodes += [
        # OUTER (shared): Sobol RNG banks. Carry no operand data -- just per-cycle
        # side energy billed as `leak`. instances:1 (broadcast array-wide); bypass
        # every dataspace so it holds nothing. Energy from the sobol_bank estimator.
        # class: storage (a recognized Timeloop storage class) + subclass names
        # the Accelergy estimator (like the stock `storage`/`aladdin_register`).
        Tagged("!Component", {
            "name": "SobolBank",
            "class": "storage",
            "subclass": "sobol_bank",
            "attributes": {"instances": 1, "depth": 1, "width": 8, "datawidth": 8},
            "constraints": {"dataspace": {
                "bypass": FlowList(["Inputs", "Weights", "Outputs"])}},
        }),
        # OUTER (per-operand): binary->stochastic conversion. A level in the
        # operand path -- keeps Inputs/Weights, bypasses Outputs -- so Timeloop
        # bills its `read` per operand delivered (mapping-dependent via reuse).
        Tagged("!Component", {
            "name": "Peripheral",
            "class": "storage",
            "subclass": "peripheral",
            "attributes": {"depth": 64, "width": 8, "datawidth": 8},
            "constraints": {"dataspace": {
                "keep": FlowList(["Inputs", "Weights"]),
                "bypass": FlowList(["Outputs"])}},
        }),
        # 2D output-tile mesh: M -> rows (meshY), N -> cols (meshX).
        Tagged("!Container", {
            "name": "PE",
            "spatial": {"meshX": mesh_x, "meshY": mesh_y},
        }),
        # Spatial-K reduction: the k_depth-wide popcount lanes inside each PE
        # (the SC array reduces k_depth K-elements in parallel per window).
        Tagged("!Container", {
            "name": "Klane",
            "spatial": {"meshX": arch.k_depth},
        }),
        Tagged("!Component", {
            "name": "MACC",
            "class": "sc_mac_inner",
            "attributes": {
                "mag_bits": arch.mag_bits,
                "datawidth": arch.mag_bits + 1,
            },
        }),
    ]
    return {"architecture": {"version": 0.4, "nodes": nodes}}


def build_mapper_spec() -> dict[str, Any]:
    """Mapper search settings (metric, algorithm, threads, termination)."""
    # timeloopfe v4 uses `optimization_metrics` (plural) + snake_case keys --
    # confirmed against example_designs/_include/mapper.yaml. `energy` makes the
    # search minimise total energy (the energy-optimal mapping).
    return {
        "mapper": {
            "version": 0.4,
            "optimization_metrics": ["energy"],
            "algorithm": "random_pruned",
            "num_threads": 8,
            "timeout": 15000,
            "victory_condition": 300,
            "live_status": False,
            "diagnostics": True,   # report why mappings are invalid if none found
        }
    }


def build_constraints_spec(problem: GemmProblem, arch: ArchConfig) -> dict[str, Any]:
    """Pin the mapper to the SC array's physical dataflow.

    timeloopfe-v4 top-level `constraints`: a `targets` list of
    {type, target, factors} (factors as a list, e.g. [M=4, N=4]).
    """
    mesh_y = arch.p_rows * arch.n_h        # spatial-M (mesh rows)
    mesh_x = arch.p_cols * arch.n_w        # spatial-N (mesh cols)
    # Pin the physical, fixed dataflow of the SC array so the mapper can't
    # explore physically-unrealizable spatial mappings:
    #   PE mesh:  M -> rows (meshY), N -> cols (meshX)  [output-stationary]
    #   Klane:    K -> k_depth parallel popcount lanes  [spatial reduction]
    # `split`+`permutation` assign factors to the mesh axes: split=1 sends the
    # first permuted factor to meshX and the rest to meshY.
    return {
        "constraints": {
            "targets": [
                {"type": "spatial", "target": "PE",
                 "factors": FlowList([f"M={mesh_y}", f"N={mesh_x}"]),
                 "permutation": FlowList(["N", "M"]),
                 "split": 1},
                {"type": "spatial", "target": "Klane",
                 "factors": FlowList([f"K={arch.k_depth}"]),
                 "permutation": FlowList(["K"]),
                 "split": 1},
            ]
        }
    }


def validate_divisibility(problem: GemmProblem, arch: ArchConfig) -> None:
    """Fail loudly if the problem dims don't fit the fixed array geometry.

    The spatial factors are pinned to the mesh (M=P_ROWS*N_H, N=P_COLS*N_W) and
    K to k_depth, so M/N/K must divide evenly -- there's no padding support. A
    non-divisible size would otherwise yield an infeasible/silently-wrong
    mapping.
    """
    mesh_y = arch.p_rows * arch.n_h
    mesh_x = arch.p_cols * arch.n_w
    errs = []
    if problem.M % mesh_y:
        errs.append(f"M={problem.M} not divisible by spatial-M = P_ROWS*N_H = {mesh_y}")
    if problem.N % mesh_x:
        errs.append(f"N={problem.N} not divisible by spatial-N = P_COLS*N_W = {mesh_x}")
    if problem.K % arch.k_depth:
        errs.append(f"K={problem.K} not divisible by k_depth = {arch.k_depth}")
    if errs:
        raise ValueError(
            "Problem dims incompatible with the array geometry (no padding yet):\n  "
            + "\n  ".join(errs)
        )


# ---------------------------------------------------------------------------
# Config I/O + timeloop-mapper invocation
# ---------------------------------------------------------------------------

def write_configs(
    problem: GemmProblem,
    arch: ArchConfig,
    out_dir: Path,
    with_glb: bool = True,
) -> list[Path]:
    """Emit the 4 timeloopfe-v4 config files (arch/problem/mapper/constraints).

    arch.yaml carries the `globals` + `variables` sections alongside
    `architecture` (timeloopfe merges top-level keys across the files).
    with_glb=False drops the on-chip SRAM (simple-tiling baseline).
    """
    validate_divisibility(problem, arch)
    out_dir.mkdir(parents=True, exist_ok=True)
    arch_file = {**build_globals_spec(), **build_variables_spec(),
                 **build_arch_spec(arch, with_glb=with_glb)}
    specs = {
        "arch.yaml": arch_file,
        "problem.yaml": build_problem_spec(problem),
        "mapper.yaml": build_mapper_spec(),
        "constraints.yaml": build_constraints_spec(problem, arch),
    }
    paths = []
    for filename, spec in specs.items():
        path = out_dir / filename
        path.write_text(yaml.dump(spec, Dumper=TimeloopDumper, sort_keys=False))
        paths.append(path)
    return paths


def find_mapper_binary() -> Path:
    """Locate the timeloop-mapper binary.

    Checks $TIMELOOP_BASE_PATH/bin first (the convention orojenesis/src/utils.py's
    RunMapper uses), then falls back to the vendored submodule's own build
    output (both `build/` and `bin/` show up across the scons build docs/
    scripts in this repo -- scripts/timeloop.py assumes `build/`).
    """
    candidates = []
    base = os.environ.get("TIMELOOP_BASE_PATH")
    if base:
        candidates.append(Path(base) / "bin" / "timeloop-mapper")
    candidates += [
        THIS_DIR / "timeloop" / "bin" / "timeloop-mapper",
        THIS_DIR / "timeloop" / "build" / "timeloop-mapper",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "timeloop-mapper binary not found. Build it from simulator/timeloop "
        "(scons -j$(nproc), see simulator/timeloop/README.md for the ordinary "
        "Timeloop dependencies -- no islpy/Barvinok needed for the plain "
        "binary), or set TIMELOOP_BASE_PATH to a directory with a built bin/."
    )


def run_mapper(config_paths: list[Path], output_dir: Path) -> Path:
    """Run timeloop-mapper directly via subprocess and return output_dir.

    Invocation confirmed against simulator/timeloop/orojenesis/src/utils.py's
    RunMapper: `timeloop-mapper <config files...> -o <output_dir>`. Output
    files (timeloop-mapper.map.yaml, .stats.txt, ...) land directly in
    output_dir; see extract_mapping for their format.
    """
    binary = find_mapper_binary()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "timeloop-mapper.log"
    cmd = [str(binary), *(str(p.resolve()) for p in config_paths), "-o", str(output_dir)]
    with open(log_path, "w") as log:
        result = subprocess.run(cmd, cwd=output_dir, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(
            f"timeloop-mapper failed (exit {result.returncode}). "
            f"Command: {' '.join(cmd)}\nSee {log_path} for details."
        )
    return output_dir


# ---------------------------------------------------------------------------
# Mapping extraction
# ---------------------------------------------------------------------------

@dataclass
class MappingResult:
    """Optimal mapping, expressed as per-dimension tile factors at each level.

    per_level_factors is keyed "<target>:<type>" (e.g. "PE:spatial",
    "DRAM:temporal") -- a level can appear once per loop type -- with each
    value a {dim_name: factor} dict (e.g. {"M": 4, "K": 1, "N": 4}), parsed
    straight out of timeloop-mapper.map.yaml's `factors: "M4 K1 N4"` strings.

    cycles is Timeloop's own count until apply_hardware_timing() rescales it
    (see that function for why: Timeloop assumes 1 cycle per innermost-K-loop
    iteration, but each of those iterations is really one bit-serial SC
    accumulate window taking cycles_per_mac_window(arch) real cycles).
    raw_cycles preserves Timeloop's pre-rescale number for reference.
    """
    m: int
    k: int
    n: int
    per_level_factors: dict[str, dict[str, int]]
    energy: float | None = None  # uJ, from timeloop-mapper.stats.txt
    cycles: int | None = None
    raw_cycles: int | None = None


_FACTOR_RE = re.compile(r"([A-Za-z]+)(\d+)")


def _parse_factors(factors_str: str) -> dict[str, int]:
    return {dim: int(val) for dim, val in _FACTOR_RE.findall(factors_str)}


def extract_mapping(problem: GemmProblem, output_dir: Path) -> MappingResult:
    """Parse timeloop-mapper.map.yaml + .stats.txt (in output_dir) into a MappingResult.

    map.yaml format (list of {target, type, factors, permutation, ...} plus
    datatype/keep-bypass entries we skip) and stats.txt's trailing
    "Summary Stats" block (Cycles: <n>, Energy: <x> uJ) are both confirmed
    against real reference output in the sibling timeloop-accelergy-exercises
    repo's workspace/example_designs/*/ref_outputs.
    """
    map_path = output_dir / "timeloop-mapper.map.yaml"
    stats_path = output_dir / "timeloop-mapper.stats.txt"

    mapping_entries = yaml.safe_load(map_path.read_text())["mapping"]
    per_level_factors: dict[str, dict[str, int]] = {}
    for entry in mapping_entries:
        if entry.get("type") not in ("temporal", "spatial"):
            continue  # skip "datatype" (keep/bypass) entries
        key = f"{entry['target']}:{entry['type']}"
        per_level_factors[key] = _parse_factors(entry.get("factors", ""))

    stats_text = stats_path.read_text()
    cycles_match = re.search(r"Cycles:\s*(\d+)", stats_text)
    energy_match = re.search(r"Energy:\s*([\d.eE+-]+)\s*uJ", stats_text)

    return MappingResult(
        m=problem.M,
        k=problem.K,
        n=problem.N,
        per_level_factors=per_level_factors,
        cycles=int(cycles_match.group(1)) if cycles_match else None,
        energy=float(energy_match.group(1)) if energy_match else None,
    )


def apply_hardware_timing(mapping: MappingResult, arch: ArchConfig) -> MappingResult:
    """Rescale Timeloop's naive cycle count to real SC hardware cycles.

    Timeloop's default timing model charges 1 cycle per innermost-loop
    iteration -- here, one iteration of the temporal/spatial K-loop pinned to
    k_depth (see build_constraints_spec). But each such iteration is really
    one bit-serial SC accumulate window, which takes cycles_per_mac_window(arch)
    real cycles (T/m_parallel + 1), not 1. Scale Timeloop's reported cycle
    count by that factor to get a real-hardware cycle estimate.
    """
    if mapping.cycles is None:
        return mapping
    scale = cycles_per_mac_window(arch)
    return replace(mapping, raw_cycles=mapping.cycles, cycles=mapping.cycles * scale)


# ---------------------------------------------------------------------------
# Bridge to the cycle-accurate simulator
# ---------------------------------------------------------------------------

def mapping_to_workload_tiles(mapping: MappingResult, arch: ArchConfig) -> dict[str, int]:
    """Convert a MappingResult into the tile sizes gemm_cycle_accurate_sim.cpp expects.

    The simulator's --binary-file workload format tiles the GEMM into
    N_TILES chunks of A_ROWS x W_COLS x K each (see the format doc at the top
    of run_workload_binary in gemm_cycle_accurate_sim.cpp), with A_ROWS =
    P_ROWS*N_H and W_COLS = P_COLS*N_W fixed by the array's geometry.

    Deliberately does NOT read mapping.per_level_factors for this: with the
    PE's spatial M/N and temporal K factors already pinned to the array's
    fixed geometry (see build_constraints_spec) and GlobalBuffer having no
    real capacity constraint, Timeloop is free to split the leftover M/N/K
    blocking arbitrarily between GlobalBuffer and DRAM's temporal loops --
    so a single level's factors aren't a reliable source for the total tile
    count. n_tiles/k_blocks are fully determined by problem/arch alone given
    the current constraints, so they're computed directly here instead.

    Requires M/N/K to divide evenly by A_ROWS/W_COLS/k_depth -- no padding
    support yet (raises ValueError otherwise).
    """
    a_rows = arch.p_rows * arch.n_h
    w_cols = arch.p_cols * arch.n_w
    if mapping.m % a_rows != 0:
        raise ValueError(
            f"M={mapping.m} must be a multiple of A_ROWS=P_ROWS*N_H={a_rows} "
            "(no padding support yet)"
        )
    if mapping.n % w_cols != 0:
        raise ValueError(
            f"N={mapping.n} must be a multiple of W_COLS=P_COLS*N_W={w_cols} "
            "(no padding support yet)"
        )
    if mapping.k % arch.k_depth != 0:
        raise ValueError(
            f"K={mapping.k} must be a multiple of k_depth={arch.k_depth} "
            "(no padding support yet)"
        )
    return {
        "n_tiles": (mapping.m // a_rows) * (mapping.n // w_cols),
        "k_blocks": mapping.k // arch.k_depth,
        "a_rows": a_rows,
        "w_cols": w_cols,
    }


def write_workload_binary(
    problem: GemmProblem,
    arch: ArchConfig,
    mapping: MappingResult,
    out_path: Path,
) -> None:
    """Pack a mapping-tiled, quantized GEMM instance into the --binary-file
    format gemm_cycle_accurate_sim.cpp's run_workload_binary expects:

        header (ASCII): "N_TILES=<n> N_CHUNKS=<c> A_ROWS=<r> W_COLS=<w> K=<k> K_BLOCKS=<kb>\\n"
        then, per (tile, chunk, k-block): uint16 A_bnd[A_ROWS][K], uint8 A_sgn[A_ROWS][K],
                                          uint16 W_bnd[W_COLS][K], uint8 W_sgn[W_COLS][K]

    (see the format doc at the top of run_workload_binary in the .cpp file).
    """
    # TODO: this needs (a) an operand source -- synthetic random tensors vs.
    # real activations/weights captured elsewhere in the repo (bench/
    # capture_real_tensors.py does this for other kernels) -- and (b) a Python
    # port of the .cpp file's quantize() (per-tensor abs-max scale, remap onto
    # the RNG grid) to turn operands into (boundary, sign) pairs. Wire both up
    # once mapping_to_workload_tiles() is implemented.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def find_optimal_mapping(
    problem: GemmProblem,
    arch: ArchConfig,
    work_dir: Path,
) -> MappingResult:
    config_paths = write_configs(problem, arch, work_dir)
    raw_result = run_mapper(config_paths, output_dir=work_dir)
    mapping = extract_mapping(problem, raw_result)
    return apply_hardware_timing(mapping, arch)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find an optimal GEMM mapping and emit a workload binary "
        "for gemm_cycle_accurate_sim.cpp's --binary-file mode."
    )
    p.add_argument("--M", type=int, default=1024, help="GEMM M dimension")
    p.add_argument("--K", type=int, default=1024, help="GEMM K dimension")
    p.add_argument("--N", type=int, default=1024, help="GEMM N dimension")
    p.add_argument("--p-rows", type=int, default=4, help="matches CFG_P_ROWS")
    p.add_argument("--p-cols", type=int, default=4, help="matches CFG_P_COLS")
    p.add_argument("--n-h", type=int, default=1, help="matches CFG_N_H")
    p.add_argument("--n-w", type=int, default=1, help="matches CFG_N_W")
    p.add_argument("--k-depth", type=int, default=4, help="matches CFG_K")
    p.add_argument("--m-parallel", type=int, default=1, help="matches CFG_M")
    p.add_argument("--mag-bits", type=int, default=7, help="matches CFG_MAG_BITS")
    p.add_argument("--work-dir", type=Path, default=Path("timeloop_workspace"),
                    help="where timeloop problem/arch/mapper YAML + outputs go")
    p.add_argument("--workload-out", type=Path, default=Path("workload.bin"),
                    help="path to write the binary workload for gemm_sim.exe --binary-file")
    p.add_argument("--emit-configs", action="store_true",
                    help="write the 4 timeloopfe-v4 YAML files to --work-dir and exit "
                         "(the container flow / `make mapping` runs the mapper on them)")
    p.add_argument("--no-glb", action="store_true",
                    help="drop the on-chip SRAM (simple-tiling baseline: stream from DRAM)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    problem = GemmProblem(M=args.M, K=args.K, N=args.N)
    arch = ArchConfig(
        p_rows=args.p_rows, p_cols=args.p_cols,
        n_h=args.n_h, n_w=args.n_w,
        k_depth=args.k_depth, m_parallel=args.m_parallel, mag_bits=args.mag_bits,
    )

    if args.emit_configs:
        paths = write_configs(problem, arch, args.work_dir, with_glb=not args.no_glb)
        print("Wrote timeloop configs:")
        for p in paths:
            print(f"  {p}")
        return

    mapping = find_optimal_mapping(problem, arch, args.work_dir)
    print(f"cycles={mapping.cycles} (raw timeloop cycles={mapping.raw_cycles}, "
          f"cycles/MAC-window={cycles_per_mac_window(arch)}) energy={mapping.energy} uJ")
    write_workload_binary(problem, arch, mapping, args.workload_out)
    print(f"Wrote workload binary to {args.workload_out}")


if __name__ == "__main__":
    main()

"""Accelergy estimator for the INNER SC compute datapath (SNG-free).

SKELETON -- structure is real, coefficients are PLACEHOLDERS to be replaced by
the single-PE PrimeTime run (which is SNG-free: the stochastic-number
generators live at the array perimeter, not inside the PE -- see array_3d.sv
rng_bank/sobol vs. sc_inner_pe_2bit_lane). Characterize `E_per_MAC` and
leakage from that run and drop them in.

This models ONLY the inner datapath of one PE: per-lane AND multiply -> k-wide
popcount -> signed sum-tree -> accumulator. It deliberately excludes SNG
energy, which is a separate `rng_bank` component/estimator (the OUTER split)
charged per operand stream, not per MAC. See find_mapping.build_arch_spec.

Per-MAC energy from a single-PE power run:
    E_per_MAC = P_avg * t_window / N_MACs
    N_MACs    = N_TILES * N_H * N_W * K
    t_window  = N_TILES * CYC_PER_TILE * T_clk
Timeloop counts logical MACs, so the length-T stream scaling lives inside this
per-`compute` energy (the cycle-side analogue of apply_hardware_timing()).
"""

from accelergy.plug_in_interface.estimator import (
    Estimator,
    actionDynamicEnergy,
    add_estimator_path,
    remove_estimator_path,
)

import os


# ---------------------------------------------------------------------------
# PLACEHOLDER coefficients -- REPLACE from the single-PE PrimeTime run.
# Per-bit dynamic energies in Joules at REF_TECH_NM (uncalibrated guesses so
# the plug-in runs). Once you have P_avg from PT, prefer setting E_COMPUTE_J
# directly (measured pJ/MAC) and ignore the per-bit breakdown.
# ---------------------------------------------------------------------------
REF_TECH_NM = 45.0
E_AND_BIT = 0.5e-15       # one AND-gate eval (one stream bit)                [J]
E_POPCOUNT_BIT = 1.5e-15  # k-wide popcount + sum-tree, per product bit       [J]
E_ACC_BIT = 2.0e-15       # one accumulator update, per bit, at 1b acc width  [J]
LEAK_POWER_W = 5.0e-6     # static leakage power of one PE                     [W]
AREA_PER_BIT_UM2 = 2.0    # crude area proxy                             [um^2]

# If set (non-None), used verbatim as the compute energy [J] -- this is where a
# measured pJ/MAC from PrimeTime goes: E_COMPUTE_J = <pJ/MAC> * 1e-12.
#
# MEASURED: 3.543 pJ/MAC (dynamic) from a back-annotated DC report_power on
# pe_single_int_2bit_lane_K32M8N4 (NanGate45, synth-level, 2.5ns, random-operand
# SAIF over 131072 MACs / 11.52us). SNG-free -- inner datapath only. Leakage was
# ~0.094 pJ/MAC-equivalent (see leak()). Re-characterize per config/tech; a
# layout-accurate PT-PX number needs an APR run (make power_apr).
E_COMPUTE_J = 3.543e-12


def _tech_nm(technology) -> float:
    if isinstance(technology, (int, float)):
        return float(technology)
    return float(str(technology).lower().replace("nm", "").strip())


class SCMacInner(Estimator):
    name = "sc_mac_inner"
    percent_accuracy_0_to_100 = 60

    def __init__(
        self,
        technology,
        mag_bits: int = 7,
        datawidth: int = 8,
        halve_bipolar_stoc_len: bool = True,
    ):
        self.mag_bits = int(mag_bits)
        self.datawidth = int(datawidth)
        self.tech_nm = _tech_nm(technology)
        self.halve_bipolar_stoc_len = bool(halve_bipolar_stoc_len)
        assert self.mag_bits >= 1, f"mag_bits {mag_bits} must be >= 1"

    def _stream_length(self) -> int:
        extra = 0 if self.halve_bipolar_stoc_len else 1
        return 1 << (self.mag_bits + extra)

    def _tech_scale(self) -> float:
        return self.tech_nm / REF_TECH_NM

    @actionDynamicEnergy
    def compute(self) -> float:
        """Energy of one logical inner MAC (length-T stream: AND+popcount+accum)."""
        if E_COMPUTE_J is not None:
            return E_COMPUTE_J
        T = self._stream_length()
        and_stream = T * E_AND_BIT
        popcount = T * E_POPCOUNT_BIT
        accumulate = T * E_ACC_BIT * self.datawidth
        energy = (and_stream + popcount + accumulate) * self._tech_scale()
        self.logger.info(f"sc_mac_inner.compute: T={T} -> {energy:.3e} J")
        return energy

    @actionDynamicEnergy
    def gated_compute(self) -> float:
        """Operand-gated compute: enable/clock overhead only (no stream)."""
        return E_POPCOUNT_BIT * self._tech_scale()

    def leak(self, global_cycle_seconds: float) -> float:
        return LEAK_POWER_W * global_cycle_seconds

    def get_area(self) -> float:
        area_um2 = AREA_PER_BIT_UM2 * (self.datawidth + self.mag_bits) * self._tech_scale()
        return area_um2 * 1e-12

    @staticmethod
    def quick_install_this_file():
        add_estimator_path(os.path.abspath(__file__))

    @staticmethod
    def quick_uninstall_this_file():
        remove_estimator_path(os.path.abspath(__file__))

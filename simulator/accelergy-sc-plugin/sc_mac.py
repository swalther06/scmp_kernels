"""Accelergy estimator for the INNER SC compute datapath (SNG-free).

Models PaYN's `InnerPE`/`InnerTile`: per-lane AND multiply -> k-wide popcount ->
signed sum-tree -> accumulator. It deliberately EXCLUDES stochastic-number
generation, which lives in two separate components -- `peripheral` (the
comparators, charged per operand read) and `sobol_bank` (the shared RNG, charged
per cycle). See find_mapping.build_arch_spec.

Action driver: `compute`, once per logical MAC. Timeloop counts logical MACs and
has no notion of a bitstream, so the length-L stream scaling lives INSIDE this
per-`compute` energy (the cycle-side analogue of apply_hardware_timing()).

Energy model (FOR NOW): a single measured per-MAC value at L_REF=128 on the
k8m16n8 PE, scaled LINEARLY with stream length -- E(L) = E_REF * L / L_REF. The
characterization gave four per-MAC buckets (core / conversion / Sobol / shared)
but NO dynamic-vs-leakage split, so all four are lumped into this per-`compute`
action (the peripheral and sobol_bank estimators are correspondingly zeroed to
avoid double-counting). See E_COMPUTE_REF_J below.
"""

from accelergy.plug_in_interface.estimator import (
    Estimator,
    actionDynamicEnergy,
    add_estimator_path,
    remove_estimator_path,
)

import os


REF_TECH_NM = 45.0
AREA_PER_BIT_UM2 = 2.0    # crude area proxy                             [um^2]

# --- Measured energy (k8m16n8 PE, PrimeTime) --------------------------------
# Reference stream length the buckets below were characterized at. Energy scales
# LINEARLY with L for now: E(L) = E_REF * L / L_REF (a first-order stand-in until
# per-L measurements exist).
L_REF = 128

# Per-MAC energy buckets at L_REF [Joules]. There was NO dynamic-vs-leakage
# split -- each is a TOTAL per-MAC value (dynamic + leakage lumped). Timeloop's
# only per-MAC action is `compute`, and per-MAC is exactly how these are
# normalized, so all four are summed into compute() and leak() charges nothing.
# The peripheral/sobol_bank estimators are zeroed (their share lives here) to
# avoid double-counting; the buckets stay named for provenance and re-splitting.
PJ = 1e-12
E_CORE_J   = 0.601948 * PJ   # inner compute datapath (popcount + sum-tree + accumulate)
E_CONV_J   = 0.047377 * PJ   # binary->stochastic conversion (peripheral comparators)
E_SOBOL_J  = 0.028485 * PJ   # shared Sobol RNG banks
E_SHARED_J = 0.003626 * PJ   # shared / top-level glue
E_COMPUTE_REF_J = E_CORE_J + E_CONV_J + E_SOBOL_J + E_SHARED_J  # 0.681436 pJ/MAC @ L_REF


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
        stream_length: int = 128,
    ):
        self.mag_bits = int(mag_bits)
        self.datawidth = int(datawidth)
        self.stream_length = int(stream_length)
        self.tech_nm = _tech_nm(technology)
        assert self.mag_bits >= 1, f"mag_bits {mag_bits} must be >= 1"

    def _tech_scale(self) -> float:
        return self.tech_nm / REF_TECH_NM

    @actionDynamicEnergy
    def compute(self) -> float:
        """Total per-MAC SC energy at this run's stream length L (linear in L).

        Carries all four measured buckets (core + conversion + Sobol + shared);
        see the module header for why they are lumped here rather than split
        across the peripheral/sobol_bank components.
        """
        return E_COMPUTE_REF_J * (self.stream_length / L_REF)

    def leak(self, global_cycle_seconds: float) -> float:
        # No separate leakage number was characterized: leakage is already folded
        # into the per-MAC compute buckets, so charge nothing per cycle here.
        return 0.0

    def get_area(self) -> float:
        area_um2 = AREA_PER_BIT_UM2 * (self.datawidth + self.mag_bits) * self._tech_scale()
        return area_um2 * 1e-12

    @staticmethod
    def quick_install_this_file():
        add_estimator_path(os.path.abspath(__file__))

    @staticmethod
    def quick_uninstall_this_file():
        remove_estimator_path(os.path.abspath(__file__))

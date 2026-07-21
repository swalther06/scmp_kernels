"""Accelergy estimator for the INNER SC compute datapath (SNG-free).

Models PaYN's `InnerPE`/`InnerTile`: per-lane AND multiply -> k-wide popcount ->
signed sum-tree -> accumulator. It deliberately EXCLUDES stochastic-number
generation, which lives in two separate components -- `peripheral` (the
comparators, charged per operand read) and `sobol_bank` (the shared RNG, charged
per cycle). See find_mapping.build_arch_spec.

Action driver: `compute`, once per logical MAC. Timeloop counts logical MACs and
has no notion of a bitstream, so the length-T stream scaling lives INSIDE this
per-`compute` energy (the cycle-side analogue of apply_hardware_timing()).

Energies are MEASURED ONLY -- there is no analytical fallback. Set E_COMPUTE_J
from the PrimeTime run:
    E_COMPUTE_J = (u_pe window energy) / (N_H * N_W * K MACs)
Until it is set, the estimator raises so an uncharacterized number can never
silently reach the ERT.
"""

from accelergy.plug_in_interface.estimator import (
    Estimator,
    actionDynamicEnergy,
    add_estimator_path,
    remove_estimator_path,
)

import os


REF_TECH_NM = 45.0
LEAK_POWER_W = None     # TODO 
AREA_PER_BIT_UM2 = 2.0    # crude area proxy                             [um^2]

# MEASURED, from PrimeTime: (u_pe energy) / (N_H*N_W*K MACs), in Joules.
#
# Current value: 3.543 pJ/MAC (dynamic), back-annotated DC report_power on
# pe_single_int_2bit_lane_K32M8N4 (ASTRAEA, NanGate45, synth-level, 2.5ns,
# random-operand SAIF over 131072 MACs / 11.52us). SNG-free, inner datapath only.
# NOTE: this is the ASTRAEA PE at 45nm -- re-characterize against PaYN's InnerPE
# (TSMC22, `SC_INNER_PE` / `u_pe` in power_payn_array) before trusting it.
E_COMPUTE_J = 3.543e-12


class EnergyNotCharacterized(RuntimeError):
    """Raised when a measured PrimeTime energy has not been filled in yet.

    Deliberate: this plug-in has no analytical fallback, so an uncharacterized
    number can never silently reach the ERT. Accelergy treats a raised error as
    "this estimator cannot estimate" and will report it if nothing else can.
    """


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
    ):
        self.mag_bits = int(mag_bits)
        self.datawidth = int(datawidth)
        self.tech_nm = _tech_nm(technology)
        assert self.mag_bits >= 1, f"mag_bits {mag_bits} must be >= 1"

    def _tech_scale(self) -> float:
        return self.tech_nm / REF_TECH_NM

    @actionDynamicEnergy
    def compute(self) -> float:
        """Energy of one logical inner MAC (length-T stream: AND+popcount+accum)."""
        if E_COMPUTE_J is None:
            raise EnergyNotCharacterized(
                "sc_mac_inner.compute energy is not characterized."
            )
        return E_COMPUTE_J

    def leak(self, global_cycle_seconds: float) -> float:
        if LEAK_POWER_W is None:
            raise EnergyNotCharacterized(
                "sc_mac_inner.leak energy is not characterized."
            )
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

"""Accelergy estimator for the INNER SC compute datapath (SNG-free).

Models PaYN's `InnerPE`/`InnerTile`: per-lane AND multiply -> k-wide popcount ->
signed sum-tree -> accumulator. It deliberately EXCLUDES stochastic-number
generation, which lives in two separate components -- `peripheral` (the
comparators, charged per operand read) and `sobol_bank` (the shared RNG, charged
per cycle). See find_mapping.build_arch_spec.

Action driver: `compute`, once per logical MAC. Timeloop counts logical MACs and
has no notion of a bitstream, so the length-T stream scaling lives INSIDE this
per-`compute` energy (the cycle-side analogue of apply_hardware_timing()).

Energies are MEASURED ONLY -- there is no analytical fallback, and one entry per
stream length L (no interpolation). From a power_payn_array PT run at each SC_T:
    E_COMPUTE_J_BY_L[L] = (u_pe dynamic energy) / (N_H * N_W * K MACs)
compute() picks the entry for the run's stream length; an unset L raises so an
uncharacterized number can never silently reach the ERT.
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

E_COMPUTE_J_BY_L = {
    16: None, 32: None, 48: None, 64: None, 96: None, 128: None, 192: None,
}
# PE leakage POWER [W] -- static, so L-independent (one value).
LEAK_POWER_W = None     # TODO


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
        """Energy of one logical inner MAC at this run's stream length L."""
        L = self.stream_length
        e = E_COMPUTE_J_BY_L.get(L)
        if e is None:
            raise EnergyNotCharacterized(
                f"sc_mac_inner.compute energy for stream length L={L} is not "
                f"characterized. Set E_COMPUTE_J_BY_L[{L}] in sc_mac.py from the "
                f"power_payn_array PT run at SC_T={L}."
            )
        return e

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

"""Accelergy estimator for the SC edge PERIPHERAL (binary -> stochastic streams).

Models PaYN's `sc_pe_peripheral`: the comparators (+ scramble XOR) that convert
a binary magnitude operand into an M-wide stochastic bitstream, comparing it
against the shared Sobol values each cycle. The Sobol RNG itself is a SEPARATE
component (sobol_bank.py) -- the peripheral only does the per-operand comparison.

Action driver: `read` fires once per operand STREAM CONVERTED (one operand
delivered into the array), NOT per MAC. Timeloop charges it on operand traffic
(Inputs/Weights reads), so it is mapping-dependent -- operand reuse (a[h,k] feeds
a whole output row, w[v,k] a whole column) means far fewer conversions than MACs,
and better temporal reuse lowers it further. This is the term that actually moves
the energy-optimal mapping.

Energies are MEASURED ONLY -- there is no analytical fallback. Set E_READ_J from
the power_payn_array PrimeTime run:
    E_READ_J = (u_peripheral window energy) / (operand-reads in window)
             = (u_peripheral window energy) / ((N_H + N_W) * K)
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
LEAK_POWER_W = None    # TODO
AREA_PER_LANE_UM2 = 4.0   # rough estimate, maybe useful for area comarison [um^2]

# MEASURED, from PrimeTime: (u_peripheral energy) / ((N_H+N_W)*K operand-reads).
E_READ_J = None  # TODO: fill in from the power_payn_array PT run


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


class Peripheral(Estimator):
    name = "peripheral"
    percent_accuracy_0_to_100 = 60

    def __init__(
        self,
        technology,
        width: int = 8,           # operand magnitude width (WIDTH)
        stream_length: int = 128, # T: cycles one operand is streamed
        M: int = 16,              # parallel stochastic lanes per operand
    ):
        self.width = int(width)
        self.stream_length = int(stream_length)
        self.M = int(M)
        self.tech_nm = _tech_nm(technology)
        assert self.stream_length >= 1 and self.M >= 1

    def _tech_scale(self) -> float:
        return self.tech_nm / REF_TECH_NM

    @actionDynamicEnergy
    def read(self) -> float:
        """Energy to convert one operand into its stochastic stream."""
        if E_READ_J is None:
            raise EnergyNotCharacterized(
                "peripheral.read energy is not characterized."
            )
        return E_READ_J

    # Alias in case Timeloop bills operand delivery through this level as `write`.
    @actionDynamicEnergy
    def write(self) -> float:
        return self.read()

    # Required storage action; the peripheral keeps read-only operands (no
    # accumulation), so update never actually fires -- define it for the ERT.
    @actionDynamicEnergy
    def update(self) -> float:
        return self.read()

    def leak(self, global_cycle_seconds: float) -> float:
        if LEAK_POWER_W is None:
            raise EnergyNotCharacterized(
                "peripheral.leak energy is not characterized."
            )
        return LEAK_POWER_W * global_cycle_seconds
        

    def get_area(self) -> float:
        area_um2 = AREA_PER_LANE_UM2 * self.M * self._tech_scale()
        return area_um2 * 1e-12

    @staticmethod
    def quick_install_this_file():
        add_estimator_path(os.path.abspath(__file__))

    @staticmethod
    def quick_uninstall_this_file():
        remove_estimator_path(os.path.abspath(__file__))

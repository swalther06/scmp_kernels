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

Energies are MEASURED ONLY -- one entry per stream length L (no interpolation).
From a power_payn_array PT run at each SC_T:
    E_READ_J_BY_L[L] = (u_peripheral dynamic energy) / ((N_H + N_W) * K operands)
read() picks the entry for the run's stream length (via the SC_STREAM_LENGTH env
var, since a storage component can't take a stream_length attribute); an unset L
raises so an uncharacterized number can never silently reach the ERT.
"""

from accelergy.plug_in_interface.estimator import (
    Estimator,
    actionDynamicEnergy,
    add_estimator_path,
    remove_estimator_path,
)

import os


REF_TECH_NM = 45.0
AREA_PER_LANE_UM2 = 4.0   # rough estimate, maybe useful for area comarison [um^2]

E_READ_J_BY_L = {
    16: None, 32: None, 48: None, 64: None, 96: None, 128: None, 192: None,
}
# peripheral leakage POWER [W] -- static, so L-independent (one value).
LEAK_POWER_W = None    # TODO


def _current_L() -> int:
    """The run's SC stream length. The peripheral is a *storage* component, so it
    can't take a `stream_length` attribute (strict StorageAttributes); the driver
    passes it via the SC_STREAM_LENGTH env var (make mapping STREAM_LEN=...)."""
    return int(os.environ.get("SC_STREAM_LENGTH", "128"))


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
        """Energy to convert one operand into its stochastic stream at length L."""
        L = _current_L()
        e = E_READ_J_BY_L.get(L)
        if e is None:
            raise EnergyNotCharacterized(
                f"peripheral.read energy for stream length L={L} is not "
                f"characterized. Set E_READ_J_BY_L[{L}] in peripheral.py from the "
                f"power_payn_array PT run at SC_T={L}."
            )
        return e

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

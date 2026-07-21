"""Accelergy estimator for the shared Sobol RNG BANKS.

Models PaYN's two `sobol_bank`s (A and W): M parallel `sobol_generator`s each,
advancing once per stochastic cycle and BROADCASTING their values to every
peripheral in the array. Because one advance serves the whole array that cycle,
the banks are a fixed, ARRAY-SHARED, PER-CYCLE cost -- independent of operand
count, MAC count, and array size (which is exactly why sharing them is the win).

Action driver: the banks are always-on per cycle and carry no operand data, so
there is no per-read or per-MAC action for them. Their dynamic per-cycle energy
is carried in `leak` -- mechanically leak = energy/cycle/instance, which is the
banks' profile. Model them as ONE shared component (instances: 1). read/write
are ~0 (they are not a level in the operand dataflow).

Energies are MEASURED ONLY -- there is no analytical fallback. Set
E_LEAK_PER_CYCLE_J from the power_payn_array PrimeTime run:
    E_LEAK_PER_CYCLE_J = (u_a_rng + u_w_rng window energy) / (window cycles)
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
N_BANKS = 2               # A-bank + W-bank
AREA_PER_BIT_UM2 = 6.0    # rough area estimate per generator bit, maybe useful later [um^2]

# MEASURED, from PrimeTime: (u_a_rng + u_w_rng energy) / (window cycles).
E_LEAK_PER_CYCLE_J = None  # TODO: fill in from the power_payn_array PT run


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


class SobolBank(Estimator):
    name = "sobol_bank"
    percent_accuracy_0_to_100 = 60

    def __init__(
        self,
        technology,
        width: int = 8,   # WIDTH per generator
        M: int = 16,      # generators per bank
    ):
        self.width = int(width)
        self.M = int(M)
        self.tech_nm = _tech_nm(technology)
        assert self.width >= 1 and self.M >= 1

    def _tech_scale(self) -> float:
        return self.tech_nm / REF_TECH_NM

    def leak(self, global_cycle_seconds: float) -> float:
        """Per-cycle dynamic energy of the always-on shared banks (see docstring)."""
        if E_LEAK_PER_CYCLE_J is None:
            raise EnergyNotCharacterized(
                "sobol_bank.leak energy is not characterized."
            )
        return E_LEAK_PER_CYCLE_J

    # The banks are not a data level -- read/write/update cost nothing here
    # (all energy is in leak). Defined because Timeloop storage requires them.
    @actionDynamicEnergy
    def read(self) -> float:
        return 0.0

    @actionDynamicEnergy
    def write(self) -> float:
        return 0.0

    @actionDynamicEnergy
    def update(self) -> float:
        return 0.0

    def get_area(self) -> float:
        area_um2 = AREA_PER_BIT_UM2 * N_BANKS * self.M * self.width * self._tech_scale()
        return area_um2 * 1e-12

    @staticmethod
    def quick_install_this_file():
        add_estimator_path(os.path.abspath(__file__))

    @staticmethod
    def quick_uninstall_this_file():
        remove_estimator_path(os.path.abspath(__file__))

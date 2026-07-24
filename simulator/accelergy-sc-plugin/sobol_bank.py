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

FOR NOW this estimator is ZEROED: the characterization gave the Sobol cost only
as a per-MAC bucket (0.028485 pJ/MAC @ L=128), not a per-cycle number, so it is
folded into sc_mac_inner.compute (see sc_mac.py header) rather than billed as
leak here. Splitting it back out needs a per-cycle (dynamic/leak) measurement --
until then leak returns 0 to avoid double-counting the value already in compute.
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

# The Sobol RNG energy (0.028485 pJ/MAC @ L=128) is currently folded into
# sc_mac_inner.compute (see sc_mac.py header): the characterization gave a
# per-MAC bucket, not a per-cycle number, so it can't be billed as leak here
# without a cycles/MAC ratio. Zeroed to avoid double-counting for now.
E_LEAK_PER_CYCLE_J = 0.0


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
        """Per-cycle energy of the shared banks -- folded into sc_mac_inner.compute
        for now (see module header), so zero here to avoid double-counting."""
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

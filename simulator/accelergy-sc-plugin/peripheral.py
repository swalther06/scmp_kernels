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

FOR NOW this estimator is ZEROED: the characterization gave the conversion cost
only as a per-MAC bucket (0.047377 pJ/MAC @ L=128), not a per-operand-read
number, so it is folded into sc_mac_inner.compute (see sc_mac.py header) rather
than billed on this level's `read`. Splitting it back out here needs a per-read
(reuse-aware) measurement -- until then, read/leak return 0 to avoid double-
counting the value that already lives in compute.
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

# The conversion energy (0.047377 pJ/MAC @ L=128) is currently folded into
# sc_mac_inner.compute (see sc_mac.py header): the characterization gave only a
# per-MAC bucket, not a per-operand-read number, so it can't be billed on this
# storage level's `read` without a reuse ratio. Zeroed here to avoid double-
# counting until a real per-read (and dynamic/leak) split exists.
E_READ_J = 0.0


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
        """Per-operand conversion energy -- folded into sc_mac_inner.compute for
        now (see module header), so zero here to avoid double-counting."""
        return E_READ_J

    # Aliases: whichever action Timeloop bills operand delivery through, it's the
    # same (zeroed) conversion energy.
    @actionDynamicEnergy
    def write(self) -> float:
        return self.read()

    @actionDynamicEnergy
    def update(self) -> float:
        return self.read()

    def leak(self, global_cycle_seconds: float) -> float:
        # Leakage is lumped into the per-MAC compute buckets; nothing per cycle.
        return 0.0


    def get_area(self) -> float:
        area_um2 = AREA_PER_LANE_UM2 * self.M * self._tech_scale()
        return area_um2 * 1e-12

    @staticmethod
    def quick_install_this_file():
        add_estimator_path(os.path.abspath(__file__))

    @staticmethod
    def quick_uninstall_this_file():
        remove_estimator_path(os.path.abspath(__file__))

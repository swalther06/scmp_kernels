"""Accelergy estimator for the OUTER stochastic-number generation (SNG).

SKELETON -- coefficients are PLACEHOLDERS to be replaced by the outer-PE
PrimeTime run (the edge PE that includes rng_bank/sobol + cmp_lane). This is
the perimeter half of the inner/outer split: one `rng_bank` component with
`instances = P_H + P_W - 1` sits between GlobalBuffer and the PE array (see
find_mapping.build_arch_spec).

Action driver: `read` fires once per operand stream converted (an operand
delivered into the array), NOT per MAC -- so Timeloop charges SNG energy on
operand traffic, which is mapping-dependent (reuse/stationarity change how many
operands cross the boundary). One conversion streams a length-T bitstream, M
bits/cycle, via a Sobol/LFSR + comparator.

Per-conversion energy from the outer-PE power run:
    E_per_conversion = P_sng_avg * t_window / N_conversions
where N_conversions = operands streamed into the array during the SAIF window.
Set E_GEN_J directly from that measurement.
"""

from accelergy.plug_in_interface.estimator import (
    Estimator,
    actionDynamicEnergy,
    add_estimator_path,
    remove_estimator_path,
)

import os


REF_TECH_NM = 45.0
E_SOBOL_BIT = 1.0e-15     # one Sobol/LFSR RNG step, per stream bit           [J]
E_CMP_BIT = 0.5e-15       # one comparator eval, per stream bit               [J]
LEAK_POWER_W = 3.0e-6     # static leakage of one SNG unit                    [W]
AREA_PER_BIT_UM2 = 3.0    # crude area proxy                             [um^2]

# Energy [J] per operand stream generated. When set, used verbatim.
#
# PLACEHOLDER (user directive): "outer PE = 1.5x inner PE energy", pending a
# real outer-PE PrimeTime run. In the two-component arch, MACC.compute already
# charges the inner datapath on EVERY PE (inner + perimeter), so this SNG
# component charges only the DELTA that makes a perimeter PE 1.5x an inner one:
#     SNG overhead = (1.5 - 1.0) * E_inner = 0.5 * 3.543 pJ/MAC = 1.772 pJ
# where 3.543 pJ/MAC is the measured inner compute (see sc_mac.E_COMPUTE_J).
# Caveat: this level's action is per operand *conversion*; setting it to the
# per-MAC delta assumes ~1 conversion per MAC (an upper bound -- operand reuse
# makes real SNG traffic lower). Replace with the measured outer-PE number.
E_INNER_J = 3.543e-12          # measured inner compute (sc_mac.E_COMPUTE_J)
OUTER_RATIO = 1.5              # outer PE energy / inner PE energy (placeholder)
E_GEN_J = (OUTER_RATIO - 1.0) * E_INNER_J


def _tech_nm(technology) -> float:
    if isinstance(technology, (int, float)):
        return float(technology)
    return float(str(technology).lower().replace("nm", "").strip())


class RngBank(Estimator):
    name = "rng_bank"
    percent_accuracy_0_to_100 = 60

    def __init__(
        self,
        technology,
        width: int = 8,
        stream_length: int = 128,
    ):
        self.width = int(width)
        self.stream_length = int(stream_length)
        self.tech_nm = _tech_nm(technology)
        assert self.stream_length >= 1

    def _tech_scale(self) -> float:
        return self.tech_nm / REF_TECH_NM

    @actionDynamicEnergy
    def read(self) -> float:
        """Energy to convert one operand into a length-T stochastic bitstream."""
        if E_GEN_J is not None:
            return E_GEN_J
        T = self.stream_length
        energy = T * (E_SOBOL_BIT + E_CMP_BIT) * self._tech_scale()
        self.logger.info(f"rng_bank.read: T={T} width={self.width} -> {energy:.3e} J")
        return energy

    # Alias: if Timeloop charges this level's operand delivery as `write`.
    @actionDynamicEnergy
    def write(self) -> float:
        return self.read()

    def leak(self, global_cycle_seconds: float) -> float:
        return LEAK_POWER_W * global_cycle_seconds

    def get_area(self) -> float:
        area_um2 = AREA_PER_BIT_UM2 * self.width * self._tech_scale()
        return area_um2 * 1e-12

    @staticmethod
    def quick_install_this_file():
        add_estimator_path(os.path.abspath(__file__))

    @staticmethod
    def quick_uninstall_this_file():
        remove_estimator_path(os.path.abspath(__file__))

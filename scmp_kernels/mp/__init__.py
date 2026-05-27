"""Mixed-precision dispatch + config.

Re-exports the public API from `scmp_kernels.mp.config`.
"""

from .config import (
    MPConfig,
    AdaptiveMPConfig,
    FreeBoundaryMPConfig,
    RangeMPConfig,
    RowAssignment,
    classify_rows_by_metric,
    adaptive_classify_rows,
    classify_groups_by_range,
    MPDistributionLogger,
    AutoMPBudgetLogger,
    MetricProfiler,
    set_current_block_idx,
    get_current_block_idx,
)
from .auto_calibrator import (
    auto_calibrate_mp,
    RidgeFitter,
    oracle_search_op,
    oracle_search_block,
)

__all__ = [
    "MPConfig",
    "AdaptiveMPConfig",
    "FreeBoundaryMPConfig",
    "RangeMPConfig",
    "RowAssignment",
    "classify_rows_by_metric",
    "adaptive_classify_rows",
    "classify_groups_by_range",
    "MPDistributionLogger",
    "AutoMPBudgetLogger",
    "MetricProfiler",
    "set_current_block_idx",
    "get_current_block_idx",
    "auto_calibrate_mp",
    "RidgeFitter",
    "oracle_search_op",
    "oracle_search_block",
]

"""
Mixed Precision configuration for per-token-row SC.

Each token row gets assigned a stoc_len level based on its importance metric.
Rows with higher importance use longer stoc_len (higher precision), while
less important rows use shorter stoc_len for faster computation.

Includes:
- MPConfig: Fixed-fraction quantile-based assignment (original).
- AdaptiveMPConfig: Timestep-adaptive thresholds with per-operator and
  per-layer control, inspired by HPCA APT's APDT algorithm.
- FreeBoundaryMPConfig: Zero-hyperparameter per-(block, op) free boundaries
  (k-1 for k levels), filled in by an offline oracle search.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


# ---------------------------------------------------------------------
# Per-block context: classifiers that index by (block, op) read this
# global; the auto-calibrator and runtime pre-hooks set it per forward.
# ---------------------------------------------------------------------
_CURRENT_BLOCK_IDX: int = 0


def set_current_block_idx(i: int) -> None:
    global _CURRENT_BLOCK_IDX
    _CURRENT_BLOCK_IDX = int(i)


def get_current_block_idx() -> int:
    return _CURRENT_BLOCK_IDX


@dataclass
class MPConfig:
    """Configuration for per-token-row mixed precision SC."""
    stoc_len_levels: list[int]                  # e.g. [256, 128, 64, 32], sorted descending
    level_fractions: Optional[list[float]] = None  # e.g. [0.25, 0.25, 0.25, 0.25]; None = equal
    qk_metric: str = "q_row_max"               # "q_row_max" (||Q_row||_inf)
    av_metric: str = "attn_row_max"             # "attn_row_max" (max of attn row)
    mlp_metric: str = "x_row_max"              # "x_row_max" (||x_row||_inf)

    def __post_init__(self):
        if self.level_fractions is None:
            n = len(self.stoc_len_levels)
            self.level_fractions = [1.0 / n] * n
        if len(self.level_fractions) != len(self.stoc_len_levels):
            raise ValueError(
                f"level_fractions length ({len(self.level_fractions)}) must match "
                f"stoc_len_levels length ({len(self.stoc_len_levels)})"
            )
        if abs(sum(self.level_fractions) - 1.0) >= 1e-6:
            raise ValueError(
                f"level_fractions must sum to 1.0, got {sum(self.level_fractions)}"
            )


@dataclass
class RowAssignment:
    """Per-head row-to-level assignment for one (batch, head) pair."""
    row_levels: torch.Tensor                        # [N] int, index into stoc_len_levels
    level_row_indices: dict[int, torch.Tensor]       # stoc_len -> LongTensor of row indices


def _bucket_index(value: int, total: int, num_buckets: int) -> int:
    """Map an absolute timestep / block index to a calibration bucket."""
    if num_buckets <= 1 or total <= 1:
        return 0
    ratio = value / max(total - 1, 1)
    return min(num_buckets - 1, int(ratio * num_buckets))


def _parse_bucket_key(bucket_key: str) -> tuple[str, int, int]:
    """Parse calibration keys like 'proj:t3:l1'."""
    try:
        operator, t_part, l_part = bucket_key.split(":")
        if not t_part.startswith("t") or not l_part.startswith("l"):
            raise ValueError
        return operator, int(t_part[1:]), int(l_part[1:])
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise ValueError(
            f"Invalid adaptive MP bucket key '{bucket_key}'. "
            "Expected format '<operator>:t<int>:l<int>'."
        ) from exc


def _extract_thresholds(payload, n_levels: int, source: str) -> list[float]:
    """Extract a threshold list of length n_levels-1 from a table payload."""
    raw_thresholds = payload.get("thresholds") if isinstance(payload, dict) else payload
    if raw_thresholds is None:
        raise ValueError(f"Missing 'thresholds' in adaptive MP payload for {source}.")
    thresholds = [float(x) for x in raw_thresholds]
    expected = max(n_levels - 1, 0)
    if len(thresholds) != expected:
        raise ValueError(
            f"Adaptive MP thresholds for {source} have length {len(thresholds)}, "
            f"expected {expected} for {n_levels} stoc_len levels."
        )
    for idx, threshold in enumerate(thresholds):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Adaptive MP threshold {threshold} for {source} is outside [0, 1]."
            )
        if idx > 0 and threshold > thresholds[idx - 1] + 1e-6:
            raise ValueError(
                f"Adaptive MP thresholds for {source} must be non-increasing, "
                f"got {thresholds}."
            )
    return thresholds


def _classify_rows_by_thresholds(
    metric_norm: torch.Tensor,
    stoc_len_levels: list[int],
    thresholds: list[float],
) -> RowAssignment:
    """Assign levels from explicit non-uniform thresholds."""
    n_levels = len(stoc_len_levels)
    expected = max(n_levels - 1, 0)
    if len(thresholds) != expected:
        raise ValueError(
            f"Expected {expected} thresholds for {n_levels} levels, got {len(thresholds)}."
        )

    row_levels = torch.full(
        (metric_norm.shape[0],),
        n_levels - 1,
        dtype=torch.long,
        device=metric_norm.device,
    )
    for level_idx, threshold in enumerate(thresholds):
        lower = metric_norm.new_tensor(threshold)
        if level_idx == 0:
            mask = metric_norm >= lower
        else:
            upper = metric_norm.new_tensor(thresholds[level_idx - 1])
            mask = (metric_norm >= lower) & (metric_norm < upper)
        row_levels[mask] = level_idx

    level_row_indices: dict[int, torch.Tensor] = {}
    for level_idx, stoc_len in enumerate(stoc_len_levels):
        level_row_indices[stoc_len] = torch.where(row_levels == level_idx)[0]

    return RowAssignment(row_levels=row_levels, level_row_indices=level_row_indices)


def classify_rows_by_metric(
    metric: torch.Tensor,
    stoc_len_levels: list[int],
    level_fractions: list[float],
) -> RowAssignment:
    """
    Rank rows by metric, bucket into levels by quantile fractions.

    Top fraction[0] rows -> levels[0] (highest stoc_len)
    Next fraction[1] rows -> levels[1]
    ...

    Args:
        metric: [N] importance values per row
        stoc_len_levels: sorted descending list of stoc_len values
        level_fractions: fraction of rows per level
    """
    N = metric.shape[0]
    sorted_indices = metric.argsort(descending=True)

    row_levels = torch.empty(N, dtype=torch.long, device=metric.device)
    level_row_indices = {}
    offset = 0
    for i, (sl, frac) in enumerate(zip(stoc_len_levels, level_fractions)):
        if i < len(stoc_len_levels) - 1:
            count = round(frac * N)
        else:
            count = N - offset
        # Clamp to remaining rows: cumulative round() on small N + many
        # levels can otherwise drive the final count negative, or have
        # earlier levels overrun N and leave later levels with no rows.
        count = max(0, min(count, N - offset))
        rows = sorted_indices[offset:offset + count]
        row_levels[rows] = i
        level_row_indices[sl] = rows
        offset += count

    return RowAssignment(row_levels=row_levels, level_row_indices=level_row_indices)


# =====================================================================
# Adaptive Mixed Precision (inspired by HPCA APT APDT)
# =====================================================================

@dataclass
class AdaptiveMPConfig:
    """Timestep-adaptive mixed precision with true thresholds and per-operator
    parameters.

    Uses absolute thresholds on normalized metric values instead of fixed
    fractions.  The number of rows per level adapts to the actual data
    distribution.

    Threshold: base_threshold(t) = α · progress(t) + β
    where progress(t) = t / (T-1)  ∈ [0, 1] (high at noisy, low at clean).

    Rows with high normalized metric → high stoc_len (precise).
    Rows with low normalized metric → low stoc_len or pruned.

    Args:
        stoc_len_levels: Descending list of stoc_len values.
            Use 0 as the last level to enable pruning (skip).
        alpha: Global default sensitivity to timestep progress.
        beta: Global default base threshold offset.
        enable_pruning: Allow stoc_len=0 (skip) level.
        operator_params: Per-operator (alpha, beta) overrides.
            Keys: "qk", "av", "mlp_fc1", "mlp_fc2", "input_proj", "proj".
    """
    stoc_len_levels: list[int]
    alpha: float = 0.3
    beta: float = 0.05
    enable_pruning: bool = True
    operator_params: dict[str, tuple[float, float]] = field(default_factory=dict)
    threshold_table_path: Optional[str] = None
    timestep_buckets: int = 1
    layer_buckets: int = 1
    operator_default_thresholds: dict[str, list[float]] = field(default_factory=dict)
    bucket_thresholds: dict[tuple[str, int, int], list[float]] = field(default_factory=dict)
    # When set, bypass the linear-threshold classifier and use these fractions
    # as quantile targets per level (top frac[0] rows -> levels[0], etc.).
    # Length must match stoc_len_levels; sums to 1.
    target_fractions: Optional[list[float]] = None

    def __post_init__(self):
        assert len(self.stoc_len_levels) >= 2, (
            "Need at least 2 levels (high + low or high + skip)")
        for i in range(len(self.stoc_len_levels) - 1):
            assert self.stoc_len_levels[i] > self.stoc_len_levels[i + 1], (
                f"stoc_len_levels must be sorted descending, "
                f"got {self.stoc_len_levels}")
        if not self.enable_pruning and 0 in self.stoc_len_levels:
            self.stoc_len_levels = [s for s in self.stoc_len_levels if s > 0]
        if self.threshold_table_path:
            self.load_threshold_table(self.threshold_table_path)
        if self.target_fractions is not None:
            assert len(self.target_fractions) == len(self.stoc_len_levels), (
                f"target_fractions length {len(self.target_fractions)} "
                f"must match stoc_len_levels length {len(self.stoc_len_levels)}")
            s = sum(self.target_fractions)
            assert abs(s - 1.0) < 1e-6, (
                f"target_fractions must sum to 1.0, got {s}")

    def get_params(self, operator: Optional[str] = None) -> tuple[float, float]:
        """Get (alpha, beta) for an operator, falling back to global."""
        if operator and operator in self.operator_params:
            return self.operator_params[operator]
        return (self.alpha, self.beta)

    def load_threshold_table(self, path: str):
        """Load calibrated thresholds exported by calibrate_mp_thresholds.py."""
        table_path = Path(path)
        with open(table_path) as f:
            payload = json.load(f)

        table_levels = [int(x) for x in payload["stoc_len_levels"]]
        if table_levels != self.stoc_len_levels:
            raise ValueError(
                f"Adaptive MP table levels {table_levels} do not match runtime "
                f"levels {self.stoc_len_levels}."
            )

        self.timestep_buckets = int(payload.get("timestep_buckets", 1))
        self.layer_buckets = int(payload.get("layer_buckets", 1))
        self.operator_default_thresholds = {}
        self.bucket_thresholds = {}

        for operator, operator_payload in payload.get("operator_defaults", {}).items():
            self.operator_default_thresholds[operator] = _extract_thresholds(
                operator_payload,
                len(self.stoc_len_levels),
                f"operator_default:{operator}",
            )

        for bucket_key, bucket_payload in payload.get("buckets", {}).items():
            operator, t_bucket, l_bucket = _parse_bucket_key(bucket_key)
            self.bucket_thresholds[(operator, t_bucket, l_bucket)] = _extract_thresholds(
                bucket_payload,
                len(self.stoc_len_levels),
                bucket_key,
            )

    def get_thresholds(
        self,
        timestep: int,
        total_timesteps: int,
        operator: Optional[str] = None,
        block_idx: Optional[int] = None,
        total_blocks: Optional[int] = None,
    ) -> Optional[list[float]]:
        """Get calibrated thresholds for one operator/timestep/block bucket."""
        if self.bucket_thresholds and operator and block_idx is not None and total_blocks is not None:
            t_bucket = _bucket_index(timestep, total_timesteps, self.timestep_buckets)
            l_bucket = _bucket_index(block_idx, total_blocks, self.layer_buckets)
            thresholds = self.bucket_thresholds.get((operator, t_bucket, l_bucket))
            if thresholds is not None:
                return thresholds
        if operator and operator in self.operator_default_thresholds:
            return self.operator_default_thresholds[operator]
        return None


def adaptive_classify_rows(
    metric: torch.Tensor,
    timestep: int,
    total_timesteps: int,
    config: AdaptiveMPConfig,
    operator: Optional[str] = None,
    block_idx: Optional[int] = None,
    total_blocks: Optional[int] = None,
) -> RowAssignment:
    """Classify rows using true absolute thresholds on normalized metrics.

    Unlike quantile-based classification, the number of rows per level
    adapts to the actual metric distribution.  Per-operator α/β allows
    different aggressiveness for different operators.

    Args:
        metric: [N] per-row importance values (e.g. row abs-max).
        timestep: Current diffusion timestep (T-1 = noisiest, 0 = cleanest).
        total_timesteps: Total number of diffusion timesteps T.
        config: AdaptiveMPConfig instance.
        operator: Operator name for per-operator α/β lookup.

    Returns:
        RowAssignment compatible with existing code.
    """
    N = metric.shape[0]
    levels = config.stoc_len_levels
    n_levels = len(levels)

    # ---------- Free-boundary path (FreeBoundaryMPConfig) ----------
    # Per-(block, op) learned boundaries; no alpha/beta/progress dependency.
    # Check subclass first so inherited isinstance(cfg, AdaptiveMPConfig)
    # dispatch still works elsewhere while we dispatch correctly here.
    if isinstance(config, FreeBoundaryMPConfig):
        fixed_level = config.get_fixed_level(operator or "")
        if fixed_level is not None:
            return _classify_all_rows_to_level(metric, levels, fixed_level)
        boundaries = config.get_boundaries(operator or "")
        return _classify_with_free_boundaries(metric, boundaries, levels)

    # ---------- Quantile path (target_fractions set) ----------
    # Independent of (t, T) / alpha / beta. Top frac[0] rows -> levels[0], etc.
    if config.target_fractions is not None:
        sorted_idx = metric.argsort(descending=True)
        row_levels_q = torch.empty(N, dtype=torch.long, device=metric.device)
        level_row_indices_q: dict[int, torch.Tensor] = {}
        offset = 0
        for i, (sl, frac) in enumerate(zip(levels, config.target_fractions)):
            if i < n_levels - 1:
                count = round(frac * N)
            else:
                count = N - offset
            rows_q = sorted_idx[offset:offset + count]
            row_levels_q[rows_q] = i
            level_row_indices_q[sl] = rows_q
            offset += count
        return RowAssignment(row_levels=row_levels_q,
                             level_row_indices=level_row_indices_q)

    # progress: 1 at noisiest (t=T-1), 0 at cleanest (t=0)
    # Early (noisy) steps → high progress → high base_threshold → aggressive
    # Late (clean) steps → low progress → low base_threshold → conservative
    progress = timestep / max(total_timesteps - 1, 1)

    # Per-operator α/β
    alpha, beta = config.get_params(operator)

    # Base threshold: the cutoff on normalized metric [0,1].
    # Rows with metric_norm >= base_threshold → level 0 (highest precision).
    # Rows with metric_norm < base_threshold → split among lower levels.
    # Higher base_threshold = more rows get lower precision.
    # base_threshold=0.95 (very aggressive) → only top 5% get level 0.
    base_threshold = alpha * progress + beta
    base_threshold = min(base_threshold, 0.95)

    # Normalize metric to [0, 1]
    m_min = metric.min()
    m_max = metric.max()
    if (m_max - m_min).item() < 1e-8:
        # All metric values are equal — no meaningful ranking.
        # Default to highest precision (all rows at level 0).
        row_levels = torch.zeros(N, dtype=torch.long, device=metric.device)
        level_row_indices = {}
        for idx, sl in enumerate(levels):
            if idx == 0:
                level_row_indices[sl] = torch.arange(N, device=metric.device)
            else:
                level_row_indices[sl] = torch.empty(0, dtype=torch.long,
                                                     device=metric.device)
        return RowAssignment(row_levels=row_levels,
                             level_row_indices=level_row_indices)
    metric_norm = (metric - m_min) / (m_max - m_min)

    calibrated_thresholds = config.get_thresholds(
        timestep=timestep,
        total_timesteps=total_timesteps,
        operator=operator,
        block_idx=block_idx,
        total_blocks=total_blocks,
    )
    if calibrated_thresholds is not None:
        return _classify_rows_by_thresholds(metric_norm, levels, calibrated_thresholds)

    # Split [0, base_threshold] evenly among non-highest levels.
    # For 3 levels [256, 64, 0] with base_threshold=0.3:
    #   metric_norm >= 0.3  → level 0 (sl=256)
    #   0.15 <= metric_norm < 0.3  → level 1 (sl=64)
    #   metric_norm < 0.15  → level 2 (sl=0, pruned)
    #
    # N-1 boundaries from high to low:
    #   boundaries[0] = base_threshold  (between level 0 and level 1)
    #   boundaries[k] = base_threshold * (n_levels - 1 - k) / (n_levels - 1)
    row_levels = torch.zeros(N, dtype=torch.long, device=metric.device)  # default: level 0

    boundaries = []
    for k in range(n_levels - 1):
        # boundary[0] = base_threshold (highest, between level 0 and 1)
        # boundary[n-2] = base_threshold / (n-1) (lowest, between level n-2 and n-1)
        b = base_threshold * (n_levels - 1 - k) / (n_levels - 1)
        boundaries.append(b)

    # Assign from lowest precision upward:
    # Everything starts at level 0 (highest precision).
    # Then demote rows below each boundary.
    for k in range(n_levels - 1):
        # Rows with metric_norm < boundaries[k] get demoted to level k+1 or lower
        row_levels[metric_norm < boundaries[k]] = k + 1

    # Build level_row_indices
    level_row_indices: dict[int, torch.Tensor] = {}
    for i, sl in enumerate(levels):
        level_row_indices[sl] = torch.where(row_levels == i)[0]

    return RowAssignment(row_levels=row_levels, level_row_indices=level_row_indices)


# =====================================================================
# Free-boundary MP (zero hyperparameter; offline oracle-search populated)
# =====================================================================

@dataclass
class FreeBoundaryMPConfig(AdaptiveMPConfig):
    """Per-(block, op) k-1 free boundaries on normalized metric in [0, 1].

    Subclasses ``AdaptiveMPConfig`` so existing ``isinstance(cfg,
    AdaptiveMPConfig)`` dispatch in the SC attention patch continues to
    fire. The inherited ``alpha`` / ``beta`` / ``target_fractions`` fields
    are ignored when the classifier takes the free-boundary branch.

    Boundaries are keyed by ``(block_idx, op_name)``; block_idx is read
    from the module-level ``_CURRENT_BLOCK_IDX`` at classification time
    (set by forward pre-hooks installed by the auto-calibrator).

    Missing entries fall back to ``default_boundaries`` (equal spacing).
    Callers may also pin an op to a fixed level index via ``fixed_levels``;
    this is useful when some ops should stay coarse/static while others are
    searched by auto-MP.
    """
    # {(block_idx, op_name): tensor of k-1 boundaries, descending in (0, 1)}
    boundaries: dict = field(default_factory=dict)
    # {(block_idx, op_name): level_idx}, where level_idx indexes stoc_len_levels
    fixed_levels: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        # sanity-check any pre-populated entries
        k = len(self.stoc_len_levels)
        for key, b in self.boundaries.items():
            assert isinstance(key, tuple) and len(key) == 2, (
                f"boundaries key must be (block_idx, op_name), got {key!r}")
            bt = b if isinstance(b, torch.Tensor) else torch.as_tensor(b)
            assert bt.numel() == k - 1, (
                f"boundaries[{key!r}] must have length {k-1}, got {bt.numel()}")
        for key, level_idx in self.fixed_levels.items():
            assert isinstance(key, tuple) and len(key) == 2, (
                f"fixed_levels key must be (block_idx, op_name), got {key!r}")
            li = int(level_idx)
            assert 0 <= li < k, (
                f"fixed_levels[{key!r}] must be in [0, {k}), got {li}")

    def default_boundaries(self) -> torch.Tensor:
        """Equal-spacing boundaries in (0, 1) descending, length k-1."""
        k = len(self.stoc_len_levels)
        return torch.tensor(
            [(k - 1 - i) / k for i in range(k - 1)], dtype=torch.float32)

    def get_boundaries(self, operator: str,
                       block_idx: Optional[int] = None) -> torch.Tensor:
        if block_idx is None:
            block_idx = _CURRENT_BLOCK_IDX
        key = (int(block_idx), operator)
        if key in self.boundaries:
            b = self.boundaries[key]
            return b if isinstance(b, torch.Tensor) else torch.as_tensor(b)
        return self.default_boundaries()

    def set_boundaries(self, operator: str, block_idx: int,
                       boundaries: torch.Tensor) -> None:
        bt = (boundaries.detach().cpu().float() if isinstance(boundaries, torch.Tensor)
              else torch.as_tensor(boundaries, dtype=torch.float32))
        k = len(self.stoc_len_levels)
        assert bt.numel() == k - 1, (
            f"expected {k-1} boundaries, got {bt.numel()}")
        self.fixed_levels.pop((int(block_idx), operator), None)
        self.boundaries[(int(block_idx), operator)] = bt

    def get_fixed_level(self, operator: str,
                        block_idx: Optional[int] = None) -> Optional[int]:
        if block_idx is None:
            block_idx = _CURRENT_BLOCK_IDX
        level_idx = self.fixed_levels.get((int(block_idx), operator))
        return None if level_idx is None else int(level_idx)

    def set_fixed_level(self, operator: str, block_idx: int,
                        level_idx: int) -> None:
        li = int(level_idx)
        k = len(self.stoc_len_levels)
        assert 0 <= li < k, f"level_idx must be in [0, {k}), got {li}"
        key = (int(block_idx), operator)
        self.boundaries.pop(key, None)
        self.fixed_levels[key] = li

    def clear_fixed_level(self, operator: str, block_idx: int) -> None:
        self.fixed_levels.pop((int(block_idx), operator), None)


def _classify_all_rows_to_level(
    metric: torch.Tensor,
    stoc_len_levels: list[int],
    level_idx: int,
) -> "RowAssignment":
    """Assign every row/head to one fixed level index."""
    N = metric.shape[0]
    row_levels = torch.full(
        (N,), int(level_idx), dtype=torch.long, device=metric.device)
    level_row_indices: dict[int, torch.Tensor] = {}
    for idx, sl in enumerate(stoc_len_levels):
        if idx == int(level_idx):
            level_row_indices[sl] = torch.arange(N, device=metric.device)
        else:
            level_row_indices[sl] = torch.empty(
                0, dtype=torch.long, device=metric.device)
    return RowAssignment(row_levels=row_levels,
                         level_row_indices=level_row_indices)


def _classify_with_free_boundaries(
    metric: torch.Tensor,
    boundaries: torch.Tensor,
    stoc_len_levels: list[int],
) -> "RowAssignment":
    """Bucket rows by normalized metric against free, non-equal-spaced
    descending boundaries. See ``adaptive_classify_rows`` for the semantics
    (level 0 = highest stoc_len, assigned to rows above the first boundary).
    """
    N = metric.shape[0]
    n_levels = len(stoc_len_levels)

    m_min = metric.min()
    m_max = metric.max()
    if (m_max - m_min).item() < 1e-8:
        # Degenerate distribution: default to level 0.
        row_levels = torch.zeros(N, dtype=torch.long, device=metric.device)
        level_row_indices: dict[int, torch.Tensor] = {}
        for idx, sl in enumerate(stoc_len_levels):
            if idx == 0:
                level_row_indices[sl] = torch.arange(N, device=metric.device)
            else:
                level_row_indices[sl] = torch.empty(
                    0, dtype=torch.long, device=metric.device)
        return RowAssignment(row_levels=row_levels,
                             level_row_indices=level_row_indices)

    metric_norm = (metric - m_min) / (m_max - m_min)

    # Ensure descending order for safety — boundaries may come from a
    # coord-descent step that hasn't yet re-sorted.
    b_sorted, _ = torch.sort(boundaries.to(metric.device).float(),
                             descending=True)
    row_levels = torch.zeros(N, dtype=torch.long, device=metric.device)
    for k in range(n_levels - 1):
        row_levels[metric_norm < b_sorted[k]] = k + 1

    level_row_indices = {}
    for i, sl in enumerate(stoc_len_levels):
        level_row_indices[sl] = torch.where(row_levels == i)[0]
    return RowAssignment(row_levels=row_levels,
                         level_row_indices=level_row_indices)


# =====================================================================
# Auto-MP budget logger (compute savings tracking during oracle search)
# =====================================================================

class AutoMPBudgetLogger:
    """Lightweight per-forward compute logger for budget-aware auto-MP.

    SC operators record a baseline cost (all rows/heads at max stoc_len) and
    the actual weighted stoc_len cost induced by the current assignment. The
    auto-MP calibrator enables this logger only while scoring candidate
    boundaries, so it sees the true block-local compute for that candidate.
    """

    _enabled: bool = False
    _log: list[dict] = []

    @classmethod
    def enable(cls):
        cls._enabled = True

    @classmethod
    def disable(cls):
        cls._enabled = False

    @classmethod
    def clear(cls):
        cls._log.clear()

    @classmethod
    def record(cls, block_idx: int, operator: str,
               baseline: float, actual: float):
        if not cls._enabled:
            return
        cls._log.append({
            "block": int(block_idx),
            "operator": operator,
            "baseline": float(baseline),
            "actual": float(actual),
        })

    @classmethod
    def snapshot(cls, clear: bool = False) -> list[dict]:
        out = list(cls._log)
        if clear:
            cls.clear()
        return out

    @classmethod
    def totals(cls, clear: bool = False) -> dict[str, float]:
        total_baseline = 0.0
        total_actual = 0.0
        for entry in cls._log:
            total_baseline += entry["baseline"]
            total_actual += entry["actual"]
        out = {"baseline": total_baseline, "actual": total_actual}
        if clear:
            cls.clear()
        return out


# =====================================================================
# MP Distribution Logger
# =====================================================================

class MPDistributionLogger:
    """Logs the fraction of rows/heads assigned to each precision level.

    Collects per-(timestep, block, operator) distribution and dumps to CSV.
    Also tracks actual compute cost for accurate savings when range-based MP
    is used (where per-row stoc_len varies across weight groups).
    """

    _log: list[dict] = []
    _compute_log: list[dict] = []  # {timestep, block, operator, baseline, actual}

    @classmethod
    def log(cls, timestep: int, block_idx: int, operator: str,
            assignment: RowAssignment, total_rows: int):
        """Record one distribution entry.

        Args:
            timestep: Current diffusion timestep.
            block_idx: Block index.
            operator: Operator name (qk, av, mlp_fc1, mlp_fc2).
            assignment: RowAssignment from classify_rows_by_metric.
            total_rows: Total number of rows/heads being classified.
        """
        entry = {
            "timestep": timestep,
            "block": block_idx,
            "operator": operator,
            "total_rows": total_rows,
        }
        for sl, rows in sorted(assignment.level_row_indices.items(), reverse=True):
            count = len(rows)
            entry[f"sl_{sl}_count"] = count
            entry[f"sl_{sl}_frac"] = round(count / max(total_rows, 1), 4)
        cls._log.append(entry)

    @classmethod
    def log_compute(cls, timestep: int, block_idx: int, operator: str,
                    baseline: int, actual: float):
        """Record actual compute cost (stoc_len * elements) for accurate savings.

        Use this instead of / in addition to log() when range-based MP is active,
        since per-row stoc_len varies across weight groups.

        Args:
            baseline: Total cost if all at max_stoc_len (M * out_features * max_sl).
            actual: Sum of effective_stoc_len * num_rows * num_out_channels per group.
        """
        cls._compute_log.append({
            "timestep": timestep,
            "block": block_idx,
            "operator": operator,
            "baseline": baseline,
            "actual": actual,
        })

    @classmethod
    def dump_csv(cls, path: str = "debug_mp_distribution.csv"):
        """Write collected distribution stats to CSV and clear."""
        if not cls._log:
            return
        import csv
        # Gather all column names (stoc_len columns vary)
        all_keys = {}
        for entry in cls._log:
            for k in entry:
                all_keys[k] = True
        # Sort: fixed columns first, then sl_* columns sorted descending
        fixed = ["timestep", "block", "operator", "total_rows"]
        sl_keys = sorted(
            [k for k in all_keys if k.startswith("sl_")],
            key=lambda k: (-int(k.split("_")[1]), k.split("_")[2]))
        fieldnames = fixed + sl_keys

        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for entry in cls._log:
                w.writerow(entry)
        print(f"[MPDistributionLogger] Wrote {len(cls._log)} rows to {path}")
        cls._log.clear()

    @classmethod
    def summary(cls, max_stoc_len: int = 256, save_path: str | None = None):
        """Print compute savings summary from collected logs.

        When _compute_log has data (range-based or combined MP), uses those
        exact baseline/actual values for accurate savings.  Otherwise falls
        back to per-row _log entries (dynamic MP only).

        Args:
            max_stoc_len: The baseline stoc_len if no MP were used.
            save_path: If provided, also save the summary to this file.
        """
        if not cls._log and not cls._compute_log:
            print("[MPDistributionLogger] No data for summary.")
            return

        total_baseline = 0
        total_actual = 0.0
        per_op_baseline: dict[str, int] = {}
        per_op_actual: dict[str, float] = {}

        # Use compute log (accurate for range-based / combined MP)
        if cls._compute_log:
            for entry in cls._compute_log:
                op = entry["operator"]
                b = entry["baseline"]
                a = entry["actual"]
                total_baseline += b
                total_actual += a
                per_op_baseline[op] = per_op_baseline.get(op, 0) + b
                per_op_actual[op] = per_op_actual.get(op, 0.0) + a

            # Also include operators that only appear in _log (e.g. qk/av
            # which may still use dynamic-only MP)
            compute_ops = {e["operator"] for e in cls._compute_log}
            for entry in cls._log:
                op = entry["operator"]
                if op in compute_ops:
                    continue  # already counted via compute_log
                n = entry["total_rows"]
                baseline = n * max_stoc_len
                total_baseline += baseline
                per_op_baseline[op] = per_op_baseline.get(op, 0) + baseline

                actual = 0.0
                for k, v in entry.items():
                    if k.startswith("sl_") and k.endswith("_count"):
                        sl = int(k.split("_")[1])
                        actual += sl * v
                total_actual += actual
                per_op_actual[op] = per_op_actual.get(op, 0.0) + actual
        else:
            # Fallback: dynamic MP only (old behaviour)
            for entry in cls._log:
                n = entry["total_rows"]
                op = entry["operator"]
                baseline = n * max_stoc_len
                total_baseline += baseline
                per_op_baseline[op] = per_op_baseline.get(op, 0) + baseline

                actual = 0.0
                for k, v in entry.items():
                    if k.startswith("sl_") and k.endswith("_count"):
                        sl = int(k.split("_")[1])
                        actual += sl * v
                total_actual += actual
                per_op_actual[op] = per_op_actual.get(op, 0.0) + actual

        savings = 1.0 - total_actual / max(total_baseline, 1)
        lines = []
        lines.append(f"{'=' * 70}")
        lines.append(f"{'MP Compute Savings Summary':^70}")
        lines.append(f"{'=' * 70}")
        lines.append(f"  Baseline (all sl={max_stoc_len}): {total_baseline:>14,}")
        lines.append(f"  Actual weighted stoc_len:         {total_actual:>14,.0f}")
        lines.append(f"  Total savings:                    {savings:>14.1%}")
        lines.append(f"  {'-' * 66}")
        lines.append(f"  {'Operator':<15s}  {'Baseline':>12s}  {'Actual':>12s}  {'Savings':>8s}")
        lines.append(f"  {'-' * 66}")
        for op in sorted(per_op_baseline.keys()):
            b = per_op_baseline[op]
            a = per_op_actual[op]
            s = 1.0 - a / max(b, 1)
            lines.append(f"  {op:<15s}  {b:>12,}  {a:>12,.0f}  {s:>8.1%}")
        lines.append(f"{'=' * 70}")

        text = "\n".join(lines)
        print(f"\n{text}\n")

        if save_path:
            with open(save_path, "w") as f:
                f.write(text + "\n")
            print(f"[MPDistributionLogger] Summary saved to {save_path}")

    @classmethod
    def clear(cls):
        cls._log.clear()
        cls._compute_log.clear()


# =====================================================================
# Metric Profiler — collects μ/σ of importance metrics per (t, block, op)
# =====================================================================

class MetricProfiler:
    """Lightweight profiler: records per-(timestep, block, operator) metric stats.

    Call MetricProfiler.record(metric, timestep, block, operator) from
    the MP classification functions.  At the end of inference, call
    MetricProfiler.dump_csv() to write the collected statistics.

    The CSV contains: timestep, block, operator, N, mean, std, min, max,
    q25, q75, q95, q99.
    """

    _log: list[dict] = []
    _enabled: bool = False

    @classmethod
    def enable(cls):
        cls._enabled = True

    @classmethod
    def disable(cls):
        cls._enabled = False

    @classmethod
    def record(cls, metric: torch.Tensor, timestep: int, block_idx: int,
               operator: str):
        """Record statistics for a single metric vector."""
        if not cls._enabled:
            return

        m = metric.float()
        cls._log.append({
            "timestep": timestep,
            "block": block_idx,
            "operator": operator,
            "N": m.numel(),
            "mean": m.mean().item(),
            "std": m.std().item(),
            "min": m.min().item(),
            "max": m.max().item(),
            "q25": m.quantile(0.25).item(),
            "q75": m.quantile(0.75).item(),
            "q95": m.quantile(0.95).item(),
            "q99": m.quantile(0.99).item(),
        })

    @classmethod
    def dump_csv(cls, path: str = "profile_metric_sigma.csv"):
        """Write collected metric statistics to CSV and clear."""
        if not cls._log:
            print("[MetricProfiler] No data to dump.")
            return
        import csv
        fieldnames = ["timestep", "block", "operator", "N",
                      "mean", "std", "min", "max",
                      "q25", "q75", "q95", "q99"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(cls._log)
        print(f"[MetricProfiler] Wrote {len(cls._log)} rows to {path}")
        cls._log.clear()

    @classmethod
    def clear(cls):
        cls._log.clear()


# =====================================================================
# Range-based Mixed Precision (weight min/max range)
# =====================================================================

@dataclass
class RangeMPConfig:
    """Range-based mixed precision: assigns stoc_len levels based on
    per-group weight (max-min) range.

    Groups with small range -> low stoc_len (tight values, low precision ok).
    Groups with large range -> high stoc_len (spread values, need precision).

    Uses threshold-based mapping similar to AdaptiveMPConfig:
    - Normalize ranges to [0, 1]
    - base_threshold controls the cutoff between highest and lower levels
    - Ranges with normalized value >= base_threshold -> highest stoc_len
    - Ranges below -> split among lower levels via evenly-spaced boundaries

    Args:
        stoc_len_levels: Descending list of stoc_len values.
        base_threshold: Normalized range threshold (0-1). Higher = more
            groups get lower precision (more aggressive).
        operator_thresholds: Per-operator threshold overrides.
            Keys: "qk", "av", "mlp_fc1", "mlp_fc2", "input_proj", "proj".
    """
    stoc_len_levels: list[int]
    base_threshold: float = 0.3
    operator_thresholds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        assert len(self.stoc_len_levels) >= 2, (
            "Need at least 2 levels (high + low)")
        for i in range(len(self.stoc_len_levels) - 1):
            assert self.stoc_len_levels[i] > self.stoc_len_levels[i + 1], (
                f"stoc_len_levels must be sorted descending, "
                f"got {self.stoc_len_levels}")

    def get_threshold(self, operator: Optional[str] = None) -> float:
        """Get threshold for an operator, falling back to global."""
        if operator and operator in self.operator_thresholds:
            return self.operator_thresholds[operator]
        return self.base_threshold


def classify_groups_by_range(
    weight: torch.Tensor,
    group_size: int,
    config: RangeMPConfig,
    operator: Optional[str] = None,
) -> list[int]:
    """Compute per-group (max-min) range and assign stoc_len levels.

    Groups with large range need more SC precision (high stoc_len),
    groups with small range can use lower precision (low stoc_len).

    The mapping uses threshold-based classification analogous to
    adaptive_classify_rows:
    - Normalize per-group ranges to [0, 1]
    - range_norm >= base_threshold -> level 0 (highest stoc_len)
    - Below base_threshold -> split evenly among lower levels

    Args:
        weight: [out_features, in_features] weight tensor (already quantized).
        group_size: Number of output rows per group.

            * ``1``  selects true per-row grouping (``num_groups == out_features``).
            * Values ``<= 0`` or ``>= out_features`` collapse to a single
              per-tensor group (``num_groups == 1``) — one ``stoc_len`` for
              the whole weight matrix.
            * Any other value ``g`` produces ``out_features // g`` groups and
              currently requires ``out_features % g == 0`` (the reshape below
              will raise otherwise).
        config: RangeMPConfig instance.
        operator: Operator name for per-op threshold lookup.

    Returns:
        List of stoc_len values, one per group (length ``num_groups``).
    """
    out_features, in_features = weight.shape
    if group_size <= 0 or group_size >= out_features:
        group_size = out_features

    num_groups = out_features // group_size
    levels = config.stoc_len_levels
    n_levels = len(levels)
    threshold = config.get_threshold(operator)
    threshold = min(threshold, 0.95)

    # Reshape to [num_groups, group_size * in_features]
    w = weight.reshape(num_groups, -1).float()
    group_max = w.amax(dim=-1)   # [num_groups]
    group_min = w.amin(dim=-1)   # [num_groups]
    group_range = group_max - group_min  # [num_groups]

    # Normalize to [0, 1]
    r_min = group_range.min()
    r_max = group_range.max()
    range_norm = (group_range - r_min) / (r_max - r_min + 1e-8)

    # Threshold-based classification (same logic as adaptive_classify_rows)
    # range_norm >= threshold -> level 0 (highest stoc_len, needs high precision)
    # Below threshold -> split evenly among lower levels
    group_levels = torch.zeros(num_groups, dtype=torch.long, device=weight.device)

    boundaries = []
    for k in range(n_levels - 1):
        b = threshold * (n_levels - 1 - k) / (n_levels - 1)
        boundaries.append(b)

    for k in range(n_levels - 1):
        group_levels[range_norm < boundaries[k]] = k + 1

    # Convert level indices to stoc_len values
    result = [levels[group_levels[g].item()] for g in range(num_groups)]

    # Log distribution
    dist = {}
    for sl in levels:
        count = result.count(sl)
        dist[sl] = count
    print(f"  [RangeMP] {operator or 'unknown'}: "
          f"groups={num_groups}, threshold={threshold:.2f}, "
          f"distribution={dist}, "
          f"range_stats: min={group_range.min().item():.4f}, "
          f"max={group_range.max().item():.4f}, "
          f"mean={group_range.mean().item():.4f}")

    return result

"""Zero-hyperparameter automatic MP calibration.

Finds per-(block, op) free boundaries that minimize the post-QwT-compensation
residual ``||Y_fp - Y_sc - (X W + b)||^2`` on a small calibration set, via
coordinate descent in a Gauss-Seidel-over-blocks schedule.

Public entry points:

* ``oracle_search_op``   — search k-1 boundaries for one (block, op).
* ``oracle_search_block`` — outer loop over all ops in one block.
* ``auto_calibrate_mp``   — block-by-block calibrator that also installs QwT
                            ``CompensationBlock`` wrappers, mirroring the
                            structure of ``third_party/QwT-SC/QwT-vit-sc/
                            qwt_sc/compensation.py:calibrate_qwt``.

Nothing here trains with gradients. Oracle search = forward-only grid-plus-
coord-descent; comp fit = closed-form ridge. Both are borrowed straight
from QwT's philosophy ("no back-prop, calibration in minutes").
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn as nn

from .config import (
    AutoMPBudgetLogger,
    FreeBoundaryMPConfig,
    set_current_block_idx,
)


# =====================================================================
# Ridge fitter with precomputed factorization
# =====================================================================

class RidgeFitter:
    """Cache the factorization of ``(X_aug^T X_aug + ridge I)`` so that
    residual norms can be scored in O(N D D_out) per call without re-solving.

    Inputs are expected **on GPU** and in float32 / float64. For large
    calibration sets keep the batch in-memory but chunk if it doesn't fit.
    """

    def __init__(self, X: torch.Tensor, ridge: float = 1e-2):
        N, D = X.shape
        ones = torch.ones(N, 1, device=X.device, dtype=X.dtype)
        self.X_aug = torch.cat([X, ones], dim=-1)  # [N, D+1]
        G = self.X_aug.t() @ self.X_aug            # [D+1, D+1]
        reg = torch.zeros_like(G)
        reg[:D, :D] = ridge * torch.eye(D, device=G.device, dtype=G.dtype)
        self.G_inv = torch.linalg.inv(G + reg)     # [D+1, D+1]

    def residual_norm_sq(self, R: torch.Tensor) -> float:
        """Return ``||R - X_aug @ sol||^2`` where sol = argmin_w ||R - X_aug w||^2."""
        rhs = self.X_aug.t() @ R                   # [D+1, D_out]
        sol = self.G_inv @ rhs
        resid = R - self.X_aug @ sol
        return float((resid.pow(2).sum()).item())


# =====================================================================
# Coordinate-descent oracle search
# =====================================================================

def _candidate_values(
    boundaries: torch.Tensor,
    j: int,
    n_candidates: int,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Linear grid of candidate values for boundaries[j], strictly between
    its neighbors (or the [eps, 1-eps] envelope at the ends).
    """
    k_minus_1 = boundaries.numel()
    lo = float(boundaries[j + 1]) + eps if j + 1 < k_minus_1 else eps
    hi = float(boundaries[j - 1]) - eps if j > 0 else 1.0 - eps
    if hi <= lo:
        # Degenerate; return current value so coord descent is a no-op here.
        return boundaries[j:j + 1].clone()
    return torch.linspace(lo, hi, n_candidates)


def _pack_budget_info(
    score: float,
    budget_stats: Optional[dict[str, Any]],
    budget_target_actual: Optional[float],
) -> dict[str, Any]:
    actual = None if budget_stats is None else float(budget_stats.get("actual", 0.0))
    baseline = None if budget_stats is None else float(budget_stats.get("baseline", 0.0))
    feasible = True
    over = 0.0
    if budget_target_actual is not None and actual is not None:
        feasible = actual <= budget_target_actual + 1e-9
        over = max(actual - budget_target_actual, 0.0)
    return {
        "score": float(score),
        "actual": actual,
        "baseline": baseline,
        "feasible": feasible,
        "over": float(over),
    }


def _budget_better(new_info: dict[str, Any], best_info: dict[str, Any]) -> bool:
    """Order candidates by feasibility first, then score / budget overflow."""
    if new_info["feasible"] and not best_info["feasible"]:
        return True
    if best_info["feasible"] and not new_info["feasible"]:
        return False

    if new_info["feasible"] and best_info["feasible"]:
        if new_info["score"] < best_info["score"] - 1e-9:
            return True
        if best_info["score"] < new_info["score"] - 1e-9:
            return False
        if (new_info["actual"] is not None and best_info["actual"] is not None
                and new_info["actual"] < best_info["actual"] - 1e-9):
            return True
        return False

    if new_info["over"] < best_info["over"] - 1e-9:
        return True
    if best_info["over"] < new_info["over"] - 1e-9:
        return False
    if new_info["score"] < best_info["score"] - 1e-9:
        return True
    return False


def oracle_search_op(
    *,
    run_block_sc: Callable[[], torch.Tensor],
    Y_fp: torch.Tensor,
    fitter: RidgeFitter,
    cfg: FreeBoundaryMPConfig,
    block_idx: int,
    op: str,
    init_boundaries: Optional[torch.Tensor] = None,
    n_candidates: int = 10,
    n_outer: int = 2,
    eps: float = 1e-3,
    run_block_sc_with_budget: Optional[
        Callable[[], tuple[torch.Tensor, Optional[dict[str, Any]]]]
    ] = None,
    budget_target_actual: Optional[float] = None,
    score_fn: Optional[Callable[[torch.Tensor], float]] = None,
    log_fn: Callable[[str], None] = lambda _m: None,
) -> tuple[torch.Tensor, float]:
    """Coord-descent search for the k-1 free boundaries of one (block, op).

    Parameters
    ----------
    run_block_sc : callable with no args
        Runs the SC block on the pre-staged calibration input and returns
        Y_sc of shape ``[N, D_out]``. The caller is responsible for making
        sure the block's config reads ``cfg`` (by passing the same ``cfg``
        instance into ``patch_model``) and that ``set_current_block_idx``
        has been called for ``block_idx``.
    Y_fp : [N, D_out]
        FP target on the same input.
    fitter : RidgeFitter
        Pre-fit on the block's input X.
    cfg : FreeBoundaryMPConfig
        Mutated in place — boundaries for ``(block_idx, op)`` end up set
        to the best discovered values on return.

    Returns
    -------
    (best_boundaries, best_score)
    """
    k = len(cfg.stoc_len_levels)
    if init_boundaries is None:
        init_boundaries = cfg.default_boundaries()
    b = init_boundaries.detach().clone().float()

    def _eval_current() -> tuple[float, dict[str, Any]]:
        if run_block_sc_with_budget is not None:
            Y_sc_eval, budget_stats = run_block_sc_with_budget()
        else:
            Y_sc_eval = run_block_sc()
            budget_stats = None
        if score_fn is not None:
            score_eval = float(score_fn(Y_sc_eval))
        else:
            score_eval = fitter.residual_norm_sq(Y_fp - Y_sc_eval)
        info_eval = _pack_budget_info(
            score_eval, budget_stats, budget_target_actual)
        return score_eval, info_eval

    cfg.set_boundaries(op, block_idx, b)
    best_score, best_info = _eval_current()
    if budget_target_actual is None or best_info["actual"] is None:
        log_fn(f"    [op={op}] init score={best_score:.4e}")
    else:
        feas = "yes" if best_info["feasible"] else "no"
        log_fn(f"    [op={op}] init score={best_score:.4e} "
               f"cost={best_info['actual']:.2f}/{budget_target_actual:.2f} "
               f"feasible={feas}")

    for outer in range(n_outer):
        improved_this_round = False
        for j in range(k - 1):
            cands = _candidate_values(b, j, n_candidates, eps=eps)
            best_cand = float(b[j])
            best_cand_score = best_score
            best_cand_info = dict(best_info)
            for c in cands.tolist():
                b_trial = b.clone()
                b_trial[j] = c
                cfg.set_boundaries(op, block_idx, b_trial)
                score, cand_info = _eval_current()
                if _budget_better(cand_info, best_cand_info):
                    best_cand_score = score
                    best_cand_info = cand_info
                    best_cand = c
            chosen_info = dict(best_cand_info)
            if best_cand != float(b[j]) or _budget_better(chosen_info, best_info):
                b[j] = best_cand
                best_score = best_cand_score
                best_info = chosen_info
                cfg.set_boundaries(op, block_idx, b)
                improved_this_round = True
            else:
                # restore best boundaries in cfg after sweeping candidates
                cfg.set_boundaries(op, block_idx, b)
        if budget_target_actual is None or best_info["actual"] is None:
            log_fn(f"    [op={op}] outer {outer}: score={best_score:.4e} "
                   f"b={b.tolist()}")
        else:
            feas = "yes" if best_info["feasible"] else "no"
            log_fn(f"    [op={op}] outer {outer}: score={best_score:.4e} "
                   f"cost={best_info['actual']:.2f}/{budget_target_actual:.2f} "
                   f"feasible={feas} b={b.tolist()}")
        if not improved_this_round:
            break

    return b, best_score


def oracle_search_block(
    *,
    run_block_sc_for_op: Callable[[str], Callable[[], torch.Tensor]],
    Y_fp: torch.Tensor,
    fitter: RidgeFitter,
    cfg: FreeBoundaryMPConfig,
    block_idx: int,
    ops: Sequence[str],
    n_candidates: int = 10,
    n_outer_per_op: int = 2,
    n_outer_block: int = 2,
    log_fn: Callable[[str], None] = lambda _m: None,
) -> dict[str, dict[str, Any]]:
    """Search boundaries for every op in a block. Wraps multiple ops in an
    outer coord-descent (``n_outer_block``) since ops within a block couple
    through the block's output.

    Parameters
    ----------
    run_block_sc_for_op : callable
        ``op -> () -> Y_sc`` — closure producing a zero-arg forward that the
        op's boundaries affect. In practice the driver uses a single forward
        of the patched block (all ops feed into it), but different ops may
        want to swap the input feed — this signature leaves that flexible.

    Returns
    -------
    dict mapping op -> {"boundaries": Tensor[k-1], "score": float}
    """
    results: dict[str, dict[str, Any]] = {}
    # Initial pass: equal spacing for everyone.
    for op in ops:
        cfg.set_boundaries(op, block_idx, cfg.default_boundaries())

    for outer in range(n_outer_block):
        log_fn(f"  [block {block_idx}] outer block round {outer}")
        for op in ops:
            run_block_sc = run_block_sc_for_op(op)
            b, score = oracle_search_op(
                run_block_sc=run_block_sc,
                Y_fp=Y_fp,
                fitter=fitter,
                cfg=cfg,
                block_idx=block_idx,
                op=op,
                init_boundaries=cfg.get_boundaries(op, block_idx),
                n_candidates=n_candidates,
                n_outer=n_outer_per_op,
                log_fn=log_fn,
            )
            results[op] = {"boundaries": b, "score": score}

    return results


# =====================================================================
# Gauss-Seidel block calibrator (mirrors QwT calibrate_qwt)
# =====================================================================

@torch.no_grad()
def auto_calibrate_mp(
    *,
    model_fp: nn.Module,
    model_sc: nn.Module,
    blocks_fp: Sequence[nn.Module],
    blocks_sc_container,                     # list-like: __getitem__/__setitem__
    calib_loader,
    device: torch.device,
    n_calib: int,
    cfg: FreeBoundaryMPConfig,
    ops_per_block: Sequence[str],
    ridge: float = 1e-2,
    start_block: int = 0,
    fwd_chunk: int = 32,
    avg_sc_draws: int = 1,
    n_candidates: int = 10,
    n_outer_per_op: int = 2,
    n_outer_block: int = 2,
    budget_ratio: Optional[float] = None,
    search_objective: str = "comp_residual",
    comp_factory: Optional[Callable[[torch.Tensor, torch.Tensor], nn.Module]] = None,
    comp_factory_variants: Optional[list[tuple]] = None,
    comp_refit_iters: int = 0,
    fit_compensation: bool = True,
    install_compensation: bool = True,
    log_fn: Callable[[str], None] = print,
) -> list[dict]:
    """Gauss-Seidel sequential calibration: per block i, (1) search oracle
    boundaries over each op, (2) fit QwT compensation W, b, (3) install the
    compensated block, (4) propagate X to i+1.

    This is the direct analog of ``calibrate_qwt`` with an oracle-search step
    interleaved before the compensation fit.

    Parameters
    ----------
    cfg : FreeBoundaryMPConfig
        **Mutated in place** — at return, ``cfg.boundaries`` is populated
        for every (block_idx, op) in ``0..n_blocks × ops_per_block``.
    ops_per_block : list[str]
        The ops whose boundaries should be searched on every block (same list
        reused for all blocks; detection of which ops are actually SC-patched
        is the caller's responsibility).

    Returns
    -------
    per-block report dicts: {block, r2, enabled, rmse_before, rmse_after,
                             raw_rmse_after, objective,
                             block_budget_target, block_compute_baseline,
                             block_compute_actual, boundaries_per_op, variant, variant_rmses,
                             noise_aware_refit, dt_s}
    """
    from qwt_sc.compensation import (  # type: ignore
        CompensationBlock, closed_form_ridge,
    )

    assert len(blocks_fp) == len(blocks_sc_container), (
        f"fp blocks ({len(blocks_fp)}) vs sc blocks "
        f"({len(blocks_sc_container)}) length mismatch")
    assert search_objective in ("comp_residual", "raw_mse"), (
        f"unknown search_objective: {search_objective}")
    if install_compensation:
        assert fit_compensation, (
            "install_compensation=True requires fit_compensation=True")
    if budget_ratio is not None:
        assert budget_ratio >= 0.0, f"budget_ratio must be >= 0, got {budget_ratio}"
        budget_ratio = min(float(budget_ratio), 1.0)
    n_blocks = len(blocks_fp)

    # Install block-index pre-hooks so classifiers see the right (block,op).
    # We stash handles to keep them alive; they remain installed post-calib.
    hook_handles = []
    for i, blk in enumerate(blocks_sc_container):
        def _make_hook(idx):
            def _hook(_m, _args):
                set_current_block_idx(idx)
            return _hook
        hook_handles.append(blk.register_forward_pre_hook(_make_hook(i)))

    # Collect block-0 inputs via a pre-hook on model_sc's first block.
    first_block = blocks_sc_container[0]
    captured: list[torch.Tensor] = []

    def _cap_hook(_m, args):
        captured.append(args[0].detach().float().cpu())

    log_fn(f"[auto_calib] collecting first-block inputs from model_sc")
    handle = first_block.register_forward_pre_hook(_cap_hook)
    seen = 0
    try:
        for batch in calib_loader:
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            imgs = imgs.to(device, non_blocking=True)
            model_sc(imgs)
            seen += imgs.size(0)
            if seen >= n_calib:
                break
    finally:
        handle.remove()
    X_cur = torch.cat(captured, dim=0)[:n_calib]
    log_fn(f"[auto_calib] X_0 shape={tuple(X_cur.shape)}")

    report: list[dict] = []
    total_compute_baseline = 0.0
    total_compute_actual = 0.0

    for i in range(n_blocks):
        t0 = time.time()
        blk_fp = blocks_fp[i]
        blk_sc = blocks_sc_container[i]
        set_current_block_idx(i)

        # ---- Y_fp target ----
        Y_fp_cpu = _forward_batched(blk_fp, X_cur, device, fwd_chunk)

        # ---- Pre-stage X and Y_fp on device for fitter + scoring ----
        X_flat_cpu = X_cur.reshape(-1, X_cur.size(-1))
        Y_fp_flat_cpu = Y_fp_cpu.reshape(-1, Y_fp_cpu.size(-1))
        X_flat = X_flat_cpu.to(device)
        Y_fp_flat = Y_fp_flat_cpu.to(device)
        fitter = RidgeFitter(X_flat, ridge=ridge)

        # ---- Baseline Y_sc (current boundaries — default for block i) ----
        def _run_block_sc():
            out = _forward_batched(blk_sc, X_cur, device, fwd_chunk)
            return out.reshape(-1, out.size(-1)).to(device)

        # optional SC averaging to reduce stochastic variance
        def _run_block_sc_avg():
            if avg_sc_draws <= 1:
                return _run_block_sc()
            acc = None
            for _ in range(avg_sc_draws):
                y = _run_block_sc()
                acc = y if acc is None else acc + y
            return acc / avg_sc_draws

        def _run_block_sc_avg_with_budget():
            AutoMPBudgetLogger.clear()
            AutoMPBudgetLogger.enable()
            try:
                y0 = _run_block_sc()
            finally:
                AutoMPBudgetLogger.disable()
            budget_entries = AutoMPBudgetLogger.snapshot(clear=True)
            budget_stats = {
                "baseline": sum(e["baseline"] for e in budget_entries),
                "actual": sum(e["actual"] for e in budget_entries),
                "entries": budget_entries,
            }
            if avg_sc_draws <= 1:
                return y0, budget_stats
            acc = y0
            for _ in range(avg_sc_draws - 1):
                acc = acc + _run_block_sc()
            return acc / avg_sc_draws, budget_stats

        if budget_ratio is not None:
            raw_Y_sc, raw_budget_stats = _run_block_sc_avg_with_budget()
        else:
            raw_Y_sc = _run_block_sc_avg()
            raw_budget_stats = None
        raw_rmse = float((Y_fp_flat - raw_Y_sc).pow(2).mean().sqrt().item())
        if search_objective == "raw_mse":
            baseline_score = float((Y_fp_flat - raw_Y_sc).pow(2).mean().item())
        else:
            baseline_score = fitter.residual_norm_sq(Y_fp_flat - raw_Y_sc)

        block_compute_baseline = (
            0.0 if raw_budget_stats is None else float(raw_budget_stats["baseline"]))
        if budget_ratio is not None:
            carryover = budget_ratio * total_compute_baseline - total_compute_actual
            block_budget_target = budget_ratio * block_compute_baseline + carryover
            block_budget_target = float(
                min(max(block_budget_target, 0.0), block_compute_baseline))
        else:
            block_budget_target = None

        log_fn(f"[auto_calib] block {i:2d}: baseline raw_rmse={raw_rmse:.4e} "
               f"score={baseline_score:.4e}")
        if block_budget_target is not None:
            log_fn(f"[auto_calib] block {i:2d}: budget target="
                   f"{block_budget_target:.2f}/{block_compute_baseline:.2f}",
                   )

        # ---- Oracle search over ops in this block ----
        per_op: dict[str, dict[str, Any]] = {}
        if search_objective == "raw_mse":
            score_fn = lambda Y_sc_eval: float(  # noqa: E731
                (Y_fp_flat - Y_sc_eval).pow(2).mean().item())
        else:
            score_fn = None
        for outer in range(n_outer_block):
            for op in ops_per_block:
                b_init = cfg.get_boundaries(op, i)
                b_new, score = oracle_search_op(
                    run_block_sc=_run_block_sc_avg,
                    Y_fp=Y_fp_flat,
                    fitter=fitter,
                    cfg=cfg,
                    block_idx=i,
                    op=op,
                    init_boundaries=b_init,
                    n_candidates=n_candidates,
                    n_outer=n_outer_per_op,
                    run_block_sc_with_budget=(
                        _run_block_sc_avg_with_budget if budget_ratio is not None else None),
                    budget_target_actual=block_budget_target,
                    score_fn=score_fn,
                    log_fn=log_fn,
                )
                per_op[op] = {"boundaries": b_new.tolist(), "score": score}

        # ---- Refit QwT compensation W, b with final boundaries ----
        if budget_ratio is not None:
            final_Y_sc, final_budget_stats = _run_block_sc_avg_with_budget()
        else:
            final_Y_sc = _run_block_sc_avg()
            final_budget_stats = None
        raw_after_rmse = float((Y_fp_flat - final_Y_sc).pow(2).mean().sqrt().item())
        block_compute_actual = (
            0.0 if final_budget_stats is None else float(final_budget_stats["actual"]))
        R_flat = Y_fp_flat - final_Y_sc
        if fit_compensation:
            W, b_vec, r2 = closed_form_ridge(X_flat, R_flat, ridge=ridge)
            after_rmse = float(
                (R_flat - (X_flat @ W + b_vec)).pow(2).mean().sqrt().item())
        else:
            W = torch.zeros(
                X_flat.size(-1), R_flat.size(-1), device=device, dtype=X_flat.dtype)
            b_vec = torch.zeros(R_flat.size(-1), device=device, dtype=X_flat.dtype)
            r2 = 0.0
            after_rmse = raw_after_rmse

        # Noise-aware refit copied from QwT's SC-comp path: if the
        # compensator itself runs in SC, subtract its measured SC noise from
        # the LS target and refit W,b for a few iterations.
        refit_log = []
        if fit_compensation and comp_factory is not None and comp_refit_iters > 0:
            for it in range(comp_refit_iters):
                tmp_comp = comp_factory(W.detach().cpu(),
                                        b_vec.detach().cpu()).to(device)
                with torch.no_grad():
                    c_actual_chunks = []
                    for s in range(0, X_flat.size(0), fwd_chunk * 257):
                        e = min(s + fwd_chunk * 257, X_flat.size(0))
                        c_actual_chunks.append(tmp_comp(X_flat[s:e]))
                    c_actual = torch.cat(c_actual_chunks, dim=0)
                fp_pred = X_flat @ W + b_vec
                delta = c_actual - fp_pred
                R_corr = R_flat - delta
                W_new, b_new, r2_new = closed_form_ridge(
                    X_flat, R_corr, ridge=ridge)
                after_rmse_new = float(
                    (R_flat - (X_flat @ W_new + b_new) - delta)
                    .pow(2).mean().sqrt().item()
                )
                refit_log.append({
                    "iter": it,
                    "r2": r2_new,
                    "after_rmse_with_delta": after_rmse_new,
                    "delta_rmse": float(delta.pow(2).mean().sqrt().item()),
                })
                W, b_vec = W_new, b_new
                r2 = r2_new
                after_rmse = after_rmse_new
                del tmp_comp, c_actual, c_actual_chunks, fp_pred, delta, R_corr
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Align the gating semantics with QwT's calibrate_qwt.
        enabled = fit_compensation and (i >= start_block) and (r2 > 0.0)
        W_cpu = W.detach().cpu()
        b_cpu = b_vec.detach().cpu()

        chosen_variant = None
        variant_rmses: dict[str, float] = {}
        comp_module = None

        def _measure(mod: nn.Module) -> float:
            mod = mod.to(device)
            with torch.no_grad():
                chunks = []
                for s in range(0, X_flat.size(0), fwd_chunk * 257):
                    e = min(s + fwd_chunk * 257, X_flat.size(0))
                    chunks.append(mod(X_flat[s:e]))
                c_act = torch.cat(chunks, dim=0)
            return float((R_flat - c_act).pow(2).mean().sqrt().item())

        if install_compensation:
            if comp_factory_variants is not None and len(comp_factory_variants) > 0:
                best = None
                for vname, vfac in comp_factory_variants:
                    cand = vfac(W_cpu, b_cpu)
                    rmse_v = _measure(cand)
                    variant_rmses[vname] = rmse_v
                    if best is None or rmse_v < best[0]:
                        best = (rmse_v, vname, cand)
                if best is not None:
                    _, chosen_variant, comp_module = best
            elif comp_factory is not None:
                comp_module = comp_factory(W_cpu, b_cpu)
            new_block = CompensationBlock(
                block=blk_sc, W=W_cpu, b=b_cpu,
                r2=r2, enabled=enabled, comp_module=comp_module,
            ).to(device)
            blocks_sc_container[i] = new_block
            # pre-hook was attached to blk_sc; re-attach to new_block
            # (old hook still fires on blk_sc if it's still referenced, but
            # forwarding now goes through new_block; add a fresh hook).
            def _make_hook(idx):
                def _hook(_m, _args):
                    set_current_block_idx(idx)
                return _hook
            new_block.register_forward_pre_hook(_make_hook(i))
            X_next = _forward_batched(new_block, X_cur, device, fwd_chunk)
        else:
            X_next = _forward_batched(blk_sc, X_cur, device, fwd_chunk)

        dt = time.time() - t0
        var_str = f"  var={chosen_variant}" if chosen_variant is not None else ""
        if fit_compensation:
            log_fn(f"[auto_calib] block {i:2d}  r2={r2:+.4f}  "
                   f"rmse {raw_rmse:.4e} -> {after_rmse:.4e}{var_str}  "
                   f"enabled={enabled}  ({dt:.1f}s)")
        else:
            log_fn(f"[auto_calib] block {i:2d}  raw_rmse "
                   f"{raw_rmse:.4e} -> {raw_after_rmse:.4e}  ({dt:.1f}s)")

        report.append({
            "block": i,
            "r2": r2,
            "enabled": enabled,
            "objective": search_objective,
            "rmse_before": raw_rmse,
            "rmse_after": after_rmse,
            "raw_rmse_after": raw_after_rmse,
            "block_budget_target": block_budget_target,
            "block_compute_baseline": block_compute_baseline,
            "block_compute_actual": block_compute_actual,
            "boundaries_per_op": per_op,
            "variant": chosen_variant,
            "variant_rmses": variant_rmses,
            "noise_aware_refit": refit_log,
            "dt_s": dt,
        })

        X_cur = X_next
        total_compute_baseline += block_compute_baseline
        total_compute_actual += block_compute_actual
        del X_flat, Y_fp_flat, fitter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return report


def _forward_batched(block: nn.Module, X: torch.Tensor, device: torch.device,
                     chunk: int) -> torch.Tensor:
    """Run `block` on CPU-resident X batch-by-batch, return CPU output."""
    outs = []
    for s in range(0, X.size(0), chunk):
        xb = X[s:s + chunk].to(device, non_blocking=True)
        y = block(xb)
        outs.append(y.detach().float().cpu())
    return torch.cat(outs, dim=0)

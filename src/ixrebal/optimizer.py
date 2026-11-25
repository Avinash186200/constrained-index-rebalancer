"""Minimum-variance rebalance under the index rules, with a relaxation ladder.

One rebalance runs in two convex stages:

1. **P0** on every holdable name: minimise ex-ante variance subject to the
   budget, long-only, per-stock caps, sector bands, country bands and the
   turnover limit against the drifted holdings.
2. **Support step.** The minimum-holding rule (a held name weighs at least
   ``min_weight``) and the minimum number of names are not convex. Names that
   P0 gives at least half the minimum weight are kept; if fewer than
   ``min_names`` remain, the next names by P0 weight (then by lowest own
   variance) are added. If P1 is infeasible on that support, small P0 weights
   are rounded up instead of down and P1 is tried once more. **P1** re-solves on that fixed support with
   ``w >= min_weight`` on every kept name. P1 is convex, so its dual values are
   exact for P1; the support itself is a heuristic, not the optimum of the
   mixed-integer problem.

If either stage is infeasible, the ladder in the rulebook relaxes one rule by
one step (the first rule in ladder order that is active and below its limit)
and the whole rebalance is retried. Every rung is logged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import warnings

import cvxpy as cp
import numpy as np

from .risk import CovModel
from .rules import Bounds, Context, Rulebook, bounds, one_way_turnover

SOLVER = "CLARABEL"
FAMILIES = ("cap", "sector", "country", "turnover", "holding")
LADDER_FAMILY = {"turnover": "turnover", "country_band": "country",
                 "sector_band": "sector", "min_names": "holding"}
DUST = 1e-6              # weights below this are zeroed when no holding rule applies
ACCEPT_VIOLATION = 1e-8   # max constraint residual for an "optimal_inaccurate" solve to be used
SOLVER_LOG: list = []     # inaccurate or failed solves, read and cleared by the backtest
BIND_SLACK = 1e-6     # a constraint within this of its bound is "at the bound"
BIND_DUAL = 1e-6      # and binds if its dual is above this, in (vol %)^2 per unit of weight
# Support rounding attempts, in order: keep names P0 gives at least half the
# minimum weight (round the rest down to 0); if P1 is then infeasible (a group at
# its lower band can lose the small weights it relied on), keep every name P0
# gives at least 1% of the minimum weight (round them up).
SUPPORT_ROUNDING = (("half_min", 0.5), ("round_up", 0.01))


class RebalanceInfeasible(RuntimeError):
    def __init__(self, msg: str, ladder: list):
        super().__init__(msg)
        self.ladder = ladder


@dataclass
class ConvexSolve:
    status: str
    w: np.ndarray | None            # full-length weights (zeros outside the index set)
    objective: float | None         # ex-ante variance in (annualised vol %)^2
    binding: dict = field(default_factory=dict)


@dataclass
class RebalanceResult:
    weights: np.ndarray
    rulebook: Rulebook              # effective rules after any relaxation
    ladder: list                    # one dict per rung
    exante_vol: float               # annualised %, from the shrunk covariance
    binding: dict                   # family -> {n, n_binding, dual_per_pp}
    n_p0_support: int               # names P0 holds at >= min_weight / 2
    n_promoted: int                 # names added to meet min_names
    stage_status: dict


def _group_matrix(g: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    A = np.zeros((n, idx.size))
    A[g[idx], np.arange(idx.size)] = 1.0
    return A


def solve_convex(cov: CovModel, ctx: Context, b: Bounds, idx: np.ndarray,
                 floor: float, drop=frozenset(), solver: str = SOLVER) -> ConvexSolve:
    """Minimum variance over names ``idx`` (all others fixed at 0)."""
    N = ctx.parent.size
    n = idx.size
    if n == 0:
        return ConvexSolve("infeasible", None, None)
    A, ridge = cov.factor(idx)
    w = cp.Variable(n)
    obj = cp.sum_squares(A @ w) + ridge * cp.sum_squares(w)
    cons = {"budget": cp.sum(w) == 1}
    cons["floor"] = w >= floor
    if "cap" not in drop:
        cons["cap"] = w <= b.cap[idx]
    if "sector" not in drop:
        G = _group_matrix(ctx.sector, idx, ctx.n_sectors)
        cons["sector_hi"] = G @ w <= b.sector_hi
        cons["sector_lo"] = G @ w >= b.sector_lo
    if "country" not in drop:
        G = _group_matrix(ctx.country, idx, ctx.n_countries)
        cons["country_hi"] = G @ w <= b.country_hi
        cons["country_lo"] = G @ w >= b.country_lo
    if "turnover" not in drop and b.turnover is not None:
        wd = ctx.drifted
        outside = float(wd.sum() - wd[idx].sum())   # drifted weight that must be sold
        cons["turnover"] = 0.5 * (cp.sum(cp.abs(w - wd[idx])) + outside) <= b.turnover
    prob = cp.Problem(cp.Minimize(obj), list(cons.values()))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # inaccurate solutions are handled below
            prob.solve(solver=solver)
    except cp.error.SolverError:
        SOLVER_LOG.append({"status": "solver_error", "n": n, "drop": sorted(drop)})
        return ConvexSolve("solver_error", None, None)
    if prob.status == "optimal_inaccurate":
        # accept only if the returned point actually satisfies every constraint
        ok = (w.value is not None and bool(np.all(np.isfinite(w.value)))
              and max(float(np.max(c.violation())) for c in cons.values()) <= ACCEPT_VIOLATION)
        SOLVER_LOG.append({"status": "optimal_inaccurate", "accepted": bool(ok), "n": n,
                           "drop": sorted(drop)})
        if not ok:
            return ConvexSolve("inaccurate_rejected", None, None)
    elif prob.status != "optimal":
        return ConvexSolve(prob.status, None, None)
    wf = np.zeros(N)
    wf[idx] = np.maximum(w.value, 0.0)
    return ConvexSolve(prob.status, wf, float(prob.value), _binding(cons, w.value, b, ctx, idx, floor))


def _binding(cons, wv, b: Bounds, ctx: Context, idx, floor) -> dict:
    """Which constraints bind, from slack and dual value, per rule family.

    ``dual_per_pp`` is the first-order fall in ex-ante variance, in (vol %)^2,
    if every bound in the family were loosened by one percentage point.
    """
    out = {}

    def fam(name, parts):
        n = nb = 0
        dsum = 0.0
        for key, slack in parts:
            if key not in cons:
                continue
            d = np.atleast_1d(np.asarray(cons[key].dual_value, float))
            s = np.atleast_1d(slack)
            n += d.size
            nb += int(np.sum((s <= BIND_SLACK) & (d > BIND_DUAL)))
            dsum += float(np.sum(np.maximum(d, 0.0)))
        if n:
            out[name] = {"n": n, "n_binding": nb, "dual_per_pp": dsum * 0.01}

    fam("cap", [("cap", b.cap[idx] - wv)])
    Gs = np.bincount(ctx.sector[idx], weights=wv, minlength=ctx.n_sectors)
    fam("sector", [("sector_hi", b.sector_hi - Gs), ("sector_lo", Gs - b.sector_lo)])
    Gc = np.bincount(ctx.country[idx], weights=wv, minlength=ctx.n_countries)
    fam("country", [("country_hi", b.country_hi - Gc), ("country_lo", Gc - b.country_lo)])
    if "turnover" in cons:
        wd = ctx.drifted
        to = 0.5 * (np.abs(wv - wd[idx]).sum() + wd.sum() - wd[idx].sum())
        fam("turnover", [("turnover", b.turnover - to)])
    if floor > 0:
        fam("min_weight", [("floor", wv - floor)])
    return out


def _pipeline(cov: CovModel, ctx: Context, rb: Rulebook, drop, solver):
    b = bounds(ctx, rb)
    if "turnover" in drop:
        b.turnover = None
    if "cap" in drop:
        holdable = np.flatnonzero(ctx.eligible)
    else:
        holdable = np.flatnonzero(b.cap > 0)
    p0 = solve_convex(cov, ctx, b, holdable, 0.0, drop, solver)
    status = {"p0": p0.status}
    if p0.w is None:
        return None, "p0", status, 0, 0, None
    if "holding" in drop:
        # no holding rules: only remove interior-point dust (weights below 1e-6)
        p0.w = np.where(p0.w >= DUST, p0.w, 0.0)
        return p0, None, status, int(np.sum(p0.w >= rb.min_weight / 2)), 0, b
    if holdable.size < rb.min_names:
        status["support"] = "too_few_holdable"
        return None, "support", status, 0, 0, None
    n_p0 = int(np.sum(p0.w[holdable] >= rb.min_weight / 2))
    p1 = None
    for rounding, thresh in SUPPORT_ROUNDING:
        keep, promoted = _support(cov, p0.w, holdable, rb, thresh)
        p1 = solve_convex(cov, ctx, b, keep, rb.min_weight, drop, solver)
        status["p1"] = p1.status
        status["support_rounding"] = rounding
        if p1.w is not None:
            return p1, None, status, n_p0, promoted, b
    return None, "p1", status, n_p0, promoted, None


def _support(cov: CovModel, w0: np.ndarray, holdable: np.ndarray, rb: Rulebook, thresh):
    """Names kept for P1: P0 weight >= thresh, topped up to min_names."""
    keep = holdable[w0[holdable] >= thresh * rb.min_weight]
    promoted = 0
    if keep.size < rb.min_names:
        rest = np.setdiff1d(holdable, keep)
        A, ridge = cov.factor(rest)
        own_var = np.sum(A ** 2, axis=0) + ridge
        order = np.lexsort((own_var, -w0[rest]))   # by P0 weight, then lowest variance
        add = rest[order[: rb.min_names - keep.size]]
        promoted = int(add.size)
        keep = np.sort(np.concatenate([keep, add]))
    return keep, promoted


def rebalance(cov: CovModel, ctx: Context, rb: Rulebook, drop=frozenset(),
              solver: str = SOLVER) -> RebalanceResult:
    """Solve one rebalance; relax rules along the ladder until it is feasible."""
    drop = frozenset(drop)
    unknown = drop - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown rule families {sorted(unknown)}")
    log: list = []
    cur = rb
    while True:
        sol, stage, status, n_p0, promoted, b = _pipeline(cov, ctx, cur, drop, solver)
        if sol is not None:
            w = sol.w / sol.w.sum()
            return RebalanceResult(weights=w, rulebook=cur, ladder=log,
                                   exante_vol=cov.vol(w), binding=sol.binding,
                                   n_p0_support=n_p0, n_promoted=promoted,
                                   stage_status=status)
        step = None
        for s in cur.ladder:
            if LADDER_FAMILY[s.rule] in drop:
                continue
            if s.rule == "turnover" and ctx.drifted is None:
                continue          # no turnover rule at inception
            if cur.can_relax(s):
                step = s
                break
        if step is None:
            raise RebalanceInfeasible(f"infeasible at stage {stage} with every ladder rung used", log)
        new = cur.relaxed(step.rule, step.step, step.limit)
        log.append({"rule": step.rule, "from": getattr(cur, step.rule),
                    "to": getattr(new, step.rule), "failed_stage": stage})
        cur = new


def unconstrained(cov: CovModel, ctx: Context, solver: str = SOLVER) -> RebalanceResult:
    """Long-only, fully invested minimum variance over eligible names, no index rules."""
    return rebalance(cov, ctx, Rulebook(), drop=frozenset(FAMILIES), solver=solver)


def replay_ladder(rb: Rulebook, ladder: list) -> Rulebook:
    """Apply a logged ladder to the original rulebook (used to audit a result)."""
    cur = rb
    for rung in ladder:
        s = next(x for x in rb.ladder if x.rule == rung["rule"])
        cur = cur.relaxed(s.rule, s.step, s.limit)
    return cur


__all__ = ["rebalance", "unconstrained", "solve_convex", "replay_ladder",
           "RebalanceInfeasible", "RebalanceResult", "one_way_turnover", "FAMILIES"]

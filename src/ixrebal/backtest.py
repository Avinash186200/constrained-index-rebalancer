"""Quarterly rebalance loop: point-in-time contexts, drifted holdings, daily returns.

Timing convention. Review ``k`` happens at the close of day ``t_k``. Everything
used there (parent weights from closing caps, the covariance window of days
``t_k - lookback + 1 .. t_k``, membership, drifted holdings) is known at that
close. The new weights earn returns from day ``t_k + 1``. No data after ``t_k``
is read when solving review ``k``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import optimizer as opt
from .checker import relaxations, verify
from .risk import CovModel, estimate
from .rules import Context, Rulebook, group_sums, one_way_turnover
from .universe import Universe

FIRST_REVIEW = 8          # day 504: two years of history before the first build
DEV_LAST_REVIEW = 31      # reviews 8..31 (6 years) are development; 32.. are held out
DROP_SETS = {             # leave-one-rule-out comparisons
    "cap": {"cap"}, "sector": {"sector"}, "country": {"country"},
    "turnover": {"turnover"}, "holding": {"holding"}, "all": set(opt.FAMILIES),
}


@dataclass(frozen=True)
class RunConfig:
    lookback: int = 504
    shrink: str = "ledoit_wolf"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def key(self) -> str:
        return f"{self.lookback}/{self.shrink}"


def period_of(k: int) -> str:
    return "development" if k <= DEV_LAST_REVIEW else "held_out"


def reviews(u: Universe, start: int = FIRST_REVIEW, stop: int | None = None) -> list:
    last = len(u.review_days) - 1 if stop is None else stop
    return list(range(start, last + 1))


def context_at(u: Universe, k: int, cfg: RunConfig, drifted) -> tuple[Context, CovModel]:
    """Point-in-time context and covariance at review k."""
    t = int(u.review_days[k])
    win = u.history(t, cfg.lookback)
    members = u.members(k)
    eligible = members & ~np.isnan(win).any(axis=0)
    cov = estimate(win[:, eligible], cfg.shrink)
    X = np.zeros((cfg.lookback, u.n_stocks))
    X[:, eligible] = cov.X
    cov.X = X
    ctx = Context(parent=u.parent_weights(k), eligible=eligible, drifted=drifted,
                  sector=u.sector, country=u.country,
                  n_sectors=u.cfg.n_sectors, n_countries=u.cfg.n_countries)
    return ctx, cov


def drift(u: Universe, w: np.ndarray, t0: int, t1: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold w from the close of t0 to the close of t1: (drifted weights, daily returns)."""
    h = w.copy()
    rets = np.empty(t1 - t0)
    for j, t in enumerate(range(t0 + 1, t1 + 1)):
        r = u.returns[t]
        held = h > 0
        if np.isnan(r[held]).any():
            raise RuntimeError(f"held name has no return on day {t}")
        r = np.where(held, r, 0.0)
        R = float(h @ r)
        h = h * (1.0 + r) / (1.0 + R)
        rets[j] = R
    return h, rets


def _end_day(u: Universe, k: int) -> int:
    rd = u.review_days
    return int(rd[k + 1]) if k + 1 < len(rd) else u.n_days - 1


def _sparse(w: np.ndarray) -> dict:
    return {str(i): float(w[i]) for i in np.flatnonzero(w > 0)}


def dense(d: dict, n: int) -> np.ndarray:
    w = np.zeros(n)
    for i, v in d.items():
        w[int(i)] = v
    return w


def _deviation(w, ctx: Context) -> tuple[float, float]:
    ds = group_sums(w, ctx.sector, ctx.n_sectors) - group_sums(ctx.parent, ctx.sector, ctx.n_sectors)
    dc = group_sums(w, ctx.country, ctx.n_countries) - group_sums(ctx.parent, ctx.country, ctx.n_countries)
    return float(np.abs(ds).max()), float(np.abs(dc).max())


def solve_review(u: Universe, k: int, rb: Rulebook, cfg: RunConfig, drifted,
                 variant: str = "index", costs: bool = False) -> tuple[dict, np.ndarray]:
    """One review: solve, verify, describe. Returns (record, published weights)."""
    ctx, cov = context_at(u, k, cfg, drifted)
    opt.SOLVER_LOG.clear()
    if variant == "index":
        res = opt.rebalance(cov, ctx, rb)
    elif variant == "unconstrained":
        res = opt.unconstrained(cov, ctx)
    else:
        raise ValueError(variant)
    # published weights are rounded to 12 decimals; everything below uses the published book
    w = np.round(res.weights, 12)
    sdev, cdev = _deviation(w, ctx)
    rec = {
        "review": k, "day": int(u.review_days[k]), "period": period_of(k),
        "n_members": int((ctx.parent > 0).sum()), "n_eligible": int(ctx.eligible.sum()),
        "shrink_delta": cov.delta, "exante_vol": res.exante_vol,
        "names": int((w > 0).sum()),
        "turnover": None if drifted is None else one_way_turnover(w, drifted),
        "max_sector_dev": sdev, "max_country_dev": cdev,
        "weights": _sparse(w),
    }
    if variant == "index":
        report = verify(w, ctx, res.rulebook)
        strict = verify(w, ctx, rb)
        rec.update({
            "ladder": res.ladder,
            "relaxed": {k2: list(v) for k2, v in relaxations(rb, res.rulebook).items()},
            "binding": res.binding,
            "n_p0_support": res.n_p0_support, "n_promoted": res.n_promoted,
            "support_rounding": res.stage_status.get("support_rounding"),
            "check_ok": report.ok, "check_failures": len(report.failures),
            "strict_rules_failed": sorted({c.rule for c in strict.failures}),
        })
        if costs:
            rec["vol_without"] = {}
            # costs are measured at the effective rulebook, so a review the ladder relaxed
            # is compared against the same relaxed rules, minus one family
            for name, fams in DROP_SETS.items():
                alt = opt.rebalance(cov, ctx, res.rulebook, drop=frozenset(fams))
                rec["vol_without"][name] = alt.exante_vol
    rec["solver_notes"] = list(opt.SOLVER_LOG)
    return rec, w


def run_path(u: Universe, rb: Rulebook, cfg: RunConfig, variant: str = "index",
             ks: list | None = None, costs: bool = False, start_weights=None) -> dict:
    """Rebalance at each review in ks, hold between reviews. Daily returns per holding period.

    ``start_weights`` are the published weights of review ks[0] - 1, for rerunning a
    path from the middle (None means ks[0] is the inception build).
    """
    ks = reviews(u) if ks is None else ks
    rd = u.review_days
    w_prev = start_weights
    recs = []
    for k in ks:
        drifted = None
        if w_prev is not None:
            drifted, _ = drift(u, w_prev, int(rd[k - 1]), int(rd[k]))
        rec, w = solve_review(u, k, rb, cfg, drifted, variant, costs)
        _, daily = drift(u, w, int(rd[k]), _end_day(u, k))
        rec["daily"] = [round(float(x), 10) for x in daily]
        recs.append(rec)
        w_prev = w
    return {"variant": variant, "config": cfg.to_dict(), "rulebook": rb.to_dict(), "records": recs}


def parent_path(u: Universe, ks: list | None = None) -> dict:
    ks = reviews(u) if ks is None else ks
    recs = []
    for k in ks:
        _, daily = drift(u, u.parent_weights(k), int(u.review_days[k]), _end_day(u, k))
        recs.append({"review": k, "period": period_of(k),
                     "daily": [round(float(x), 10) for x in daily]})
    return {"variant": "parent", "records": recs}

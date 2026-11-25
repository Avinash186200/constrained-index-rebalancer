"""The index rulebook as data, and the bounds it implies for one rebalance.

Each rule in the methodology document becomes one field here. ``bounds`` turns
the rulebook plus the parent index at the review into numbers: a cap per stock,
a lower and upper weight per sector and per country, and the turnover budget.
The optimiser and the checker both read these numbers, but the checker
recomputes every quantity it checks from the published weights itself.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np

RELAXABLE = ("turnover", "country_band", "sector_band", "min_names")


@dataclass(frozen=True)
class LadderStep:
    rule: str        # one of RELAXABLE
    step: float      # how much to relax per rung (min_names: names to drop)
    limit: float     # never relax beyond this value

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_LADDER = (
    LadderStep("turnover", 0.05, 0.30),
    LadderStep("country_band", 0.025, 0.10),
    LadderStep("sector_band", 0.025, 0.10),
    LadderStep("min_names", 10, 50),
)


@dataclass(frozen=True)
class Rulebook:
    cap_abs: float = 0.015        # no stock above 1.5% ...
    cap_mult: float = 20.0        # ... or above 20x its parent weight
    min_weight: float = 0.0005    # a held name weighs at least 0.05%
    sector_band: float = 0.05     # sector weight within +/- 5 points of the parent's
    country_band: float = 0.05    # country weight within +/- 5 points of the parent's
    min_names: int = 120          # at least 120 names held
    turnover: float = 0.10        # one-way turnover vs drifted holdings, per rebalance
    ladder: tuple = field(default=DEFAULT_LADDER)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ladder"] = [s.to_dict() if isinstance(s, LadderStep) else s for s in self.ladder]
        return d

    def relaxed(self, rule: str, step: float, limit: float) -> "Rulebook":
        """One rung of the ladder; the value never passes ``limit``."""
        cur = getattr(self, rule)
        if rule == "min_names":
            new = int(max(limit, cur - step))
        else:
            new = float(min(limit, cur + step))
        return replace(self, **{rule: new})

    def can_relax(self, s: LadderStep) -> bool:
        cur = getattr(self, s.rule)
        return cur > s.limit if s.rule == "min_names" else cur < s.limit - 1e-12


@dataclass
class Context:
    """Everything known at a review's close that the rules depend on."""
    parent: np.ndarray            # (N,) parent index weights after the review
    eligible: np.ndarray          # (N,) bool: member with enough history to estimate risk
    drifted: np.ndarray | None    # (N,) holdings drifted since the last rebalance; None at inception
    sector: np.ndarray            # (N,) int
    country: np.ndarray           # (N,) int
    n_sectors: int
    n_countries: int


@dataclass
class Bounds:
    cap: np.ndarray               # (N,) per-stock upper bound (0 for ineligible names)
    sector_lo: np.ndarray
    sector_hi: np.ndarray
    country_lo: np.ndarray
    country_hi: np.ndarray
    turnover: float | None        # None: no turnover rule (inception)


def group_sums(w: np.ndarray, g: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(g, weights=w, minlength=n)


def bounds(ctx: Context, rb: Rulebook) -> Bounds:
    cap = np.minimum(rb.cap_abs, rb.cap_mult * ctx.parent)
    # a name whose cap is below the minimum holding can never be held
    cap = np.where(ctx.eligible & (cap >= rb.min_weight), cap, 0.0)
    ps = group_sums(ctx.parent, ctx.sector, ctx.n_sectors)
    pc = group_sums(ctx.parent, ctx.country, ctx.n_countries)
    return Bounds(
        cap=cap,
        sector_lo=np.maximum(0.0, ps - rb.sector_band), sector_hi=ps + rb.sector_band,
        country_lo=np.maximum(0.0, pc - rb.country_band), country_hi=pc + rb.country_band,
        turnover=None if ctx.drifted is None else rb.turnover,
    )


def one_way_turnover(w: np.ndarray, before: np.ndarray) -> float:
    return 0.5 * float(np.abs(w - before).sum())

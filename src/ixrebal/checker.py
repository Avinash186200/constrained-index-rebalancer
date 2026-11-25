"""Independent post-solve verification of a published book.

The checker does not trust the optimiser. It takes the published weights, the
parent index and the drifted holdings, recomputes every quantity a rule
constrains with plain numpy, and compares it with the rule's limit plus a
tolerance. It never imports cvxpy or reads solver output.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rules import Context, Rulebook

TOL = 1e-7   # absolute, in weight units (1e-7 = 0.00001 percentage points)


@dataclass
class Check:
    rule: str
    item: str
    value: float
    lower: float
    upper: float

    @property
    def ok(self) -> bool:
        return self.lower - TOL <= self.value <= self.upper + TOL

    @property
    def slack(self) -> float:
        return min(self.value - self.lower, self.upper - self.value)


@dataclass
class Report:
    checks: list

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def failures(self) -> list:
        return [c for c in self.checks if not c.ok]

    def by_rule(self) -> dict:
        out: dict = {}
        for c in self.checks:
            d = out.setdefault(c.rule, {"n": 0, "failed": 0, "min_slack": np.inf})
            d["n"] += 1
            d["failed"] += int(not c.ok)
            d["min_slack"] = min(d["min_slack"], c.slack)
        return out


def verify(w: np.ndarray, ctx: Context, rb: Rulebook) -> Report:
    w = np.asarray(w, float)
    p = ctx.parent
    inf = np.inf
    held = w > TOL
    checks = [Check("budget", "sum", float(w.sum()), 1.0, 1.0),
              Check("long_only", "min weight", float(w.min()), 0.0, inf)]
    # names outside the eligible set must not be held
    checks.append(Check("eligible", "weight outside eligible set",
                        float(np.abs(w[~ctx.eligible]).sum()), 0.0, 0.0))
    for i in np.flatnonzero(held):
        cap = min(rb.cap_abs, rb.cap_mult * p[i])
        checks.append(Check("cap", f"stock {i}", float(w[i]), 0.0, cap))
        checks.append(Check("min_weight", f"stock {i}", float(w[i]), rb.min_weight, inf))
    checks.append(Check("min_names", "names held", float(held.sum()), rb.min_names, inf))
    for rule, g, n, band in (("sector", ctx.sector, ctx.n_sectors, rb.sector_band),
                             ("country", ctx.country, ctx.n_countries, rb.country_band)):
        for j in range(n):
            mask = g == j
            pj = float(p[mask].sum())
            checks.append(Check(rule, f"{rule} {j}", float(w[mask].sum()),
                                max(0.0, pj - band), pj + band))
    if ctx.drifted is not None:
        checks.append(Check("turnover", "one-way",
                            0.5 * float(np.abs(w - ctx.drifted).sum()), 0.0, rb.turnover))
    return Report(checks)


def relaxations(original: Rulebook, effective: Rulebook) -> dict:
    """Rules whose effective value differs from the rulebook: {rule: (original, effective)}."""
    out = {}
    for f in ("cap_abs", "cap_mult", "min_weight", "sector_band", "country_band",
              "min_names", "turnover"):
        a, b = getattr(original, f), getattr(effective, f)
        if a != b:
            out[f] = (a, b)
    return out

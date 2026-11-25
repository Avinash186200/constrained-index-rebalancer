from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import minimize

from ixrebal import optimizer as opt
from ixrebal.checker import verify
from ixrebal.rules import Rulebook, bounds

from conftest import make_case

RB = Rulebook(cap_abs=0.10, cap_mult=8.0, min_weight=0.005, sector_band=0.05,
              country_band=0.08, min_names=12, turnover=0.25)


def test_rebalance_passes_the_checker_without_relaxation():
    ctx, cov = make_case(0)
    res = opt.rebalance(cov, ctx, RB)
    assert res.ladder == [] and res.rulebook == RB
    rep = verify(res.weights, ctx, RB)
    assert rep.ok, [(c.rule, c.item, c.value, c.lower, c.upper) for c in rep.failures]
    assert (res.weights > 0).sum() >= RB.min_names
    assert res.exante_vol == pytest.approx(cov.vol(res.weights))


def test_unconstrained_is_never_riskier_than_the_index():
    for seed in range(4):
        ctx, cov = make_case(seed)
        idx = opt.rebalance(cov, ctx, RB)
        unc = opt.unconstrained(cov, ctx)
        assert unc.exante_vol <= idx.exante_vol + 1e-6
        assert np.all(unc.weights[~ctx.eligible] == 0)


def test_p0_relaxations_never_raise_variance():
    ctx, cov = make_case(5)
    b = bounds(ctx, RB)
    hold = np.flatnonzero(b.cap > 0)
    full = opt.solve_convex(cov, ctx, b, hold, 0.0).objective
    for fam in ("cap", "sector", "country", "turnover"):
        assert opt.solve_convex(cov, ctx, b, hold, 0.0, {fam}).objective <= full + 1e-7


def test_ladder_relaxes_turnover_first_and_logs_each_rung():
    ctx, cov = make_case(6)
    # put 18% of the drifted book in a name that has left the eligible set: it must be sold
    j = int(np.flatnonzero(~ctx.eligible)[0])
    d = ctx.drifted * 0.82
    d[j] += 0.18
    ctx.drifted = d
    rb = replace(RB, turnover=0.10)
    res = opt.rebalance(cov, ctx, rb)
    assert res.ladder and all(r["rule"] == "turnover" for r in res.ladder)
    assert [r["to"] for r in res.ladder] == pytest.approx(
        [0.10 + 0.05 * (i + 1) for i in range(len(res.ladder))])
    assert res.rulebook == opt.replay_ladder(rb, res.ladder)
    assert verify(res.weights, ctx, res.rulebook).ok
    assert {c.rule for c in verify(res.weights, ctx, rb).failures} == {"turnover"}


def test_exhausted_ladder_raises_with_the_log():
    ctx, cov = make_case(7, eligible_share=1.0)
    # sector 0 is half the parent but has no holdable name: bands cannot reach it
    ctx.parent[ctx.sector == 0] *= 50
    ctx.parent /= ctx.parent.sum()
    ctx.eligible[ctx.sector == 0] = False
    with pytest.raises(opt.RebalanceInfeasible) as e:
        opt.rebalance(cov, ctx, RB)
    rules = [r["rule"] for r in e.value.ladder]
    assert rules.index("turnover") < rules.index("country_band") < rules.index("sector_band")
    final = opt.replay_ladder(RB, e.value.ladder)
    assert final.sector_band == pytest.approx(0.10) and final.turnover == pytest.approx(0.30)


def test_unknown_rule_family_is_rejected():
    ctx, cov = make_case(0)
    with pytest.raises(ValueError):
        opt.rebalance(cov, ctx, RB, drop={"sectors"})


def _p0(ctx, cov, rb):
    b = bounds(ctx, rb)
    return opt.solve_convex(cov, ctx, b, np.flatnonzero(b.cap > 0), 0.0)


@pytest.mark.parametrize("field,family", [("sector_band", "sector"), ("turnover", "turnover"),
                                          ("country_band", "country")])
def test_duals_match_a_finite_difference(field, family):
    """dual_per_pp predicts the fall in variance from loosening the whole family by 1pp."""
    ctx, cov = make_case(11, n=60)
    rb = replace(RB, sector_band=0.02, country_band=0.03, turnover=0.15)
    base = _p0(ctx, cov, rb)
    pred = base.binding[family]["dual_per_pp"]
    assert pred > 0.01, "the test case must make this family bind"
    eps = 2e-4
    up = _p0(ctx, cov, replace(rb, **{field: getattr(rb, field) + eps})).objective
    dn = _p0(ctx, cov, replace(rb, **{field: getattr(rb, field) - eps})).objective
    fd_per_pp = -(up - dn) / (2 * eps) * 0.01
    assert fd_per_pp == pytest.approx(pred, rel=0.02)


def test_p0_agrees_with_scipy_slsqp():
    """Cross-check on a tiny problem without the non-smooth turnover term."""
    ctx, cov = make_case(12, n=15, n_sectors=3, n_countries=2, drifted=False, eligible_share=1.0)
    rb = replace(RB, cap_abs=0.2, cap_mult=10.0, sector_band=0.04, country_band=0.1)
    ours = _p0(ctx, cov, rb)
    b = bounds(ctx, rb)
    S = cov.matrix() * 252 * 1e4
    scale = ours.objective          # SLSQP needs an objective of order 1
    Gs = np.eye(ctx.n_sectors)[ctx.sector].T
    Gc = np.eye(ctx.n_countries)[ctx.country].T
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "ineq", "fun": lambda w: b.sector_hi - Gs @ w},
            {"type": "ineq", "fun": lambda w: Gs @ w - b.sector_lo},
            {"type": "ineq", "fun": lambda w: b.country_hi - Gc @ w},
            {"type": "ineq", "fun": lambda w: Gc @ w - b.country_lo}]
    x0 = np.minimum(b.cap, 1 / 15)
    x0 /= x0.sum()
    sp = minimize(lambda w: w @ S @ w / scale, x0, jac=lambda w: 2 * S @ w / scale, method="SLSQP",
                  bounds=[(0, c) for c in b.cap], constraints=cons,
                  options={"ftol": 1e-12, "maxiter": 500})
    assert sp.success
    assert ours.objective == pytest.approx(sp.fun * scale, rel=1e-5)
    np.testing.assert_allclose(ours.w, sp.x, atol=1e-4)


def test_min_names_top_up_adds_lowest_risk_names():
    ctx, cov = make_case(13, n=50)
    rb = replace(RB, min_names=40, turnover=1.0)
    res = opt.rebalance(cov, ctx, rb)
    assert res.n_promoted > 0
    assert (res.weights > 0).sum() >= 40
    assert verify(res.weights, ctx, res.rulebook).ok

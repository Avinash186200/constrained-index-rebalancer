"""Property-based tests on random small universes (hypothesis, derandomised).

Every case either publishes a book that the independent checker passes against
the effective (logged) rules, or raises with a ladder that has used every rung.
"""
import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ixrebal import optimizer as opt
from ixrebal.checker import verify
from ixrebal.rules import Rulebook, bounds

from conftest import make_case

RULE_OF = {"turnover": "turnover", "country_band": "country", "sector_band": "sector",
           "min_names": "min_names"}

SETTINGS = settings(max_examples=50, deadline=None, derandomize=True,
                    suppress_health_check=[HealthCheck.too_slow])


@st.composite
def cases(draw):
    seed = draw(st.integers(0, 10_000))
    n = draw(st.integers(15, 45))
    ctx, cov = make_case(seed, n=n, n_sectors=draw(st.integers(2, 5)),
                         n_countries=draw(st.integers(2, 4)), T=draw(st.integers(30, 90)),
                         drifted=draw(st.booleans()),
                         eligible_share=draw(st.floats(0.5, 1.0)))
    rb = Rulebook(
        cap_abs=draw(st.floats(0.04, 0.5)), cap_mult=draw(st.floats(2.0, 30.0)),
        min_weight=draw(st.floats(0.001, 0.01)),
        sector_band=draw(st.floats(0.0, 0.2)), country_band=draw(st.floats(0.0, 0.2)),
        min_names=draw(st.integers(1, max(1, 3 * n // 5))), turnover=draw(st.floats(0.0, 0.6)))
    return ctx, cov, rb


@SETTINGS
@given(cases())
def test_published_books_pass_the_checker_or_the_ladder_is_exhausted(c):
    ctx, cov, rb = c
    try:
        res = opt.rebalance(cov, ctx, rb)
    except opt.RebalanceInfeasible as e:
        final = opt.replay_ladder(rb, e.ladder)
        for s in rb.ladder:
            if s.rule == "turnover" and ctx.drifted is None:
                continue
            assert not final.can_relax(s), f"{s.rule} could still relax"
        return
    rep = verify(res.weights, ctx, res.rulebook)
    assert rep.ok, [(f.rule, f.item, f.value, f.lower, f.upper) for f in rep.failures]
    assert res.rulebook == opt.replay_ladder(rb, res.ladder)
    # against the original rulebook, only rules the ladder relaxed may fail
    strict = {f.rule for f in verify(res.weights, ctx, rb).failures}
    assert strict <= {RULE_OF[r["rule"]] for r in res.ladder}
    assert abs(res.weights.sum() - 1) < 1e-12 and res.weights.min() >= 0


@SETTINGS
@given(cases())
def test_unconstrained_minimum_variance_is_a_lower_bound(c):
    ctx, cov, rb = c
    try:
        res = opt.rebalance(cov, ctx, rb)
    except opt.RebalanceInfeasible:
        return
    unc = opt.unconstrained(cov, ctx)
    assert unc.exante_vol <= res.exante_vol + 1e-6


@SETTINGS
@given(cases(), st.sampled_from(["cap", "sector", "country", "turnover"]))
def test_dropping_a_rule_never_raises_p0_variance(c, fam):
    ctx, cov, rb = c
    b = bounds(ctx, rb)
    hold = np.flatnonzero(b.cap > 0)
    full = opt.solve_convex(cov, ctx, b, hold, 0.0)
    if full.objective is None:
        return
    alt = opt.solve_convex(cov, ctx, b, hold, 0.0, {fam})
    assert alt.objective <= full.objective * (1 + 1e-6) + 1e-8

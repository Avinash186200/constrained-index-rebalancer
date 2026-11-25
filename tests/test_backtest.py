import copy

import numpy as np
import pytest

from ixrebal.backtest import RunConfig, context_at, drift, run_path, solve_review
from ixrebal.rules import Rulebook
from ixrebal.universe import UniverseConfig, make_universe

CFG = RunConfig(lookback=126)
RB = Rulebook(cap_abs=0.04, cap_mult=20.0, min_names=30, sector_band=0.08, country_band=0.08,
              turnover=0.20)


@pytest.fixture(scope="module")
def u():
    return make_universe(UniverseConfig(n_stocks=120, years=4, seed=5, crises=((300, 20, 3.0),),
                                        booms=((0, 600, 63, 0.8),), mass_exits=((6, 0.1),)))


def test_drift_matches_share_count_bookkeeping(u):
    rd = u.review_days
    w = u.parent_weights(3)
    t0, t1 = int(rd[3]), int(rd[4])
    h, rets = drift(u, w, t0, t1)
    shares = w / u.caps[t0]                       # weight per unit of market cap = shares
    shares = np.nan_to_num(shares)
    value = shares * np.nan_to_num(u.caps[t1])
    np.testing.assert_allclose(h, value / value.sum(), atol=1e-12)
    level = np.prod(1 + rets)
    assert level == pytest.approx(value.sum() / (shares * np.nan_to_num(u.caps[t0])).sum(), rel=1e-10)


def test_review_uses_no_data_after_its_close(u):
    """Scramble every return after review k's close: the review's solve must not change."""
    k = 4
    t = int(u.review_days[k])
    rec, w = solve_review(u, k, RB, CFG, None)
    v = copy.deepcopy(u)
    rng = np.random.default_rng(0)
    v.returns[t + 1:] = rng.normal(0, 0.05, v.returns[t + 1:].shape)
    v.caps[t + 1:] *= 3.0
    rec2, w2 = solve_review(v, k, RB, CFG, None)
    np.testing.assert_array_equal(w, w2)
    assert rec["exante_vol"] == rec2["exante_vol"]


def test_path_with_drifted_holdings_uses_no_data_after_each_close(u):
    """Over several reviews with drifted books: scrambling the future after review 5 changes
    nothing up to and including review 5's published weights."""
    ks = [2, 3, 4, 5]
    t5 = int(u.review_days[5])
    a = run_path(u, RB, CFG, ks=ks)
    v = copy.deepcopy(u)
    rng = np.random.default_rng(1)
    v.returns[t5 + 1:] = rng.normal(0, 0.05, v.returns[t5 + 1:].shape)
    v.caps[t5 + 1:] *= 0.5
    b = run_path(v, RB, CFG, ks=ks)
    for ra, rb_ in zip(a["records"], b["records"]):
        assert ra["weights"] == rb_["weights"] and ra["turnover"] == rb_["turnover"]
        assert ra["exante_vol"] == rb_["exante_vol"]
    for ra, rb_ in zip(a["records"][:-1], b["records"][:-1]):
        assert ra["daily"] == rb_["daily"]            # holding periods before t5 are untouched


def test_eligibility_needs_full_history_and_membership(u):
    ctx, _ = context_at(u, 5, CFG, None)
    t = int(u.review_days[5])
    win = u.returns[t - CFG.lookback + 1: t + 1]
    expect = u.members(5) & ~np.isnan(win).any(axis=0)
    np.testing.assert_array_equal(ctx.eligible, expect)
    assert np.all(ctx.parent[~u.members(5)] == 0)


def test_path_is_continuous_and_rerunnable_from_the_middle(u):
    ks = [2, 3, 4, 5, 6, 7]
    full = run_path(u, RB, CFG, ks=ks)
    recs = full["records"]
    assert recs[0]["turnover"] is None and all(r["turnover"] is not None for r in recs[1:])
    assert all(r["check_ok"] for r in recs)
    # mass exit at review 6: drifted holdings in leavers must be sold
    w5 = np.zeros(u.n_stocks)
    for i, x in recs[3]["weights"].items():
        w5[int(i)] = x
    leavers = u.members(5) & ~u.members(6)
    assert w5[leavers].sum() > 0
    w6 = np.zeros(u.n_stocks)
    for i, x in recs[4]["weights"].items():
        w6[int(i)] = x
    assert w6[leavers].sum() == 0
    part = run_path(u, RB, CFG, ks=[6, 7], start_weights=w5)
    for a, b in zip(part["records"], recs[4:]):
        assert a["weights"] == b["weights"] and a["daily"] == b["daily"]


def test_unconstrained_path_is_less_volatile_ex_ante(u):
    idx = run_path(u, RB, CFG, ks=[3, 4])
    unc = run_path(u, RB, CFG, "unconstrained", ks=[3, 4])
    for a, b in zip(idx["records"], unc["records"]):
        assert b["exante_vol"] <= a["exante_vol"] + 1e-6

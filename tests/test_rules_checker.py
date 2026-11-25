from dataclasses import replace

import numpy as np
import pytest

from ixrebal.checker import relaxations, verify
from ixrebal.rules import LadderStep, Rulebook, bounds, one_way_turnover

from conftest import make_case


def test_bounds_follow_the_parent():
    ctx, _ = make_case(1)
    rb = Rulebook(cap_abs=0.05, cap_mult=3.0, sector_band=0.04, country_band=0.06)
    b = bounds(ctx, rb)
    expect = np.minimum(0.05, 3.0 * ctx.parent)
    expect = np.where(ctx.eligible & (expect >= rb.min_weight), expect, 0.0)
    np.testing.assert_allclose(b.cap, expect)
    ps = np.bincount(ctx.sector, ctx.parent, ctx.n_sectors)
    np.testing.assert_allclose(b.sector_hi, ps + 0.04)
    np.testing.assert_allclose(b.sector_lo, np.maximum(0, ps - 0.04))
    assert b.turnover == rb.turnover


def test_no_turnover_rule_at_inception():
    ctx, _ = make_case(1, drifted=False)
    assert bounds(ctx, Rulebook()).turnover is None
    rep = verify(np.where(ctx.eligible, 1.0, 0) / ctx.eligible.sum(), ctx, Rulebook(min_names=1))
    assert "turnover" not in {c.rule for c in rep.checks}


def test_relaxation_steps_stop_at_their_limit():
    rb = Rulebook(turnover=0.10, min_names=60)
    a = rb.relaxed("turnover", 0.05, 0.12)
    assert a.turnover == pytest.approx(0.12)
    assert not a.can_relax(LadderStep("turnover", 0.05, 0.12))
    b = rb.relaxed("min_names", 25, 50)
    assert b.min_names == 50
    assert relaxations(rb, b) == {"min_names": (60, 50)}


def test_one_way_turnover():
    assert one_way_turnover(np.array([0.5, 0.5, 0]), np.array([0.5, 0, 0.5])) == pytest.approx(0.5)


def _passing_book(ctx, rb):
    """Parent weights restricted to eligible names, used as a hand-made book."""
    w = np.where(ctx.eligible, ctx.parent, 0)
    return w / w.sum()


def test_checker_flags_each_kind_of_violation():
    ctx, _ = make_case(2, n=30, drifted=False, eligible_share=1.0)
    rb = Rulebook(cap_abs=1.0, cap_mult=100.0, min_weight=0.0, sector_band=1.0,
                  country_band=1.0, min_names=1)
    w = _passing_book(ctx, rb)
    assert verify(w, ctx, rb).ok

    def fails(w2, rb2=rb, ctx2=ctx):
        return {c.rule for c in verify(w2, ctx2, rb2).failures}

    assert fails(w * 1.01) == {"budget"}
    w_neg = w.copy()
    w_neg[1] += w[0] + 0.001
    w_neg[0] = -0.001
    assert "long_only" in fails(w_neg)
    assert "cap" in fails(w, replace(rb, cap_abs=float(w.max()) - 1e-4))
    assert "min_names" in fails(w, replace(rb, min_names=31))
    dust = w.copy()
    j = int(np.argmin(w))
    dust[(j + 1) % 30] += dust[j] - 1e-6
    dust[j] = 1e-6
    assert "min_weight" in fails(dust, replace(rb, min_weight=1e-4))
    one = np.eye(30)[0]
    assert "sector" in fails(one, replace(rb, sector_band=0.01))
    assert "country" in fails(one, replace(rb, country_band=0.01))
    ctx.eligible[0] = False
    assert "eligible" in fails(w)
    ctx.eligible[0] = True
    ctx.drifted = np.roll(w, 7)
    assert "turnover" in fails(w, replace(rb, turnover=0.0))


def test_checker_tolerance_is_absolute_and_small():
    ctx, _ = make_case(3, n=20, drifted=False, eligible_share=1.0)
    rb = Rulebook(cap_abs=1.0, cap_mult=100.0, min_weight=0.0, sector_band=1.0,
                  country_band=1.0, min_names=1)
    w = _passing_book(ctx, rb)
    assert verify(w * (1 + 5e-8), ctx, rb).ok          # inside 1e-7
    assert not verify(w * (1 + 5e-7), ctx, rb).ok      # outside

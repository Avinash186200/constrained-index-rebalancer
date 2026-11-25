import numpy as np
import pytest

from ixrebal.universe import UniverseConfig, make_universe


@pytest.fixture(scope="module")
def u():
    return make_universe(UniverseConfig(n_stocks=120, years=6, seed=3,
                                        crises=((300, 20, 3.0),), booms=((0, 700, 63, 0.8),)))


def test_shape_and_determinism(u):
    v = make_universe(UniverseConfig(n_stocks=120, years=6, seed=3,
                                     crises=((300, 20, 3.0),), booms=((0, 700, 63, 0.8),)))
    assert u.returns.shape == (6 * 252, 120)
    np.testing.assert_array_equal(np.isnan(u.returns), np.isnan(v.returns))
    np.testing.assert_array_equal(np.nan_to_num(u.returns), np.nan_to_num(v.returns))


def test_parent_weights_are_cap_weights_of_members(u):
    for k in (0, 5, 10):
        p = u.parent_weights(k)
        m = u.members(k)
        assert p.sum() == pytest.approx(1.0)
        assert np.all(p[~m] == 0) and np.all(p[m] > 0)
        t = u.review_days[k]
        np.testing.assert_allclose(p[m], u.caps[t, m] / u.caps[t, m].sum())


def test_members_have_returns_until_the_next_review(u):
    rd = u.review_days
    for k in range(len(rd) - 1):
        m = u.members(k)
        block = u.returns[rd[k] + 1: rd[k + 1] + 1][:, m]
        assert not np.isnan(block).any()


def test_membership_changes_happen(u):
    counts = [u.members(k).sum() for k in range(len(u.review_days))]
    assert len(set(counts)) > 1


def test_history_is_point_in_time(u):
    t = 400
    h = u.history(t, 100)
    np.testing.assert_array_equal(np.nan_to_num(h[-1]), np.nan_to_num(u.returns[t]))
    assert h.shape[0] == 100
    with pytest.raises(ValueError):
        u.history(50, 100)


def test_crisis_raises_true_volatility(u):
    tv = u.true_vol
    assert tv[300:320].mean() > 1.5 * tv[200:280].mean()

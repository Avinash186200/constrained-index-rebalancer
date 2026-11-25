import numpy as np
import pytest

from ixrebal.risk import SCALE, estimate, ledoit_wolf_delta


def _lw_by_loops(X):
    """Ledoit-Wolf (2004) intensity written out term by term, for tiny inputs."""
    T, N = X.shape
    S = X.T @ X / T
    mu = np.trace(S) / N
    F = mu * np.eye(N)
    d2 = np.sum((S - F) ** 2) / N
    b2 = 0.0
    for t in range(T):
        x = X[t][:, None]
        b2 += np.sum((x @ x.T - S) ** 2) / N
    b2 /= T ** 2
    return min(b2, d2) / d2, mu


def test_ledoit_wolf_matches_term_by_term_formula():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 8)) @ rng.standard_normal((8, 8))
    X -= X.mean(0)
    d, mu = ledoit_wolf_delta(X)
    d_ref, mu_ref = _lw_by_loops(X)
    assert d == pytest.approx(d_ref, rel=1e-10)
    assert mu == pytest.approx(mu_ref, rel=1e-12)
    assert 0.0 <= d <= 1.0


def test_shrinkage_rises_when_history_is_short():
    rng = np.random.default_rng(1)
    load = rng.standard_normal((3, 40))
    vols = rng.uniform(0.5, 2.0, 40)

    def draw(T):
        return (rng.standard_normal((T, 3)) @ load + rng.standard_normal((T, 40)) * vols) * 0.01

    long = estimate(draw(2000))
    short = estimate(draw(50))
    assert short.delta > long.delta


def test_factor_form_equals_matrix_form():
    rng = np.random.default_rng(2)
    cov = estimate(rng.standard_normal((60, 25)) * 0.01)
    w = rng.dirichlet(np.ones(25))
    A, ridge = cov.factor()
    quad = np.sum((A @ w) ** 2) + ridge * np.sum(w ** 2)
    assert quad == pytest.approx(SCALE * w @ cov.matrix() @ w, rel=1e-10)
    assert cov.vol(w) == pytest.approx(np.sqrt(quad), rel=1e-12)
    assert np.linalg.eigvalsh(cov.matrix()).min() > 0      # shrinkage makes it positive definite


def test_estimate_rejects_missing_values_and_unknown_shrink():
    X = np.ones((10, 3))
    X[2, 1] = np.nan
    with pytest.raises(ValueError):
        estimate(X)
    with pytest.raises(ValueError):
        estimate(np.ones((10, 3)), shrink="magic")


def test_no_shrink_is_the_sample_covariance():
    rng = np.random.default_rng(3)
    W = rng.standard_normal((40, 6))
    cov = estimate(W, "none")
    assert cov.delta == 0.0
    np.testing.assert_allclose(cov.matrix(), np.cov(W.T, ddof=0), rtol=1e-12, atol=1e-14)

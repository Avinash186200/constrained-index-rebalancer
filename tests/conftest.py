import os

# one BLAS thread before numpy is imported, as in CI and the Makefile
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ixrebal.risk import estimate  # noqa: E402
from ixrebal.rules import Context  # noqa: E402


def make_case(seed: int, n: int = 40, n_sectors: int = 4, n_countries: int = 3, T: int = 80,
              drifted: bool = True, eligible_share: float = 0.9):
    """A small random universe: factor-model returns, cap-weighted parent, drifted book."""
    rng = np.random.default_rng(seed)
    sector = rng.integers(0, n_sectors, n)
    country = rng.integers(0, n_countries, n)
    sector[:n_sectors] = np.arange(n_sectors)      # every group has at least one name
    country[:n_countries] = np.arange(n_countries)
    f = rng.standard_normal((T, 1 + n_sectors)) * 0.01
    load = np.zeros((n, 1 + n_sectors))
    load[:, 0] = rng.uniform(0.5, 1.5, n)
    load[np.arange(n), 1 + sector] = rng.uniform(0.5, 1.2, n)
    R = f @ load.T + rng.standard_normal((T, n)) * rng.uniform(0.005, 0.03, n)
    parent = rng.lognormal(0, 1, n)
    parent /= parent.sum()
    eligible = rng.random(n) < eligible_share
    eligible[:n_sectors] = True
    cov = estimate(R[:, eligible])
    X = np.zeros((T, n))
    X[:, eligible] = cov.X
    cov.X = X
    d = None
    if drifted:
        # a book that started near the parent and drifted: parent on eligible names plus noise
        d = 0.8 * np.where(eligible, parent, 0) / parent[eligible].sum() + 0.2 * rng.dirichlet(np.ones(n))
    ctx = Context(parent=parent, eligible=eligible, drifted=d, sector=sector, country=country,
                  n_sectors=n_sectors, n_countries=n_countries)
    return ctx, cov


@pytest.fixture
def case():
    return make_case(0)

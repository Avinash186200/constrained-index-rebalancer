"""Covariance estimate for the optimiser: sample or Ledoit-Wolf shrunk.

The covariance is never passed to the solver as an N x N matrix (the Ledoit-Wolf
intensity below is computed from the N x N sample covariance, once per review). With
``X`` the demeaned (T x N) return window, the shrunk estimate is

    Sigma = (1 - delta) * X'X / T + delta * mu * I

so a portfolio's variance is ``(1 - delta) / T * ||X w||^2 + delta * mu * ||w||^2``,
a sum of squares that a conic solver handles directly (T + N terms instead of
N^2). ``delta`` is the Ledoit-Wolf (2004) intensity towards the scaled identity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ANNUAL = 252.0
SCALE = ANNUAL * 1e4   # objective in (annualised vol in %)^2


@dataclass
class CovModel:
    X: np.ndarray      # (T, N) demeaned returns
    delta: float       # shrinkage intensity in [0, 1]
    mu: float          # target scale: average sample variance

    @property
    def n(self) -> int:
        return self.X.shape[1]

    def factor(self, idx=None):
        """(A, ridge) with w' Sigma w * SCALE = ||A w||^2 + ridge * ||w||^2 on columns idx."""
        X = self.X if idx is None else self.X[:, idx]
        T = X.shape[0]
        A = np.sqrt(SCALE * (1.0 - self.delta) / T) * X
        return A, SCALE * self.delta * self.mu

    def matrix(self) -> np.ndarray:
        T = self.X.shape[0]
        S = self.X.T @ self.X / T
        return (1.0 - self.delta) * S + self.delta * self.mu * np.eye(self.n)

    def vol(self, w: np.ndarray) -> float:
        """Ex-ante annualised volatility in % of a full-length weight vector."""
        A, ridge = self.factor()
        return float(np.sqrt(np.sum((A @ w) ** 2) + ridge * np.sum(w ** 2)))


def ledoit_wolf_delta(X: np.ndarray) -> tuple[float, float]:
    """Ledoit-Wolf intensity towards mu * I for demeaned X (T x N). Returns (delta, mu).

    Uses the normalised Frobenius norm ||A||^2 = tr(A A') / N, as in the paper.
    """
    T, N = X.shape
    S = X.T @ X / T
    mu = float(np.trace(S) / N)
    d2 = float(np.sum((S - mu * np.eye(N)) ** 2) / N)
    # b2_bar = 1/T^2 sum_t ||x_t x_t' - S||^2, expanded so no N x N matrix per row
    sq = np.sum(X ** 2, axis=1)
    xSx = np.einsum("ti,ij,tj->t", X, S, X)
    b2_bar = float(np.sum(sq ** 2 - 2.0 * xSx + np.sum(S ** 2)) / N / T ** 2)
    b2 = min(b2_bar, d2)
    delta = 0.0 if d2 == 0 else b2 / d2
    return delta, mu


def estimate(window: np.ndarray, shrink: str = "ledoit_wolf") -> CovModel:
    """Covariance model from a (T x N) window with no missing values."""
    if np.isnan(window).any():
        raise ValueError("window has missing values")
    X = window - window.mean(axis=0)
    T, N = X.shape
    if shrink == "ledoit_wolf":
        delta, mu = ledoit_wolf_delta(X)
    elif shrink == "none":
        delta, mu = 0.0, float(np.sum(X ** 2) / (T * N))
    else:
        raise ValueError(f"unknown shrink {shrink!r}")
    return CovModel(X=X, delta=delta, mu=mu)

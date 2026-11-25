"""Performance and rule statistics computed from per-review records."""
from __future__ import annotations

import numpy as np

ANNUAL = 252


def daily_series(path: dict, period: str | None = None) -> np.ndarray:
    recs = [r for r in path["records"] if period is None or r["period"] == period]
    return np.concatenate([np.asarray(r["daily"], float) for r in recs]) if recs else np.array([])


def ann_return(r: np.ndarray) -> float:
    return float(np.prod(1.0 + r) ** (ANNUAL / r.size) - 1.0)


def ann_vol(r: np.ndarray) -> float:
    return float(np.std(r, ddof=1) * np.sqrt(ANNUAL))


def max_drawdown(r: np.ndarray) -> float:
    level = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], level]))[1:]
    return float(np.min(level / peak - 1.0))


def tracking_error(r: np.ndarray, bench: np.ndarray) -> float:
    return ann_vol(r - bench)


def beta(r: np.ndarray, bench: np.ndarray) -> float:
    c = np.cov(r, bench, ddof=1)
    return float(c[0, 1] / c[1, 1])


def series_stats(r: np.ndarray, parent: np.ndarray | None = None) -> dict:
    out = {"days": int(r.size), "ann_return": ann_return(r), "ann_vol": ann_vol(r),
           "max_drawdown": max_drawdown(r)}
    if parent is not None:
        out["vol_reduction"] = 1.0 - out["ann_vol"] / ann_vol(parent)
        out["tracking_error"] = tracking_error(r, parent)
        out["beta"] = beta(r, parent)
    return out


def mean_or_none(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None

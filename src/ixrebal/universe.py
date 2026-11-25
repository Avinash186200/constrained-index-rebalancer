"""Synthetic equity universe with a factor-model return process.

The universe stands in for the constituents of a broad, cap-weighted parent
index. Returns come from a factor model (market, one factor per sector, one per
country, a size factor) plus stock-specific noise. Factor and stock volatilities
move over time (log-AR(1) stochastic volatility), and two kinds of event are
planted on purpose so that the index rules come under stress:

* volatility crises: the market factor's volatility is multiplied for a window;
* a sector boom: one high-volatility sector gets a strong positive drift for a
  window, so its weight in the parent index rises quickly;
* parent reconstitutions: at a chosen review a share of the members leave at
  once, so the index must sell those holdings within its turnover budget.

Constituent changes happen only at quarterly reviews: some names list late
(their returns are missing before they join) and some leave the parent at a
review (their returns stop after it). Nothing in this module is read by the
optimiser except returns, market caps and the membership calendar, all through
point-in-time accessors.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class UniverseConfig:
    n_stocks: int = 500
    n_sectors: int = 11
    n_countries: int = 6
    years: int = 13
    days_per_year: int = 252
    review_every: int = 63          # trading days between quarterly reviews
    seed: int = 7
    late_joiner_share: float = 0.15  # names that list after day 0
    leaver_share: float = 0.10       # names that leave the parent at some review
    market_vol: float = 0.15         # annualised, before stochastic-vol scaling
    market_drift: float = 0.06       # annualised
    # (start_day, length_days, market-vol multiplier)
    crises: tuple = ((1100, 40, 3.0), (2850, 30, 2.5))
    # (sector, start_day, length_days, extra annualised drift of the sector factor)
    booms: tuple = ((0, 2300, 126, 0.80),)
    # (review, share of members): a parent reconstitution that removes many names at once
    mass_exits: tuple = ((20, 0.08), (42, 0.08))
    country_shares: tuple = (0.55, 0.15, 0.10, 0.08, 0.07, 0.05)

    @property
    def n_days(self) -> int:
        return self.years * self.days_per_year

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Universe:
    cfg: UniverseConfig
    returns: np.ndarray        # (T, N) daily simple returns; NaN when not listed
    caps: np.ndarray           # (T, N) market cap at each close; NaN when not listed
    sector: np.ndarray         # (N,) int
    country: np.ndarray        # (N,) int
    join_review: np.ndarray    # (N,) review index at which the name enters the parent
    leave_review: np.ndarray   # (N,) review index after which it is gone (n_reviews = never)
    true_vol: np.ndarray = field(repr=False, default=None)  # (T, N) ann. vol, diagnostics only

    @property
    def n_days(self) -> int:
        return self.returns.shape[0]

    @property
    def n_stocks(self) -> int:
        return self.returns.shape[1]

    @property
    def review_days(self) -> np.ndarray:
        """Close of day ``k * review_every`` is review ``k``."""
        return np.arange(0, self.n_days, self.cfg.review_every)

    def members(self, k: int) -> np.ndarray:
        """Parent membership decided at review k (holds until review k + 1)."""
        return (self.join_review <= k) & (self.leave_review > k)

    def parent_weights(self, k: int) -> np.ndarray:
        """Cap weights of the members chosen at review k, at that review's close."""
        t = self.review_days[k]
        m = self.members(k)
        w = np.where(m, np.nan_to_num(self.caps[t]), 0.0)
        return w / w.sum()

    def history(self, t: int, lookback: int) -> np.ndarray:
        """Returns for days t - lookback + 1 .. t (inclusive): only data known at t's close."""
        if t + 1 < lookback:
            raise ValueError("not enough history")
        return self.returns[t - lookback + 1: t + 1]


def _log_ar1(rng, n_days, size, phi, sigma):
    """exp of a stationary AR(1), normalised so its mean is about 1."""
    h = np.empty((n_days, size))
    s0 = sigma / np.sqrt(1 - phi ** 2)
    h[0] = rng.normal(0, s0, size)
    eps = rng.normal(0, sigma, (n_days, size))
    for t in range(1, n_days):
        h[t] = phi * h[t - 1] + eps[t]
    return np.exp(h - 0.5 * s0 ** 2)


def make_universe(cfg: UniverseConfig = UniverseConfig()) -> Universe:
    rng = np.random.default_rng(cfg.seed)
    N, T, S, C = cfg.n_stocks, cfg.n_days, cfg.n_sectors, cfg.n_countries
    dpy = cfg.days_per_year
    n_reviews = len(range(0, T, cfg.review_every))

    # --- static layout -------------------------------------------------------
    sector_p = rng.dirichlet(np.full(S, 4.0))
    sector = rng.choice(S, size=N, p=sector_p)
    cs = np.asarray(cfg.country_shares[:C], float)
    country = rng.choice(C, size=N, p=cs / cs.sum())
    log_cap0 = rng.normal(0.0, 1.3, N)
    size_z = (log_cap0 - log_cap0.mean()) / log_cap0.std()

    beta_m = rng.uniform(0.5, 1.5, N)
    beta_s = rng.uniform(0.6, 1.2, N)
    beta_c = rng.uniform(0.4, 1.0, N)
    beta_z = -0.4 * size_z                      # small caps load on the size factor
    sector_vol = np.sort(rng.uniform(0.06, 0.22, S))[::-1]  # sector 0 is the most volatile
    country_vol = rng.uniform(0.05, 0.12, C)
    size_vol = 0.05
    # large caps are quieter: idiosyncratic vol 15% .. 45% by size rank, plus noise
    rank = np.argsort(np.argsort(-log_cap0)) / (N - 1)
    idio_vol = 0.15 + 0.30 * rank + rng.normal(0, 0.03, N)
    idio_vol = np.clip(idio_vol, 0.10, None)

    # --- time-varying volatility ---------------------------------------------
    mkt_sv = _log_ar1(rng, T, 1, 0.985, 0.06)[:, 0]
    for start, length, mult in cfg.crises:
        mkt_sv[start: start + length] *= mult
    sec_sv = _log_ar1(rng, T, S, 0.98, 0.05)
    idio_sv = _log_ar1(rng, T, N, 0.97, 0.04)

    sq = np.sqrt(dpy)
    f_m = rng.standard_normal(T) * cfg.market_vol / sq * mkt_sv + cfg.market_drift / dpy
    f_s = rng.standard_normal((T, S)) * sector_vol / sq * sec_sv
    for s, start, length, drift in cfg.booms:
        f_s[start: start + length, s] += drift / dpy
    f_c = rng.standard_normal((T, C)) * country_vol / sq
    f_z = rng.standard_normal(T) * size_vol / sq
    eps = rng.standard_normal((T, N)) * idio_vol / sq * idio_sv

    r = (f_m[:, None] * beta_m + f_s[:, sector] * beta_s + f_c[:, country] * beta_c
         + f_z[:, None] * beta_z + eps)
    r = np.clip(r, -0.9, None)

    var_t = ((cfg.market_vol * mkt_sv[:, None] * beta_m) ** 2
             + (sector_vol[sector] * sec_sv[:, sector] * beta_s) ** 2
             + (country_vol[country] * beta_c) ** 2
             + (size_vol * beta_z) ** 2 + (idio_vol * idio_sv) ** 2)
    true_vol = np.sqrt(var_t)

    # --- membership calendar ---------------------------------------------------
    join_review = np.zeros(N, dtype=int)
    leave_review = np.full(N, n_reviews, dtype=int)
    late = rng.choice(N, size=int(round(cfg.late_joiner_share * N)), replace=False)
    join_review[late] = rng.integers(1, n_reviews - 1, size=late.size)
    rest = np.setdiff1d(np.arange(N), late)
    leavers = rng.choice(rest, size=int(round(cfg.leaver_share * N)), replace=False)
    leave_review[leavers] = rng.integers(1, n_reviews - 1, size=leavers.size)
    for k, share in cfg.mass_exits:
        live = np.flatnonzero((join_review <= k) & (leave_review > k))
        out = rng.choice(live, size=int(round(share * live.size)), replace=False)
        leave_review[out] = k

    review_days = np.arange(0, T, cfg.review_every)
    days = np.arange(T)[:, None]
    first_day = np.where(join_review > 0, review_days[np.minimum(join_review, n_reviews - 1)], 0)
    # a joiner lists one quarter before it enters the parent, a leaver stops trading
    # after the review that removes it
    first_day = np.where(join_review > 0, first_day - cfg.review_every, 0)
    last_day = np.where(leave_review < n_reviews,
                        review_days[np.minimum(leave_review, n_reviews - 1)], T - 1)
    listed = (days >= first_day) & (days <= last_day)
    r = np.where(listed, r, np.nan)
    r[np.arange(T)[:, None] == first_day[None, :]] = np.nan  # no return on the listing day

    growth = np.cumprod(1.0 + np.nan_to_num(r), axis=0)
    caps = np.where(listed, np.exp(log_cap0) * growth, np.nan)
    return Universe(cfg=cfg, returns=r, caps=caps, sector=sector, country=country,
                    join_review=join_review, leave_review=leave_review, true_vol=true_vol)

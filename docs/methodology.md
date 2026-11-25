# Methodology

The rules behind every number in the README, and what was decided where.

## Universe (`universe.py`)

One synthetic universe, seed 7: 500 stocks, 11 sectors, 6 countries, 13 years of 252 trading days, a review every 63 days (52 reviews, numbered 0-51).

- **Layout.** Sector shares drawn from a Dirichlet; country shares 55 / 15 / 10 / 8 / 7 / 5%; log market caps normal with sd 1.3, so a few names dominate the parent (the largest is about 5% at the first build).
- **Returns.** `r = beta_m f_market + beta_s f_sector + beta_c f_country + beta_z f_size + e`. Market vol 15% a year with a 6% drift; sector vols 6-22% (sector 0 is the most volatile); country vols 5-12%; size factor 5%, loaded negatively by large caps; stock-specific vol 15-45% rising as size falls.
- **Time-varying volatility.** The market factor, each sector factor and each stock's specific term are scaled by their own exp(log-AR(1)) processes (daily persistence 0.97-0.985).
- **Planted events.** Two market-volatility crises (days 1100-1139 at 3x, days 2850-2879 at 2.5x), a boom in the most volatile sector (days 2300-2425, +80% a year of extra drift), and two parent reconstitutions where 8% of the members leave at once (reviews 20 and 42). They are planted so that the rules come under stress in both periods. They are not calibrated to any real episode.
- **Constituent changes.** 15% of names list late (one quarter before they join the parent, so they have little history); 10% leave the parent at a random review and stop trading after it. Membership changes only at reviews.
- **Parent index.** Cap-weighted members, drifting with prices between reviews.

## Point-in-time discipline

Review `k` happens at the close of day `t_k`. The solve reads only:

- parent weights from closing caps at `t_k`;
- the covariance window, days `t_k - L + 1 .. t_k`;
- membership decided at review `k`;
- the index holdings drifted from the last rebalance to `t_k` with realised returns up to `t_k`.

New weights earn from day `t_k + 1`. `test_review_uses_no_data_after_its_close` replaces every return and cap after `t_k` with noise and checks that the published weights are bit-for-bit unchanged. Names need a full window of returns to be eligible, so a late listing cannot be held until it has `L` days of history.

## Risk model (`risk.py`)

Sample covariance of the demeaned window, optionally shrunk towards `mu I` with the Ledoit-Wolf (2004) intensity (checked against a term-by-term implementation in the tests). The solver's objective never contains the N x N matrix: variance is `(1 - delta) / T ||X w||^2 + delta mu ||w||^2`. The objective is scaled to (annualised vol in %)^2, so dual values are in those units.

## Rulebook (`rules.py`)

| rule | value | form |
|---|---|---|
| eligibility | parent member with a full covariance window (504 days, two years) | which names get a variable |
| stock cap | min(1.5%, 20 x parent weight) | `w_i <= cap_i`; names with cap below the minimum weight are not holdable |
| minimum holding | 0.05% | support step + `w_i >= 0.0005` on the kept names |
| minimum names | 120 | support step |
| sector band | parent sector weight +/- 5 points, floored at 0 | two rows per sector |
| country band | parent country weight +/- 5 points, floored at 0 | two rows per country |
| turnover | one-way, 10% per review, against drifted holdings; none at the first build | `0.5 sum |w - w_drift| <= 0.10` |
| budget, long-only | | `sum w = 1`, `w >= 0` |

The ladder (turnover, then country band, then sector band, then minimum names) is described in [rulebook.md](rulebook.md).

## Optimiser (`optimizer.py`)

cvxpy with the CLARABEL interior-point solver, pinned. P0, the support step and P1 are described in [rulebook.md](rulebook.md). A solve counts only if the status is `optimal`, or `optimal_inaccurate` with every constraint residual at most 1e-8; anything else is treated as infeasible. Every inaccurate or failed solve is logged in the review record (`solver_notes`).

The **unconstrained** book is the same objective with only the budget and long-only constraints, over the same eligible names and the same covariance. It has no turnover limit, so its turnover is large.

## Binding rules and rule costs

- **Binding** comes from the final convex solve: slack at most 1e-6 and a dual value above 1e-6. For the minimum-names rule there is no dual; the report counts the reviews where the top-up was needed.
- **`dual_per_pp`** is the family's summed dual value times 0.01: the first-order fall in ex-ante variance, in (vol %)^2, if every bound in the family moved out by one percentage point. `test_duals_match_a_finite_difference` checks it against a central finite difference for the sector, country and turnover families (within 2%).
- **Leave-one-rule-out cost**: at every review of the frozen run, the same context is re-solved with one rule family removed (stock caps, sector bands, country bands, turnover, holding rules) and with all of them removed. The re-solve starts from the review's effective rulebook (after any ladder relaxation), so at review 42 every variant that keeps a turnover rule uses the relaxed 20% limit. Cost = ex-ante vol of the index minus ex-ante vol without the rule, in vol points. These are ex-ante numbers on the index's own state; they are not separate backtests.

## Development vs held-out

| | reviews | days | used for |
|---|---|---|---|
| burn-in | 0-7 | 0-503 | covariance history only |
| development | 8-31 | 504-2016 | the one data-driven choice below, the rulebook values, the support rule, the ladder |
| held-out | 32-51 | 2017-3275 | the reported held-out tables, with the three disclosed exceptions below |

The held-out run is one continuous path: its first review starts from the drifted development book, as a live index would.

**The data-driven choice.** Covariance window (126, 252 or 504 days) x shrinkage (Ledoit-Wolf or none), scored by the realised volatility of the rule-constrained index over the development reviews, rounded to 0.01 vol points. The lowest wins; ties go to Ledoit-Wolf shrinkage, then to the longer window. It is made in code from development records (`results/dev_choices.json`), and `test_frozen_run_reproduces_the_development_run_of_its_config` checks that the development half of the final run is identical to the run the choice was made on.

**Made by hand while building, before the final run:**

- The rulebook values are scenario settings, not tuned for performance. `min_names` was first 100. A prototype run over all reviews showed that P0 alone always held more than 100 names, so the rule never did anything; it was raised to 120, a value inside the range of P0 support sizes seen over the development reviews (107-133 in that prototype). Because that prototype covered the held-out reviews too, I state it here: the held-out results of that prototype were seen, but only support sizes and ladder events were inspected, and nothing was chosen to improve a held-out number.
- The second support attempt ("round up") was added after the same prototype hit a review (in the held-out period) where rounding small P0 weights down made a sector's lower band unreachable, and the fixed-order ladder then relaxed turnover and country bands that were not in conflict. The fix is a correctness change to the heuristic, not a tuned parameter.
- **The tie rule was added after the first complete run.** That run used plain "lowest realised volatility" and picked 504 days with no shrinkage, ahead of 504 days with Ledoit-Wolf by 0.0001 vol points (12.8468% against 12.8469%), a difference far below anything one path can resolve. Its held-out run had already been computed and its numbers seen. I then added the rounding and the tie-break towards shrinkage (the estimator the scenario asks for, and the one whose forecasts are better conditioned), reran everything, and the committed results come from that second run. The held-out numbers of the two runs differ only slightly, but this is a choice made after held-out numbers were seen, and I state it as such.
- **Measured effect of all three.** `make bench` reruns the full path under the first run's choice (504 days, no shrinkage) and under `min_names = 100`, and writes them to `results/sensitivity.json`. Held-out volatility: 12.397% committed, 12.397% under the first run's choice, 12.395% with `min_names = 100`; tracking error 7.428%, 7.433% and 7.432%. The round-up support retry fired in 0 of 44 reviews of the committed run (`support_round_up` in `results/rules.json`). None of the three decisions moves a held-out headline number.
- The planted reconstitutions were added at the same time as the round-up support attempt, so that the turnover rule and the ladder are exercised by the data and not only by the unit tests.

## Metrics (`metrics.py`, `bench.py`)

- Annualised return: geometric, `prod(1 + r)^(252 / days) - 1`. Volatility: daily sd x sqrt(252). Tracking error: the same on daily active returns. Beta: daily covariance over parent variance. Max drawdown on the daily level.
- Turnover: one-way, against drifted holdings, per review, averaged over reviews (the first build is excluded).
- Sector and country deviation: per review, the largest absolute difference between the book's and the parent's group weight, in percentage points; mean and max over reviews.
- Realized / ex-ante: realised volatility over the period divided by the mean ex-ante forecast at its reviews; above 1 means the risk model under-forecast.

There are no standard errors in the tables. There is one universe and one path, the daily returns are autocorrelated in volatility, and the periods are 6 and 5 years; differences of a few tenths of a vol point between configurations should be read as ties.

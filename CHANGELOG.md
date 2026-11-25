# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-25

### Added

- Synthetic universe: 500 stocks, 11 sectors, 6 countries, factor-model returns (market, sector, country, size, specific) with stochastic volatility, cap-weighted parent index, late listings and leavers at quarterly reviews, and planted stress events (volatility crises, a sector boom, two parent reconstitutions).
- Ledoit-Wolf shrunk covariance passed to the solver in factor form, so the objective holds no N x N matrix (the shrinkage intensity itself is computed from the N x N sample covariance).
- Rulebook as data: stock cap tied to the parent weight, minimum holding weight, minimum number of names, sector and country bands around the parent, one-way turnover against drifted holdings, and a fixed-order relaxation ladder.
- cvxpy (CLARABEL) minimum-variance rebalance in two convex stages with a support step for the non-convex holding rules; every ladder rung is logged; inaccurate solves are accepted only after a residual check.
- Independent numpy rule checker with an absolute tolerance, run against the effective and the original rulebook.
- Binding rules and first-order rule costs from dual values (sector, country and turnover duals checked against finite differences), and leave-one-rule-out ex-ante costs at every review.
- Backtest against the parent and an unconstrained minimum-variance book, with a development / held-out split by time and one data-driven choice (covariance window and shrinkage) made on development reviews; a descriptive turnover-limit sweep; sensitivity runs that measure the decisions made after held-out reviews were seen (`results/sensitivity.json`).
- Tests: unit tests, hypothesis property tests on random universes, a look-ahead test, an SLSQP cross-check, summaries recomputed from `results/runs.json`, every published book re-verified, a subset of reviews re-solved from scratch (all of them with `FULL=1`); `constraints.txt` with exact versions; CI on ubuntu (Python 3.11) with a secret scan.
- Docs: methodology, walkthrough, and reading a rulebook into constraints.

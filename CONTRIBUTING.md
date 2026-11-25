# Contributing

Thanks for taking a look. Changes are judged on whether they keep the repo offline, honest about its numbers, free of look-ahead, and every published book checkable.

## Ground rules

1. **Everything in CI runs offline and on CPU.** Tests and `make bench` need no network and no GPU.
2. **Numbers come from results.** Any number in a doc must be a field in a `results/*.json` file written by `make bench`. If you change code that affects results, rerun `make bench`, commit the new JSON and `RESULTS.md` with the change, and run `make verify`.
3. **Never tune on held-out reviews.** The covariance window, the shrinkage and any new design choice are made on development reviews (8-31). If a change needs a new choice after held-out results exist, make it on development reviews and record the history in `docs/methodology.md`.
4. **No look-ahead.** A review may read only data known at its close. `tests/test_backtest.py::test_review_uses_no_data_after_its_close` scrambles the future and checks the solve does not move; keep it passing.
5. **Every rule has two implementations.** The optimiser's constraint and the checker's re-test are written separately; the checker never imports cvxpy or reads solver output.
6. **No timing assertions in tests.** CI runners are shared and slow; tests check correctness, never speed.
7. **Determinism.** Generators take a seed, BLAS runs on one thread (set in `ixrebal/__init__.py`, the Makefile and CI), and the solver is pinned to CLARABEL.
8. **No secrets.** Nothing here needs credentials; `.env` is git-ignored and CI runs a secret scan.

## Set up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -c constraints.txt -e ".[dev]"
pytest -q
```

## Adding a rule

Follow the checklist at the end of [docs/rulebook.md](docs/rulebook.md): constraint in `optimizer.solve_convex`, re-test in `checker.verify`, dual report in `optimizer._binding`, a leave-one-rule-out entry in `backtest.DROP_SETS`, a ladder position if it may relax, and a unit test that the checker catches a violation.

.PHONY: install test verify bench report demo

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

install:
	pip install -c constraints.txt -e ".[dev]"

# Unit and property tests; summaries recomputed from results/runs.json; a subset of reviews re-solved.
test:
	pytest -q

# Re-solve every review of the frozen run and check it against results/runs.json (several minutes).
verify:
	FULL=1 pytest -q tests/test_results.py

# Offline, deterministic: development choice, then the frozen run over all reviews,
# the turnover sweep and the held-out sensitivity runs.
# Writes results/{runs,dev_choices,backtest,rules,sweep,sensitivity}.json and RESULTS.md.
bench:
	python -m ixrebal.cli bench --out results

report:
	python -m ixrebal.cli report --out results

demo:
	python -m ixrebal.cli demo

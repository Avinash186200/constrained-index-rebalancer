# Walkthrough

From install to reading one review's output, then running a rebalance on your own data.

## 1. Install and test

```bash
python -m venv .venv && source .venv/bin/activate
pip install -c constraints.txt -e ".[dev]"
pytest -q
```

The tests run on CPU. They include hypothesis property tests on random small universes, a look-ahead test, a re-verification of every committed book, and a re-solve of a subset of reviews from `results/runs.json`.

## 2. One build and one rebalance

```bash
python -m ixrebal.cli demo
```

Output (seed 7, the committed universe):

```
review 8 (inception build): 120 names, ex-ante vol 10.93%, checker ok=True
review 9: 429 parent members, 420 eligible, 120 names held, turnover 10.0% (limit 10%), ex-ante vol 11.02%
checker ok=True, ladder=not needed
binding (family: binding/constraints, variance saving per +1pp):
  cap          76/120  30.572
  sector        1/22   0.293
  country       4/12   0.477
  turnover      1/1    0.056
  min_weight   11/120  0.978
ex-ante vol with one rule family removed:
  cap       10.67%  (cost +0.354)
  sector    11.00%  (cost +0.020)
  country   11.01%  (cost +0.014)
  turnover  11.02%  (cost +0.002)
  holding   11.02%  (cost +0.001)
  all       10.01%  (cost +1.014)
```

How to read it:

- **Inception build.** There are no holdings yet, so there is no turnover rule. P0 held fewer than 120 names, so the support step topped it up to exactly 120.
- **Review 9.** The book from review 8 has drifted for a quarter. Rebalancing it uses the whole 10% turnover budget (the turnover constraint binds). 9 of the 429 parent members are not eligible: they listed recently and lack two years of returns.
- **Binding.** 76 of the 120 held names sit at their cap. The caps are what most separates this index from unconstrained minimum variance: removing them alone lowers ex-ante vol by 0.35 points, removing everything by 1.01 points.
- **`dual_per_pp`** is a first-order number: for caps it says what loosening all 120 caps by one point each would save if the dual stayed constant. That is a large move, so the linear extrapolation (30.6 (vol %)^2) overstates the real effect of dropping caps (11.02^2 - 10.67^2 = 7.6 (vol %)^2). Duals are good for ranking rules and for small changes; leave-one-rule-out is the honest number for removing a rule.
- **Costs do not add up.** The single-rule costs sum to about 0.39 points; removing all rules saves 1.01. Rules overlap: with the caps gone, the sector and country bands still keep the book spread out.

## 3. Rebuild every committed number

```bash
make bench      # several minutes; writes results/*.json and RESULTS.md
make verify     # recompute runs.json from scratch and compare with the committed file
```

`make bench` first runs the six development configurations over reviews 8-31, chooses one by code, then runs the frozen configuration over all 44 reviews with leave-one-rule-out costs, the unconstrained book, the parent and the turnover sweep.

## 4. Run a rebalance on your own data

You need, at one review date: a (T x N) window of daily returns with no gaps for the names you allow, parent weights, sector and country codes, and the current holdings drifted to today's prices.

```python
import numpy as np
from ixrebal.risk import estimate
from ixrebal.rules import Context, Rulebook
from ixrebal.optimizer import rebalance, RebalanceInfeasible
from ixrebal.checker import verify

eligible = ~np.isnan(window).any(axis=0)           # plus your own eligibility rules
cov = estimate(window[:, eligible], "ledoit_wolf")
X = np.zeros((window.shape[0], window.shape[1])); X[:, eligible] = cov.X; cov.X = X

ctx = Context(parent=parent_w, eligible=eligible, drifted=current_w,
              sector=sector_codes, country=country_codes,
              n_sectors=sector_codes.max() + 1, n_countries=country_codes.max() + 1)
rb = Rulebook(cap_abs=0.02, cap_mult=10, sector_band=0.03, turnover=0.08)

try:
    res = rebalance(cov, ctx, rb)
except RebalanceInfeasible as e:
    print("could not publish; ladder used:", e.ladder)
else:
    report = verify(res.weights, ctx, res.rulebook)
    print(report.ok, res.ladder, res.binding)
    print(report.by_rule())            # min slack per rule, for the publication log
```

`drifted=None` means a first build with no turnover rule. To change the order of relaxation, pass your own `ladder=(LadderStep(...), ...)` to `Rulebook`.

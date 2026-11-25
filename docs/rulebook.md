# Reading a rulebook into constraints

An index methodology document is written for people: "the weight of each
constituent is capped at the lower of 1.5% and 20 times its weight in the
parent index". An optimiser needs a matrix. This page goes through the rules
used here, one at a time, and shows the sentence, the arithmetic, the cvxpy
line, and how the checker re-tests it. The rule values are illustrative, not
any index provider's.

## Step 0: decide what the variables are

`w` is the vector of new weights, one per stock that is allowed to be held at
this review. Everything not in that vector is fixed at zero. Three inputs come
from the review date, all known at that day's close:

| input | meaning |
|---|---|
| `p` | parent index weights after the review (cap weights of the members) |
| `w_drift` | the current index holdings, drifted with prices since the last rebalance (share counts times today's prices, normalised) |
| `Sigma` | a shrunk covariance estimated from the trailing window |

The word "current" in a rulebook almost always means `w_drift`, not the
weights published last quarter. Turnover against last quarter's published
weights would charge the index for price moves it never traded.

## Step 1: sort sentences into five kinds

| kind | example sentence | becomes |
|---|---|---|
| eligibility | "constituents of the parent with at least one year of price history" | which names get a variable at all |
| bound on one weight | "capped at the lower of 1.5% and 20x its parent weight" | box constraint per stock |
| bound on a sum | "sector weights within 5 points of the parent's" | one linear row per group, two inequalities |
| path-dependent | "one-way turnover at most 10% per review" | a constraint involving `w_drift` |
| combinatorial | "at least 120 constituents", "no weight below 0.05%" | not convex: handled outside the solver |

The last row is the one to notice early. Count and minimum-holding rules make
the problem mixed-integer. This repo handles them with a support step (below)
and says so; it does not pretend they are linear.

## Step 2: the rules here

### Eligibility

> A parent constituent is eligible if it has a full covariance window of daily
> returns (504 days, two years, in the committed run).

```python
eligible = members & ~np.isnan(window).any(axis=0)
```

Ineligible members still count in the parent's sector and country weights, so
the bands are relative to the whole parent, not to the eligible part. That can
make a band unreachable (a sector full of new listings), which is one way the
problem becomes infeasible.

### Stock cap

> Each constituent weighs at most min(1.5%, 20 x parent weight).

```python
cap = np.minimum(rb.cap_abs, rb.cap_mult * p)
cap = np.where(eligible & (cap >= rb.min_weight), cap, 0.0)   # cannot be held
cons["cap"] = w <= cap[idx]
```

The second line is an interaction between two rules: a name whose cap is below
the minimum holding weight can never be held, so it gets no variable.

### Sector and country bands

> Each sector's weight is within +/- 5 percentage points of its parent weight
> (floored at zero). The same applies to countries.

```python
G = one_hot(sector)[idx].T                    # (n_sectors x n)
cons["sector_hi"] = G @ w <= p_sector + band
cons["sector_lo"] = G @ w >= np.maximum(0, p_sector - band)
```

Real methodologies often use a relative band for small groups ("within 5
points or 3x the parent weight, whichever is tighter"). That is still one
number per group: compute it first, then write the same two rows.

### Turnover against drifted holdings

> One-way turnover at a review is at most 10%.

One-way turnover is `0.5 * sum |w - w_drift|`. The absolute value is convex,
so cvxpy accepts it directly; under the hood it adds one auxiliary variable per
name (`t_i >= w_i - w_drift_i`, `t_i >= w_drift_i - w_i`). Names that are not in
`w` (they left the parent, or they are no longer eligible) must be sold in
full, and that sale counts:

```python
outside = w_drift.sum() - w_drift[idx].sum()
cons["turnover"] = 0.5 * (cp.sum(cp.abs(w - w_drift[idx])) + outside) <= rb.turnover
```

At the first build there is no `w_drift`, and no turnover rule.

### Budget and long-only

```python
cons["budget"] = cp.sum(w) == 1
cons["floor"]  = w >= 0          # or >= min_weight on the kept support, see below
```

### Minimum holding weight and minimum number of names

> No constituent weighs less than 0.05%. The index holds at least 120 names.

Both are "either zero or at least x" statements. The support step:

1. Solve the convex problem with only `w >= 0` (P0).
2. Keep the names P0 gives at least half the minimum weight. If fewer than
   `min_names` are kept, add the next names by P0 weight, then by lowest own
   variance.
3. Re-solve on that support with `w >= min_weight` (P1). If P1 is infeasible,
   round small P0 weights up instead of down and try once more.

P1 is a convex problem, so its dual values are exact for P1. The support is a
heuristic: a mixed-integer solver could find a lower-variance book with a
different set of names.

## Step 3: write the order of relaxation down

A rulebook should say what happens when the rules cannot all hold. Here that
is a ladder, read in order:

| rung | rule | step | never beyond |
|---|---|---|---|
| 1 | turnover limit | +5 points | 30% |
| 2 | country band | +2.5 points | 10 points |
| 3 | sector band | +2.5 points | 10 points |
| 4 | minimum names | -10 names | 50 |

Each failed attempt relaxes the first rule in the ladder that is active and
not yet at its limit, by one step, and the whole rebalance is retried. Every
rung is logged with the stage that failed. If the ladder runs out, the
rebalance raises instead of publishing. The ladder is fixed-order: it does not
look for the rule that is actually in conflict, so it can relax a rule that did
not need it (`test_exhausted_ladder_raises_with_the_log` relaxes turnover and country bands on the way to a sector conflict). An elastic formulation that
minimises the weighted relaxation in one solve is the obvious next step.

## Step 4: check every rule again, without the solver

`checker.verify` takes the published weights, the parent and the drifted
holdings, recomputes every constrained quantity with numpy, and compares with
the effective rule plus an absolute tolerance of 1e-7. It reports each check
with its value, its bounds and its slack. It is run twice per review: against
the effective rules (must pass) and against the original rulebook (any failure
there must be a rule the ladder relaxed: a subset of the relaxed rules).

The checker trusts its inputs: the parent weights, the eligible set and the drifted
holdings. Those are covered separately, by the drift (share-count bookkeeping),
eligibility and look-ahead tests in `tests/test_backtest.py`, and the results test
rebuilds them from the generator rather than taking them from the solver.

## Step 5: say which rules bind, and what they cost

From the final solve:

- **Binding**: a constraint is at its bound (slack under 1e-6) and its dual
  value is positive.
- **Marginal cost** (`dual_per_pp`): the sum of a family's dual values times
  0.01, the first-order fall in ex-ante variance, in (vol %)^2, if every bound
  in the family moved out by one percentage point. A test checks it against a
  finite difference for the sector, country and turnover families (not caps).
- **Total cost** (leave-one-rule-out): re-solve the same review with one rule
  family removed and compare ex-ante volatility. The re-solve starts from the
  effective rulebook of that review, so a review the ladder relaxed is compared
  against the same relaxed rules minus one family. This is not additive across
  rules, and because of the support step and the ladder it can even be
  negative; both are reported.

## Checklist for a new rule

1. Which kind is it (eligibility, box, group sum, path-dependent, combinatorial)?
2. Which inputs does it read, and are they all known at the review close?
3. Write the cvxpy line in `optimizer.solve_convex` and give it a name in `cons`.
4. Add the independent re-check to `checker.verify`.
5. Add it to `_binding` so its dual is reported, and to the leave-one-rule-out set.
6. Decide where it sits in the ladder, if it may relax at all.
7. Add a unit test that the checker flags a violation, and rerun `make bench`.

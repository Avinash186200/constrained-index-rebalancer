"""The committed results are reproducible and leak-free.

* Every summary and RESULTS.md is recomputed from results/runs.json.
* Every published index book is re-verified by the independent checker, with the
  context rebuilt from the generator and the previous published book.
* A fixed subset of reviews is re-solved from scratch (the inception build, the
  first held-out review, every review that needed the ladder, the last review,
  plus unconstrained and development reviews) and must match its record.
  FULL=1 recomputes the whole of runs.json instead (several minutes; this is
  what ``make verify`` does).
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from ixrebal import bench as B
from ixrebal.backtest import (DEV_LAST_REVIEW, FIRST_REVIEW, RunConfig, context_at, dense,
                              drift, run_path)
from ixrebal.checker import verify
from ixrebal.optimizer import replay_ladder
from ixrebal.rules import Rulebook
from ixrebal.universe import UniverseConfig, make_universe

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FULL = os.environ.get("FULL") == "1"
TOL = 1e-6


def _diff(a, b, path="$"):
    """Differences between two JSON trees; floats match to TOL absolute."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(a.keys() | b.keys()):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: missing")
            else:
                out += _diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]")]
    if isinstance(a, bool) or isinstance(b, bool):
        return [] if a == b else [f"{path}: {a!r} != {b!r}"]
    if isinstance(a, float) or isinstance(b, float):
        # duals are summed over ~120 constraints and can be ~40: compare them relatively
        rel = 1e-6 if path.endswith("dual_per_pp") else 0.0
        ok = isinstance(a, (int, float)) and isinstance(b, (int, float)) and \
            math.isclose(a, b, rel_tol=rel, abs_tol=TOL)
        return [] if ok else [f"{path}: {a} != {b}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def _norm(x):
    return json.loads(json.dumps(x))


@pytest.fixture(scope="module")
def runs():
    return json.loads((RES / "runs.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def universe(runs):
    cfg = UniverseConfig()
    assert _norm(cfg.to_dict()) == runs["universe"]
    return make_universe(cfg)


@pytest.mark.parametrize("name", B.SUMMARY_FILES)
def test_summaries_are_computed_from_the_records(runs, name):
    fresh = _norm(B.SUMMARIZERS[name](runs))
    committed = json.loads((RES / f"{name}.json").read_text(encoding="utf-8"))
    diffs = _diff(fresh, committed)
    assert not diffs, "\n".join(diffs[:20])


def test_results_md_is_rendered_from_the_summaries():
    assert B.render(RES) == (RES / "RESULTS.md").read_text(encoding="utf-8")


def test_results_written_with_lf_only_and_small():
    for f in RES.iterdir():
        assert b"\r\n" not in f.read_bytes(), f.name
        assert f.stat().st_size < 1_000_000, f.name


def test_library_defaults_are_the_frozen_choices(runs):
    dev = json.loads((RES / "dev_choices.json").read_text(encoding="utf-8"))
    assert RunConfig().to_dict() == dev["chosen"] == runs["index"]["config"]
    assert _norm(Rulebook().to_dict()) == runs["rulebook"] == runs["index"]["rulebook"]


def test_development_runs_never_see_held_out_reviews(runs):
    for path in runs["dev"].values():
        ks = [r["review"] for r in path["records"]]
        assert ks == list(range(FIRST_REVIEW, DEV_LAST_REVIEW + 1))
    held = [r for r in runs["index"]["records"] if r["review"] > DEV_LAST_REVIEW]
    assert held and all(r["period"] == "held_out" for r in held)


def test_frozen_run_reproduces_the_development_run_of_its_config(runs):
    """The development half of the final run is the run the choice was made on."""
    key = RunConfig().key
    dev = runs["dev"][key]["records"]
    main = [r for r in runs["index"]["records"] if r["period"] == "development"]
    assert len(dev) == len(main)
    for a, b in zip(dev, main):
        for f in ("exante_vol", "turnover", "names", "daily", "ladder", "binding"):
            assert not _diff(a[f], b[f]), (a["review"], f)


def test_every_published_book_passes_the_checker(runs, universe):
    """Contexts rebuilt from the generator and the previous published book, not from the solver."""
    rb = Rulebook()
    rd = universe.review_days
    recs = runs["index"]["records"]
    prev = None
    for rec in recs:
        k = rec["review"]
        drifted = None if prev is None else drift(universe, prev, int(rd[k - 1]), int(rd[k]))[0]
        ctx, _ = context_at(universe, k, RunConfig(), drifted)
        w = dense(rec["weights"], universe.n_stocks)
        eff = replay_ladder(rb, rec["ladder"])
        rep = verify(w, ctx, eff)
        assert rep.ok, (k, [(c.rule, c.item, c.value, c.lower, c.upper) for c in rep.failures])
        if drifted is not None:
            assert rec["turnover"] == pytest.approx(0.5 * np.abs(w - drifted).sum(), abs=1e-9)
        prev = w


def _reruns(runs, universe):
    rb, cfg = Rulebook(), RunConfig()
    idx = runs["index"]["records"]
    first_held = next(i for i, r in enumerate(idx) if r["period"] == "held_out")
    pick = {0, 1, first_held, len(idx) - 1}
    pick |= {i for i, r in enumerate(idx) if r["ladder"]}
    jobs = [("index", i, True) for i in sorted(pick)]
    jobs += [("unconstrained", i, False) for i in (0, 1, len(idx) - 1)]
    out = []
    for variant, i, costs in jobs:
        recs = runs[variant]["records"]
        start = None if i == 0 else dense(recs[i - 1]["weights"], universe.n_stocks)
        p = run_path(universe, rb, cfg, variant, ks=[recs[i]["review"]], costs=costs,
                     start_weights=start)
        out.append(((variant, i), p["records"][0]))
    # the first two reviews of every development configuration
    for key, path in runs["dev"].items():
        c = RunConfig(path["config"]["lookback"], path["config"]["shrink"])
        p = B._strip_weights(run_path(universe, rb, c, ks=[FIRST_REVIEW, FIRST_REVIEW + 1]))
        for j, r in enumerate(p["records"]):
            out.append((("dev", key, j), r))
    return out


@pytest.mark.skipif(FULL, reason="FULL=1 recomputes everything in test_full_recompute")
def test_rerun_reviews_match_their_records(runs, universe):
    reruns = _reruns(runs, universe)
    assert any(k[0] == "index" and runs["index"]["records"][k[1]]["ladder"] for k, _ in reruns), \
        "a review that needed the ladder must be re-solved"
    diffs = []
    for key, fresh in reruns:
        if key[0] == "dev":
            committed = runs["dev"][key[1]]["records"][key[2]]
        else:
            committed = runs[key[0]]["records"][key[1]]
        diffs += _diff(_norm(fresh), committed, str(key))
    assert not diffs, "\n".join(diffs[:20])


@pytest.mark.skipif(not FULL, reason="set FULL=1 (make verify)")
def test_full_recompute():
    fresh = _norm(B.compute_runs(log=lambda *_: None))
    committed = json.loads((RES / "runs.json").read_text(encoding="utf-8"))
    diffs = _diff(fresh, committed)
    assert not diffs, "\n".join(diffs[:20])

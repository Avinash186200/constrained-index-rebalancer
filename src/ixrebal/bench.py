"""Full benchmark: development choice, frozen held-out run, rule costs, turnover sweep.

``run`` writes ``results/runs.json`` (every per-review record) and then the
summaries, each a pure function of runs.json, and RESULTS.md, a pure function of
the summaries. The tests recompute the summaries and RESULTS.md from the
committed files and rerun a subset of reviews from scratch.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import metrics as M
from .backtest import (DEV_LAST_REVIEW, DROP_SETS, FIRST_REVIEW, RunConfig, parent_path,
                       reviews, run_path)
from .rules import Rulebook
from .universe import UniverseConfig, make_universe

DEV_CONFIGS = tuple(RunConfig(L, s) for L in (126, 252, 504) for s in ("ledoit_wolf", "none"))
TURNOVER_SWEEP = (0.05, 0.20, 1.0)       # the rulebook's own 0.10 is the main run
PERIODS = ("development", "held_out", "full")
SUMMARY_FILES = ("dev_choices", "backtest", "rules", "sweep", "sensitivity")


def _strip_weights(path: dict) -> dict:
    for r in path["records"]:
        r.pop("weights", None)
    return path


SLIM_FIELDS = ("review", "period", "exante_vol", "turnover", "names", "ladder",
               "support_rounding", "check_ok", "daily")


def _slim(path: dict) -> dict:
    """Keep only what the sensitivity summary needs (runs.json must stay under 1 MB)."""
    path["records"] = [{k: r[k] for k in SLIM_FIELDS} for r in path["records"]]
    return path


def _write_json(path: Path, obj, indent=1) -> None:
    path.write_text(json.dumps(obj, indent=indent, sort_keys=False) + "\n",
                    encoding="utf-8", newline="\n")


def compute_runs(log=print) -> dict:
    ucfg = UniverseConfig()
    u = make_universe(ucfg)
    rb = Rulebook()
    runs = {"universe": ucfg.to_dict(), "rulebook": rb.to_dict(),
            "first_review": FIRST_REVIEW, "dev_last_review": DEV_LAST_REVIEW, "dev": {}}
    dev_ks = reviews(u, stop=DEV_LAST_REVIEW)
    for cfg in DEV_CONFIGS:
        log(f"development: {cfg.key}")
        runs["dev"][cfg.key] = _strip_weights(run_path(u, rb, cfg, ks=dev_ks))
    runs["parent_dev"] = parent_path(u, dev_ks)
    chosen = frozen_choice(dev_choices(runs))
    log(f"chosen: {chosen.key}; held-out run")
    runs["index"] = run_path(u, rb, chosen, costs=True)
    runs["unconstrained"] = run_path(u, rb, chosen, "unconstrained")
    runs["parent"] = parent_path(u)
    runs["sweep"] = {}
    for tau in TURNOVER_SWEEP:
        log(f"turnover sweep: {tau}")
        runs["sweep"][f"{tau:.2f}"] = _strip_weights(run_path(u, replace(rb, turnover=tau), chosen))
    # Held-out sensitivity to the three disclosed decisions made after held-out reviews were
    # seen (docs/methodology.md): the first full run's choice and the original min_names.
    # The third, the round-up support retry, is counted in rules.json (support_round_up).
    log("sensitivity runs")
    runs["sensitivity"] = {
        "first_run_choice_504_none": _slim(run_path(u, rb, RunConfig(504, "none"))),
        "min_names_100": _slim(run_path(u, replace(rb, min_names=100), chosen)),
    }
    return runs


def frozen_choice(dev: dict) -> RunConfig:
    d = dev["chosen"]
    return RunConfig(lookback=d["lookback"], shrink=d["shrink"])


# --------------------------------------------------------------------------- summaries
def dev_choices(runs: dict) -> dict:
    parent = M.daily_series(runs["parent_dev"])
    rows = []
    for key, path in runs["dev"].items():
        r = M.daily_series(path)
        recs = path["records"]
        rows.append({
            "lookback": path["config"]["lookback"], "shrink": path["config"]["shrink"],
            "realized_vol": M.ann_vol(r),
            "mean_exante_vol": float(np.mean([x["exante_vol"] for x in recs])),
            "mean_shrink_delta": float(np.mean([x["shrink_delta"] for x in recs])),
            "mean_turnover": M.mean_or_none(x["turnover"] for x in recs),
            "ladder_rungs": int(sum(len(x["ladder"]) for x in recs)),
        })
    # differences below 0.01 vol points are noise on one path: round, then prefer
    # Ledoit-Wolf shrinkage, then the longer window
    best = min(rows, key=lambda x: (round(100 * x["realized_vol"], 2),
                                    x["shrink"] != "ledoit_wolf", -x["lookback"]))
    return {"reviews": [runs["first_review"], runs["dev_last_review"]],
            "parent_realized_vol": M.ann_vol(parent), "rows": rows,
            "rule": ("lowest realized volatility of the rule-constrained index over the development "
                     "reviews, rounded to 0.01 vol points; ties go to Ledoit-Wolf shrinkage, then "
                     "to the longer window"),
            "chosen": {"lookback": best["lookback"], "shrink": best["shrink"]}}


def _sel(path, period):
    return [r for r in path["records"] if period == "full" or r["period"] == period]


def _daily(path, period):
    return M.daily_series(path, None if period == "full" else period)


def _book_stats(path, period) -> dict:
    recs = _sel(path, period)
    return {
        "reviews": len(recs),
        "mean_turnover": M.mean_or_none(r["turnover"] for r in recs),
        "mean_names": float(np.mean([r["names"] for r in recs])),
        "mean_max_sector_dev": float(np.mean([r["max_sector_dev"] for r in recs])),
        "max_sector_dev": float(np.max([r["max_sector_dev"] for r in recs])),
        "mean_max_country_dev": float(np.mean([r["max_country_dev"] for r in recs])),
        "max_country_dev": float(np.max([r["max_country_dev"] for r in recs])),
        "mean_exante_vol": float(np.mean([r["exante_vol"] for r in recs])),
    }


def backtest(runs: dict) -> dict:
    out = {"chosen": runs["index"]["config"]}
    for period in PERIODS:
        par = _daily(runs["parent"], period)
        idx = _daily(runs["index"], period)
        unc = _daily(runs["unconstrained"], period)
        out[period] = {
            "parent": M.series_stats(par),
            "index": {**M.series_stats(idx, par), **_book_stats(runs["index"], period)},
            "unconstrained": {**M.series_stats(unc, par), **_book_stats(runs["unconstrained"], period)},
            "index_vs_unconstrained_te": M.tracking_error(idx, unc),
        }
        for k in ("index", "unconstrained"):
            d = out[period][k]
            d["realized_over_exante"] = d["ann_vol"] * 100 / d["mean_exante_vol"]
    return out


def rules(runs: dict) -> dict:
    out = {}
    for period in PERIODS:
        recs = _sel(runs["index"], period)
        n = len(recs)
        fams = sorted({f for r in recs for f in r["binding"]})
        binding = {}
        for f in fams:
            got = [r["binding"][f] for r in recs if f in r["binding"]]
            binding[f] = {
                "reviews_with_rule": len(got),
                "share_reviews_binding": sum(g["n_binding"] > 0 for g in got) / n,
                "mean_n_binding": float(np.mean([g["n_binding"] for g in got])),
                "mean_constraints": float(np.mean([g["n"] for g in got])),
                "mean_dual_per_pp": float(np.mean([g["dual_per_pp"] for g in got])),
            }
        costs = {}
        for name in DROP_SETS:
            diffs = [r["exante_vol"] - r["vol_without"][name] for r in recs]
            costs[name] = {"mean_vol_cost": float(np.mean(diffs)),
                           "max_vol_cost": float(np.max(diffs)),
                           "reviews_negative": int(sum(d < -1e-9 for d in diffs))}
        single = [k for k in DROP_SETS if k != "all"]
        out[period] = {
            "reviews": n,
            "checker_pass": int(sum(r["check_ok"] for r in recs)),
            "checker_failures": int(sum(r["check_failures"] for r in recs)),
            "reviews_relaxed": int(sum(bool(r["ladder"]) for r in recs)),
            "ladder": [{"review": r["review"], **rung} for r in recs for rung in r["ladder"]],
            "strict_rules_failed": sorted({x for r in recs for x in r["strict_rules_failed"]}),
            "min_names_promotions": int(sum(r["n_promoted"] > 0 for r in recs)),
            "mean_promoted": float(np.mean([r["n_promoted"] for r in recs])),
            "support_round_up": int(sum(r["support_rounding"] == "round_up" for r in recs)),
            "solver_inaccurate_accepted": int(sum(x.get("accepted", False)
                                                  for r in recs for x in r["solver_notes"])),
            "solver_failed_or_rejected": int(sum(not x.get("accepted", False)
                                                 for r in recs for x in r["solver_notes"])),
            "binding": binding,
            "costs": costs,
            "sum_single_rule_costs": float(sum(costs[k]["mean_vol_cost"] for k in single)),
        }
    return out


def sweep(runs: dict) -> dict:
    paths = {f"{runs['index']['rulebook']['turnover']:.2f}": runs["index"], **runs["sweep"]}
    out = {}
    for tau in sorted(paths, key=float):
        path = paths[tau]
        out[tau] = {}
        for period in PERIODS:
            par = _daily(runs["parent"], period)
            idx = _daily(path, period)
            recs = _sel(path, period)
            out[tau][period] = {
                "ann_vol": M.ann_vol(idx), "tracking_error": M.tracking_error(idx, par),
                "ann_return": M.ann_return(idx),
                "mean_turnover": M.mean_or_none(r["turnover"] for r in recs),
                "reviews_relaxed": int(sum(bool(r["ladder"]) for r in recs)),
                "checker_pass": int(sum(r["check_ok"] for r in recs)), "reviews": len(recs),
            }
    return out


def sensitivity(runs: dict) -> dict:
    paths = {"committed": runs["index"], **runs["sensitivity"]}
    out = {}
    for name, path in paths.items():
        out[name] = {}
        for period in ("development", "held_out"):
            par = _daily(runs["parent"], period)
            idx = _daily(path, period)
            recs = _sel(path, period)
            out[name][period] = {
                "ann_vol": M.ann_vol(idx), "tracking_error": M.tracking_error(idx, par),
                "mean_turnover": M.mean_or_none(r["turnover"] for r in recs),
                "reviews_relaxed": int(sum(bool(r["ladder"]) for r in recs)),
                "support_round_up": int(sum(r["support_rounding"] == "round_up" for r in recs)),
                "checker_pass": int(sum(r["check_ok"] for r in recs)), "reviews": len(recs),
            }
    return out


SUMMARIZERS = {"dev_choices": dev_choices, "backtest": backtest, "rules": rules, "sweep": sweep,
               "sensitivity": sensitivity}

SENS_LABEL = {"committed": "committed run (504 days, Ledoit-Wolf, min_names 120)",
              "first_run_choice_504_none": "first full run's choice (504 days, no shrinkage)",
              "min_names_100": "original min_names = 100"}


# --------------------------------------------------------------------------- report
def _pct(x, d=2):
    if x is None:
        return "-"
    s = f"{100 * x:.{d}f}"
    return (s[1:] if s.startswith("-") and float(s) == 0 else s) + "%"


def _pp(x, d=2):
    return "-" if x is None else f"{100 * x:.{d}f}"


FAMILY_LABEL = {"cap": "stock caps", "sector": "sector bands", "country": "country bands",
                "turnover": "turnover limit", "holding": "min weight + min names",
                "min_weight": "min weight (floor)", "all": "all rules at once"}


def render(res: Path) -> str:
    dev = json.loads((res / "dev_choices.json").read_text(encoding="utf-8"))
    bt = json.loads((res / "backtest.json").read_text(encoding="utf-8"))
    ru = json.loads((res / "rules.json").read_text(encoding="utf-8"))
    sw = json.loads((res / "sweep.json").read_text(encoding="utf-8"))
    se = json.loads((res / "sensitivity.json").read_text(encoding="utf-8"))
    L = ["# Results", "",
         "Synthetic data only, generated by `make bench` (`python -m ixrebal.cli bench --out results`).",
         "One universe (seed 7, 500 names, 13 years of daily returns), quarterly reviews 8-51.",
         f"Development: reviews {dev['reviews'][0]}-{dev['reviews'][1]}; held out: reviews "
         f"{dev['reviews'][1] + 1}-51, with three disclosed exceptions: held-out reviews were seen in a "
         "prototype and in a first full run before the tie rule was fixed (see docs/methodology.md "
         "and the sensitivity table at the end).", ""]
    L += ["## Development choice (covariance window and shrinkage)", "",
          f"Rule: {dev['rule']}. Parent realized volatility over the same days: "
          f"{_pct(dev['parent_realized_vol'])}.", "",
          "| lookback (days) | shrinkage | realized vol | mean ex-ante vol | mean LW intensity | mean one-way turnover | ladder rungs |",
          "|---|---|---|---|---|---|---|"]
    for r in sorted(dev["rows"], key=lambda x: x["realized_vol"]):
        L.append(f"| {r['lookback']} | {r['shrink']} | {_pct(r['realized_vol'])} | "
                 f"{r['mean_exante_vol']:.2f}% | {r['mean_shrink_delta']:.3f} | "
                 f"{_pct(r['mean_turnover'])} | {r['ladder_rungs']} |")
    c = dev["chosen"]
    L += ["", f"Chosen: **{c['lookback']} days, {c['shrink']}**.", ""]

    L += ["## Backtest: parent vs rule-constrained index vs unconstrained minimum variance", ""]
    for period in PERIODS:
        b = bt[period]
        L += [f"### {period.replace('_', ' ')} ({b['index']['reviews']} reviews, {b['parent']['days']} days)", "",
              "| book | ann. return | ann. vol | vol reduction vs parent | tracking error vs parent | beta | max drawdown | mean one-way turnover | mean names | mean / max abs sector dev (pp) | mean / max abs country dev (pp) | realized / ex-ante vol |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        p = b["parent"]
        L.append(f"| parent | {_pct(p['ann_return'])} | {_pct(p['ann_vol'])} | - | - | 1.00 | "
                 f"{_pct(p['max_drawdown'], 1)} | - | - | - | - | - |")
        for k, lab in (("index", "rule-constrained index"), ("unconstrained", "unconstrained min variance")):
            d = b[k]
            L.append(f"| {lab} | {_pct(d['ann_return'])} | {_pct(d['ann_vol'])} | "
                     f"{_pct(d['vol_reduction'], 1)} | {_pct(d['tracking_error'])} | {d['beta']:.2f} | "
                     f"{_pct(d['max_drawdown'], 1)} | {_pct(d['mean_turnover'], 1)} | {d['mean_names']:.0f} | "
                     f"{_pp(d['mean_max_sector_dev'])} / {_pp(d['max_sector_dev'])} | "
                     f"{_pp(d['mean_max_country_dev'])} / {_pp(d['max_country_dev'])} | "
                     f"{d['realized_over_exante']:.2f} |")
        L += ["", f"Tracking error of the index against unconstrained minimum variance: "
              f"{_pct(b['index_vs_unconstrained_te'])}.", ""]

    L += ["## Rules: verification, relaxation, binding, cost", ""]
    for period in PERIODS:
        r = ru[period]
        L += [f"### {period.replace('_', ' ')} ({r['reviews']} reviews)", "",
              f"- Checker: {r['checker_pass']} of {r['reviews']} published books pass every rule "
              f"(after any logged relaxation); {r['checker_failures']} failed checks.",
              f"- Reviews needing the relaxation ladder: {r['reviews_relaxed']}; "
              f"rules missed against the original rulebook: {', '.join(r['strict_rules_failed']) or 'none'}.",
              f"- Minimum-names top-up needed in {r['min_names_promotions']} reviews "
              f"(mean {r['mean_promoted']:.1f} names added); support rounded up in {r['support_round_up']} reviews.",
              f"- Solver: {r['solver_inaccurate_accepted']} 'optimal_inaccurate' solves accepted after a residual check, "
              f"{r['solver_failed_or_rejected']} failed or rejected (all solves of the review, including the leave-one-rule-out ones).",
              ""]
        if r["ladder"]:
            L += ["| review | rule relaxed | from | to | stage that failed |", "|---|---|---|---|---|"]
            for x in r["ladder"]:
                L.append(f"| {x['review']} | {x['rule']} | {x['from']:g} | {x['to']:g} | {x['failed_stage']} |")
            L.append("")
        L += ["| rule family | share of reviews binding | mean binding / constraints | first-order variance saving per +1pp, (vol %)^2 |",
              "|---|---|---|---|"]
        for f, d in r["binding"].items():
            L.append(f"| {FAMILY_LABEL.get(f, f)} | {_pct(d['share_reviews_binding'], 0)} | "
                     f"{d['mean_n_binding']:.1f} / {d['mean_constraints']:.1f} | {d['mean_dual_per_pp']:.3f} |")
        L += ["", "| rule removed | mean ex-ante vol cost (vol points) | max | reviews where removing it raised vol |",
              "|---|---|---|---|"]
        for f, d in r["costs"].items():
            L.append(f"| {FAMILY_LABEL[f]} | {d['mean_vol_cost']:.3f} | {d['max_vol_cost']:.3f} | {d['reviews_negative']} |")
        L += ["", f"Sum of the five single-rule costs: {r['sum_single_rule_costs']:.3f} vol points "
              f"(against {r['costs']['all']['mean_vol_cost']:.3f} for removing all rules at once).", ""]

    L += ["## Turnover-limit sweep (descriptive, not a choice)", "",
          "| one-way limit per review | period | ann. vol | tracking error | ann. return | mean turnover | reviews relaxed | checker pass |",
          "|---|---|---|---|---|---|---|---|"]
    for tau, d in sw.items():
        for period in ("development", "held_out"):
            x = d[period]
            L.append(f"| {_pct(float(tau), 0)} | {period.replace('_', ' ')} | {_pct(x['ann_vol'])} | "
                     f"{_pct(x['tracking_error'])} | {_pct(x['ann_return'])} | {_pct(x['mean_turnover'], 1)} | "
                     f"{x['reviews_relaxed']} | {x['checker_pass']}/{x['reviews']} |")
    L += ["", "## Sensitivity to the decisions made after held-out reviews were seen", "",
          "| run | period | ann. vol | tracking error | mean turnover | reviews relaxed | support rounded up | checker pass |",
          "|---|---|---|---|---|---|---|---|"]
    for name, d in se.items():
        for period in ("development", "held_out"):
            x = d[period]
            L.append(f"| {SENS_LABEL[name]} | {period.replace('_', ' ')} | {_pct(x['ann_vol'], 3)} | "
                     f"{_pct(x['tracking_error'], 3)} | {_pct(x['mean_turnover'], 1)} | {x['reviews_relaxed']} | "
                     f"{x['support_round_up']} | {x['checker_pass']}/{x['reviews']} |")
    return "\n".join(L) + "\n"


def write_summaries(res: Path, runs: dict) -> None:
    for name in SUMMARY_FILES:
        _write_json(res / f"{name}.json", SUMMARIZERS[name](runs))
    (res / "RESULTS.md").write_text(render(res), encoding="utf-8", newline="\n")


def run(res: Path, log=print) -> None:
    res.mkdir(parents=True, exist_ok=True)
    runs = compute_runs(log)
    _write_json(res / "runs.json", runs, indent=None)
    # summaries are always computed from the file, exactly as the tests do
    write_summaries(res, json.loads((res / "runs.json").read_text(encoding="utf-8")))

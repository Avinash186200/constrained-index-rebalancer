"""ixrebal command line: bench | report | demo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _demo(args) -> None:
    from .backtest import RunConfig, drift, solve_review
    from .rules import Rulebook
    from .universe import UniverseConfig, make_universe

    u = make_universe(UniverseConfig(seed=args.seed))
    rb = Rulebook(turnover=args.turnover)
    cfg = RunConfig()
    k0 = args.review
    rec0, w0 = solve_review(u, k0, rb, cfg, None)
    print(f"review {k0} (inception build): {rec0['names']} names, ex-ante vol "
          f"{rec0['exante_vol']:.2f}%, checker ok={rec0['check_ok']}")
    rd = u.review_days
    drifted, _ = drift(u, w0, int(rd[k0]), int(rd[k0 + 1]))
    rec, _ = solve_review(u, k0 + 1, rb, cfg, drifted, costs=True)
    print(f"review {k0 + 1}: {rec['n_members']} parent members, {rec['n_eligible']} eligible, "
          f"{rec['names']} names held, turnover {100 * rec['turnover']:.1f}% "
          f"(limit {100 * rb.turnover:.0f}%), ex-ante vol {rec['exante_vol']:.2f}%")
    print(f"checker ok={rec['check_ok']}, ladder={rec['ladder'] or 'not needed'}")
    print("binding (family: binding/constraints, variance saving per +1pp):")
    for f, d in rec["binding"].items():
        print(f"  {f:<11} {d['n_binding']:>3}/{d['n']:<4} {d['dual_per_pp']:.3f}")
    print("ex-ante vol with one rule family removed:")
    for f, v in rec["vol_without"].items():
        print(f"  {f:<9} {v:.2f}%  (cost {rec['exante_vol'] - v:+.3f})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="ixrebal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="full deterministic benchmark -> results/*.json, RESULTS.md")
    b.add_argument("--out", default="results")
    r = sub.add_parser("report", help="recompute summaries and RESULTS.md from results/runs.json")
    r.add_argument("--out", default="results")
    d = sub.add_parser("demo", help="build the index at one review and rebalance it at the next")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--review", type=int, default=8)
    d.add_argument("--turnover", type=float, default=0.10)
    args = ap.parse_args(argv)

    if args.cmd == "bench":
        from .bench import run
        run(Path(args.out))
    elif args.cmd == "report":
        from .bench import write_summaries
        res = Path(args.out)
        write_summaries(res, json.loads((res / "runs.json").read_text(encoding="utf-8")))
    elif args.cmd == "demo":
        _demo(args)


if __name__ == "__main__":
    main()

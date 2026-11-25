import json
import math
import shutil
from pathlib import Path

from ixrebal.cli import main

RES = Path(__file__).resolve().parents[1] / "results"


def test_demo_runs(capsys):
    main(["demo"])
    out = capsys.readouterr().out
    assert "inception build" in out and "checker ok=True" in out
    assert "ex-ante vol with one rule family removed" in out


def _close(a, b, path=""):
    """Compare two JSON values; floats may differ in the last bits across platforms."""
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), f"{path}: keys differ"
        for k in a:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"{path}: length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def test_report_rebuilds_the_committed_files(tmp_path):
    shutil.copy(RES / "runs.json", tmp_path / "runs.json")
    main(["report", "--out", str(tmp_path)])
    # Summaries are recomputed from runs.json; on another platform the last printed digit of a
    # float can differ, so JSON files are compared value by value with a tight tolerance.
    for name in ("dev_choices.json", "backtest.json", "rules.json", "sweep.json", "sensitivity.json"):
        _close(json.loads((tmp_path / name).read_text()), json.loads((RES / name).read_text()), name)
    # RESULTS.md shows rounded numbers only, so it must match exactly.
    assert (tmp_path / "RESULTS.md").read_bytes() == (RES / "RESULTS.md").read_bytes()

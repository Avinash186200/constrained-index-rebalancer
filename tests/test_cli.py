import shutil
from pathlib import Path

from ixrebal.cli import main

RES = Path(__file__).resolve().parents[1] / "results"


def test_demo_runs(capsys):
    main(["demo"])
    out = capsys.readouterr().out
    assert "inception build" in out and "checker ok=True" in out
    assert "ex-ante vol with one rule family removed" in out


def test_report_rebuilds_the_committed_files(tmp_path):
    shutil.copy(RES / "runs.json", tmp_path / "runs.json")
    main(["report", "--out", str(tmp_path)])
    for name in ("dev_choices.json", "backtest.json", "rules.json", "sweep.json", "sensitivity.json", "RESULTS.md"):
        assert (tmp_path / name).read_bytes() == (RES / name).read_bytes(), name

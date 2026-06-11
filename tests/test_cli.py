from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataset_audit_kit.cli import main


def test_cli_outputs_json(tmp_path, capsys) -> None:
    data = pd.DataFrame(
        {
            "feature_a": [1.0, 1.1, 1.2],
            "target": [0, 0, 1],
        }
    )
    reference = pd.DataFrame(
        {
            "feature_a": [1.0, 1.1, 1.2],
            "target": [0, 1, 1],
        }
    )

    data_path = tmp_path / "data.csv"
    reference_path = tmp_path / "reference.csv"
    data.to_csv(data_path, index=False)
    reference.to_csv(reference_path, index=False)

    exit_code = main(
        [
            "audit",
            str(data_path),
            "--reference",
            str(reference_path),
            "--label-column",
            "target",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"status": "pass"' in captured.out or '"status": "warn"' in captured.out


def test_cli_writes_html_report(tmp_path, capsys) -> None:
    data = pd.DataFrame(
        {
            "feature_a": [1.0, 1.1, 1.2],
            "target": [0, 0, 1],
        }
    )
    data_path = tmp_path / "data.csv"
    html_path = tmp_path / "report.html"
    data.to_csv(data_path, index=False)

    exit_code = main(
        [
            "audit",
            str(data_path),
            "--label-column",
            "target",
            "--html-out",
            str(html_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "<!doctype html>" in html
    assert "HTML report written to" in captured.out


def test_cli_accepts_parquet_input(tmp_path, capsys) -> None:
    data = pd.DataFrame(
        {
            "feature_a": [1.0, 1.1, 1.2],
            "target": [0, 0, 1],
        }
    )
    data_path = tmp_path / "data.parquet"
    data.to_parquet(data_path, index=False)

    exit_code = main(
        [
            "audit",
            str(data_path),
            "--label-column",
            "target",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"rows": 3' in captured.out


def test_demo_script_runs() -> None:
    import runpy

    runpy.run_path(str(Path("examples/demo.py")), run_name="__main__")

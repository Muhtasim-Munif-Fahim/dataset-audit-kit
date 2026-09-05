"""Tests for the `diff` subcommand, including --json output."""

from __future__ import annotations

import json

import pandas as pd

from dataset_audit_kit.cli import main
from dataset_audit_kit.core import DatasetAuditor


CLEAN = pd.DataFrame(
    {"id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"], "label": ["x", "y", "x", "y"]}
)
DIRTY = pd.DataFrame(
    {"id": [1, 1, 2, 3], "name": ["a", "a", "b", "c"], "label": ["x", "x", "y", "y"]}
)


def _write_reports(tmp_path) -> tuple[str, str]:
    baseline = tmp_path / "clean.json"
    current = tmp_path / "dirty.json"
    baseline.write_text(
        DatasetAuditor().audit_dataframe(CLEAN).to_json(), encoding="utf-8"
    )
    current.write_text(
        DatasetAuditor().audit_dataframe(DIRTY).to_json(), encoding="utf-8"
    )
    return str(baseline), str(current)


class TestDiffCommand:
    def test_default_table_output_runs_and_lists_regressions(self, tmp_path, capsys) -> None:
        baseline, current = _write_reports(tmp_path)
        assert main(["diff", baseline, current]) == 0
        out = capsys.readouterr().out
        assert "quality_score" in out
        assert "duplicate_rows" in out
        assert "<-- worse" in out
        assert "Issues: 0 -> 1" in out
        assert "duplicates:" in out

    def test_json_output_is_valid_machine_readable_report(self, tmp_path, capsys) -> None:
        baseline, current = _write_reports(tmp_path)
        assert main(["diff", "--json", baseline, current]) == 0
        payload = json.loads(capsys.readouterr().out)

        assert set(payload) == {"metrics", "issues", "schema_changes", "regression_detected"}
        for key in ("quality_score", "rows", "columns", "duplicate_rows", "missing_cells"):
            entry = payload["metrics"][key]
            assert {"baseline", "current", "change", "worsened"} <= set(entry)
        assert payload["metrics"]["quality_score"]["worsened"] is True
        assert payload["metrics"]["duplicate_rows"]["worsened"] is True
        assert payload["metrics"]["duplicate_rows"]["current"] == 1
        assert payload["metrics"]["duplicate_rows"]["baseline"] == 0

        issues = payload["issues"]
        assert issues["baseline_count"] == 0
        assert issues["current_count"] == 1
        assert issues["added"] == ["duplicates:"]
        assert issues["resolved"] == []

        assert payload["schema_changes"] == {"dropped": [], "gained": []}
        assert payload["regression_detected"] is True

    def test_fail_on_regression_exits_nonzero_on_regression(self, tmp_path) -> None:
        baseline, current = _write_reports(tmp_path)
        assert main(["diff", "--fail-on-regression", baseline, current]) == 1

    def test_fail_on_regression_passes_when_stable(self, tmp_path) -> None:
        baseline, _ = _write_reports(tmp_path)
        assert main(["diff", "--fail-on-regression", baseline, baseline]) == 0

    def test_json_with_fail_on_regression_exits_nonzero(self, tmp_path, capsys) -> None:
        baseline, current = _write_reports(tmp_path)
        assert main(["diff", "--json", "--fail-on-regression", baseline, current]) == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["regression_detected"] is True

    def test_missing_report_file_exits_two(self, tmp_path, capsys) -> None:
        _, current = _write_reports(tmp_path)
        assert main(["diff", str(tmp_path / "nope.json"), current]) == 2
        assert "Cannot read baseline report" in capsys.readouterr().err

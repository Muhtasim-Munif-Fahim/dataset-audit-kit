"""Tests for AuditReport.batch_summary_csv."""

from __future__ import annotations

import csv
from pathlib import Path

from dataset_audit_kit.core import AuditReport


def _report(rows=10, columns=2, **profiles):
    report = AuditReport(rows=rows, columns=columns, duplicate_rows=0, missing_cells=0)
    report.column_profiles = dict(profiles)
    return report


class TestBatchSummaryCsv:
    def test_writes_a_single_report(self, tmp_path: Path) -> None:
        report = _report(rows=100, columns=3, x={"dtype": "numeric"})
        report.risk_score = 0.42
        target = tmp_path / "summary.csv"
        result_path = report.batch_summary_csv(str(target))
        assert result_path == str(target)
        with open(target, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        assert len(rows) == 1
        assert int(rows[0]["rows"]) == 100
        assert int(rows[0]["columns"]) == 3
        assert float(rows[0]["risk_score"]) == 0.42

    def test_writes_multiple_reports(self, tmp_path: Path) -> None:
        first = _report(rows=10, columns=1, x={"dtype": "numeric"})
        second = _report(rows=20, columns=2, y={"dtype": "categorical", "top_5": {"a": 1}})
        target = tmp_path / "summary.csv"
        first.batch_summary_csv(str(target), reports=[first, second])
        with open(target, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        assert len(rows) == 2
        assert int(rows[0]["rows"]) == 10
        assert int(rows[1]["rows"]) == 20

    def test_writes_header_even_when_empty(self, tmp_path: Path) -> None:
        report = _report()
        target = tmp_path / "summary.csv"
        report.batch_summary_csv(str(target), reports=[])
        text = target.read_text(encoding="utf-8")
        assert text.splitlines()[0].startswith("audit_id")
        assert len(text.splitlines()) == 1

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        report = _report()
        target = tmp_path / "subdir" / "summary.csv"
        report.batch_summary_csv(str(target))
        assert target.exists()

    def test_picks_up_blocking_issues(self, tmp_path: Path) -> None:
        from dataset_audit_kit.core import AuditIssue
        report = _report()
        report.issues = [
            AuditIssue(check="missingness", column="a", severity="error", message="x", observed=0.5, threshold=0.1),
            AuditIssue(check="outliers", column="b", severity="warning", message="y", observed=0.2, threshold=0.1),
        ]
        target = tmp_path / "summary.csv"
        report.batch_summary_csv(str(target))
        with open(target, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            row = reader.__next__()
        assert int(row["blocking_issues"]) == 2
        assert "missingness:a" in row["failing_checks"]
        assert "outliers:b" in row["failing_checks"]
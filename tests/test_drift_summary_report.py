"""Tests for AuditReport.drift_summary_report."""

from __future__ import annotations

import pytest

from dataset_audit_kit.core import AuditReport


def _report(rows=10, columns=2, **profiles):
    report = AuditReport(rows=rows, columns=columns, duplicate_rows=0, missing_cells=0)
    report.column_profiles = dict(profiles)
    return report


class TestDriftSummary:
    def test_filters_by_min_score(self) -> None:
        report = _report()
        report.drift_scores = {"a": 0.1, "b": 0.3, "c": 0.05}
        rows = report.drift_summary_report(min_score=0.2)
        assert len(rows) == 1
        assert rows[0]["column"] == "b"
        assert rows[0]["drift_score"] == 0.3

    def test_default_min_score_is_0(self) -> None:
        report = _report()
        report.drift_scores = {"a": 0.0, "b": 0.15}
        rows = report.drift_summary_report()
        assert len(rows) == 2

    def test_sorted_by_score_desc(self) -> None:
        report = _report()
        report.drift_scores = {"a": 0.1, "b": 0.3, "c": 0.2}
        rows = report.drift_summary_report()
        assert [row["column"] for row in rows] == ["b", "c", "a"]

    def test_empty_drift_scores(self) -> None:
        report = _report()
        assert report.drift_summary_report() == []


class TestDriftSummaryValidation:
    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3"])
    def test_top_must_be_a_positive_integer(self, bad: object) -> None:
        with pytest.raises(ValueError, match="top must be a positive integer"):
            _report().drift_summary_report(top=bad)

"""Tests for AuditReport.empty_columns_report."""

from __future__ import annotations

import pytest

from dataset_audit_kit.core import AuditReport


def _report(rows=10, columns=2, **profiles):
    report = AuditReport(rows=rows, columns=columns, duplicate_rows=0, missing_cells=0)
    report.column_profiles = dict(profiles)
    return report


class TestEmptyColumns:
    def test_identifies_all_null_columns(self) -> None:
        report = _report(
            a={"dtype": "other", "count": 100, "missing": 100, "unique": 0},
            b={"dtype": "numeric", "count": 100, "missing": 0, "unique": 50},
        )
        rows = report.empty_columns_report()
        assert len(rows) == 1
        assert rows[0]["column"] == "a"
        assert rows[0]["missing_ratio"] == 1.0

    def test_no_empty_columns_returns_empty(self) -> None:
        report = _report(
            a={"dtype": "numeric", "count": 100, "missing": 0, "unique": 50},
            b={"dtype": "categorical", "count": 100, "missing": 5, "unique": 20},
        )
        assert report.empty_columns_report() == []

    def test_empty_report_returns_empty(self) -> None:
        report = _report()
        assert report.empty_columns_report() == []


class TestEmptyColumnsValidation:
    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3"])
    def test_top_must_be_a_positive_integer(self, bad: object) -> None:
        with pytest.raises(ValueError, match="top must be a positive integer"):
            _report().empty_columns_report(top=bad)


class TestEmptyColumnsSorting:
    def test_sorted_alphabetically(self) -> None:
        report = _report(
            z={"dtype": "other", "count": 5, "missing": 5, "unique": 0},
            a={"dtype": "other", "count": 5, "missing": 5, "unique": 0},
        )
        names = [row["column"] for row in report.empty_columns_report()]
        assert names == ["a", "z"]

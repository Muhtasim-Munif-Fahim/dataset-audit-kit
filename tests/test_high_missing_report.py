"""Tests for AuditReport.high_missing_report."""

from __future__ import annotations

import pytest

from dataset_audit_kit.core import AuditReport


def _report(rows=10, columns=2, **profiles):
    report = AuditReport(rows=rows, columns=columns, duplicate_rows=0, missing_cells=0)
    report.column_profiles = dict(profiles)
    return report


class TestHighMissing:
    def test_filters_by_threshold(self) -> None:
        report = _report(
            a={"dtype": "numeric", "count": 100, "missing": 0, "unique": 50},
            b={"dtype": "categorical", "count": 100, "missing": 20, "unique": 20},
            c={"dtype": "other", "count": 100, "missing": 5, "unique": 0},
        )
        rows = report.high_missing_report(threshold=0.1)
        assert len(rows) == 1
        assert rows[0]["column"] == "b"
        assert rows[0]["missing_ratio"] == 0.2

    def test_default_threshold_is_0_1(self) -> None:
        report = _report(
            a={"dtype": "numeric", "count": 100, "missing": 5, "unique": 50},
            b={"dtype": "categorical", "count": 100, "missing": 15, "unique": 20},
        )
        rows = report.high_missing_report()
        assert len(rows) == 1
        assert rows[0]["column"] == "b"

    def test_sorted_by_missing_ratio_desc(self) -> None:
        report = _report(
            a={"dtype": "numeric", "count": 100, "missing": 30, "unique": 50},
            b={"dtype": "categorical", "count": 100, "missing": 10, "unique": 20},
        )
        rows = report.high_missing_report(threshold=0.05)
        assert [row["column"] for row in rows] == ["a", "b"]


class TestHighMissingValidation:
    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3"])
    def test_top_must_be_a_positive_integer(self, bad: object) -> None:
        with pytest.raises(ValueError, match="top must be a positive integer"):
            _report().high_missing_report(top=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.1, True, "0.5"])
    def test_threshold_must_be_a_fraction(self, bad: object) -> None:
        with pytest.raises(ValueError, match="threshold must be a number"):
            _report().high_missing_report(threshold=bad)

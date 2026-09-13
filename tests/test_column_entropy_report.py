"""Tests for AuditReport.column_entropy_report."""

from __future__ import annotations

import pytest

from dataset_audit_kit.core import AuditReport


def _report(rows=10, columns=2, **profiles):
    report = AuditReport(rows=rows, columns=columns, duplicate_rows=0, missing_cells=0)
    report.column_profiles = dict(profiles)
    return report


class TestColumnEntropy:
    def test_only_columns_with_entropy(self) -> None:
        report = _report(
            a={"dtype": "categorical", "count": 100, "missing": 0, "unique": 3,
               "entropy": 1.1, "normalized_entropy": 0.9},
            b={"dtype": "numeric", "count": 100, "missing": 0, "unique": 50,
               "min": 0, "max": 10, "mean": 5.0},
        )
        rows = report.column_entropy_report()
        assert len(rows) == 1
        assert rows[0]["column"] == "a"

    def test_sorted_by_normalized_entropy_desc(self) -> None:
        report = _report(
            a={"dtype": "categorical", "count": 100, "missing": 0, "unique": 10,
               "entropy": 2.0, "normalized_entropy": 0.9},
            b={"dtype": "categorical", "count": 100, "missing": 0, "unique": 3,
               "entropy": 1.0, "normalized_entropy": 0.8},
            c={"dtype": "categorical", "count": 100, "missing": 0, "unique": 2,
               "entropy": 0.5, "normalized_entropy": 0.5},
        )
        rows = report.column_entropy_report()
        assert [row["column"] for row in rows] == ["a", "b", "c"]

    def test_no_entropy_columns_returns_empty(self) -> None:
        report = _report(
            a={"dtype": "numeric", "count": 100, "missing": 0, "unique": 50},
        )
        assert report.column_entropy_report() == []


class TestColumnEntropyValidation:
    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3"])
    def test_top_must_be_a_positive_integer(self, bad: object) -> None:
        with pytest.raises(ValueError, match="top must be a positive integer"):
            _report().column_entropy_report(top=bad)

"""Tests for AuditReport.low_cardinality_report."""

from __future__ import annotations

import pytest

from dataset_audit_kit.core import AuditReport


def _report(rows=10, columns=2, **profiles):
    report = AuditReport(rows=rows, columns=columns, duplicate_rows=0, missing_cells=0)
    report.column_profiles = dict(profiles)
    return report


class TestLowCardinality:
    def test_filters_by_max_ratio(self) -> None:
        report = _report(
            a={"dtype": "categorical", "count": 100, "missing": 0, "unique": 3},
            b={"dtype": "numeric", "count": 100, "missing": 0, "unique": 90},
        )
        rows = report.low_cardinality_report(max_ratio=0.05)
        assert len(rows) == 1
        assert rows[0]["column"] == "a"
        assert rows[0]["unique_ratio"] == 0.03

    def test_default_max_ratio_is_0_05(self) -> None:
        report = _report(
            a={"dtype": "categorical", "count": 100, "missing": 0, "unique": 2},
            b={"dtype": "categorical", "count": 100, "missing": 0, "unique": 8},
        )
        rows = report.low_cardinality_report()
        assert len(rows) == 1
        assert rows[0]["column"] == "a"

    def test_sorted_by_ratio_desc(self) -> None:
        report = _report(
            a={"dtype": "categorical", "count": 100, "missing": 0, "unique": 3},
            b={"dtype": "categorical", "count": 100, "missing": 0, "unique": 1},
        )
        rows = report.low_cardinality_report(max_ratio=0.05)
        assert [row["column"] for row in rows] == ["a", "b"]


class TestLowCardinalityValidation:
    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3"])
    def test_top_must_be_a_positive_integer(self, bad: object) -> None:
        with pytest.raises(ValueError, match="top must be a positive integer"):
            _report().low_cardinality_report(top=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.1, True, "0.5"])
    def test_max_ratio_must_be_a_fraction(self, bad: object) -> None:
        with pytest.raises(ValueError, match="max_ratio must be a number"):
            _report().low_cardinality_report(max_ratio=bad)

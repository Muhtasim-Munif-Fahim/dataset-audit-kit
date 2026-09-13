"""Tests for AuditReport.zero_variance_report."""

from __future__ import annotations

import pytest

from dataset_audit_kit.core import AuditReport


def _report(rows=10, columns=2, **profiles):
    report = AuditReport(rows=rows, columns=columns, duplicate_rows=0, missing_cells=0)
    report.column_profiles = dict(profiles)
    return report


class TestZeroVariance:
    def test_identifies_constant_columns(self) -> None:
        report = _report(
            a={"dtype": "categorical", "count": 100, "missing": 0, "unique": 1, "top": "yes"},
            b={"dtype": "numeric", "count": 100, "missing": 0, "unique": 50},
        )
        rows = report.zero_variance_report()
        assert len(rows) == 1
        assert rows[0]["column"] == "a"
        assert rows[0]["unique"] == 1

    def test_all_null_column_is_included(self) -> None:
        report = _report(
            a={"dtype": "other", "count": 10, "missing": 10, "unique": 0},
            b={"dtype": "numeric", "count": 10, "missing": 0, "unique": 10},
        )
        rows = report.zero_variance_report()
        assert len(rows) == 1
        assert rows[0]["column"] == "a"

    def test_no_zero_variance_returns_empty(self) -> None:
        report = _report(
            a={"dtype": "numeric", "count": 10, "missing": 0, "unique": 10},
            b={"dtype": "categorical", "count": 10, "missing": 0, "unique": 3},
        )
        assert report.zero_variance_report() == []


class TestZeroVarianceValidation:
    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3"])
    def test_top_must_be_a_positive_integer(self, bad: object) -> None:
        with pytest.raises(ValueError, match="top must be a positive integer"):
            _report().zero_variance_report(top=bad)

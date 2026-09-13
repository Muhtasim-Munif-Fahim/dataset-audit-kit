"""Tests for AuditReport.numeric_stats_report."""

from __future__ import annotations

import pytest

from dataset_audit_kit.core import AuditReport


def _report(rows=10, columns=2, **profiles):
    report = AuditReport(rows=rows, columns=columns, duplicate_rows=0, missing_cells=0)
    report.column_profiles = dict(profiles)
    return report


class TestNumericStats:
    def test_only_numeric_columns_included(self) -> None:
        report = _report(
            a={"dtype": "numeric", "count": 100, "missing": 0, "unique": 50,
               "min": 0, "max": 10, "mean": 5.0, "median": 5.0, "std": 2.0,
               "q25": 2.5, "q50": 5.0, "q75": 7.5, "skewness": 0.1, "kurtosis": -0.5,
               "outliers_iqr": 2, "outlier_ratio": 0.02},
            b={"dtype": "categorical", "count": 100, "missing": 0, "unique": 3},
        )
        rows = report.numeric_stats_report()
        assert len(rows) == 1
        assert rows[0]["column"] == "a"
        assert rows[0]["mean"] == 5.0
        assert rows[0]["std"] == 2.0

    def test_skips_numeric_columns_without_stats(self) -> None:
        report = _report(
            a={"dtype": "numeric", "count": 10, "missing": 0, "unique": 5},
            b={"dtype": "numeric", "count": 100, "missing": 0, "unique": 50,
               "min": 0, "max": 10, "mean": 5.0, "median": 5.0, "std": 2.0,
               "q25": 2.5, "q50": 5.0, "q75": 7.5, "skewness": 0.1, "kurtosis": -0.5,
               "outliers_iqr": 2, "outlier_ratio": 0.02},
        )
        rows = report.numeric_stats_report()
        assert len(rows) == 1
        assert rows[0]["column"] == "b"

    def test_sorted_by_skewness_desc(self) -> None:
        report = _report(
            a={"dtype": "numeric", "count": 100, "missing": 0, "unique": 50,
               "min": 0, "max": 10, "mean": 5.0, "std": 2.0,
               "q25": 2.5, "q50": 5.0, "q75": 7.5, "skewness": 2.0, "kurtosis": 1.0},
            b={"dtype": "numeric", "count": 100, "missing": 0, "unique": 50,
               "min": 0, "max": 10, "mean": 5.0, "std": 2.0,
               "q25": 2.5, "q50": 5.0, "q75": 7.5, "skewness": 0.5, "kurtosis": 1.0},
        )
        rows = report.numeric_stats_report()
        assert [row["column"] for row in rows] == ["a", "b"]


class TestNumericStatsValidation:
    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3"])
    def test_top_must_be_a_positive_integer(self, bad: object) -> None:
        with pytest.raises(ValueError, match="top must be a positive integer"):
            _report().numeric_stats_report(top=bad)

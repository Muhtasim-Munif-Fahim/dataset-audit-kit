"""Tests for AuditReport.outlier_detail_report."""

from __future__ import annotations

import pytest

from dataset_audit_kit.core import AuditReport


def _report(rows=10, columns=2, **profiles):
    report = AuditReport(rows=rows, columns=columns, duplicate_rows=0, missing_cells=0)
    report.column_profiles = dict(profiles)
    return report


class TestOutlierDetail:
    def test_only_numeric_with_outlier_data(self) -> None:
        report = _report(
            a={"dtype": "numeric", "count": 100, "missing": 0, "unique": 50,
               "q25": 2.5, "q75": 7.5, "outliers_iqr": 5, "outlier_ratio": 0.05},
            b={"dtype": "categorical", "count": 100, "missing": 0, "unique": 3,
               "entropy": 1.0, "normalized_entropy": 0.8},
        )
        rows = report.outlier_detail_report()
        assert len(rows) == 1
        assert rows[0]["column"] == "a"
        assert rows[0]["outliers"] == 5
        assert rows[0]["outlier_ratio"] == 0.05
        assert rows[0]["lower_fence"] == 2.5 - 1.5 * 5.0
        assert rows[0]["upper_fence"] == 7.5 + 1.5 * 5.0

    def test_skips_numeric_without_outlier_ratio(self) -> None:
        report = _report(
            a={"dtype": "numeric", "count": 100, "missing": 0, "unique": 50},
            b={"dtype": "numeric", "count": 100, "missing": 0, "unique": 50,
               "q25": 2.5, "q75": 7.5, "outliers_iqr": 3, "outlier_ratio": 0.03},
        )
        rows = report.outlier_detail_report()
        assert len(rows) == 1
        assert rows[0]["column"] == "b"

    def test_sorted_by_outlier_ratio_desc(self) -> None:
        report = _report(
            a={"dtype": "numeric", "count": 100, "missing": 0, "unique": 50,
               "q25": 10, "q75": 20, "outliers_iqr": 1, "outlier_ratio": 0.01},
            b={"dtype": "numeric", "count": 100, "missing": 0, "unique": 50,
               "q25": 10, "q75": 20, "outliers_iqr": 5, "outlier_ratio": 0.05},
        )
        rows = report.outlier_detail_report()
        assert [row["column"] for row in rows] == ["b", "a"]


class TestOutlierDetailValidation:
    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3"])
    def test_top_must_be_a_positive_integer(self, bad: object) -> None:
        with pytest.raises(ValueError, match="top must be a positive integer"):
            _report().outlier_detail_report(top=bad)

"""Tests for AuditReport.constant_columns_report."""

from __future__ import annotations

import pytest

from dataset_audit_kit.core import AuditReport


def _report(**profiles: object) -> AuditReport:
    report = AuditReport(rows=10, columns=len(profiles), duplicate_rows=0, missing_cells=0)
    report.column_profiles = dict(profiles)
    return report


def test_constant_columns_report_excludes_variable_and_empty_columns() -> None:
    report = _report(
        country={"dtype": "categorical", "count": 10, "missing": 2, "unique": 1},
        score={"dtype": "numeric", "count": 10, "missing": 0, "unique": 4},
        empty={"dtype": "other", "count": 10, "missing": 10, "unique": 0},
    )
    assert report.constant_columns_report() == [
        {"column": "country", "non_null": 8, "missing": 2, "dtype": "categorical"}
    ]


def test_constant_columns_report_is_sorted_and_limited() -> None:
    report = _report(
        z={"count": 10, "missing": 0, "unique": 1},
        a={"count": 10, "missing": 0, "unique": 1},
    )
    assert [row["column"] for row in report.constant_columns_report(top=1)] == ["a"]


@pytest.mark.parametrize("top", [0, True, "1"])
def test_constant_columns_report_validates_top(top: object) -> None:
    with pytest.raises(ValueError, match="top must be a positive integer"):
        _report().constant_columns_report(top=top)

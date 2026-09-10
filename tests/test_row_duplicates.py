"""Tests for row_duplicates_report."""
from __future__ import annotations

import pandas as pd
import pytest

from dataset_audit_kit import AuditReport, DatasetAuditor


def _make_report(duplicate_rows: int, duplicate_groups: list[dict]) -> AuditReport:
    report = AuditReport(
        rows=10,
        columns=2,
        duplicate_rows=duplicate_rows,
        duplicate_groups=duplicate_groups,
        missing_cells=0,
    )
    return report


def test_returns_groups_sorted_by_count_desc() -> None:
    groups = [
        {"indices": [0, 1], "count": 2, "columns": ["a", "b"]},
        {"indices": [2, 3, 4], "count": 3, "columns": ["a", "b"]},
    ]
    report = _make_report(5, groups)
    rows = report.row_duplicates_report()
    assert [row["count"] for row in rows] == [3, 2]


def test_top_argument_limits_results() -> None:
    groups = [{"indices": [i], "count": i + 1, "columns": ["a"]} for i in range(5)]
    report = _make_report(15, groups)
    rows = report.row_duplicates_report(top=2)
    assert len(rows) == 2
    assert rows[0]["count"] == 5


def test_empty_groups_returns_empty() -> None:
    report = _make_report(0, [])
    assert report.row_duplicates_report() == []


def test_invalid_top_raises() -> None:
    report = _make_report(0, [])
    with pytest.raises(ValueError, match="top must be a positive integer"):
        report.row_duplicates_report(top=0)
    with pytest.raises(ValueError, match="top must be a positive integer"):
        report.row_duplicates_report(top=-1)


def test_end_to_end_with_dataframe() -> None:
    df = pd.DataFrame({
        "a": [1, 2, 1, 3, 2],
        "b": ["x", "y", "x", "z", "y"],
    })
    auditor = DatasetAuditor(max_duplicate_ratio=0.0)
    report = auditor.audit_dataframe(df)
    rows = report.row_duplicates_report()
    assert report.duplicate_rows == 2
    assert len(rows) == 2
    assert rows[0]["count"] == 2
    assert rows[1]["count"] == 2


def test_indices_are_integers() -> None:
    groups = [{"indices": [0, 1], "count": 2, "columns": ["a", "b"]}]
    report = _make_report(2, groups)
    rows = report.row_duplicates_report()
    assert all(isinstance(i, int) for i in rows[0]["indices"])

"""Tests for AuditReport.column_overlap_table."""

from __future__ import annotations

import pytest

from dataset_audit_kit.core import AuditReport


def _report(rows=10, columns=2, **profiles):
    report = AuditReport(rows=rows, columns=columns, duplicate_rows=0, missing_cells=0)
    report.column_profiles = dict(profiles)
    return report


class TestColumnOverlapTable:
    def test_levels_left_and_right_take_precedence(self) -> None:
        left = _report(
            x={"dtype": "categorical", "top_5": {"a": 1, "b": 1}},
            y={"dtype": "categorical", "top_5": {"c": 1}},
        )
        right = _report(
            x={"dtype": "categorical", "top_5": {"b": 1, "c": 1}},
            y={"dtype": "categorical", "top_5": {"c": 1}},
        )
        rows = left.column_overlap_table(
            right,
            levels_left={"x": ["a", "b", "c"]},
            levels_right={"x": ["b", "c", "d"]},
        )
        x = next(row for row in rows if row["column"] == "x")
        assert sorted(x["intersection"]) == ["b", "c"]
        assert sorted(x["only_left"]) == ["a"]
        assert sorted(x["only_right"]) == ["d"]
        assert x["union_size"] == 4
        assert x["jaccard"] == pytest.approx(0.5)

    def test_falls_back_to_top_5_keys(self) -> None:
        left = _report(
            x={"dtype": "categorical", "top_5": {"a": 1, "b": 1, "c": 1}},
            y={"dtype": "categorical", "top_5": {"a": 1, "b": 1}},
        )
        right = _report(
            x={"dtype": "categorical", "top_5": {"a": 1, "b": 1, "d": 1}},
            y={"dtype": "categorical", "top_5": {"a": 1, "b": 1}},
        )
        rows = left.column_overlap_table(right)
        x = next(row for row in rows if row["column"] == "x")
        assert sorted(x["intersection"]) == ["a", "b"]
        assert sorted(x["only_left"]) == ["c"]
        assert sorted(x["only_right"]) == ["d"]
        assert x["only_left"] or x["only_right"]

    def test_columns_subset_restricts(self) -> None:
        left = _report(
            a={"dtype": "categorical", "top_5": {"x": 1}},
            b={"dtype": "categorical", "top_5": {"x": 1}},
        )
        right = _report(
            a={"dtype": "categorical", "top_5": {"x": 1}},
            b={"dtype": "categorical", "top_5": {"x": 1}},
        )
        rows = left.column_overlap_table(right, columns=["a"])
        assert [row["column"] for row in rows] == ["a"]

    def test_unknown_columns_raise(self) -> None:
        left = _report(a={"dtype": "categorical"})
        right = _report(a={"dtype": "categorical"})
        with pytest.raises(ValueError, match="unknown columns"):
            left.column_overlap_table(right, columns=["a", "ghost"])

    def test_only_shared_columns_appear(self) -> None:
        left = _report(a={"dtype": "categorical", "top_5": {"x": 1}}, b={"dtype": "categorical", "top_5": {"x": 1}})
        right = _report(a={"dtype": "categorical", "top_5": {"x": 1}}, c={"dtype": "categorical", "top_5": {"x": 1}})
        rows = left.column_overlap_table(right)
        assert [row["column"] for row in rows] == ["a"]

    def test_jaccard_is_one_when_both_levels_empty(self) -> None:
        left = _report(x={"dtype": "numeric"})
        right = _report(x={"dtype": "numeric"})
        rows = left.column_overlap_table(right)
        assert rows[0]["jaccard"] == 1.0

    def test_rows_are_sorted_by_column(self) -> None:
        left = _report(b={"dtype": "categorical", "top_5": {"x": 1}}, a={"dtype": "categorical", "top_5": {"x": 1}})
        right = _report(b={"dtype": "categorical", "top_5": {"x": 1}}, a={"dtype": "categorical", "top_5": {"x": 1}})
        rows = left.column_overlap_table(right)
        assert [row["column"] for row in rows] == ["a", "b"]

    def test_levels_dict_accepts_dict_keys(self) -> None:
        left = _report(x={"dtype": "categorical", "top_5": {"a": 1}})
        right = _report(x={"dtype": "categorical", "top_5": {"a": 1}})
        rows = left.column_overlap_table(
            right,
            levels_left={"x": {"a": 1, "b": 1}},
            levels_right={"x": {"a": 1, "c": 1}},
        )
        x = rows[0]
        assert sorted(x["only_left"]) == ["b"]
        assert sorted(x["only_right"]) == ["c"]
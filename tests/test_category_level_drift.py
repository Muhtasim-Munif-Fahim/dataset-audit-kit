"""Tests for categorical level-set (new/removed category) drift detection."""

from __future__ import annotations

import json

import pandas as pd

from dataset_audit_kit.core import DatasetAuditor


def _baseline_current(baseline, current, column="flag"):
    report = DatasetAuditor().audit_dataframe(
        pd.DataFrame({column: current}),
        reference=pd.DataFrame({column: baseline}),
    )
    return report


class TestCategoricalLevelDrift:
    def test_new_categories_are_flagged(self) -> None:
        report = _baseline_current(
            ["a", "b", "a", "b"], ["a", "b", "d", "a"]
        )
        issue = next(i for i in report.issues if i.check == "category_level_drift")
        assert issue.column == "flag"
        assert "1 new" in issue.message
        assert "d" in issue.message
        assert "removed" not in issue.message
        assert issue.observed == 1

    def test_removed_categories_are_flagged(self) -> None:
        report = _baseline_current(
            ["a", "b", "c", "a", "b", "c"], ["a", "b", "a", "b"]
        )
        issue = next(i for i in report.issues if i.check == "category_level_drift")
        assert "1 removed" in issue.message
        assert "c" in issue.message
        assert "new (" not in issue.message
        assert issue.observed == 1

    def test_new_and_removed_categories_reported_together(self) -> None:
        report = _baseline_current(
            ["a", "b", "c"], ["a", "b", "d"]
        )
        issue = next(i for i in report.issues if i.check == "category_level_drift")
        assert "1 new" in issue.message
        assert "1 removed" in issue.message
        assert "d" in issue.message and "c" in issue.message
        assert issue.observed == 2

    def test_no_level_change_is_silent(self) -> None:
        report = _baseline_current(
            ["a", "b", "c", "a", "b", "c"], ["a", "b", "c", "a", "b", "c"]
        )
        assert not [i for i in report.issues if i.check == "category_level_drift"]

    def test_missing_values_do_not_count_as_levels(self) -> None:
        report = _baseline_current(
            pd.Series(["a", "b", None, "a", "b", None]),
            pd.Series(["a", "b", None, "a", "b", None]),
        )
        assert not [i for i in report.issues if i.check == "category_level_drift"]

    def test_numeric_columns_skip_level_drift(self) -> None:
        report = DatasetAuditor().audit_dataframe(
            pd.DataFrame({"revenue": [1.0, 2.0, 3.0]}),
            reference=pd.DataFrame({"revenue": [1.0, 2.0, 3.0]}),
        )
        assert not [i for i in report.issues if i.check == "category_level_drift"]


class TestCategoricalLevelDriftSurface:
    def test_finding_serializes_to_json(self) -> None:
        report = _baseline_current(["a", "b", "c"], ["a", "b", "d"])
        payload = json.loads(report.to_json())
        level = [i for i in payload["issues"] if i["check"] == "category_level_drift"]
        assert level and level[0]["column"] == "flag"

    def test_finding_renders_in_markdown(self) -> None:
        report = _baseline_current(["a", "b", "c"], ["a", "b", "d"])
        assert "category_level_drift" in report.to_markdown()

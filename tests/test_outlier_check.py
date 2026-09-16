"""Tests for the opt-in DatasetAuditor outlier / extreme-value check."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from dataset_audit_kit.core import DatasetAuditor


def _messages(report, check: str = "outliers") -> list[str]:
    return [issue.message for issue in report.issues if issue.check == check]


def _spiked_frame() -> pd.DataFrame:
    """Tight numeric cluster plus one extreme value, with a text column."""

    return pd.DataFrame(
        {
            "score": list(range(10, 30)) + [1000],
            "name": ["x"] * 21,
        }
    )


class TestOutlierCheckIqr:
    def test_flags_an_extreme_value_with_default_iqr_fences(self) -> None:
        report = DatasetAuditor(outlier_check=True).audit_dataframe(_spiked_frame())
        issues = [i for i in report.issues if i.check == "outliers"]
        assert len(issues) == 1
        issue = issues[0]
        assert issue.column == "score"
        assert issue.severity == "warning"
        assert issue.observed == 1
        assert issue.threshold == 0.0
        assert "IQR outlier" in issue.message
        assert "max=1000" in issue.message
        assert report.status == "warn"

    def test_clustered_values_stay_silent(self) -> None:
        data = pd.DataFrame({"score": list(range(100))})
        report = DatasetAuditor(outlier_check=True).audit_dataframe(data)
        assert _messages(report) == []

    def test_the_check_is_off_by_default(self) -> None:
        report = DatasetAuditor().audit_dataframe(_spiked_frame())
        assert _messages(report) == []

    def test_non_numeric_and_boolean_columns_are_skipped(self) -> None:
        data = pd.DataFrame(
            {
                "name": ["ann", "bo", "cy", "di", "eve", "fran"],
                "flag": [True, False, True, False, True, False],
            }
        )
        report = DatasetAuditor(outlier_check=True).audit_dataframe(data)
        assert _messages(report) == []

    def test_constant_column_is_skipped(self) -> None:
        data = pd.DataFrame({"score": [5] * 20})
        report = DatasetAuditor(outlier_check=True).audit_dataframe(data)
        assert _messages(report) == []

    def test_missing_and_non_finite_values_are_ignored(self) -> None:
        data = pd.DataFrame(
            {"score": list(range(10, 30)) + [None, float("nan"), 1000]}
        )
        report = DatasetAuditor(outlier_check=True).audit_dataframe(data)
        issues = [i for i in report.issues if i.check == "outliers"]
        assert len(issues) == 1
        assert issues[0].observed == 1
        assert "of 21 values" in issues[0].message

    def test_ratio_below_allowance_stays_silent(self) -> None:
        report = DatasetAuditor(
            outlier_check=True, outlier_max_ratio=0.10
        ).audit_dataframe(_spiked_frame())
        assert _messages(report) == []

    def test_wider_iqr_fence_can_absorb_the_spike(self) -> None:
        report = DatasetAuditor(
            outlier_check=True, outlier_method="iqr", outlier_threshold=200.0
        ).audit_dataframe(_spiked_frame())
        assert _messages(report) == []

    def test_short_series_is_skipped(self) -> None:
        data = pd.DataFrame({"score": [1.0, 2.0, 100.0]})
        report = DatasetAuditor(outlier_check=True).audit_dataframe(data)
        assert _messages(report) == []


class TestOutlierCheckZscore:
    def test_flags_values_beyond_the_zscore_cutoff(self) -> None:
        data = pd.DataFrame({"score": list(range(99)) + [1000]})
        report = DatasetAuditor(
            outlier_check=True, outlier_method="zscore", outlier_threshold=2.0
        ).audit_dataframe(data)
        issues = [i for i in report.issues if i.check == "outliers"]
        assert len(issues) == 1
        assert issues[0].column == "score"
        assert issues[0].observed == 1
        assert "z-score" in issues[0].message
        assert "|z| > 2" in issues[0].message

    def test_default_zscore_cutoff_is_three(self) -> None:
        auditor = DatasetAuditor(outlier_check=True, outlier_method="zscore")
        assert auditor.outlier_threshold == 3.0

    def test_no_flag_when_all_values_are_within_zscore(self) -> None:
        data = pd.DataFrame({"score": list(range(100))})
        report = DatasetAuditor(
            outlier_check=True, outlier_method="zscore", outlier_threshold=3.0
        ).audit_dataframe(data)
        assert _messages(report) == []


class TestOutlierCheckValidation:
    def test_unknown_method_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="outlier_method"):
            DatasetAuditor(outlier_method="mad")

    @pytest.mark.parametrize("bad", [0, -1.0, True, float("nan"), float("inf")])
    def test_threshold_must_be_a_positive_finite_number(self, bad: object) -> None:
        with pytest.raises(ValueError, match="outlier_threshold"):
            DatasetAuditor(outlier_threshold=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.1, True])
    def test_max_ratio_must_be_a_unit_interval(self, bad: object) -> None:
        with pytest.raises(ValueError, match="outlier_max_ratio"):
            DatasetAuditor(outlier_max_ratio=bad)


class TestOutlierCheckReports:
    def test_json_markdown_and_html_carry_column_level_detail(self) -> None:
        # Several extremes so the JSON outlier_summary (5% floor) also lists the column.
        data = pd.DataFrame(
            {"score": list(range(10, 30)) + [1000, 2000, -800], "name": ["x"] * 23}
        )
        report = DatasetAuditor(outlier_check=True).audit_dataframe(data)
        payload = json.loads(report.to_json())
        findings = [i for i in payload["issues"] if i["check"] == "outliers"]
        assert findings and findings[0]["column"] == "score"
        assert findings[0]["observed"] == 3
        assert payload["outlier_summary"]
        assert payload["outlier_summary"][0]["column"] == "score"

        markdown = report.to_markdown()
        assert "## Outliers" in markdown
        assert "`score`" in markdown
        assert "IQR outlier" in markdown
        assert "`outliers`" in markdown

        html = report.to_html()
        assert "<h2>Outliers</h2>" in html
        assert "score" in html
        assert "IQR outliers" in html

    def test_profile_iqr_detail_appears_even_when_the_check_is_off(self) -> None:
        report = DatasetAuditor().audit_dataframe(_spiked_frame())
        markdown = report.to_markdown()
        assert "## Outliers" in markdown
        assert "IQR outliers" in markdown
        html = report.to_html()
        assert "<h2>Outliers</h2>" in html

    def test_fix_suggestion_recommends_winsorizing(self) -> None:
        report = DatasetAuditor(outlier_check=True).audit_dataframe(_spiked_frame())
        suggestion = next(s for s in report.fix_suggestions if s["action"] == "winsorize")
        assert "score" in suggestion["code"]
        assert "clip" in suggestion["code"]

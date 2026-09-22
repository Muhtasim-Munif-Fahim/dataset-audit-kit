"""Tests for the opt-in variance inflation factor multicollinearity check."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from dataset_audit_kit.core import DEFAULT_VIF_THRESHOLD, DatasetAuditor


def _vif_issues(report) -> list:
    return [issue for issue in report.issues if issue.check == "vif"]


def _perfect_pair() -> pd.DataFrame:
    return pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]}
    )


def _population_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_z = (left - left.mean()) / left.std(ddof=0)
    right_z = (right - right.mean()) / right.std(ddof=0)
    return float(np.dot(left_z, right_z) / left.size)


class TestVifCheck:
    def test_perfect_collinearity_is_an_infinite_vif(self) -> None:
        report = DatasetAuditor(vif_check=True).audit_dataframe(_perfect_pair())
        assert report.vif_scores == {"a": None, "b": None}
        issues = _vif_issues(report)
        assert {issue.column for issue in issues} == {"a", "b"}
        assert all(issue.severity == "warning" for issue in issues)
        assert all(issue.observed is None for issue in issues)
        assert all(issue.threshold == DEFAULT_VIF_THRESHOLD for issue in issues)
        assert all("infinite" in issue.message for issue in issues)
        assert any("most aligned with 'b'" in issue.message for issue in issues)
        assert report.status == "warn"

    def test_the_check_is_off_by_default(self) -> None:
        report = DatasetAuditor().audit_dataframe(_perfect_pair())
        assert report.vif_scores == {}
        assert _vif_issues(report) == []
        assert DatasetAuditor().vif_check is False
        assert DatasetAuditor().vif_threshold == DEFAULT_VIF_THRESHOLD

    def test_two_column_vif_matches_the_correlation_formula(self) -> None:
        rng = np.random.default_rng(7)
        left = rng.normal(size=500)
        right = 0.5 * left + rng.normal(size=500)
        data = pd.DataFrame({"x": left, "y": right})
        report = DatasetAuditor(vif_check=True, vif_threshold=1.01).audit_dataframe(data)
        correlation = _population_correlation(left, right)
        expected = 1.0 / (1.0 - correlation * correlation)
        assert report.vif_scores["x"] == pytest.approx(expected, rel=1e-6)
        assert report.vif_scores["y"] == pytest.approx(expected, rel=1e-6)
        assert {issue.column for issue in _vif_issues(report)} == {"x", "y"}

    def test_orthogonal_columns_stay_under_the_default_threshold(self) -> None:
        rng = np.random.default_rng(0)
        data = pd.DataFrame(
            {
                "a": rng.normal(size=400),
                "b": rng.normal(size=400),
                "c": rng.normal(size=400),
            }
        )
        report = DatasetAuditor(vif_check=True).audit_dataframe(data)
        assert set(report.vif_scores) == {"a", "b", "c"}
        assert all(score is not None and score < 2.0 for score in report.vif_scores.values())
        assert _vif_issues(report) == []

    def test_multivariable_collinearity_is_caught_without_a_pairwise_hit(self) -> None:
        rng = np.random.default_rng(1)
        rows = 300
        x1 = rng.normal(size=rows)
        x2 = rng.normal(size=rows)
        x3 = x1 + x2 + rng.normal(scale=0.05, size=rows)
        data = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "note": ["ok"] * rows})
        report = DatasetAuditor(vif_check=True).audit_dataframe(data)
        assert {issue.column for issue in _vif_issues(report)} == {"x1", "x2", "x3"}
        assert [issue for issue in report.issues if issue.check == "redundancy"] == []
        assert all(
            score is None or score >= DEFAULT_VIF_THRESHOLD
            for score in report.vif_scores.values()
        )

    def test_lowering_the_threshold_catches_moderate_correlation(self) -> None:
        rng = np.random.default_rng(3)
        rows = 800
        left = rng.normal(size=rows)
        right = 0.9 * left + math.sqrt(1.0 - 0.81) * rng.normal(size=rows)
        data = pd.DataFrame({"x": left, "y": right})
        silent = DatasetAuditor(vif_check=True).audit_dataframe(data)
        flagged = DatasetAuditor(vif_check=True, vif_threshold=5).audit_dataframe(data)
        correlation = _population_correlation(left, right)
        expected = 1.0 / (1.0 - correlation * correlation)
        assert 5.0 < expected < DEFAULT_VIF_THRESHOLD
        assert silent.vif_scores["x"] == pytest.approx(expected, rel=1e-6)
        assert _vif_issues(silent) == []
        assert {issue.column for issue in _vif_issues(flagged)} == {"x", "y"}
        assert all(issue.threshold == 5 for issue in _vif_issues(flagged))

    def test_label_column_is_excluded_from_the_design_matrix(self) -> None:
        rng = np.random.default_rng(2)
        feature = rng.normal(size=120)
        other = rng.normal(size=120)
        data = pd.DataFrame({"x": feature, "y": other, "target": feature * 3.0 + 1.0})
        included = DatasetAuditor(vif_check=True).audit_dataframe(data)
        excluded = DatasetAuditor(vif_check=True).audit_dataframe(
            data, label_column="target"
        )
        assert included.vif_scores["x"] is None
        assert included.vif_scores["target"] is None
        assert included.vif_scores["y"] < 2.0
        assert set(excluded.vif_scores) == {"x", "y"}
        assert all(score < 2.0 for score in excluded.vif_scores.values())
        assert _vif_issues(excluded) == []

    def test_non_numeric_boolean_and_constant_columns_are_skipped(self) -> None:
        text_only = pd.DataFrame(
            {
                "name": ["ann", "bo", "cy", "di", "eve", "fran"],
                "flag": [True, False, True, False, True, False],
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        assert DatasetAuditor(vif_check=True).audit_dataframe(text_only).vif_scores == {}

        with_constant = _perfect_pair().assign(c=[5.0] * 6)
        report = DatasetAuditor(vif_check=True).audit_dataframe(with_constant)
        assert set(report.vif_scores) == {"a", "b"}

    def test_short_and_single_column_frames_are_skipped(self) -> None:
        short = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 4.0]})
        single = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "name": list("abcd")})
        assert DatasetAuditor(vif_check=True).audit_dataframe(short).vif_scores == {}
        assert DatasetAuditor(vif_check=True).audit_dataframe(single).vif_scores == {}

    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    def test_missing_and_non_finite_rows_are_dropped(self) -> None:
        data = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, None, float("inf")],
                "b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 99.0, 0.0],
            }
        )
        report = DatasetAuditor(vif_check=True).audit_dataframe(data)
        assert report.vif_scores == {"a": None, "b": None}

    @pytest.mark.parametrize("bad", [0, -1.0, True, float("nan"), float("inf")])
    def test_threshold_must_be_a_positive_finite_number(self, bad: object) -> None:
        with pytest.raises(ValueError, match="vif_threshold"):
            DatasetAuditor(vif_threshold=bad)  # type: ignore[arg-type]


class TestVifReports:
    def test_json_markdown_html_and_fix_suggestion(self) -> None:
        report = DatasetAuditor(vif_check=True).audit_dataframe(_perfect_pair())
        payload = json.loads(report.to_json())
        assert payload["vif_scores"] == {"a": None, "b": None}
        findings = [issue for issue in payload["issues"] if issue["check"] == "vif"]
        assert findings and findings[0]["observed"] is None
        assert findings[0]["threshold"] == DEFAULT_VIF_THRESHOLD

        markdown = report.to_markdown()
        assert "## Variance inflation factors" in markdown
        assert "`a`: infinite" in markdown
        assert "`vif`" in markdown

        html = report.to_html()
        assert "<h2>Variance inflation factors</h2>" in html
        assert "infinite" in html

        suggestion = next(
            item for item in report.fix_suggestions if item["action"] == "drop_collinear"
        )
        assert "drop(columns=" in suggestion["code"]

    def test_disabled_check_omits_the_vif_section(self) -> None:
        report = DatasetAuditor().audit_dataframe(_perfect_pair())
        assert json.loads(report.to_json())["vif_scores"] == {}
        assert "Variance inflation factors" not in report.to_markdown()
        assert "Variance inflation factors" not in report.to_html()

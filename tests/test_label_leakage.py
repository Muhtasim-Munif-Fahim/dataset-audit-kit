"""Tests for the opt-in label-leakage association check."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dataset_audit_kit.core import DEFAULT_LABEL_LEAKAGE_THRESHOLD, DatasetAuditor


def _leak_issues(report) -> list:
    return [issue for issue in report.issues if issue.check == "label_leakage"]


def _copied_numeric() -> pd.DataFrame:
    values = np.arange(40, dtype=float)
    return pd.DataFrame({"amount": values, "target": values * 2.0 + 1.0})


def _coded_labels(repeats: int = 40) -> pd.DataFrame:
    labels = np.tile(np.array(["cat", "dog", "bird"]), repeats)
    codes = np.tile(np.array(["c", "d", "b"]), repeats)
    return pd.DataFrame(
        {
            "row_id": np.arange(labels.size),
            "code": codes,
            "noise": np.resize(np.array(["w", "x", "y", "z"]), labels.size),
            "target": labels,
        }
    )


class TestLabelLeakageCheck:
    def test_numeric_copy_is_perfectly_associated(self) -> None:
        report = DatasetAuditor(label_leakage_check=True).audit_dataframe(
            _copied_numeric(), label_column="target"
        )
        score = report.label_leakage_scores["amount"]
        assert score["correlation"] == pytest.approx(1.0)
        assert score["mutual_information"] == pytest.approx(1.0)
        issues = _leak_issues(report)
        assert [issue.column for issue in issues] == ["amount"]
        assert issues[0].severity == "warning"
        assert issues[0].observed == pytest.approx(1.0)
        assert issues[0].threshold == DEFAULT_LABEL_LEAKAGE_THRESHOLD
        assert "label 'target'" in issues[0].message
        assert "|r|=1.000" in issues[0].message
        assert "mutual information=1.000" in issues[0].message
        assert "target" not in report.label_leakage_scores

    def test_the_check_is_off_by_default(self) -> None:
        report = DatasetAuditor().audit_dataframe(
            _copied_numeric(), label_column="target"
        )
        assert report.label_leakage_scores == {}
        assert _leak_issues(report) == []
        assert DatasetAuditor().label_leakage_check is False
        assert DatasetAuditor().label_leakage_threshold == DEFAULT_LABEL_LEAKAGE_THRESHOLD

    def test_categorical_bijection_is_mutual_information_only(self) -> None:
        report = DatasetAuditor(label_leakage_check=True).audit_dataframe(
            _coded_labels(), label_column="target"
        )
        code = report.label_leakage_scores["code"]
        assert code["correlation"] is None
        assert code["mutual_information"] == pytest.approx(1.0)
        assert [issue.column for issue in _leak_issues(report)] == ["code"]
        message = _leak_issues(report)[0].message
        assert "mutual information=1.000" in message
        assert "|r|" not in message
        noise = report.label_leakage_scores["noise"]
        assert noise["correlation"] is None
        assert noise["mutual_information"] < 0.2
        assert "row_id" in report.label_leakage_scores
        assert report.label_leakage_scores["row_id"]["mutual_information"] < 0.2

    def test_mutual_information_flags_a_nonlinear_numeric_relationship(self) -> None:
        rng = np.random.default_rng(0)
        label = rng.normal(size=800)
        data = pd.DataFrame({"squared": label**2, "target": label})
        silent = DatasetAuditor(label_leakage_check=True).audit_dataframe(
            data, label_column="target"
        )
        flagged = DatasetAuditor(
            label_leakage_check=True, label_leakage_threshold=0.5
        ).audit_dataframe(data, label_column="target")
        score = silent.label_leakage_scores["squared"]
        assert score["correlation"] < 0.3
        assert 0.5 < score["mutual_information"] < DEFAULT_LABEL_LEAKAGE_THRESHOLD
        assert _leak_issues(silent) == []
        issues = _leak_issues(flagged)
        assert [issue.column for issue in issues] == ["squared"]
        assert issues[0].threshold == 0.5
        assert issues[0].observed == pytest.approx(score["mutual_information"], abs=1e-4)

    def test_numeric_feature_that_determines_a_categorical_label(self) -> None:
        rng = np.random.default_rng(1)
        measurement = rng.normal(size=800)
        data = pd.DataFrame(
            {
                "measurement": measurement,
                "other": rng.normal(size=800),
                "target": np.where(measurement > 0.0, "pos", "neg"),
            }
        )
        report = DatasetAuditor(label_leakage_check=True).audit_dataframe(
            data, label_column="target"
        )
        leaked = report.label_leakage_scores["measurement"]
        assert leaked["correlation"] is None
        assert leaked["mutual_information"] >= DEFAULT_LABEL_LEAKAGE_THRESHOLD
        assert [issue.column for issue in _leak_issues(report)] == ["measurement"]
        other = report.label_leakage_scores["other"]
        assert other["mutual_information"] < 0.2

    def test_lowering_the_threshold_catches_moderate_correlation(self) -> None:
        rng = np.random.default_rng(0)
        rows = 1000
        label = np.array([0.0, 1.0] * (rows // 2))
        feature = label + rng.normal(scale=0.32, size=rows)
        data = pd.DataFrame({"score": feature, "target": label})
        silent = DatasetAuditor(label_leakage_check=True).audit_dataframe(
            data, label_column="target"
        )
        flagged = DatasetAuditor(
            label_leakage_check=True, label_leakage_threshold=0.8
        ).audit_dataframe(data, label_column="target")
        correlation = silent.label_leakage_scores["score"]["correlation"]
        assert 0.8 < correlation < DEFAULT_LABEL_LEAKAGE_THRESHOLD
        assert _leak_issues(silent) == []
        issues = _leak_issues(flagged)
        assert [issue.column for issue in issues] == ["score"]
        assert issues[0].threshold == 0.8
        assert issues[0].observed == pytest.approx(correlation, abs=1e-4)

    def test_independent_features_stay_under_the_default_threshold(self) -> None:
        rng = np.random.default_rng(2)
        rows = 400
        data = pd.DataFrame(
            {
                "a": rng.normal(size=rows),
                "b": rng.normal(size=rows),
                "target": rng.normal(size=rows),
            }
        )
        report = DatasetAuditor(label_leakage_check=True).audit_dataframe(
            data, label_column="target"
        )
        assert set(report.label_leakage_scores) == {"a", "b"}
        for score in report.label_leakage_scores.values():
            assert score["correlation"] < 0.3
            assert score["mutual_information"] < 0.3
        assert _leak_issues(report) == []

    def test_boolean_label_copy_is_flagged(self) -> None:
        label = np.array([True, False] * 30)
        data = pd.DataFrame({"flag": label.copy(), "target": label})
        report = DatasetAuditor(label_leakage_check=True).audit_dataframe(
            data, label_column="target"
        )
        score = report.label_leakage_scores["flag"]
        assert score["correlation"] == pytest.approx(1.0)
        assert score["mutual_information"] == pytest.approx(1.0)
        assert [issue.column for issue in _leak_issues(report)] == ["flag"]

    def test_no_label_column_scores_nothing(self) -> None:
        data = _copied_numeric()
        missing = DatasetAuditor(label_leakage_check=True).audit_dataframe(data)
        unknown = DatasetAuditor(label_leakage_check=True).audit_dataframe(
            data, label_column="missing"
        )
        assert missing.label_leakage_scores == {}
        assert unknown.label_leakage_scores == {}
        assert _leak_issues(missing) == []
        assert _leak_issues(unknown) == []

    def test_identifier_like_categorical_is_not_scored(self) -> None:
        rows = 40
        data = pd.DataFrame(
            {
                "user_id": [f"u{i}" for i in range(rows)],
                "target": ["yes", "no"] * (rows // 2),
            }
        )
        report = DatasetAuditor(label_leakage_check=True).audit_dataframe(
            data, label_column="target"
        )
        assert report.label_leakage_scores == {}
        assert _leak_issues(report) == []

    def test_datetime_label_is_skipped(self) -> None:
        data = pd.DataFrame(
            {
                "amount": [1.0, 2.0, 3.0, 4.0],
                "target": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
                ),
            }
        )
        report = DatasetAuditor(label_leakage_check=True).audit_dataframe(
            data, label_column="target"
        )
        assert report.label_leakage_scores == {}
        assert _leak_issues(report) == []

    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    def test_constant_short_and_non_finite_rows(self) -> None:
        constant = pd.DataFrame(
            {"amount": [1.0, 1.0, 1.0, 1.0], "target": [0.0, 1.0, 0.0, 1.0]}
        )
        assert (
            DatasetAuditor(label_leakage_check=True)
            .audit_dataframe(constant, label_column="target")
            .label_leakage_scores
            == {}
        )

        short = pd.DataFrame({"amount": [1.0, 2.0], "target": [1.0, 2.0]})
        assert (
            DatasetAuditor(label_leakage_check=True)
            .audit_dataframe(short, label_column="target")
            .label_leakage_scores
            == {}
        )

        leaky = pd.DataFrame(
            {
                "amount": [1.0, 2.0, 3.0, 4.0, 5.0, float("inf"), None],
                "target": [2.0, 4.0, 6.0, 8.0, 10.0, 0.0, 99.0],
            }
        )
        report = DatasetAuditor(label_leakage_check=True).audit_dataframe(
            leaky, label_column="target"
        )
        assert report.label_leakage_scores["amount"]["correlation"] == pytest.approx(1.0)

    def test_duplicate_feature_names_are_scored_once(self) -> None:
        frame = _copied_numeric()
        frame = pd.concat([frame["amount"], frame["amount"], frame["target"]], axis=1)
        frame.columns = ["amount", "amount", "target"]
        report = DatasetAuditor(label_leakage_check=True).audit_dataframe(
            frame, label_column="target"
        )
        assert list(report.label_leakage_scores) == ["amount"]

    @pytest.mark.parametrize(
        "bad", [0, -0.1, 1.1, True, float("nan"), float("inf")]
    )
    def test_threshold_must_be_in_the_open_unit_interval(self, bad: object) -> None:
        with pytest.raises(ValueError, match="label_leakage_threshold"):
            DatasetAuditor(label_leakage_threshold=bad)  # type: ignore[arg-type]

    def test_threshold_of_one_accepts_only_a_perfect_association(self) -> None:
        perfect = DatasetAuditor(
            label_leakage_check=True, label_leakage_threshold=1
        ).audit_dataframe(_copied_numeric(), label_column="target")
        assert _leak_issues(perfect)

        rng = np.random.default_rng(0)
        rows = 1000
        label = np.array([0.0, 1.0] * (rows // 2))
        moderate = pd.DataFrame(
            {
                "score": label + rng.normal(scale=0.32, size=rows),
                "target": label,
            }
        )
        report = DatasetAuditor(
            label_leakage_check=True, label_leakage_threshold=1
        ).audit_dataframe(moderate, label_column="target")
        assert _leak_issues(report) == []


class TestLabelLeakageReports:
    def test_json_markdown_html_and_fix_suggestion(self) -> None:
        report = DatasetAuditor(label_leakage_check=True).audit_dataframe(
            _coded_labels(), label_column="target"
        )
        payload = json.loads(report.to_json())
        assert payload["label_leakage_scores"]["code"] == {
            "correlation": None,
            "mutual_information": 1.0,
        }
        findings = [
            issue for issue in payload["issues"] if issue["check"] == "label_leakage"
        ]
        assert findings and findings[0]["column"] == "code"
        assert findings[0]["threshold"] == DEFAULT_LABEL_LEAKAGE_THRESHOLD

        markdown = report.to_markdown()
        assert "## Label leakage" in markdown
        assert "`code`: |r|=n/a, mutual information=1.000" in markdown
        assert "`label_leakage`" in markdown

        html = report.to_html()
        assert "<h2>Label leakage</h2>" in html
        assert "Mutual information" in html
        assert ">code</td>" in html

        suggestion = next(
            item
            for item in report.fix_suggestions
            if item["action"] == "drop_leaking_feature"
        )
        assert "drop(columns=['code'])" in suggestion["code"]

    def test_disabled_check_omits_the_leakage_section(self) -> None:
        report = DatasetAuditor().audit_dataframe(
            _coded_labels(), label_column="target"
        )
        assert json.loads(report.to_json())["label_leakage_scores"] == {}
        assert "Label leakage" not in report.to_markdown()
        assert "Label leakage" not in report.to_html()

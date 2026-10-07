"""Tests for the opt-in target-encoding leakage check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataset_audit_kit.core import (
    DEFAULT_TARGET_ENCODING_LEAKAGE_THRESHOLD,
    DatasetAuditor,
)


def _te_issues(report):
    return [issue for issue in report.issues if issue.check == "target_encoding_leakage"]


def _leaky_frame(n: int = 90, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = np.array(["a", "b", "c"])[rng.integers(0, 3, size=n)]
    base = np.where(cats == "a", 0.1, np.where(cats == "b", 0.5, 0.9))
    label = base + rng.normal(0.0, 0.01, size=n)
    means = {c: float(label[cats == c].mean()) for c in ("a", "b", "c")}
    encoded = np.asarray([means[c] for c in cats], dtype=float)
    return pd.DataFrame(
        {
            "group": cats,
            "te": encoded,
            "noise": rng.normal(size=n),
            "y": label,
        }
    )


class TestTargetEncodingLeakageCheck:
    def test_exact_in_sample_means_are_flagged(self) -> None:
        data = _leaky_frame()
        report = DatasetAuditor(target_encoding_leakage_check=True).audit_dataframe(
            data, label_column="y"
        )
        score = report.target_encoding_leakage_scores["te|group"]
        assert score["match_correlation"] == pytest.approx(1.0)
        assert score["match_mae"] == pytest.approx(0.0)
        issues = _te_issues(report)
        assert len(issues) == 1
        assert issues[0].column == "te"
        assert issues[0].severity == "warning"
        assert issues[0].observed == pytest.approx(1.0)
        assert issues[0].threshold == DEFAULT_TARGET_ENCODING_LEAKAGE_THRESHOLD
        assert "group" in issues[0].message
        assert "target-encoding leakage" in issues[0].message

    def test_check_is_off_by_default(self) -> None:
        report = DatasetAuditor().audit_dataframe(_leaky_frame(), label_column="y")
        assert report.target_encoding_leakage_scores == {}
        assert _te_issues(report) == []
        assert DatasetAuditor().target_encoding_leakage_check is False
        assert (
            DatasetAuditor().target_encoding_leakage_threshold
            == DEFAULT_TARGET_ENCODING_LEAKAGE_THRESHOLD
        )

    def test_noise_feature_is_not_flagged(self) -> None:
        report = DatasetAuditor(target_encoding_leakage_check=True).audit_dataframe(
            _leaky_frame(), label_column="y"
        )
        noise = report.target_encoding_leakage_scores["noise|group"]
        assert noise["match_correlation"] < 0.5
        assert all(issue.column != "noise" for issue in _te_issues(report))

    def test_threshold_filters_borderline_matches(self) -> None:
        rng = np.random.default_rng(2)
        cats = np.array(["x", "y"])[rng.integers(0, 2, size=80)]
        label = np.where(cats == "x", 0.0, 1.0).astype(float)
        means = {c: float(label[cats == c].mean()) for c in ("x", "y")}
        encoded = np.asarray([means[c] for c in cats], dtype=float)
        # Mix the true encoding with noise so |r| is high but below 0.99.
        mixed = 0.85 * encoded + 0.15 * rng.normal(size=80)
        data = pd.DataFrame({"group": cats, "almost": mixed, "y": label})
        silent = DatasetAuditor(
            target_encoding_leakage_check=True,
            target_encoding_leakage_threshold=0.99,
        ).audit_dataframe(data, label_column="y")
        loud = DatasetAuditor(
            target_encoding_leakage_check=True,
            target_encoding_leakage_threshold=0.5,
        ).audit_dataframe(data, label_column="y")
        corr = silent.target_encoding_leakage_scores["almost|group"]["match_correlation"]
        assert 0.5 < corr < 0.99
        assert _te_issues(silent) == []
        assert [i.column for i in _te_issues(loud)] == ["almost"]

    def test_requires_label_column(self) -> None:
        report = DatasetAuditor(target_encoding_leakage_check=True).audit_dataframe(
            _leaky_frame()
        )
        assert report.target_encoding_leakage_scores == {}
        assert _te_issues(report) == []

    def test_skips_non_numeric_labels(self) -> None:
        data = pd.DataFrame(
            {
                "group": ["a", "b", "a", "b"] * 10,
                "te": [0.1, 0.9, 0.1, 0.9] * 10,
                "y": ["yes", "no", "yes", "no"] * 10,
            }
        )
        report = DatasetAuditor(target_encoding_leakage_check=True).audit_dataframe(
            data, label_column="y"
        )
        assert report.target_encoding_leakage_scores == {}

    def test_rejects_bad_threshold(self) -> None:
        with pytest.raises(ValueError, match="target_encoding_leakage_threshold"):
            DatasetAuditor(target_encoding_leakage_threshold=0.0)
        with pytest.raises(ValueError, match="target_encoding_leakage_threshold"):
            DatasetAuditor(target_encoding_leakage_threshold=1.5)

    def test_report_json_includes_scores(self) -> None:
        report = DatasetAuditor(target_encoding_leakage_check=True).audit_dataframe(
            _leaky_frame(), label_column="y"
        )
        payload = report.to_dict()
        assert "te|group" in payload["target_encoding_leakage_scores"]
        markdown = report.to_markdown()
        assert "Target-encoding leakage" in markdown
        assert "te|group" in markdown

    def test_fix_suggestion_mentions_out_of_fold(self) -> None:
        report = DatasetAuditor(target_encoding_leakage_check=True).audit_dataframe(
            _leaky_frame(), label_column="y"
        )
        suggestions = report.fix_suggestions
        te = [s for s in suggestions if s.get("action") == "refit_target_encoding_out_of_fold"]
        assert te
        assert "out-of-fold" in te[0]["description"].lower() or "out-of-fold" in te[0]["code"]

    def test_loo_style_encoding_is_less_correlated_than_in_sample(self) -> None:
        """Leave-one-out means should not match in-sample means exactly."""
        rng = np.random.default_rng(4)
        cats = np.array(["a", "b", "c"])[rng.integers(0, 3, size=120)]
        label = rng.normal(size=120)
        # Leave-one-out target encoding for each row.
        loo = np.empty(120)
        for i in range(120):
            mask = (cats == cats[i]) & (np.arange(120) != i)
            loo[i] = float(label[mask].mean()) if mask.any() else 0.0
        data = pd.DataFrame({"group": cats, "loo": loo, "y": label})
        report = DatasetAuditor(
            target_encoding_leakage_check=True,
            target_encoding_leakage_threshold=0.999,
        ).audit_dataframe(data, label_column="y")
        # LOO encodings are close but typically not a perfect match to in-sample means.
        corr = report.target_encoding_leakage_scores["loo|group"]["match_correlation"]
        assert corr < 1.0

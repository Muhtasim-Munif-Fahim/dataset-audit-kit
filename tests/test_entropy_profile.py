"""Tests for Shannon entropy in categorical column profiles."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from dataset_audit_kit.core import DatasetAuditor


class TestEntropyInProfiles:
    def test_uniform_distribution_has_max_entropy(self) -> None:
        data = pd.DataFrame({"c": ["a", "b", "c", "d"]})
        profile = DatasetAuditor._profile_columns(data)["c"]
        assert "entropy" in profile
        assert profile["entropy"] == pytest.approx(2.0, abs=1e-9)
        assert profile["normalized_entropy"] == pytest.approx(1.0, abs=1e-9)

    def test_single_category_has_zero_entropy(self) -> None:
        data = pd.DataFrame({"c": ["a", "a", "a", "a"]})
        profile = DatasetAuditor._profile_columns(data)["c"]
        assert profile["entropy"] == 0.0
        assert profile["normalized_entropy"] == 0.0

    def test_skewed_distribution_has_low_normalized_entropy(self) -> None:
        data = pd.DataFrame({"c": ["a"] * 90 + ["b"] * 10})
        profile = DatasetAuditor._profile_columns(data)["c"]
        assert profile["entropy"] < 1.0
        assert 0.0 < profile["normalized_entropy"] < 1.0
        expected = -(0.9 * math.log2(0.9) + 0.1 * math.log2(0.1))
        assert profile["entropy"] == pytest.approx(expected, rel=1e-9)

    def test_entropy_is_bounded_by_log2_of_unique(self) -> None:
        data = pd.DataFrame({"c": ["a", "b", "c", "d"]})
        profile = DatasetAuditor._profile_columns(data)["c"]
        assert profile["entropy"] <= 2.0

    def test_all_same_non_string_values(self) -> None:
        data = pd.DataFrame({"c": [1, 2, 1, 2, 1, 1]})
        profile = DatasetAuditor._profile_columns(data)["c"]
        assert profile["dtype"] == "numeric"
        assert "entropy" not in profile

    def test_entropy_handles_missing_values(self) -> None:
        data = pd.DataFrame({"c": ["a", "b", None, "a"]})
        profile = DatasetAuditor._profile_columns(data)["c"]
        assert "entropy" in profile
        assert profile["entropy"] >= 0.0

    def test_two_categories_even_split(self) -> None:
        data = pd.DataFrame({"c": ["a", "b", "a", "b"]})
        profile = DatasetAuditor._profile_columns(data)["c"]
        assert profile["entropy"] == pytest.approx(1.0, abs=1e-9)
        assert profile["normalized_entropy"] == pytest.approx(1.0, abs=1e-9)


class TestEntropyRendering:
    def test_entropy_appears_in_markdown(self) -> None:
        data = pd.DataFrame({"c": ["a", "b", "c", "d"]})
        report = DatasetAuditor().audit_dataframe(data)
        text = report.to_markdown()
        assert "Entropy" in text
        assert "normalized" in text

    def test_entropy_appears_in_html(self) -> None:
        data = pd.DataFrame({"c": ["a", "b", "c", "d"]})
        report = DatasetAuditor().audit_dataframe(data)
        html = report.to_html()
        assert "Entropy" in html
        assert "Normalized entropy" in html

    def test_entropy_not_shown_for_numeric_columns(self) -> None:
        data = pd.DataFrame({"n": [1, 2, 3, 4]})
        report = DatasetAuditor().audit_dataframe(data)
        text = report.to_markdown()
        assert "Entropy" not in text

    def test_entropy_serialized_in_json(self) -> None:
        data = pd.DataFrame({"c": ["a", "b", "c", "d"]})
        report = DatasetAuditor().audit_dataframe(data)
        import json
        payload = json.loads(report.to_json())
        profile = payload["column_profiles"]["c"]
        assert "entropy" in profile
        assert "normalized_entropy" in profile

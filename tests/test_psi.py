"""Tests for the Population Stability Index (PSI) drift metric."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataset_audit_kit.core import DatasetAuditor


class TestPopulationStabilityIndex:
    def test_identical_distributions_yield_zero_psi(self) -> None:
        baseline = pd.Series(np.linspace(0.0, 100.0, 200))
        psi = DatasetAuditor.population_stability_index(baseline, baseline)
        assert psi == pytest.approx(0.0, abs=1e-9)

    def test_shifted_distribution_yields_high_psi(self) -> None:
        baseline = pd.Series(np.linspace(0.0, 100.0, 200))
        shifted = pd.Series(np.linspace(100.0, 200.0, 200))
        psi = DatasetAuditor.population_stability_index(baseline, shifted)
        assert psi > 0.25

    def test_small_shift_yields_small_psi(self) -> None:
        baseline = pd.Series(np.linspace(0.0, 100.0, 200))
        shifted = pd.Series(np.linspace(5.0, 105.0, 200))
        psi = DatasetAuditor.population_stability_index(baseline, shifted)
        assert 0.0 < psi < 0.1

    def test_too_few_baseline_values_returns_zero(self) -> None:
        baseline = pd.Series([5.0])
        current = pd.Series([5.0, 6.0])
        assert DatasetAuditor.population_stability_index(baseline, current) == 0.0

    def test_empty_current_returns_zero(self) -> None:
        baseline = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert DatasetAuditor.population_stability_index(baseline, pd.Series([], dtype=float)) == 0.0

    def test_non_numeric_values_are_coerced(self) -> None:
        baseline = pd.Series(["1", "2", "3", "4"])
        current = pd.Series(["1", "2", "3", "4"])
        assert DatasetAuditor.population_stability_index(baseline, current) == pytest.approx(0.0, abs=1e-9)

    def test_respects_bin_count(self) -> None:
        baseline = pd.Series(np.linspace(0.0, 100.0, 200))
        shifted = pd.Series(np.linspace(100.0, 200.0, 200))
        coarse = DatasetAuditor.population_stability_index(baseline, shifted, bins=4)
        fine = DatasetAuditor.population_stability_index(baseline, shifted, bins=20)
        assert coarse > 0.25
        assert fine > 0.25


class TestPsiDriftIntegration:
    def test_psi_score_recorded_in_drift_scores(self) -> None:
        baseline = pd.DataFrame({"revenue": np.linspace(0.0, 100.0, 200)})
        current = pd.DataFrame({"revenue": np.linspace(0.0, 100.0, 200)})
        report = DatasetAuditor().audit_dataframe(
            current, reference=baseline
        )
        assert "revenue__psi" in report.drift_scores
        assert report.drift_scores["revenue__psi"] == pytest.approx(0.0, abs=1e-9)

    def test_psi_emits_no_issue_for_stable_distribution(self) -> None:
        baseline = pd.DataFrame({"revenue": np.linspace(0.0, 100.0, 200)})
        current = pd.DataFrame({"revenue": np.linspace(0.0, 100.0, 200)})
        report = DatasetAuditor().audit_dataframe(
            current, reference=baseline
        )
        assert not [i for i in report.issues if i.check == "psi"]

    def test_psi_emits_issue_when_above_threshold(self) -> None:
        baseline = pd.DataFrame({"revenue": np.linspace(0.0, 100.0, 200)})
        shifted = pd.DataFrame({"revenue": np.linspace(100.0, 200.0, 200)})
        report = DatasetAuditor(drift_threshold=0.1).audit_dataframe(
            shifted, reference=baseline
        )
        issue = next(i for i in report.issues if i.check == "psi")
        assert issue.column == "revenue"
        assert issue.observed == pytest.approx(report.drift_scores["revenue__psi"])
        assert issue.threshold == 0.1
        assert "PSI" in issue.message

    def test_psi_threshold_respects_column_rule_max_drift(self) -> None:
        from dataset_audit_kit.core import ValidationRules

        rules = ValidationRules.from_dict({"revenue": {"max_drift": 0.05}})
        baseline = pd.DataFrame({"revenue": np.linspace(0.0, 100.0, 200)})
        shifted = pd.DataFrame({"revenue": np.linspace(100.0, 200.0, 200)})
        report = DatasetAuditor(
            drift_threshold=0.20, rules=rules
        ).audit_dataframe(shifted, reference=baseline)
        issue = next(i for i in report.issues if i.check == "psi")
        assert issue.threshold == 0.05

    def test_psi_skipped_for_categorical_columns(self) -> None:
        baseline = pd.DataFrame({"flag": ["a"] * 200})
        current = pd.DataFrame({"flag": ["b"] * 200})
        report = DatasetAuditor().audit_dataframe(
            current, reference=baseline
        )
        assert "flag__psi" not in report.drift_scores
        assert not [i for i in report.issues if i.check == "psi"]

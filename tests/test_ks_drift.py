"""Tests for the Kolmogorov-Smirnov drift significance test."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dataset_audit_kit.cli import main
from dataset_audit_kit.core import DatasetAuditor


class TestKolmogorovSmirnovTest:
    def test_identical_distributions_return_zero_statistic_and_one_pvalue(self) -> None:
        baseline = pd.Series(np.linspace(0.0, 100.0, 200))
        d, p = DatasetAuditor.kolmogorov_smirnov_test(baseline, baseline)
        assert d == pytest.approx(0.0, abs=1e-9)
        assert p == pytest.approx(1.0, abs=1e-9)

    def test_disjoint_distributions_return_large_statistic_and_tiny_pvalue(self) -> None:
        baseline = pd.Series(np.linspace(0.0, 100.0, 200))
        current = pd.Series(np.linspace(150.0, 250.0, 200))
        d, p = DatasetAuditor.kolmogorov_smirnov_test(baseline, current)
        assert d == pytest.approx(1.0, abs=1e-9)
        assert p < 0.05

    def test_insufficient_samples_return_passthrough_values(self) -> None:
        assert DatasetAuditor.kolmogorov_smirnov_test(
            pd.Series([5.0]), pd.Series([5.0, 6.0])
        ) == (0.0, 1.0)

    def test_non_numeric_values_are_coerced(self) -> None:
        d, p = DatasetAuditor.kolmogorov_smirnov_test(
            pd.Series(["1", "2", "3", "4"]), pd.Series(["1", "2", "3", "4"])
        )
        assert d == pytest.approx(0.0, abs=1e-9)
        assert p == pytest.approx(1.0, abs=1e-9)

    def test_pvalue_is_bounded_between_zero_and_one(self) -> None:
        baseline = pd.Series(np.linspace(0.0, 100.0, 200))
        current = pd.Series(np.linspace(150.0, 250.0, 200))
        _, p = DatasetAuditor.kolmogorov_smirnov_test(baseline, current)
        assert 0.0 <= p <= 1.0


class TestKsDriftIntegration:
    def test_ks_scores_recorded_for_numeric_columns(self) -> None:
        baseline = pd.DataFrame({"revenue": np.linspace(0.0, 100.0, 200)})
        current = pd.DataFrame({"revenue": np.linspace(0.0, 100.0, 200)})
        report = DatasetAuditor().audit_dataframe(current, reference=baseline)
        assert "revenue__ks_stat" in report.drift_scores
        assert "revenue__ks_pvalue" in report.drift_scores
        assert report.drift_scores["revenue__ks_stat"] == pytest.approx(0.0, abs=1e-9)

    def test_no_ks_finding_for_stable_numeric_distribution(self) -> None:
        baseline = pd.DataFrame({"revenue": np.linspace(0.0, 100.0, 200)})
        current = pd.DataFrame({"revenue": np.linspace(0.0, 100.0, 200)})
        report = DatasetAuditor().audit_dataframe(current, reference=baseline)
        assert not [i for i in report.issues if i.check == "ks_drift"]

    def test_ks_finding_emitted_when_significant(self) -> None:
        baseline = pd.DataFrame({"revenue": np.linspace(0.0, 100.0, 200)})
        current = pd.DataFrame({"revenue": np.linspace(150.0, 250.0, 200)})
        report = DatasetAuditor().audit_dataframe(current, reference=baseline)
        issue = next(i for i in report.issues if i.check == "ks_drift")
        assert issue.column == "revenue"
        assert issue.observed < 0.05
        assert issue.threshold == 0.05
        assert "KS" in issue.message

    def test_ks_alpha_gates_the_finding(self) -> None:
        baseline = pd.DataFrame({"revenue": np.linspace(0.0, 100.0, 200)})
        current = pd.DataFrame({"revenue": np.linspace(150.0, 250.0, 200)})
        none = DatasetAuditor(ks_alpha=0.0).audit_dataframe(current, reference=baseline)
        assert not [i for i in none.issues if i.check == "ks_drift"]
        all_ = DatasetAuditor(ks_alpha=1.0).audit_dataframe(current, reference=baseline)
        assert next(i for i in all_.issues if i.check == "ks_drift") is not None

    def test_ks_scores_render_in_markdown_report(self) -> None:
        baseline = pd.DataFrame({"revenue": np.linspace(0.0, 100.0, 200)})
        current = pd.DataFrame({"revenue": np.linspace(150.0, 250.0, 200)})
        text = DatasetAuditor().audit_dataframe(current, reference=baseline).to_markdown()
        assert "revenue__ks_stat" in text
        assert "revenue__ks_pvalue" in text


class TestKsAlphaConfig:
    def test_default_alpha_is_0p05(self) -> None:
        assert DatasetAuditor().ks_alpha == 0.05

    def test_invalid_alpha_is_rejected(self) -> None:
        for bad in (1.5, -0.1, True):
            with pytest.raises(ValueError, match="ks_alpha"):
                DatasetAuditor(ks_alpha=bad)

    def test_cli_accepts_ks_alpha_and_changes_config_hash(self, tmp_path) -> None:
        baseline = tmp_path / "ref.csv"
        current = tmp_path / "cur.csv"
        pd.DataFrame({"revenue": np.linspace(150.0, 250.0, 200)}).to_csv(baseline, index=False)
        pd.DataFrame({"revenue": np.linspace(150.0, 250.0, 200)}).to_csv(current, index=False)

        out_a = tmp_path / "a.json"
        out_b = tmp_path / "b.json"
        assert main(["audit", str(current), "--reference", str(baseline),
                     "--ks-alpha", "0.01", "--save-json", str(out_a)]) == 0
        assert main(["audit", str(current), "--reference", str(baseline),
                     "--ks-alpha", "0.20", "--save-json", str(out_b)]) == 0
        hash_a = json.loads(out_a.read_text())["meta"]["config_hash"]
        hash_b = json.loads(out_b.read_text())["meta"]["config_hash"]
        assert hash_a != hash_b

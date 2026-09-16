"""Tests for the Population Stability Index (PSI) drift metric."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dataset_audit_kit.cli import main
from dataset_audit_kit.core import DatasetAuditor


def _numeric_shift(n: int = 400, *, loc: float = 0.0, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(loc, 1.0, n))


def _categorical_shift(
    n: int = 400, *, weights: tuple[float, float, float], seed: int = 0
) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.choice(["a", "b", "c"], size=n, p=list(weights)))


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

    def test_synthetic_gaussian_mean_shift_is_large(self) -> None:
        baseline = _numeric_shift(loc=0.0)
        shifted = _numeric_shift(loc=3.0, seed=1)
        psi = DatasetAuditor.population_stability_index(baseline, shifted)
        assert psi > 0.25

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

    def test_identical_categorical_distributions_yield_zero_psi(self) -> None:
        baseline = pd.Series(["a", "b", "a", "c"] * 50)
        psi = DatasetAuditor.population_stability_index(baseline, baseline)
        assert psi == pytest.approx(0.0, abs=1e-9)

    def test_shifted_categorical_distribution_yields_high_psi(self) -> None:
        baseline = _categorical_shift(weights=(0.70, 0.20, 0.10))
        shifted = _categorical_shift(weights=(0.10, 0.20, 0.70), seed=1)
        psi = DatasetAuditor.population_stability_index(baseline, shifted)
        assert psi > 0.25

    def test_empty_categorical_current_returns_zero(self) -> None:
        baseline = pd.Series(["a", "b", "c", "a"])
        assert DatasetAuditor.population_stability_index(
            baseline, pd.Series([], dtype=object)
        ) == 0.0


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

    def test_psi_recorded_for_categorical_columns(self) -> None:
        baseline = pd.DataFrame({"flag": ["a"] * 200})
        current = pd.DataFrame({"flag": ["b"] * 200})
        report = DatasetAuditor().audit_dataframe(
            current, reference=baseline
        )
        assert "flag__psi" in report.drift_scores
        assert report.drift_scores["flag__psi"] > 0.25
        issue = next(i for i in report.issues if i.check == "psi")
        assert issue.column == "flag"

    def test_stable_categorical_emits_no_psi_issue(self) -> None:
        baseline = pd.DataFrame({"flag": ["a", "b"] * 100})
        current = pd.DataFrame({"flag": ["a", "b"] * 100})
        report = DatasetAuditor().audit_dataframe(current, reference=baseline)
        assert report.drift_scores["flag__psi"] == pytest.approx(0.0, abs=1e-9)
        assert not [i for i in report.issues if i.check == "psi"]

    def test_psi_threshold_gates_the_finding(self) -> None:
        baseline = pd.DataFrame({"revenue": np.linspace(0.0, 100.0, 200)})
        shifted = pd.DataFrame({"revenue": np.linspace(100.0, 200.0, 200)})
        none = DatasetAuditor(psi_threshold=999.0).audit_dataframe(
            shifted, reference=baseline
        )
        assert not [i for i in none.issues if i.check == "psi"]
        flagged = DatasetAuditor(psi_threshold=0.01).audit_dataframe(
            shifted, reference=baseline
        )
        assert next(i for i in flagged.issues if i.check == "psi").threshold == 0.01

    def test_psi_bins_are_forwarded_into_the_score(self) -> None:
        baseline = pd.DataFrame({"revenue": np.linspace(0.0, 100.0, 200)})
        shifted = pd.DataFrame({"revenue": np.linspace(100.0, 200.0, 200)})
        coarse = DatasetAuditor(psi_bins=4).audit_dataframe(shifted, reference=baseline)
        fine = DatasetAuditor(psi_bins=20).audit_dataframe(shifted, reference=baseline)
        assert coarse.drift_scores["revenue__psi"] > 0.25
        assert fine.drift_scores["revenue__psi"] > 0.25

    def test_psi_findings_render_in_json_markdown_and_html(self) -> None:
        baseline = pd.DataFrame({"revenue": np.linspace(0.0, 100.0, 200)})
        shifted = pd.DataFrame({"revenue": np.linspace(100.0, 200.0, 200)})
        report = DatasetAuditor(drift_threshold=0.1).audit_dataframe(
            shifted, reference=baseline
        )
        payload = json.loads(report.to_json())
        assert "revenue__psi" in payload["drift_scores"]
        assert [i for i in payload["issues"] if i["check"] == "psi"]
        markdown = report.to_markdown()
        assert "revenue__psi" in markdown
        assert "`psi`" in markdown
        html = report.to_html()
        assert "revenue__psi" in html
        assert ">psi<" in html


class TestPsiConfig:
    def test_defaults(self) -> None:
        auditor = DatasetAuditor()
        assert auditor.psi_threshold is None
        assert auditor.psi_bins == 10

    def test_invalid_threshold_is_rejected(self) -> None:
        for bad in (-0.1, True, "0.2"):
            with pytest.raises(ValueError, match="psi_threshold"):
                DatasetAuditor(psi_threshold=bad)  # type: ignore[arg-type]

    def test_invalid_bins_are_rejected(self) -> None:
        for bad in (1, 0, -3, True, 10.0, "10"):
            with pytest.raises(ValueError, match="psi_bins"):
                DatasetAuditor(psi_bins=bad)  # type: ignore[arg-type]

    def test_cli_accepts_psi_threshold_and_changes_config_hash(self, tmp_path) -> None:
        baseline = tmp_path / "ref.csv"
        current = tmp_path / "cur.csv"
        pd.DataFrame({"revenue": np.linspace(0.0, 100.0, 200)}).to_csv(
            baseline, index=False
        )
        pd.DataFrame({"revenue": np.linspace(0.0, 100.0, 200)}).to_csv(
            current, index=False
        )

        out_a = tmp_path / "a.json"
        out_b = tmp_path / "b.json"
        assert main([
            "audit", str(current), "--reference", str(baseline),
            "--psi-threshold", "0.10", "--save-json", str(out_a),
        ]) == 0
        assert main([
            "audit", str(current), "--reference", str(baseline),
            "--psi-threshold", "0.25", "--save-json", str(out_b),
        ]) == 0
        hash_a = json.loads(out_a.read_text())["meta"]["config_hash"]
        hash_b = json.loads(out_b.read_text())["meta"]["config_hash"]
        assert hash_a != hash_b

    def test_cli_psi_threshold_gates_shifted_numeric_and_categorical(
        self, tmp_path, capsys
    ) -> None:
        baseline = tmp_path / "ref.csv"
        current = tmp_path / "cur.csv"
        pd.DataFrame({
            "revenue": np.linspace(0.0, 100.0, 200),
            "flag": ["a"] * 200,
        }).to_csv(baseline, index=False)
        pd.DataFrame({
            "revenue": np.linspace(100.0, 200.0, 200),
            "flag": ["b"] * 200,
        }).to_csv(current, index=False)

        json_path = tmp_path / "report.json"
        markdown_path = tmp_path / "report.md"
        html_path = tmp_path / "report.html"
        code = main([
            "audit", str(current), "--reference", str(baseline),
            "--psi-threshold", "0.10",
            "--json",
            "--save-json", str(json_path),
            "--save-markdown", str(markdown_path),
            "--html-out", str(html_path),
        ])
        assert code == 1
        payload = json.loads(capsys.readouterr().out)
        psi_issues = [i for i in payload["issues"] if i["check"] == "psi"]
        assert {i["column"] for i in psi_issues} >= {"revenue", "flag"}
        assert "revenue__psi" in payload["drift_scores"]
        assert "flag__psi" in payload["drift_scores"]

        saved = json.loads(json_path.read_text())
        assert [i for i in saved["issues"] if i["check"] == "psi"]
        markdown = markdown_path.read_text()
        assert "revenue__psi" in markdown
        assert "`psi`" in markdown
        html = html_path.read_text()
        assert "revenue__psi" in html
        assert ">psi<" in html

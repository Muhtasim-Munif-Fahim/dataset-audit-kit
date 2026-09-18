"""Tests for the Kolmogorov-Smirnov numeric drift significance test."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dataset_audit_kit.cli import main
from dataset_audit_kit.core import DatasetAuditor


def _shifted_numeric(n: int = 200, *, loc: float = 0.0) -> pd.Series:
    return pd.Series(np.linspace(loc, loc + 100.0, n))


class TestKolmogorovSmirnovTest:
    def test_identical_distributions_return_zero_statistic_and_one_pvalue(self) -> None:
        baseline = _shifted_numeric()
        d, p = DatasetAuditor.kolmogorov_smirnov_test(baseline, baseline)
        assert d == pytest.approx(0.0, abs=1e-9)
        assert p == pytest.approx(1.0, abs=1e-9)

    def test_disjoint_distributions_return_large_statistic_and_tiny_pvalue(self) -> None:
        baseline = _shifted_numeric(loc=0.0)
        current = _shifted_numeric(loc=150.0)
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
        baseline = _shifted_numeric(loc=0.0)
        current = _shifted_numeric(loc=150.0)
        _, p = DatasetAuditor.kolmogorov_smirnov_test(baseline, current)
        assert 0.0 <= p <= 1.0

    def test_nan_and_infinite_values_are_dropped(self) -> None:
        baseline = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan, np.inf])
        current = pd.Series([1.0, 2.0, 3.0, 4.0, -np.inf, np.nan])
        d, p = DatasetAuditor.kolmogorov_smirnov_test(baseline, current)
        assert d == pytest.approx(0.0, abs=1e-9)
        assert p == pytest.approx(1.0, abs=1e-9)

    def test_all_non_finite_returns_passthrough(self) -> None:
        assert DatasetAuditor.kolmogorov_smirnov_test(
            pd.Series([np.nan, np.inf]), pd.Series([1.0, 2.0, 3.0])
        ) == (0.0, 1.0)


class TestKsDriftIntegration:
    def test_ks_scores_recorded_for_numeric_columns(self) -> None:
        baseline = pd.DataFrame({"revenue": _shifted_numeric()})
        current = pd.DataFrame({"revenue": _shifted_numeric()})
        report = DatasetAuditor().audit_dataframe(current, reference=baseline)
        assert "revenue__ks_stat" in report.drift_scores
        assert "revenue__ks_pvalue" in report.drift_scores
        assert report.drift_scores["revenue__ks_stat"] == pytest.approx(0.0, abs=1e-9)

    def test_no_ks_finding_for_stable_numeric_distribution(self) -> None:
        baseline = pd.DataFrame({"revenue": _shifted_numeric()})
        current = pd.DataFrame({"revenue": _shifted_numeric()})
        report = DatasetAuditor().audit_dataframe(current, reference=baseline)
        assert not [i for i in report.issues if i.check == "ks_drift"]

    def test_ks_finding_emitted_when_significant(self) -> None:
        baseline = pd.DataFrame({"revenue": _shifted_numeric(loc=0.0)})
        current = pd.DataFrame({"revenue": _shifted_numeric(loc=150.0)})
        report = DatasetAuditor().audit_dataframe(current, reference=baseline)
        issue = next(i for i in report.issues if i.check == "ks_drift")
        assert issue.column == "revenue"
        assert issue.observed < 0.05
        assert issue.threshold == 0.05
        assert "KS" in issue.message

    def test_ks_alpha_gates_the_finding(self) -> None:
        baseline = pd.DataFrame({"revenue": _shifted_numeric(loc=0.0)})
        current = pd.DataFrame({"revenue": _shifted_numeric(loc=150.0)})
        none = DatasetAuditor(ks_alpha=0.0).audit_dataframe(current, reference=baseline)
        assert not [i for i in none.issues if i.check == "ks_drift"]
        all_ = DatasetAuditor(ks_alpha=1.0).audit_dataframe(current, reference=baseline)
        assert next(i for i in all_.issues if i.check == "ks_drift") is not None

    def test_ks_threshold_gates_the_finding(self) -> None:
        baseline = pd.DataFrame({"revenue": _shifted_numeric(loc=0.0)})
        current = pd.DataFrame({"revenue": _shifted_numeric(loc=50.0)})
        # A 50-unit location shift is significant, but D stays below 0.9.
        flagged = DatasetAuditor(ks_threshold=0.0).audit_dataframe(
            current, reference=baseline
        )
        d = flagged.drift_scores["revenue__ks_stat"]
        assert 0.0 < d < 0.9
        assert next(i for i in flagged.issues if i.check == "ks_drift") is not None
        silenced = DatasetAuditor(ks_threshold=0.9).audit_dataframe(
            current, reference=baseline
        )
        assert not [i for i in silenced.issues if i.check == "ks_drift"]

    def test_ks_scores_render_in_markdown_report(self) -> None:
        baseline = pd.DataFrame({"revenue": _shifted_numeric(loc=0.0)})
        current = pd.DataFrame({"revenue": _shifted_numeric(loc=150.0)})
        text = DatasetAuditor().audit_dataframe(current, reference=baseline).to_markdown()
        assert "revenue__ks_stat" in text
        assert "revenue__ks_pvalue" in text

    def test_ks_findings_render_in_json_markdown_and_html(self) -> None:
        baseline = pd.DataFrame({"revenue": _shifted_numeric(loc=0.0)})
        current = pd.DataFrame({"revenue": _shifted_numeric(loc=150.0)})
        report = DatasetAuditor().audit_dataframe(current, reference=baseline)
        payload = json.loads(report.to_json())
        assert "revenue__ks_stat" in payload["drift_scores"]
        assert "revenue__ks_pvalue" in payload["drift_scores"]
        assert [i for i in payload["issues"] if i["check"] == "ks_drift"]
        markdown = report.to_markdown()
        assert "revenue__ks_stat" in markdown
        assert "`ks_drift`" in markdown
        html = report.to_html()
        assert "revenue__ks_stat" in html
        assert ">ks_drift<" in html

    def test_categorical_and_boolean_columns_skip_ks(self) -> None:
        baseline = pd.DataFrame(
            {
                "flag": ["a", "b"] * 100,
                "on": [True, False] * 100,
                "revenue": _shifted_numeric(),
            }
        )
        current = pd.DataFrame(
            {
                "flag": ["b", "a"] * 100,
                "on": [False, True] * 100,
                "revenue": _shifted_numeric(),
            }
        )
        report = DatasetAuditor().audit_dataframe(current, reference=baseline)
        assert "flag__ks_stat" not in report.drift_scores
        assert "on__ks_stat" not in report.drift_scores
        assert "revenue__ks_stat" in report.drift_scores

    def test_label_column_is_excluded_from_ks(self) -> None:
        baseline = pd.DataFrame(
            {"revenue": _shifted_numeric(loc=0.0), "target": _shifted_numeric(loc=0.0)}
        )
        current = pd.DataFrame(
            {"revenue": _shifted_numeric(loc=150.0), "target": _shifted_numeric(loc=150.0)}
        )
        report = DatasetAuditor().audit_dataframe(
            current, reference=baseline, label_column="target"
        )
        assert "target__ks_stat" not in report.drift_scores
        assert "revenue__ks_stat" in report.drift_scores

    def test_ks_drift_offers_an_investigate_suggestion(self) -> None:
        baseline = pd.DataFrame({"revenue": _shifted_numeric(loc=0.0)})
        current = pd.DataFrame({"revenue": _shifted_numeric(loc=150.0)})
        report = DatasetAuditor().audit_dataframe(current, reference=baseline)
        suggestion = next(
            s for s in report.fix_suggestions if s["action"] == "investigate_drift"
        )
        assert "distribution drift" in suggestion["description"]


class TestKsAlphaConfig:
    def test_default_alpha_is_0p05(self) -> None:
        auditor = DatasetAuditor()
        assert auditor.ks_alpha == 0.05
        assert auditor.ks_threshold == 0.0

    def test_invalid_alpha_is_rejected(self) -> None:
        for bad in (1.5, -0.1, True):
            with pytest.raises(ValueError, match="ks_alpha"):
                DatasetAuditor(ks_alpha=bad)

    def test_invalid_threshold_is_rejected(self) -> None:
        for bad in (1.5, -0.1, True):
            with pytest.raises(ValueError, match="ks_threshold"):
                DatasetAuditor(ks_threshold=bad)

    def test_cli_accepts_ks_alpha_and_changes_config_hash(self, tmp_path) -> None:
        baseline = tmp_path / "ref.csv"
        current = tmp_path / "cur.csv"
        pd.DataFrame({"revenue": _shifted_numeric(loc=150.0)}).to_csv(
            baseline, index=False
        )
        pd.DataFrame({"revenue": _shifted_numeric(loc=150.0)}).to_csv(
            current, index=False
        )

        out_a = tmp_path / "a.json"
        out_b = tmp_path / "b.json"
        assert main(["audit", str(current), "--reference", str(baseline),
                     "--ks-alpha", "0.01", "--save-json", str(out_a)]) == 0
        assert main(["audit", str(current), "--reference", str(baseline),
                     "--ks-alpha", "0.20", "--save-json", str(out_b)]) == 0
        hash_a = json.loads(out_a.read_text())["meta"]["config_hash"]
        hash_b = json.loads(out_b.read_text())["meta"]["config_hash"]
        assert hash_a != hash_b

    def test_cli_ks_threshold_changes_config_hash(self, tmp_path) -> None:
        baseline = tmp_path / "ref.csv"
        current = tmp_path / "cur.csv"
        pd.DataFrame({"revenue": _shifted_numeric()}).to_csv(baseline, index=False)
        pd.DataFrame({"revenue": _shifted_numeric()}).to_csv(current, index=False)

        out_a = tmp_path / "a.json"
        out_b = tmp_path / "b.json"
        assert main(["audit", str(current), "--reference", str(baseline),
                     "--ks-threshold", "0.05", "--save-json", str(out_a)]) == 0
        assert main(["audit", str(current), "--reference", str(baseline),
                     "--ks-threshold", "0.20", "--save-json", str(out_b)]) == 0
        hash_a = json.loads(out_a.read_text())["meta"]["config_hash"]
        hash_b = json.loads(out_b.read_text())["meta"]["config_hash"]
        assert hash_a != hash_b

    def test_cli_rejects_out_of_range_ks_alpha(self, tmp_path, capsys) -> None:
        path = tmp_path / "data.csv"
        pd.DataFrame({"revenue": _shifted_numeric()}).to_csv(path, index=False)
        with pytest.raises(SystemExit) as exc:
            main(["audit", str(path), "--ks-alpha", "1.5"])
        assert exc.value.code == 2
        assert "between 0 and 1" in capsys.readouterr().err

    def test_cli_ks_alpha_gates_shifted_numeric(
        self, tmp_path, capsys
    ) -> None:
        baseline = tmp_path / "ref.csv"
        current = tmp_path / "cur.csv"
        pd.DataFrame({
            "revenue": _shifted_numeric(loc=0.0),
            "flag": ["a"] * 200,
        }).to_csv(baseline, index=False)
        pd.DataFrame({
            "revenue": _shifted_numeric(loc=150.0),
            "flag": ["b"] * 200,
        }).to_csv(current, index=False)

        json_path = tmp_path / "report.json"
        markdown_path = tmp_path / "report.md"
        html_path = tmp_path / "report.html"
        code = main([
            "audit", str(current), "--reference", str(baseline),
            "--ks-alpha", "0.05",
            "--json",
            "--save-json", str(json_path),
            "--save-markdown", str(markdown_path),
            "--html-out", str(html_path),
        ])
        assert code == 1
        payload = json.loads(capsys.readouterr().out)
        ks_issues = [i for i in payload["issues"] if i["check"] == "ks_drift"]
        assert {i["column"] for i in ks_issues} == {"revenue"}
        assert "revenue__ks_stat" in payload["drift_scores"]
        assert "revenue__ks_pvalue" in payload["drift_scores"]
        assert "flag__ks_stat" not in payload["drift_scores"]

        saved = json.loads(json_path.read_text())
        assert [i for i in saved["issues"] if i["check"] == "ks_drift"]
        markdown = markdown_path.read_text()
        assert "revenue__ks_stat" in markdown
        assert "`ks_drift`" in markdown
        html = html_path.read_text()
        assert "revenue__ks_stat" in html
        assert ">ks_drift<" in html

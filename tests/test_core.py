from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dataset_audit_kit import DatasetAuditor, ValidationRules


def test_audit_detects_quality_issues() -> None:
    data = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, None, 2.0],
            "feature_b": ["x", "x", "y", "x"],
            "target": [0, 0, 0, 0],
        }
    )

    auditor = DatasetAuditor(missing_threshold=0.10, drift_threshold=0.20, label_min_share=0.20)
    report = auditor.audit_dataframe(
        data,
        label_column="target",
        expected_columns=["feature_a", "feature_b", "target"],
    )

    assert report.status == "warn"
    assert report.duplicate_rows == 1
    assert report.missing_cells == 1
    assert any(issue.check == "missingness" for issue in report.issues)
    assert any(issue.check == "labels" for issue in report.issues)
    assert any(issue.check == "duplicates" for issue in report.issues)


def test_audit_detects_drift_against_reference() -> None:
    current = pd.DataFrame(
        {
            "feature_a": [10.0, 11.0, 12.0, 13.0],
            "feature_b": ["z", "z", "z", "z"],
            "target": [1, 1, 1, 1],
        }
    )
    reference = pd.DataFrame(
        {
            "feature_a": [1.0, 1.1, 1.2, 1.3],
            "feature_b": ["x", "x", "y", "x"],
            "target": [0, 0, 0, 1],
        }
    )

    auditor = DatasetAuditor(missing_threshold=0.50, drift_threshold=0.10, label_min_share=0.10)
    report = auditor.audit_dataframe(
        current,
        reference=reference,
        label_column="target",
        expected_columns=["feature_a", "feature_b", "target"],
    )

    assert report.drift_scores["feature_a"] > 0
    assert any(issue.check == "drift" for issue in report.issues)


def test_report_serializes_to_json_and_markdown() -> None:
    data = pd.DataFrame({"feature_a": [1.0, 2.0, 3.0], "target": [0, 1, 0]})
    auditor = DatasetAuditor()
    report = auditor.audit_dataframe(data, label_column="target", expected_columns=["feature_a", "target"])

    parsed = json.loads(report.to_json())
    assert parsed["status"] == "pass"
    assert "# Dataset Audit Report" in report.to_markdown()
    html = report.to_html()
    assert "<!doctype html>" in html
    assert "Dataset Audit Report" in html


def test_audit_file_supports_jsonl_and_parquet(tmp_path) -> None:
    data = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": ["x", "y", "z"],
            "target": [0, 1, 1],
        }
    )
    reference = pd.DataFrame(
        {
            "feature_a": [1.1, 2.1, 3.1],
            "feature_b": ["x", "x", "z"],
            "target": [0, 0, 1],
        }
    )

    jsonl_path = tmp_path / "data.jsonl"
    parquet_path = tmp_path / "reference.parquet"
    data.to_json(jsonl_path, orient="records", lines=True)
    reference.to_parquet(parquet_path, index=False)

    auditor = DatasetAuditor()
    report = auditor.audit_file(
        str(jsonl_path),
        reference_path=str(parquet_path),
        label_column="target",
        expected_columns=["feature_a", "feature_b", "target"],
    )

    assert report.rows == 3
    assert report.columns == 3
    assert "feature_a" in report.drift_scores


def test_load_dataframe_rejects_unknown_suffix(tmp_path) -> None:
    dataset_path = tmp_path / "data.tsv"
    dataset_path.write_text("a\tb\n1\t2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported dataset format"):
        DatasetAuditor.load_dataframe(dataset_path)


def test_demo_notebook_is_valid_json() -> None:
    notebook = json.loads(Path("examples/demo.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 2

# ------------------------------------------------------------------
# Per-column validation rules
# ------------------------------------------------------------------

class TestColumnRules:
    """Tests for per-column validation rules (issue #3)."""

    def test_clean_data_passes_rules(self):
        data = pd.DataFrame({
            "feature_1": [1.0, 2.0, 1.5],
            "feature_2": [2.5, 3.5, 2.0],
            "target": ["A", "B", "A"],
        })
        rules = ValidationRules.from_dict({
            "feature_1": {"dtype": "numeric", "min_value": 0.0, "max_value": 10.0},
            "feature_2": {"dtype": "numeric", "min_value": 0.0, "max_value": 10.0},
            "target": {"dtype": "categorical", "allowed_values": ["A", "B"]},
        })
        auditor = DatasetAuditor(rules=rules)
        report = auditor.audit_dataframe(data)
        rule_issues = [i for i in report.issues if i.check == "rule"]
        assert len(rule_issues) == 0, f"Expected no rule issues, got: {rule_issues}"

    def test_outlier_detected_by_rule(self):
        data = pd.DataFrame({
            "feature_1": [1.0, 100.0, 2.0],
            "feature_2": [2.5, 3.5, 2.0],
        })
        rules = ValidationRules.from_dict({
            "feature_1": {"dtype": "numeric", "min_value": 0.0, "max_value": 10.0},
        })
        auditor = DatasetAuditor(rules=rules)
        report = auditor.audit_dataframe(data)
        rule_issues = [i for i in report.issues if i.check == "rule"]
        assert len(rule_issues) == 1
        assert "above maximum" in rule_issues[0].message

    def test_unexpected_categorical_values(self):
        data = pd.DataFrame({
            "target": ["A", "B", "X", "Y"],
        })
        rules = ValidationRules.from_dict({
            "target": {"allowed_values": ["A", "B"]},
        })
        auditor = DatasetAuditor(rules=rules)
        report = auditor.audit_dataframe(data)
        rule_issues = [i for i in report.issues if i.check == "rule"]
        assert len(rule_issues) == 1
        assert "Unexpected values" in rule_issues[0].message

    def test_missing_column_defined_in_rule(self):
        data = pd.DataFrame({
            "feature_1": [1.0, 2.0],
        })
        rules = ValidationRules.from_dict({
            "feature_1": {"dtype": "numeric"},
            "missing_col": {"dtype": "numeric"},
        })
        auditor = DatasetAuditor(rules=rules)
        report = auditor.audit_dataframe(data)
        rule_issues = [i for i in report.issues if i.check == "rule"]
        assert len(rule_issues) == 1
        assert "missing column" in rule_issues[0].message.lower()

    def test_validation_rules_from_json(self, tmp_path):
        import json
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps({
            "score": {"dtype": "numeric", "min_value": 0.0, "max_value": 100.0},
        }))
        rules = ValidationRules.from_json(str(rules_file))
        assert "score" in rules.columns
        assert rules.columns["score"].min_value == 0.0
        assert rules.columns["score"].max_value == 100.0

    def test_validation_rules_to_dict_roundtrip(self):
        rules = ValidationRules.from_dict({
            "col": {"dtype": "numeric", "min_value": 0.0, "allowed_values": None},
        })
        d = rules.to_dict()
        assert "col" in d
        assert d["col"]["dtype"] == "numeric"
        assert d["col"]["min_value"] == 0.0
        # Fields with None should be omitted
        assert "allowed_values" not in d["col"]

    def test_per_column_missing_threshold(self):
        data = pd.DataFrame({
            "feature_1": [1.0, None, None],
            "feature_2": [2.5, 3.5, None],
        })
        rules = ValidationRules.from_dict({
            "feature_1": {"max_missing_ratio": 0.2},  # 66% missing > 20%
            "feature_2": {"max_missing_ratio": 0.5},  # 33% missing < 50%
        })
        auditor = DatasetAuditor(rules=rules, missing_threshold=1.0)  # global threshold lenient
        report = auditor.audit_dataframe(data)
        rule_issues = [i for i in report.issues if i.check == "rule"]
        assert len(rule_issues) == 1
        assert "feature_1" in rule_issues[0].column


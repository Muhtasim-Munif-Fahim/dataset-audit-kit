from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dataset_audit_kit import DatasetAuditor


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
    data = pd.DataFrame({"feature_a": [1.0, 2.0], "target": [0, 1]})
    auditor = DatasetAuditor()
    report = auditor.audit_dataframe(data, label_column="target", expected_columns=["feature_a", "target"])

    parsed = json.loads(report.to_json())
    assert parsed["status"] == "pass"
    assert "# Dataset Audit Report" in report.to_markdown()
    html = report.to_html()
    assert "<!doctype html>" in html
    assert "Dataset Audit Report" in html


def test_demo_notebook_is_valid_json() -> None:
    notebook = json.loads(Path("examples/demo.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 2

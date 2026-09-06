"""Tests for YAML report output (to_yaml and --yaml-out CLI)."""

from __future__ import annotations

import pytest

import pandas as pd

from dataset_audit_kit.core import DatasetAuditor, ValidationRules
from dataset_audit_kit.cli import main as cli_main


@pytest.fixture
def auditor() -> DatasetAuditor:
    return DatasetAuditor(rules=ValidationRules())


@pytest.fixture
def dirty_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, None],
            "score": [10.0, 20.0, 15.0, 12.0, 14.0, None],
            "category": ["a", "b", "a", None, "b", "a"],
        }
    )


class TestToYaml:
    def test_to_yaml_returns_non_empty_string(self, auditor: DatasetAuditor, dirty_df: pd.DataFrame) -> None:
        report = auditor.audit_dataframe(dirty_df)
        text = report.to_yaml()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_to_yaml_is_valid_yaml_with_top_level_keys(self, auditor: DatasetAuditor, dirty_df: pd.DataFrame) -> None:
        report = auditor.audit_dataframe(dirty_df)
        import yaml

        loaded = yaml.safe_load(report.to_yaml())
        assert isinstance(loaded, dict)
        for key in ("status", "quality_score", "rows", "columns", "duplicate_rows"):
            assert key in loaded, f"missing key: {key}"

    def test_to_yaml_issues_key_matches_issue_count(self, auditor: DatasetAuditor, dirty_df: pd.DataFrame) -> None:
        report = auditor.audit_dataframe(dirty_df)
        import yaml

        loaded = yaml.safe_load(report.to_yaml())
        assert len(loaded["issues"]) == len(report.issues)

    def test_to_yaml_clean_report_has_empty_issues(self, auditor: DatasetAuditor) -> None:
        clean_df = pd.DataFrame({"id": [1, 2, 3], "score": [10.0, 20.0, 15.0]})
        report = auditor.audit_dataframe(clean_df)
        import yaml

        loaded = yaml.safe_load(report.to_yaml())
        assert loaded["issues"] == []
        assert loaded["status"] == "pass"


class TestToFileYaml:
    def test_to_file_yaml_extension(self, auditor: DatasetAuditor, dirty_df: pd.DataFrame, tmp_path) -> None:
        report = auditor.audit_dataframe(dirty_df)
        out = tmp_path / "report.yaml"
        report.to_file(str(out))
        text = out.read_text(encoding="utf-8")
        import yaml

        assert yaml.safe_load(text)["columns"] == report.columns

    def test_to_file_yml_extension(self, auditor: DatasetAuditor, dirty_df: pd.DataFrame, tmp_path) -> None:
        report = auditor.audit_dataframe(dirty_df)
        out = tmp_path / "report.yml"
        report.to_file(str(out))
        text = out.read_text(encoding="utf-8")
        import yaml

        assert "quality_score" in yaml.safe_load(text)

    def test_to_file_creates_parent_directory(self, auditor: DatasetAuditor, dirty_df: pd.DataFrame, tmp_path) -> None:
        report = auditor.audit_dataframe(dirty_df)
        out = tmp_path / "deep" / "nested" / "report.yaml"
        report.to_file(str(out))
        assert out.exists()


class TestCliYamlOut:
    def test_audit_yaml_out_flag_writes_yaml_file(self, dirty_df: pd.DataFrame, tmp_path) -> None:
        csv_path = tmp_path / "data.csv"
        dirty_df.to_csv(csv_path, index=False)
        yaml_path = tmp_path / "out.yaml"

        rc = cli_main(["audit", str(csv_path), "--yaml-out", str(yaml_path)])
        assert rc == 1
        assert yaml_path.exists()

        import yaml

        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert loaded["columns"] == 3

    def test_check_yaml_out_flag_writes_yaml_file(self, dirty_df: pd.DataFrame, tmp_path) -> None:
        csv_path = tmp_path / "data.csv"
        dirty_df.to_csv(csv_path, index=False)
        yaml_path = tmp_path / "out.yaml"

        rc = cli_main(["check", str(csv_path), "--yaml-out", str(yaml_path)])
        assert rc == 1
        assert yaml_path.exists()

        import yaml

        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert len(loaded["issues"]) > 0

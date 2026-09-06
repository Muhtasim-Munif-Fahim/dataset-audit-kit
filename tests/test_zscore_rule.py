"""Tests for the max_zscore ColumnRule outlier detection."""

from __future__ import annotations

import pytest

import pandas as pd
from dataset_audit_kit.core import DatasetAuditor, ValidationRules


class TestMaxZscoreParsing:
    def test_zscore_round_trips_through_from_dict_and_to_dict(self) -> None:
        rules = ValidationRules.from_dict({"score": {"max_zscore": 2.5}})
        assert rules.columns["score"].max_zscore == 2.5
        assert rules.to_dict()["score"]["max_zscore"] == 2.5

    def test_zscore_none_by_default(self) -> None:
        rules = ValidationRules.from_dict({"score": {"dtype": "numeric"}})
        assert rules.columns["score"].max_zscore is None

    def test_zscore_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_zscore"):
            ValidationRules.from_dict({"score": {"max_zscore": 0.0}})
        with pytest.raises(ValueError, match="max_zscore"):
            ValidationRules.from_dict({"score": {"max_zscore": -1.0}})

    def test_zscore_must_be_finite_number(self) -> None:
        with pytest.raises(ValueError, match="max_zscore"):
            ValidationRules.from_dict({"score": {"max_zscore": "two"}})
        with pytest.raises(ValueError, match="max_zscore"):
            ValidationRules.from_dict({"score": {"max_zscore": True}})

    def test_zscore_float_accepted(self) -> None:
        rules = ValidationRules.from_dict({"score": {"max_zscore": 0.5}})
        assert rules.columns["score"].max_zscore == 0.5


class TestZscoreOutlierCheck:
    def test_flags_values_beyond_zscore_threshold(self) -> None:
        data = pd.DataFrame({"score": list(range(99)) + [1000]})
        rules = ValidationRules.from_dict({"score": {"max_zscore": 2.0}})
        report = DatasetAuditor(rules=rules).audit_dataframe(data)
        zscore_issues = [i for i in report.issues if i.observed == 1.0]
        assert len(zscore_issues) == 1
        assert zscore_issues[0].column == "score"
        assert zscore_issues[0].threshold == 2.0
        assert zscore_issues[0].severity == "warning"

    def test_no_flag_when_all_values_within_zscore(self) -> None:
        data = pd.DataFrame({"score": list(range(100))})
        rules = ValidationRules.from_dict({"score": {"max_zscore": 3.0}})
        report = DatasetAuditor(rules=rules).audit_dataframe(data)
        zscore_issues = [
            i for i in report.issues
            if i.check == "rule" and i.column == "score" and i.threshold == 3.0
        ]
        assert len(zscore_issues) == 0

    def test_non_numeric_column_skips_zscore_check(self) -> None:
        data = pd.DataFrame({"name": ["ann", "bo", "cy", "di", "eve", "fran"]})
        rules = ValidationRules.from_dict(
            {"name": {"dtype": "categorical", "max_zscore": 2.0}}
        )
        report = DatasetAuditor(rules=rules).audit_dataframe(data)
        zscore_issues = [i for i in report.issues if i.threshold == 2.0]
        assert len(zscore_issues) == 0

    def test_constant_column_skips_zscore_check(self) -> None:
        data = pd.DataFrame({"score": [5] * 10})
        rules = ValidationRules.from_dict({"score": {"max_zscore": 2.0}})
        report = DatasetAuditor(rules=rules).audit_dataframe(data)
        zscore_issues = [i for i in report.issues if i.threshold == 2.0]
        assert len(zscore_issues) == 0

    def test_zscore_uses_population_std(self) -> None:
        data = pd.DataFrame({"score": [-3.0, -1.0, 0.0, 1.0, 3.0]})
        rules = ValidationRules.from_dict({"score": {"max_zscore": 1.0}})
        report = DatasetAuditor(rules=rules).audit_dataframe(data)
        messages = [i.message for i in report.issues if i.column == "score" and i.threshold == 1.0]
        assert len(messages) == 1
        assert "2 value(s)" in messages[0]


class TestZscoreValidateConfig:
    def test_validate_config_accepts_zscore(self, tmp_path) -> None:
        from dataset_audit_kit.cli import main

        rules_file = tmp_path / "rules.json"
        rules_file.write_text('{"score": {"max_zscore": 3.0}}', encoding="utf-8")
        rc = main(["validate-config", str(rules_file)])
        assert rc == 0

    def test_validate_config_rejects_negative_zscore(self, tmp_path) -> None:
        from dataset_audit_kit.cli import main

        rules_file = tmp_path / "rules.json"
        rules_file.write_text('{"score": {"max_zscore": -1.0}}', encoding="utf-8")
        rc = main(["validate-config", str(rules_file)])
        assert rc == 1


class TestZscoreEmitConfig:
    def test_emit_config_minimal_includes_zscore(self, tmp_path) -> None:
        from dataset_audit_kit.cli import main
        import json

        out = tmp_path / "minimal.json"
        rc = main(["emit-config", "--minimal", "--output", str(out)])
        assert rc == 0
        template = json.loads(out.read_text())
        assert "max_zscore" in template["column_name"]

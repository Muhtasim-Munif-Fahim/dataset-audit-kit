"""Tests for the ``monotonic`` per-column validation rule."""

from __future__ import annotations

import json

import pandas as pd

from dataset_audit_kit.cli import main
from dataset_audit_kit.core import DatasetAuditor, ValidationRules


def _rules(monotonic: str) -> ValidationRules:
    return ValidationRules.from_dict({"x": {"monotonic": monotonic}})


class TestMonotonicRule:
    def test_increasing_passes_for_monotonic_data(self) -> None:
        data = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        report = DatasetAuditor(rules=_rules("increasing")).audit_dataframe(data)
        assert not [i for i in report.issues if i.check == "rule" and i.column == "x"]

    def test_increasing_fails_for_a_decrease(self) -> None:
        data = pd.DataFrame({"x": [1.0, 3.0, 2.0, 4.0]})
        report = DatasetAuditor(rules=_rules("increasing")).audit_dataframe(data)
        issue = next(i for i in report.issues if i.check == "rule" and i.column == "x")
        assert "increasing" in issue.message
        assert issue.severity == "warning"

    def test_decreasing_passes_for_monotonic_data(self) -> None:
        data = pd.DataFrame({"x": [4.0, 3.0, 2.0, 1.0]})
        report = DatasetAuditor(rules=_rules("decreasing")).audit_dataframe(data)
        assert not [i for i in report.issues if i.check == "rule" and i.column == "x"]

    def test_strictly_increasing_fails_on_equal_values(self) -> None:
        data = pd.DataFrame({"x": [1.0, 2.0, 2.0, 3.0]})
        report = DatasetAuditor(rules=_rules("strictly_increasing")).audit_dataframe(data)
        assert [i for i in report.issues if i.check == "rule" and i.column == "x"]

    def test_strictly_increasing_passes_when_values_are_unique(self) -> None:
        data = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        report = DatasetAuditor(rules=_rules("strictly_increasing")).audit_dataframe(data)
        assert not [i for i in report.issues if i.check == "rule" and i.column == "x"]

    def test_strictly_decreasing_fails_on_equal_values(self) -> None:
        data = pd.DataFrame({"x": [3.0, 2.0, 2.0, 1.0]})
        report = DatasetAuditor(rules=_rules("strictly_decreasing")).audit_dataframe(data)
        assert [i for i in report.issues if i.check == "rule" and i.column == "x"]

    def test_monotonic_skipped_for_non_ordered_columns(self) -> None:
        data = pd.DataFrame({"x": ["a", "b", "c", "a"]})
        report = DatasetAuditor(rules=_rules("strictly_increasing")).audit_dataframe(data)
        assert not [i for i in report.issues if i.check == "rule" and i.column == "x"]

    def test_monotonic_applies_to_datetime_columns(self) -> None:
        increasing = pd.DataFrame(
            {"x": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])}
        )
        reversed_ts = pd.DataFrame(
            {"x": pd.to_datetime(["2020-03-01", "2020-02-01", "2020-01-01"])}
        )
        passes = DatasetAuditor(rules=_rules("increasing")).audit_dataframe(increasing)
        assert not [i for i in passes.issues if i.column == "x"]
        report = DatasetAuditor(rules=_rules("increasing")).audit_dataframe(reversed_ts)
        issue = next(i for i in report.issues if i.check == "rule" and i.column == "x")
        assert "increasing" in issue.message


class TestMonotonicSerialization:
    def test_round_trips_through_rule_dictionary(self) -> None:
        rules = ValidationRules.from_dict({"x": {"monotonic": "strictly_decreasing"}})
        assert rules.columns["x"].monotonic == "strictly_decreasing"
        dumped = rules.to_dict()
        assert dumped["x"]["monotonic"] == "strictly_decreasing"
        again = ValidationRules.from_dict(dumped)
        assert again.columns["x"].monotonic == "strictly_decreasing"

    def test_invalid_monotonic_value_is_rejected(self) -> None:
        try:
            ValidationRules.from_dict({"x": {"monotonic": "bogus"}})
        except ValueError as exc:
            assert "monotonic" in str(exc)
        else:
            raise AssertionError("expected ValueError for invalid monotonic mode")


class TestValidateConfigMonotonic:
    def test_validate_config_accepts_valid_monotonic(self, tmp_path) -> None:
        path = tmp_path / "rules.json"
        path.write_text(
            json.dumps({"x": {"dtype": "numeric", "monotonic": "strictly_increasing"}}),
            encoding="utf-8",
        )
        assert main(["validate-config", str(path)]) == 0

    def test_validate_config_rejects_invalid_monotonic(self, tmp_path, capsys) -> None:
        path = tmp_path / "rules.json"
        path.write_text(json.dumps({"x": {"monotonic": "sideways"}}), encoding="utf-8")
        assert main(["validate-config", str(path)]) == 1
        err = capsys.readouterr().err
        assert "monotonic" in err

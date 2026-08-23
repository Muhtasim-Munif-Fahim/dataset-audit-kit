"""Tests for the auditing core."""

from __future__ import annotations

import json
import warnings
from xml.etree import ElementTree

import pandas as pd
import pytest

from dataset_audit_kit.core import (
    AuditIssue,
    AuditReport,
    BatchAuditReport,
    DatasetAuditor,
    DatasetBaseline,
    ValidationRules,
)


def _messages(report: AuditReport, check: str) -> list[str]:
    return [issue.message for issue in report.issues if issue.check == check]


class TestStatus:
    def test_info_only_issues_still_pass(self) -> None:
        report = AuditReport(rows=1, columns=1, duplicate_rows=0, missing_cells=0)
        report.issues.append(
            AuditIssue(check="schema_diff", severity="info", message="Column 'b' added.")
        )
        assert report.status == "pass"
        assert report.blocking_issues == []

    @pytest.mark.parametrize("severity", ["warning", "error"])
    def test_warnings_and_errors_fail(self, severity: str) -> None:
        report = AuditReport(rows=1, columns=1, duplicate_rows=0, missing_cells=0)
        report.issues.append(AuditIssue(check="duplicates", severity=severity, message="x"))
        assert report.status == "warn"
        assert len(report.blocking_issues) == 1

    def test_error_gate_can_ignore_warnings(self) -> None:
        report = AuditReport(rows=1, columns=1, duplicate_rows=0, missing_cells=0)
        report.issues.extend(
            [
                AuditIssue(check="missing", severity="warning", message="x"),
                AuditIssue(check="schema", severity="error", message="y"),
            ]
        )
        assert [issue.severity for issue in report.gated_issues("error")] == ["error"]
        assert report.exit_code("error") == 1

    def test_invalid_gate_is_rejected(self) -> None:
        report = AuditReport(rows=1, columns=1, duplicate_rows=0, missing_cells=0)
        with pytest.raises(ValueError, match="fail_on"):
            report.exit_code("info")


class TestUniqueness:
    def test_counts_every_redundant_row(self) -> None:
        # One value repeated three times leaves two redundant rows, and a value
        # repeated twice leaves one: three in total across two repeated values.
        data = pd.DataFrame({"id": [1, 1, 1, 2, 2, 3]})
        report = DatasetAuditor().audit_dataframe(data, unique_columns=["id"])
        message = _messages(report, "uniqueness")[0]
        assert "3 duplicate row(s)" in message
        assert "2 repeated value(s)" in message

    def test_unique_column_is_silent(self) -> None:
        data = pd.DataFrame({"id": [1, 2, 3]})
        report = DatasetAuditor().audit_dataframe(data, unique_columns=["id"])
        assert _messages(report, "uniqueness") == []

    def test_missing_column_is_an_error(self) -> None:
        report = DatasetAuditor().audit_dataframe(
            pd.DataFrame({"id": [1]}), unique_columns=["nope"]
        )
        assert "not found" in _messages(report, "uniqueness")[0]

    def test_composite_key_detects_duplicate_combinations(self) -> None:
        data = pd.DataFrame(
            {"site": ["A", "A", "A"], "subject": [1, 1, 2], "visit": [1, 2, 1]}
        )
        report = DatasetAuditor().audit_dataframe(
            data, unique_together=[["site", "subject"]]
        )
        assert "1 duplicate row(s)" in _messages(report, "composite_uniqueness")[0]

    def test_composite_key_can_be_unique_when_members_are_not(self) -> None:
        data = pd.DataFrame({"site": ["A", "A"], "subject": [1, 2]})
        report = DatasetAuditor().audit_dataframe(
            data, unique_together=[["site", "subject"]]
        )
        assert _messages(report, "composite_uniqueness") == []


class TestColumnNames:
    def test_exact_duplicate_names_are_flagged(self) -> None:
        data = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "b"])
        report = DatasetAuditor().audit_dataframe(data)
        assert any("Duplicate column name" in m for m in _messages(report, "column_names"))

    def test_duplicate_names_do_not_break_profiling(self) -> None:
        data = pd.DataFrame([[1, 2, "x"], [3, 4, "y"]], columns=["a", "a", "b"])
        report = DatasetAuditor().audit_dataframe(data)
        assert report.to_markdown()
        assert report.to_html()

    def test_case_only_difference_is_flagged(self) -> None:
        data = pd.DataFrame([[1, 2]], columns=["Age", "age"])
        report = DatasetAuditor().audit_dataframe(data)
        assert any("only by case" in m for m in _messages(report, "column_names"))

    def test_clean_names_are_silent(self, clean_frame: pd.DataFrame) -> None:
        report = DatasetAuditor().audit_dataframe(clean_frame)
        assert _messages(report, "column_names") == []


class TestLabelBalance:
    def test_categorical_label_column_does_not_crash(self) -> None:
        data = pd.DataFrame({"y": pd.Categorical(["a", "b", None])})
        report = DatasetAuditor().audit_dataframe(data, label_column="y")
        assert report.label_distribution == {"a": 1, "b": 1, "<missing>": 1}

    def test_missing_bucket_is_not_a_class(self) -> None:
        # Two balanced classes plus one missing row: the missing rows get their
        # own warning, but they must not be reported as a minority class.
        data = pd.DataFrame({"y": ["a"] * 10 + ["b"] * 10 + [None]})
        report = DatasetAuditor().audit_dataframe(data, label_column="y")
        messages = _messages(report, "labels")
        assert any("missing label value(s)" in m for m in messages)
        assert not any("imbalanced" in m for m in messages)

    def test_imbalance_is_still_reported(self) -> None:
        data = pd.DataFrame({"y": ["a"] * 99 + ["b"]})
        report = DatasetAuditor().audit_dataframe(data, label_column="y")
        assert any("imbalanced" in m for m in _messages(report, "labels"))


class TestProfiles:
    def test_text_columns_profile_as_categorical(self, tmp_path) -> None:
        # pandas 3 reads text into a dedicated string dtype rather than object,
        # so a dtype check against object alone loses the top-value summary.
        path = tmp_path / "text.csv"
        pd.DataFrame({"c": ["a", "a", "b"]}).to_csv(path, index=False)
        data = DatasetAuditor.load_dataframe(str(path))
        profile = DatasetAuditor._profile_columns(data)["c"]
        assert profile["dtype"] == "categorical"
        assert profile["top"] == "a"
        assert profile["freq"] == 2

    def test_no_deprecated_pandas_api_warnings(self) -> None:
        data = pd.DataFrame(
            {"c": pd.Categorical(["a", "b", "a"]), "n": [1.0, 2.0, 3.0]}
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DatasetAuditor()._profile_columns(data)
            DatasetAuditor._infer_dtype(data["c"])
        assert not [w for w in caught if "is_categorical_dtype" in str(w.message)]


class TestDatasetBaseline:
    def test_compares_new_data_without_raw_reference_rows(self) -> None:
        baseline = DatasetBaseline.from_dataframe(
            pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0], "group": ["a", "a", "b", "b"]})
        )
        current = pd.DataFrame(
            {"value": [20.0, 21.0, None, None], "new_column": [1, 2, 3, 4]}
        )
        issues = baseline.compare(current, missing_ratio_delta=0.2, mean_shift_std=2.0)
        checks = {issue.check for issue in issues}
        assert checks == {
            "baseline_schema",
            "baseline_missingness",
            "baseline_mean_shift",
        }

    def test_baseline_round_trips_as_json(self, tmp_path) -> None:
        baseline = DatasetBaseline.from_dataframe(pd.DataFrame({"value": [1.0, 2.0]}))
        path = baseline.to_file(tmp_path / "profiles" / "baseline.json")
        restored = DatasetBaseline.from_json(path)
        assert restored.rows == 2
        assert restored.column_profiles["value"]["mean"] == 1.5


class TestRuleInference:
    def test_infers_bounds_categories_and_missing_tolerance(self) -> None:
        data = pd.DataFrame(
            {
                "age": [20.0, 35.0, None],
                "group": ["control", "treated", "control"],
            }
        )
        rules = ValidationRules.infer(data, missing_tolerance=0.1)

        assert rules.columns["age"].dtype == "numeric"
        assert rules.columns["age"].min_value == 20.0
        assert rules.columns["age"].max_value == 35.0
        assert rules.columns["age"].max_missing_ratio == pytest.approx(1 / 3 + 0.1)
        assert rules.columns["group"].allowed_values == ["control", "treated"]

    def test_high_cardinality_text_does_not_freeze_values(self) -> None:
        data = pd.DataFrame({"id": [f"subject-{i}" for i in range(30)]})
        rules = ValidationRules.infer(data, max_categories=5)
        assert rules.columns["id"].dtype == "categorical"
        assert rules.columns["id"].allowed_values is None


class TestTextPatternRules:
    def test_reports_values_that_do_not_fully_match(self) -> None:
        rules = ValidationRules.from_dict(
            {"subject_id": {"pattern": r"SUBJ-\d{3}"}}
        )
        report = DatasetAuditor(rules=rules).audit_dataframe(
            pd.DataFrame({"subject_id": ["SUBJ-001", "SUBJ-12", None]})
        )
        assert "1 value(s) do not match pattern" in _messages(report, "rule")[0]

    def test_pattern_round_trips_in_rule_dictionary(self) -> None:
        rules = ValidationRules.from_dict({"code": {"pattern": "[A-Z]+"}})
        assert rules.to_dict()["code"]["pattern"] == "[A-Z]+"

    def test_invalid_pattern_identifies_the_column(self) -> None:
        rules = ValidationRules.from_dict({"code": {"pattern": "["}})
        with pytest.raises(ValueError, match="column 'code'"):
            DatasetAuditor(rules=rules).audit_dataframe(pd.DataFrame({"code": ["A"]}))


class TestCardinalityRules:
    def test_min_and_max_unique_values_are_enforced(self) -> None:
        rules = ValidationRules.from_dict(
            {
                "group": {"min_unique": 3},
                "code": {"max_unique": 2},
            }
        )
        report = DatasetAuditor(rules=rules).audit_dataframe(
            pd.DataFrame({"group": ["a", "a", "b"], "code": ["x", "y", "z"]})
        )

        messages = _messages(report, "rule")
        assert any("expected at least 3" in message for message in messages)
        assert any("expected at most 2" in message for message in messages)

    def test_cardinality_bounds_round_trip(self) -> None:
        rules = ValidationRules.from_dict(
            {"code": {"min_unique": 1, "max_unique": 4}}
        )
        assert rules.to_dict()["code"]["min_unique"] == 1
        assert rules.to_dict()["code"]["max_unique"] == 4

    def test_cardinality_bounds_must_be_ordered_non_negative_integers(self) -> None:
        with pytest.raises(ValueError, match="non-negative integer"):
            ValidationRules.from_dict({"code": {"max_unique": -1}})
        with pytest.raises(ValueError, match="cannot exceed"):
            ValidationRules.from_dict(
                {"code": {"min_unique": 4, "max_unique": 2}}
            )


class TestBatchAuditing:
    def test_audit_many_preserves_input_order_and_reports_failed_paths(self, tmp_path) -> None:
        clean = tmp_path / "clean.csv"
        duplicate = tmp_path / "duplicate.csv"
        pd.DataFrame({"id": [1, 2]}).to_csv(clean, index=False)
        pd.DataFrame({"id": [1, 1]}).to_csv(duplicate, index=False)

        batch = DatasetAuditor().audit_many([clean, duplicate])

        assert list(batch.reports) == [str(clean), str(duplicate)]
        assert batch.failed_paths == [str(duplicate)]
        assert batch.exit_code() == 1
        assert batch.exit_code("error") == 0

    def test_batch_json_contains_each_file_report(self, tmp_path) -> None:
        first = tmp_path / "first.csv"
        second = tmp_path / "second.csv"
        pd.DataFrame({"value": [1]}).to_csv(first, index=False)
        pd.DataFrame({"value": [2]}).to_csv(second, index=False)

        payload = BatchAuditReport(
            {str(first): DatasetAuditor().audit_file(first), str(second): DatasetAuditor().audit_file(second)}
        ).to_dict()

        assert payload["status"] == "pass"
        assert set(payload["files"]) == {str(first), str(second)}


class TestRendering:
    def test_csv_contains_a_stable_flat_findings_schema(self) -> None:
        report = AuditReport(rows=2, columns=1, duplicate_rows=1, missing_cells=0)
        report.issues.append(
            AuditIssue(
                check="duplicates",
                severity="warning",
                message="row, contains comma",
                column="id",
                observed=1,
            )
        )

        lines = report.to_csv().splitlines()
        assert lines[0] == "severity,check,column,message,observed,threshold"
        assert 'warning,duplicates,id,"row, contains comma",1,' in lines[1]

    def test_sarif_maps_issues_and_column_metadata(self) -> None:
        report = AuditReport(rows=2, columns=1, duplicate_rows=1, missing_cells=0)
        report.issues.append(
            AuditIssue(
                check="uniqueness",
                severity="error",
                message="Identifier is repeated.",
                column="id",
                observed=1,
                threshold=0,
            )
        )

        payload = json.loads(report.to_sarif(artifact_uri="data/train.csv"))
        assert payload["version"] == "2.1.0"
        result = payload["runs"][0]["results"][0]
        assert result["ruleId"] == "uniqueness"
        assert result["level"] == "error"
        assert result["properties"] == {
            "column": "id",
            "observed": 1,
            "threshold": 0,
        }
        location = result["locations"][0]["physicalLocation"]
        assert location["artifactLocation"]["uri"] == "data/train.csv"

    def test_html_survives_an_entirely_missing_numeric_column(self) -> None:
        data = pd.DataFrame({"n": [1.0, 2.0], "empty": pd.Series([None, None], dtype="float64")})
        report = DatasetAuditor().audit_dataframe(data)
        html = report.to_html()
        assert "<td>Mean</td><td>?</td>" in html

    def test_markdown_survives_an_entirely_missing_numeric_column(self) -> None:
        data = pd.DataFrame({"empty": pd.Series([None, None], dtype="float64")})
        report = DatasetAuditor().audit_dataframe(data)
        assert "Mean: ?" in report.to_markdown()

    def test_html_escapes_column_names(self) -> None:
        data = pd.DataFrame({"<script>": [1, 2]})
        html = DatasetAuditor().audit_dataframe(data).to_html()
        assert "<script>" not in html.split("<style>")[1]

    def test_junit_xml_maps_blocking_and_informational_findings(self) -> None:
        report = AuditReport(rows=2, columns=1, duplicate_rows=0, missing_cells=0)
        report.issues.extend(
            [
                AuditIssue("rule", "warning", "Bad <value>", column="code"),
                AuditIssue("schema_diff", "info", "Column added", column="new"),
            ]
        )
        root = ElementTree.fromstring(report.to_junit_xml(suite_name="nightly"))
        assert root.attrib == {
            "name": "nightly",
            "tests": "2",
            "failures": "1",
            "skipped": "1",
        }
        failure = root.find("./testcase/failure")
        assert failure is not None and failure.text == "Bad <value>"

    def test_junit_xml_writes_through_report_file_api(self, tmp_path) -> None:
        report = AuditReport(rows=1, columns=1, duplicate_rows=0, missing_cells=0)
        destination = tmp_path / "reports" / "audit.xml"
        report.to_file(str(destination))
        assert ElementTree.parse(destination).getroot().attrib["failures"] == "0"


class TestReportFiles:
    def test_to_file_creates_parent_directories(self, tmp_path) -> None:
        report = DatasetAuditor().audit_dataframe(pd.DataFrame({"a": [1]}))
        destination = tmp_path / "nested" / "deeper" / "report.json"
        report.to_file(str(destination))
        assert destination.exists()

    def test_to_file_rejects_unknown_extension(self, tmp_path) -> None:
        report = DatasetAuditor().audit_dataframe(pd.DataFrame({"a": [1]}))
        with pytest.raises(ValueError, match="Unsupported report format"):
            report.to_file(str(tmp_path / "report.txt"))

    def test_to_file_supports_csv_findings(self, tmp_path) -> None:
        report = AuditReport(rows=1, columns=1, duplicate_rows=0, missing_cells=0)
        destination = tmp_path / "reports" / "audit.csv"
        report.to_file(str(destination))
        assert destination.read_text(encoding="utf-8").startswith(
            "severity,check,column,message,observed,threshold\n"
        )


class TestFileLoading:
    def test_audit_file_honours_encoding(self, tmp_path) -> None:
        path = tmp_path / "latin.csv"
        pd.DataFrame({"n": ["café", "b"], "v": [1, 2]}).to_csv(
            path, index=False, encoding="latin-1"
        )
        report = DatasetAuditor().audit_file(str(path), encoding="latin-1")
        assert report.rows == 2

    def test_audit_file_honours_delimiter(self, tmp_path) -> None:
        path = tmp_path / "semi.csv"
        path.write_text("a;b\n1;2\n", encoding="utf-8")
        report = DatasetAuditor().audit_file(str(path), delimiter=";")
        assert report.columns == 2

    def test_reference_is_read_with_the_same_options(self, tmp_path) -> None:
        current = tmp_path / "cur.csv"
        reference = tmp_path / "ref.csv"
        current.write_text("a;b\n1;2\n", encoding="utf-8")
        reference.write_text("a;b\n1;2\n", encoding="utf-8")
        report = DatasetAuditor().audit_file(
            str(current), reference_path=str(reference), delimiter=";"
        )
        assert report.columns == 2

class TestDateFormatRule:
    def test_reports_unparseable_dates(self) -> None:
        rules = ValidationRules.from_dict(
            {"when": {"date_format": "%Y-%m-%d"}}
        )
        data = pd.DataFrame({"when": ["2026-01-01", "not-a-date"]})
        report = DatasetAuditor(rules=rules).audit_dataframe(data)
        message = _messages(report, "rule")[0]
        assert "do not parse with date format" in message

    def test_valid_dates_pass(self) -> None:
        rules = ValidationRules.from_dict(
            {"when": {"date_format": "%Y-%m-%d"}}
        )
        data = pd.DataFrame({"when": ["2026-01-01", "2026-12-31"]})
        report = DatasetAuditor(rules=rules).audit_dataframe(data)
        assert _messages(report, "rule") == []

    def test_missing_values_are_skipped(self) -> None:
        rules = ValidationRules.from_dict(
            {"when": {"date_format": "%Y-%m-%d"}}
        )
        data = pd.DataFrame({"when": [None, "2026-01-01"]})
        report = DatasetAuditor(rules=rules).audit_dataframe(data)
        assert _messages(report, "rule") == []

    def test_invalid_format_raises(self) -> None:
        rules = ValidationRules.from_dict(
            {"when": {"date_format": "%Q"}}
        )
        with pytest.raises(ValueError, match="date_format"):
            DatasetAuditor(rules=rules).audit_dataframe(
                pd.DataFrame({"when": ["2026-01-01"]})
            )

    def test_round_trips_through_json(self, tmp_path) -> None:
        path = tmp_path / "rules.json"
        path.write_text(
            json.dumps({"when": {"date_format": "%Y-%m-%d"}}),
            encoding="utf-8",
        )
        rules = ValidationRules.from_json(str(path))
        assert rules.columns["when"].date_format == "%Y-%m-%d"


class TestStringLengthRules:
    def test_reports_values_outside_configured_length_range(self) -> None:
        rules = ValidationRules.from_dict(
            {"code": {"min_length": 2, "max_length": 4}}
        )
        report = DatasetAuditor(rules=rules).audit_dataframe(
            pd.DataFrame({"code": ["A", "AB", "ABCDE", None]})
        )

        messages = _messages(report, "rule")
        assert "1 value(s) shorter than minimum length 2." in messages
        assert "1 value(s) longer than maximum length 4." in messages

    def test_length_bounds_round_trip_and_validate_order(self) -> None:
        rules = ValidationRules.from_dict({"code": {"min_length": 2, "max_length": 4}})
        assert rules.to_dict()["code"] == {"min_length": 2, "max_length": 4}

        with pytest.raises(ValueError, match="min_length cannot exceed max_length"):
            ValidationRules.from_dict({"code": {"min_length": 5, "max_length": 4}})


class TestOutlierRatioRules:
    def test_enforces_configured_iqr_outlier_ratio_without_numeric_bounds(self) -> None:
        rules = ValidationRules.from_dict({"score": {"max_outlier_ratio": 0.10}})
        report = DatasetAuditor(rules=rules).audit_dataframe(
            pd.DataFrame({"score": [1, 2, 2, 3, 3, 20]})
        )

        issue = next(issue for issue in report.issues if issue.column == "score")
        assert issue.severity == "info"
        assert issue.threshold == 0.10
        assert "exceeds allowed 10.0%" in issue.message

    def test_outlier_ratio_round_trips_and_requires_a_fraction(self) -> None:
        rules = ValidationRules.from_dict({"score": {"max_outlier_ratio": 0.25}})
        assert rules.to_dict()["score"]["max_outlier_ratio"] == 0.25

        with pytest.raises(ValueError, match="max_outlier_ratio"):
            ValidationRules.from_dict({"score": {"max_outlier_ratio": 1.1}})


class TestDuplicateAllowance:
    def test_allows_duplicate_rows_within_configured_ratio(self) -> None:
        data = pd.DataFrame({"id": [1, 1, 2, 3]})

        allowed = DatasetAuditor(max_duplicate_ratio=0.25).audit_dataframe(data)
        rejected = DatasetAuditor(max_duplicate_ratio=0.24).audit_dataframe(data)

        assert _messages(allowed, "duplicates") == []
        issue = next(issue for issue in rejected.issues if issue.check == "duplicates")
        assert issue.threshold == 0.24
        assert "exceeds allowed 24.0%" in issue.message

    def test_duplicate_allowance_must_be_a_fraction(self) -> None:
        with pytest.raises(ValueError, match="max_duplicate_ratio"):
            DatasetAuditor(max_duplicate_ratio=1.01)


class TestColumnDriftThresholds:
    def test_column_rule_can_be_stricter_than_global_drift_threshold(self) -> None:
        rules = ValidationRules.from_dict({"revenue": {"max_drift": 0.05}})
        report = DatasetAuditor(drift_threshold=0.20, rules=rules).audit_dataframe(
            pd.DataFrame({"revenue": [11.0, 11.0]}),
            reference=pd.DataFrame({"revenue": [10.0, 10.0]}),
        )

        issue = next(issue for issue in report.issues if issue.check == "drift")
        assert issue.threshold == 0.05
        assert "0.050 threshold" in issue.message

    def test_column_drift_threshold_round_trips_and_rejects_negative_values(self) -> None:
        rules = ValidationRules.from_dict({"revenue": {"max_drift": 0.15}})
        assert rules.to_dict()["revenue"]["max_drift"] == 0.15

        with pytest.raises(ValueError, match="max_drift"):
            ValidationRules.from_dict({"revenue": {"max_drift": -0.1}})


class TestRowCountContracts:
    def test_enforces_minimum_and_maximum_dataset_rows(self) -> None:
        too_small = DatasetAuditor(min_rows=3).audit_dataframe(pd.DataFrame({"x": [1, 2]}))
        too_large = DatasetAuditor(max_rows=2).audit_dataframe(pd.DataFrame({"x": [1, 2, 3]}))

        assert "expected at least 3" in _messages(too_small, "rows")[0]
        assert "expected at most 2" in _messages(too_large, "rows")[0]

    def test_rejects_inverted_row_count_contract(self) -> None:
        with pytest.raises(ValueError, match="min_rows"):
            DatasetAuditor(min_rows=4, max_rows=3)


class TestMissingCellAllowance:
    def test_enforces_global_missing_cell_allowance(self) -> None:
        report = DatasetAuditor(max_missing_cells=1).audit_dataframe(
            pd.DataFrame({"a": [1, None], "b": [None, 2]})
        )

        issue = next(issue for issue in report.issues if issue.check == "missing_cells")
        assert issue.observed == 2
        assert issue.threshold == 1


class TestColumnWidthAllowance:
    def test_warns_when_dataset_exceeds_column_budget(self) -> None:
        report = DatasetAuditor(max_columns=2).audit_dataframe(
            pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        )

        issue = next(issue for issue in report.issues if issue.check == "columns")
        assert issue.observed == 3
        assert issue.threshold == 2


class TestColumnOrderContract:
    def test_can_require_expected_column_order(self) -> None:
        report = DatasetAuditor().audit_dataframe(
            pd.DataFrame({"b": [1], "a": [2]}),
            expected_columns=["a", "b"],
            require_column_order=True,
        )

        assert _messages(report, "schema") == [
            "Column order does not match the expected schema contract."
        ]

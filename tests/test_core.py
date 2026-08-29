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
    ColumnRule,
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


class TestWhitespaceValueCheck:
    def test_padded_values_are_flagged_when_enabled(self) -> None:
        data = pd.DataFrame({"name": [" ann", "bo", "cy ", "di"]})
        report = DatasetAuditor(whitespace_check=True).audit_dataframe(data)
        messages = _messages(report, "whitespace")
        assert len(messages) == 1
        assert "leading or trailing" in messages[0]
        issue = next(i for i in report.issues if i.check == "whitespace")
        assert issue.observed == 2

    def test_invisible_characters_are_flagged(self) -> None:
        data = pd.DataFrame({"name": ["ann\u200b", "bo"]})
        report = DatasetAuditor(whitespace_check=True).audit_dataframe(data)
        messages = _messages(report, "whitespace")
        assert len(messages) == 1
        assert "invisible" in messages[0]

    def test_clean_text_stays_silent(self) -> None:
        data = pd.DataFrame({"name": ["ann", "bo"]})
        report = DatasetAuditor(whitespace_check=True).audit_dataframe(data)
        assert _messages(report, "whitespace") == []

    def test_numeric_columns_are_ignored(self) -> None:
        data = pd.DataFrame({"n": [1.0, 2.0]})
        report = DatasetAuditor(whitespace_check=True).audit_dataframe(data)
        assert _messages(report, "whitespace") == []

    def test_the_check_is_off_by_default(self) -> None:
        data = pd.DataFrame({"name": [" ann", "bo"]})
        report = DatasetAuditor().audit_dataframe(data)
        assert _messages(report, "whitespace") == []


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


class TestAllowedValueCaseHandling:
    def test_case_differences_are_unexpected_by_default(self) -> None:
        rules = ValidationRules.from_dict({"grade": {"allowed_values": ["A", "B"]}})
        report = DatasetAuditor(rules=rules).audit_dataframe(
            pd.DataFrame({"grade": ["a", "B"]})
        )
        assert "Unexpected values found: a." in _messages(report, "rule")[0]

    def test_ignore_case_accepts_mixed_case_values(self) -> None:
        rules = ValidationRules.from_dict(
            {"grade": {"allowed_values": ["A", "B"], "ignore_case": True}}
        )
        report = DatasetAuditor(rules=rules).audit_dataframe(
            pd.DataFrame({"grade": ["a", "b", None]})
        )
        assert _messages(report, "rule") == []

    def test_ignore_case_still_reports_genuinely_unknown_values(self) -> None:
        rules = ValidationRules.from_dict(
            {"grade": {"allowed_values": ["A"], "ignore_case": True}}
        )
        report = DatasetAuditor(rules=rules).audit_dataframe(
            pd.DataFrame({"grade": ["a", "Z"]})
        )
        assert "Unexpected values found: Z." in _messages(report, "rule")[0]

    def test_ignore_case_must_be_a_boolean(self) -> None:
        with pytest.raises(ValueError, match="ignore_case"):
            ValidationRules.from_dict(
                {"grade": {"allowed_values": ["A"], "ignore_case": "yes"}}
            )

    def test_ignore_case_round_trips_only_when_enabled(self) -> None:
        rules = ValidationRules.from_dict(
            {"grade": {"allowed_values": ["A"], "ignore_case": True}}
        )
        assert rules.to_dict()["grade"]["ignore_case"] is True

        plain = ValidationRules.from_dict({"code": {"pattern": "[A-Z]+"}})
        assert "ignore_case" not in plain.to_dict()["code"]


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


class TestDateRangeRules:
    def test_reports_values_outside_configured_date_range(self) -> None:
        rules = ValidationRules.from_dict(
            {"when": {"min_date": "2020-01-01", "max_date": "2020-12-31"}}
        )
        report = DatasetAuditor(rules=rules).audit_dataframe(
            pd.DataFrame({"when": ["2019-06-01", "2020-06-01", "2021-06-01", None]})
        )

        messages = _messages(report, "rule")
        assert "1 value(s) before minimum date '2020-01-01'." in messages
        assert "1 value(s) after maximum date '2020-12-31'." in messages

    def test_boundary_values_are_accepted(self) -> None:
        rules = ValidationRules.from_dict(
            {"when": {"min_date": "2020-01-01", "max_date": "2020-12-31"}}
        )
        report = DatasetAuditor(rules=rules).audit_dataframe(
            pd.DataFrame({"when": ["2020-01-01", "2020-12-31"]})
        )
        assert _messages(report, "rule") == []

    def test_future_dates_flagged_when_enabled(self) -> None:
        rules = ValidationRules.from_dict({"when": {"no_future_dates": True}})
        report = DatasetAuditor(rules=rules).audit_dataframe(
            pd.DataFrame({"when": ["2020-06-01", "2050-06-01"]})
        )

        messages = _messages(report, "rule")
        assert any("lie in the future" in message for message in messages)
        assert [issue.observed for issue in report.issues if issue.check == "rule"] == [1]

    def test_past_dates_pass_the_future_check(self) -> None:
        rules = ValidationRules.from_dict({"when": {"no_future_dates": True}})
        report = DatasetAuditor(rules=rules).audit_dataframe(
            pd.DataFrame({"when": ["2015-06-01", "2020-06-01"]})
        )
        assert _messages(report, "rule") == []

    def test_missing_and_unparseable_values_are_skipped(self) -> None:
        rules = ValidationRules.from_dict(
            {"when": {"min_date": "2020-01-01", "max_date": "2020-12-31"}}
        )
        report = DatasetAuditor(rules=rules).audit_dataframe(
            pd.DataFrame({"when": [None, "2020-06-01", "not-a-date"]})
        )
        assert _messages(report, "rule") == []

    def test_applies_to_datetime_dtype_columns(self) -> None:
        rules = ValidationRules.from_dict({"when": {"max_date": "2020-12-31"}})
        report = DatasetAuditor(rules=rules).audit_dataframe(
            pd.DataFrame({"when": pd.to_datetime(["2020-06-01", "2021-06-01"])})
        )

        messages = _messages(report, "rule")
        assert "1 value(s) after maximum date '2020-12-31'." in messages

    def test_honours_configured_date_format(self) -> None:
        rules = ValidationRules.from_dict(
            {"when": {"date_format": "%Y-%m-%d", "min_date": "2020-01-01"}}
        )
        report = DatasetAuditor(rules=rules).audit_dataframe(
            pd.DataFrame({"when": ["2019-06-01", "2020-06-01", "31/12/2019"]})
        )

        messages = _messages(report, "rule")
        assert "1 value(s) before minimum date '2020-01-01'." in messages

    def test_invalid_bound_raises(self) -> None:
        with pytest.raises(ValueError, match="min_date"):
            ValidationRules.from_dict({"when": {"min_date": "not-a-date"}})

    def test_invalid_bound_constructed_directly_raises_at_audit_time(self) -> None:
        rules = ValidationRules(
            columns={"when": ColumnRule(name="when", min_date="not-a-date")}
        )
        with pytest.raises(ValueError, match="Invalid min_date"):
            DatasetAuditor(rules=rules).audit_dataframe(
                pd.DataFrame({"when": ["2020-06-01"]})
            )

    def test_min_exceeding_max_raises(self) -> None:
        with pytest.raises(ValueError, match="min_date cannot exceed max_date"):
            ValidationRules.from_dict(
                {"when": {"min_date": "2021-01-01", "max_date": "2020-01-01"}}
            )

    def test_non_string_bounds_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_date"):
            ValidationRules.from_dict({"when": {"min_date": 20200101}})

    def test_round_trips_through_json(self, tmp_path) -> None:
        path = tmp_path / "rules.json"
        path.write_text(
            json.dumps(
                {
                    "when": {
                        "min_date": "2020-01-01",
                        "max_date": "2020-12-31",
                        "no_future_dates": True,
                    }
                }
            ),
            encoding="utf-8",
        )
        rules = ValidationRules.from_json(str(path))
        assert rules.columns["when"].min_date == "2020-01-01"
        assert rules.columns["when"].max_date == "2020-12-31"
        assert rules.columns["when"].no_future_dates is True
        assert rules.to_dict()["when"] == {
            "min_date": "2020-01-01",
            "max_date": "2020-12-31",
            "no_future_dates": True,
        }


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


class TestNumericBoundModes:
    def test_bounds_are_inclusive_by_default(self) -> None:
        rules = ValidationRules.from_dict({"n": {"min_value": 0.0, "max_value": 10.0}})
        report = DatasetAuditor(rules=rules).audit_dataframe(
            pd.DataFrame({"n": [0.0, 10.0, -0.1, 10.1]})
        )

        messages = _messages(report, "rule")
        assert any("below minimum 0.0" in m for m in messages)
        assert any("above maximum 10.0" in m for m in messages)
        issues = [i for i in report.issues if i.check == "rule" and i.severity == "warning"]
        assert [i.observed for i in issues] == [1.0, 1.0]

    def test_exclusive_min_flags_the_boundary_value(self) -> None:
        rules = ValidationRules.from_dict(
            {"n": {"min_value": 0.0, "min_inclusive": False}}
        )
        report = DatasetAuditor(rules=rules).audit_dataframe(
            pd.DataFrame({"n": [0.0, 1.0, -1.0]})
        )

        issue = next(i for i in report.issues if i.check == "rule")
        assert issue.observed == 2.0
        assert "exclusive bound" in issue.message
        assert issue.message.startswith("2 value(s) below minimum 0.0")

    def test_exclusive_max_flags_the_boundary_value(self) -> None:
        rules = ValidationRules.from_dict(
            {"n": {"max_value": 10.0, "max_inclusive": False}}
        )
        report = DatasetAuditor(rules=rules).audit_dataframe(
            pd.DataFrame({"n": [10.0, 9.0, 11.0]})
        )

        issue = next(i for i in report.issues if i.check == "rule")
        assert issue.observed == 2.0
        assert "exclusive bound" in issue.message
        assert issue.message.startswith("2 value(s) above maximum 10.0")

    def test_tolerance_absorbs_values_near_the_bound(self) -> None:
        rules = ValidationRules.from_dict(
            {"n": {"min_value": 0.0, "value_tolerance": 0.5}}
        )
        report = DatasetAuditor(rules=rules).audit_dataframe(
            pd.DataFrame({"n": [-0.4, 0.0, -0.6]})
        )

        issue = next(i for i in report.issues if i.check == "rule")
        assert issue.observed == 1.0
        assert "beyond 0.5 tolerance" in issue.message

    def test_tolerance_and_exclusive_mode_compose(self) -> None:
        rules = ValidationRules.from_dict(
            {
                "n": {
                    "max_value": 10.0,
                    "max_inclusive": False,
                    "value_tolerance": 0.25,
                }
            }
        )
        report = DatasetAuditor(rules=rules).audit_dataframe(
            pd.DataFrame({"n": [10.2, 10.0, 10.3]})
        )

        issue = next(i for i in report.issues if i.check == "rule")
        assert issue.observed == 1.0
        assert "exclusive bound" in issue.message
        assert "beyond 0.25 tolerance" in issue.message

    def test_new_options_round_trip_through_the_dict_form(self) -> None:
        rules = ValidationRules.from_dict(
            {
                "n": {
                    "min_value": 0.0,
                    "min_inclusive": False,
                    "max_inclusive": False,
                    "value_tolerance": 0.25,
                }
            }
        )
        entry = rules.to_dict()["n"]
        assert entry["min_inclusive"] is False
        assert entry["max_inclusive"] is False
        assert entry["value_tolerance"] == 0.25

        reloaded = ValidationRules.from_dict({"n": entry})
        assert reloaded.columns["n"].min_inclusive is False
        assert reloaded.columns["n"].max_inclusive is False
        assert reloaded.columns["n"].value_tolerance == 0.25

    def test_defaults_are_omitted_from_the_dict_form(self) -> None:
        rules = ValidationRules.from_dict({"n": {"min_value": 0.0}})
        entry = rules.to_dict()["n"]
        assert "min_inclusive" not in entry
        assert "value_tolerance" not in entry

    def test_rejects_negative_tolerance(self) -> None:
        with pytest.raises(ValueError, match="value_tolerance"):
            ValidationRules.from_dict({"n": {"min_value": 0, "value_tolerance": -0.1}})

    def test_rejects_non_boolean_modes(self) -> None:
        with pytest.raises(ValueError, match="min_inclusive"):
            ValidationRules.from_dict({"n": {"min_value": 0, "min_inclusive": "yes"}})
        with pytest.raises(ValueError, match="max_inclusive"):
            ValidationRules.from_dict({"n": {"max_value": 5, "max_inclusive": 1}})


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


class TestRedundancyThreshold:
    def test_perfect_correlation_is_flagged_with_the_default_threshold(self) -> None:
        data = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
        report = DatasetAuditor().audit_dataframe(data)

        issue = next(issue for issue in report.issues if issue.check == "redundancy")
        assert issue.threshold == 0.95

    def test_lowering_the_threshold_catches_weaker_correlations(self) -> None:
        data = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [1, 3, 2, 5, 4, 6]})

        assert _messages(DatasetAuditor().audit_dataframe(data), "redundancy") == []
        flagged = DatasetAuditor(redundancy_threshold=0.85).audit_dataframe(data)
        assert len(_messages(flagged, "redundancy")) == 1

    def test_threshold_must_be_a_fraction(self) -> None:
        with pytest.raises(ValueError, match="redundancy_threshold"):
            DatasetAuditor(redundancy_threshold=1.01)


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
class TestDuplicateColumnDetection:
    def test_copies_of_a_column_are_flagged_once_per_pair(self) -> None:
        data = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "copy": [1, 2, 3],
                "other": ["a", "b", "c"],
            }
        )
        report = DatasetAuditor().audit_dataframe(data)
        flagged = [i for i in report.issues if i.check == "duplicate_columns"]
        assert len(flagged) == 1
        assert flagged[0].column == "id,copy"

    def test_one_differing_value_breaks_the_match(self) -> None:
        data = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [1.0, 2.0, 3.5],
            }
        )
        report = DatasetAuditor().audit_dataframe(data)
        assert not [i for i in report.issues if i.check == "duplicate_columns"]

    def test_matching_gaps_still_count_as_identical(self) -> None:
        data = pd.DataFrame(
            {
                "a": pd.Series([None, "x", None], dtype=object),
                "b": pd.Series([None, "x", None], dtype=object),
            }
        )
        report = DatasetAuditor().audit_dataframe(data)
        checks = [i.check for i in report.issues]
        assert "duplicate_columns" in checks

    def test_entirely_missing_columns_are_not_compared(self) -> None:
        data = pd.DataFrame(
            {
                "a": pd.Series([None, None], dtype=object),
                "b": pd.Series([None, None], dtype=object),
                "c": ["p", "q"],
            }
        )
        report = DatasetAuditor().audit_dataframe(data)
        assert not [i for i in report.issues if i.check == "duplicate_columns"]

    def test_three_copies_yield_three_pairs(self) -> None:
        values = [7, 8, 9]
        data = pd.DataFrame(
            {"a": values, "b": list(values), "c": list(values)}
        )
        report = DatasetAuditor().audit_dataframe(data)
        flagged = [i for i in report.issues if i.check == "duplicate_columns"]
        assert len(flagged) == 3

    def test_text_column_copies_are_caught_without_correlation(self) -> None:
        data = pd.DataFrame(
            {
                "city": ["berlin", "tokyo", "lima"],
                "clone": ["berlin", "tokyo", "lima"],
            }
        )
        report = DatasetAuditor().audit_dataframe(data)
        checks = [i.check for i in report.issues]
        assert "duplicate_columns" in checks
class TestCategoryShareCheck:
    def test_dominant_category_is_flagged_when_configured(self) -> None:
        data = pd.DataFrame({"status": ["ok"] * 9 + ["no"]})
        report = DatasetAuditor(max_category_share=0.80).audit_dataframe(data)

        issue = next(i for i in report.issues if i.check == "category_share")
        assert issue.observed == 0.9
        assert issue.severity == "warning"
        assert "90.0%" in issue.message

    def test_share_at_or_below_the_threshold_stays_silent(self) -> None:
        data = pd.DataFrame({"status": ["ok"] * 7 + ["no"] * 3})
        report = DatasetAuditor(max_category_share=0.80).audit_dataframe(data)

        assert _messages(report, "category_share") == []

    def test_rare_categories_are_flagged_when_configured(self) -> None:
        data = pd.DataFrame({"status": ["a"] * 8 + ["b"] + ["c"]})
        report = DatasetAuditor(rare_category_share=0.15).audit_dataframe(data)

        issue = next(i for i in report.issues if i.check == "category_share")
        assert issue.observed == 2
        assert "rare category(ies)" in issue.message

    def test_dominance_and_rarity_can_fire_together(self) -> None:
        data = pd.DataFrame({"status": ["a"] * 9 + ["b"] + ["c"]})
        report = DatasetAuditor(
            max_category_share=0.80, rare_category_share=0.15
        ).audit_dataframe(data)

        flagged = [i for i in report.issues if i.check == "category_share"]
        assert len(flagged) == 2

    def test_missing_values_do_not_count_toward_the_share(self) -> None:
        data = pd.DataFrame({"status": ["ok"] * 9 + [None] * 5})
        report = DatasetAuditor(max_category_share=0.80).audit_dataframe(data)

        # 9 of 9 non-missing values are "ok" -> a 100% share.
        issue = next(i for i in report.issues if i.check == "category_share")
        assert issue.observed == 1.0

    def test_numeric_columns_are_ignored(self) -> None:
        data = pd.DataFrame({"n": [1, 1, 1, 1, 2]})
        report = DatasetAuditor(max_category_share=0.50).audit_dataframe(data)

        assert _messages(report, "category_share") == []

    def test_the_check_is_off_by_default(self) -> None:
        data = pd.DataFrame({"status": ["ok", "ok", "ok"]})
        report = DatasetAuditor().audit_dataframe(data)

        assert _messages(report, "category_share") == []

    def test_thresholds_must_be_fractions_between_zero_and_one(self) -> None:
        with pytest.raises(ValueError, match="max_category_share"):
            DatasetAuditor(max_category_share=1.5)
        with pytest.raises(ValueError, match="max_category_share"):
            DatasetAuditor(max_category_share=0.0)
        with pytest.raises(ValueError, match="rare_category_share"):
            DatasetAuditor(rare_category_share=0.0)
        with pytest.raises(ValueError, match="rare_category_share"):
            DatasetAuditor(rare_category_share=True)


class TestWeightedRiskScoring:
    def test_default_weights_score_an_error_at_double_a_warning(self) -> None:
        data = pd.DataFrame({"n": [-1.0, -2.0]})
        auditor = DatasetAuditor(
            min_rows=5,
            rules=ValidationRules.from_dict({"n": {"min_value": 0.0}}),
        )
        report = auditor.audit_dataframe(data)
        checks = {(i.check, i.severity) for i in report.issues}
        assert ("rows", "error") in checks
        assert ("rule", "warning") in checks
        assert report.risk_score == 15.0

    def test_configured_weights_rescale_their_check(self) -> None:
        data = pd.DataFrame({"n": [-1.0, -2.0]})
        auditor = DatasetAuditor(
            min_rows=5,
            severity_weights={"rule": 30.0},
            rules=ValidationRules.from_dict({"n": {"min_value": 0.0}}),
        )
        report = auditor.audit_dataframe(data)
        assert report.risk_score == 25.0

    def test_the_score_is_capped_at_one_hundred(self) -> None:
        data = pd.DataFrame({"n": [-1.0, -2.0]})
        auditor = DatasetAuditor(
            min_rows=5,
            severity_weights={"rule": 400.0},
            rules=ValidationRules.from_dict({"n": {"min_value": 0.0}}),
        )
        assert auditor.audit_dataframe(data).risk_score == 100.0

    def test_info_findings_add_no_risk(self) -> None:
        data = pd.DataFrame({"n": [1.0, 2.0, 3.0, 4.0, 100.0]})
        rules = ValidationRules.from_dict({"n": {"max_outlier_ratio": 0.0}})
        report = DatasetAuditor(rules=rules).audit_dataframe(data)
        severities = {i.severity for i in report.issues}
        assert "info" in severities
        blocking = [i for i in report.issues if i.severity != "info"]
        if not blocking:
            assert report.risk_score == 0.0

    def test_weights_must_be_finite_non_negative_numbers(self) -> None:
        with pytest.raises(ValueError):
            DatasetAuditor(severity_weights={"drift": -1.0})
        with pytest.raises(ValueError):
            DatasetAuditor(severity_weights={"drift": float("nan")})
        with pytest.raises(ValueError):
            DatasetAuditor(severity_weights={"drift": True})

    def test_the_json_report_carries_the_risk_score(self) -> None:
        data = pd.DataFrame({"id": [1, 2, 3, 4], "m": [1.0, None, None, None]})
        auditor = DatasetAuditor(severity_weights={"missingness": 25.0})
        payload = json.loads(auditor.audit_dataframe(data).to_json())
        assert payload["risk_score"] == 12.5
class TestSeededSampleAudit:
    def test_a_sample_of_the_requested_size_is_reported(self) -> None:
        data = pd.DataFrame({"id": range(50), "value": ["x"] * 50})
        report = DatasetAuditor().audit_dataframe(data, sample_rows=10, sample_seed=7)
        assert report.rows == 10
        sampling = [i for i in report.issues if i.check == "sampling"]
        assert len(sampling) == 1
        assert sampling[0].severity == "info"
        assert "seed 7" in sampling[0].message

    def test_the_same_seed_reproduces_the_same_sample(self) -> None:
        data = pd.DataFrame({"id": range(50)})
        first = DatasetAuditor().audit_dataframe(data, sample_rows=10, sample_seed=3)
        second = DatasetAuditor().audit_dataframe(data, sample_rows=10, sample_seed=3)
        assert first.column_profiles == second.column_profiles
        assert [i.message for i in first.issues] == [i.message for i in second.issues]

    def test_no_sampling_notice_when_the_file_fits(self) -> None:
        data = pd.DataFrame({"id": range(5)})
        report = DatasetAuditor().audit_dataframe(data, sample_rows=10, sample_seed=1)
        assert report.rows == 5
        assert not [i for i in report.issues if i.check == "sampling"]

    def test_sample_rows_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            DatasetAuditor().audit_dataframe(pd.DataFrame({"a": [1]}), sample_rows=0)

    def test_a_seed_without_a_sample_size_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            DatasetAuditor().audit_dataframe(pd.DataFrame({"a": [1]}), sample_seed=1)
class TestBatchIssueCounts:
    def test_counts_aggregate_across_files_by_check(self) -> None:
        first = AuditReport(
            rows=1,
            columns=1,
            duplicate_rows=0,
            missing_cells=0,
            issues=[
                AuditIssue(check="missingness", severity="warning", message="m"),
                AuditIssue(check="missingness", severity="warning", message="m2"),
            ],
        )
        second = AuditReport(
            rows=1,
            columns=1,
            duplicate_rows=0,
            missing_cells=0,
            issues=[
                AuditIssue(check="duplicates", severity="warning", message="d"),
            ],
        )
        batch = BatchAuditReport(reports={"a.csv": first, "b.csv": second})
        assert batch.issue_counts() == {"missingness": 2, "duplicates": 1}


class TestRiskScoreGate:
    @staticmethod
    def _warning_report(count: int) -> AuditReport:
        report = AuditReport(
            rows=4,
            columns=2,
            duplicate_rows=0,
            missing_cells=2 * count,
            issues=[
                AuditIssue(check="missingness", severity="warning", message=f"w{i}")
                for i in range(count)
            ],
        )
        # Default-weight warnings contribute 5 points each to the score.
        report.risk_score = 5.0 * count
        return report

    def test_a_score_past_the_ceiling_fails_despite_the_error_gate(self) -> None:
        report = self._warning_report(3)
        assert report.exit_code("error") == 0
        assert report.exit_code("error", max_risk=10.0) == 1

    def test_a_score_at_the_ceiling_still_passes(self) -> None:
        assert self._warning_report(3).exit_code("error", max_risk=15.0) == 0

    def test_findings_still_fail_without_reaching_the_ceiling(self) -> None:
        report = self._warning_report(3)
        assert report.exit_code("warning", max_risk=100.0) == 1

    def test_a_negative_ceiling_is_rejected(self) -> None:
        report = AuditReport(rows=1, columns=1, duplicate_rows=0, missing_cells=0)
        with pytest.raises(ValueError, match="max_risk"):
            report.exit_code("warning", max_risk=-1)

    def test_the_batch_gate_applies_the_ceiling_to_every_file(self) -> None:
        clean = AuditReport(rows=1, columns=1, duplicate_rows=0, missing_cells=0)
        heavy = self._warning_report(3)
        batch = BatchAuditReport(reports={"a.csv": clean, "b.csv": heavy})
        assert batch.exit_code("error", max_risk=10.0) == 1
        assert batch.exit_code("error", max_risk=20.0) == 0
class TestNamedRuleProfiles:
    """Reusable named rule sets stored inside one rules file."""

    @pytest.fixture
    def profiles_path(self, tmp_path):
        path = tmp_path / "rules.json"
        path.write_text(
            json.dumps(
                {
                    "profiles": {
                        "strict": {"age": {"dtype": "numeric", "min_value": 0}},
                        "loose": {"age": {"dtype": "numeric"}},
                    }
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_profile_key_selects_one_rule_set(self, profiles_path: str) -> None:
        rules = ValidationRules.from_json(profiles_path, profile="strict")
        assert list(rules.columns) == ["age"]
        assert rules.columns["age"].min_value == 0

    def test_flat_files_load_unchanged(self, tmp_path) -> None:
        path = tmp_path / "flat.json"
        path.write_text('{"age": {"dtype": "numeric"}}', encoding="utf-8")
        assert list(ValidationRules.from_json(str(path)).columns) == ["age"]

    def test_a_profiles_file_demands_a_choice(self, profiles_path: str) -> None:
        with pytest.raises(ValueError, match=r"loose, strict"):
            ValidationRules.from_json(profiles_path)

    def test_unknown_profile_lists_the_available_names(self, profiles_path: str) -> None:
        with pytest.raises(ValueError, match="'tight' not found; available profiles: loose, strict"):
            ValidationRules.from_json(profiles_path, profile="tight")

    def test_naming_a_profile_on_a_flat_file_is_rejected(self, tmp_path) -> None:
        path = tmp_path / "flat.json"
        path.write_text('{"age": {"dtype": "numeric"}}', encoding="utf-8")
        with pytest.raises(ValueError, match="no profiles section"):
            ValidationRules.from_json(str(path), profile="strict")

    def test_profiles_section_must_map_names_to_objects(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"profiles": ["strict"]}', encoding="utf-8")
        with pytest.raises(ValueError, match="mapping profile names"):
            ValidationRules.from_json(str(path))

    def test_selected_rules_drive_the_audit(self, tmp_path) -> None:
        data = pd.DataFrame({"age": [5, 200]})
        path = tmp_path / "rules.json"
        path.write_text(
            json.dumps(
                {
                    "profiles": {
                        "adults_only": {"age": {"min_value": 18}},
                        "any_age": {"age": {"dtype": "numeric"}},
                    }
                }
            ),
            encoding="utf-8",
        )
        strict = DatasetAuditor(
            rules=ValidationRules.from_json(str(path), profile="adults_only")
        ).audit_dataframe(data)
        assert len(_messages(strict, "rule")) == 1

        relaxed = DatasetAuditor(
            rules=ValidationRules.from_json(str(path), profile="any_age")
        ).audit_dataframe(data)
        assert _messages(relaxed, "rule") == []
class TestRunMetadataStamping:
    def _stamped(self) -> AuditReport:
        report = AuditReport(rows=4, columns=2, duplicate_rows=0, missing_cells=0)
        report.audit_id = "run123"
        report.created_utc = "2026-08-26T08:00:00+00:00"
        report.config_hash = "cafe" * 16
        return report

    def test_unstamped_json_keeps_its_old_shape(self) -> None:
        report = AuditReport(rows=1, columns=1, duplicate_rows=0, missing_cells=0)
        assert "meta" not in report.to_dict()

    def test_stamped_json_carries_a_meta_block(self) -> None:
        payload = json.loads(self._stamped().to_json())
        assert payload["meta"] == {
            "audit_id": "run123",
            "created_utc": "2026-08-26T08:00:00+00:00",
            "config_hash": "cafe" * 16,
        }

    def test_sarif_properties_carry_the_run_stamps(self) -> None:
        runs = json.loads(self._stamped().to_sarif())["runs"]
        properties = runs[0]["properties"]
        assert properties["auditId"] == "run123"
        assert properties["createdUtc"] == "2026-08-26T08:00:00+00:00"
        assert properties["configHash"] == "cafe" * 16

    def test_unstamped_sarif_omits_the_stamp_keys(self) -> None:
        plain = AuditReport(rows=1, columns=1, duplicate_rows=0, missing_cells=0)
        properties = json.loads(plain.to_sarif())["runs"][0]["properties"]
        assert set(properties) == {"auditStatus", "qualityScore"}

    def test_html_shows_the_provenance_line(self) -> None:
        page = self._stamped().to_html()
        assert "<code>run123</code>" in page
        assert "Generated 2026-08-26T08:00:00+00:00" in page
        assert "cafe" * 16 in page

    def test_unstamped_html_has_no_provenance_line(self) -> None:
        plain = AuditReport(rows=1, columns=1, duplicate_rows=0, missing_cells=0)
        assert "Audit ID" not in plain.to_html()

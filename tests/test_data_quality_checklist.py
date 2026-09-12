"""Tests for AuditReport.data_quality_checklist."""

from __future__ import annotations

import json

import pytest

from dataset_audit_kit.core import AuditReport, AuditIssue, DatasetAuditor


def _report(rows=10, columns=2, **profiles):
    report = AuditReport(rows=rows, columns=columns, duplicate_rows=0, missing_cells=0)
    report.column_profiles = dict(profiles)
    return report


class TestChecklistEmpty:
    def test_empty_report(self) -> None:
        report = _report()
        rows = report.data_quality_checklist()
        assert len(rows) == 10
        for row in rows:
            assert row["status"] == "pass"
            assert row["issues"] == 0

    def test_core_checks_always_present(self) -> None:
        report = _report()
        checks = {row["check"] for row in report.data_quality_checklist()}
        for name in (
            "schema", "rows", "columns", "missing_cells",
            "duplicates", "missingness", "column_names",
            "uniqueness", "composite_uniqueness", "labels",
        ):
            assert name in checks


class TestChecklistWithIssues:
    def test_failing_checks_marked_fail(self) -> None:
        report = _report()
        report.issues = [
            AuditIssue(
                check="duplicates",
                severity="warning",
                message="2 duplicate rows found",
            ),
            AuditIssue(
                check="schema",
                severity="error",
                message="Missing column id",
            ),
        ]
        rows = {row["check"]: row for row in report.data_quality_checklist()}
        assert rows["duplicates"]["status"] == "fail"
        assert rows["duplicates"]["issues"] == 1
        assert rows["schema"]["status"] == "fail"
        assert rows["schema"]["issues"] == 1

    def test_info_only_is_pass(self) -> None:
        report = _report()
        report.issues = [
            AuditIssue(
                check="missingness",
                severity="info",
                message="Informational note",
            ),
        ]
        rows = {row["check"]: row for row in report.data_quality_checklist()}
        assert rows["missingness"]["status"] == "pass"
        assert rows["missingness"]["issues"] == 1

    def test_mixed_severity_is_fail(self) -> None:
        report = _report()
        report.issues = [
            AuditIssue(
                check="schema",
                severity="error",
                message="Missing column",
            ),
            AuditIssue(
                check="schema",
                severity="info",
                message="Extra column",
            ),
        ]
        rows = {row["check"]: row for row in report.data_quality_checklist()}
        assert rows["schema"]["status"] == "fail"
        assert rows["schema"]["issues"] == 2

    def test_multiple_checks(self) -> None:
        report = _report()
        report.issues = [
            AuditIssue(
                check="duplicates",
                severity="warning",
                message="duplicates",
            ),
            AuditIssue(
                check="missingness",
                severity="warning",
                message="missingness",
            ),
            AuditIssue(
                check="missingness",
                severity="warning",
                message="missingness 2",
            ),
        ]
        rows = {row["check"]: row for row in report.data_quality_checklist()}
        assert rows["duplicates"]["status"] == "fail"
        assert rows["duplicates"]["issues"] == 1
        assert rows["missingness"]["status"] == "fail"
        assert rows["missingness"]["issues"] == 2
        assert rows["schema"]["status"] == "pass"
        assert rows["schema"]["issues"] == 0


class TestChecklistSorting:
    def test_sorted_alphabetically(self) -> None:
        report = _report()
        report.issues = [
            AuditIssue(check="schema", severity="error", message="x"),
            AuditIssue(check="duplicates", severity="error", message="x"),
        ]
        names = [row["check"] for row in report.data_quality_checklist()]
        assert names == sorted(names)


class TestChecklistEndToEnd:
    def test_real_audited_frame(self) -> None:
        frame = __import__("pandas").DataFrame(
            {
                "id": range(20),
                "flag": ["y", "n"] * 10,
                "constant": ["same"] * 20,
            }
        )
        auditor = DatasetAuditor(missing_threshold=0.05, max_duplicate_ratio=0.0)
        report = auditor.audit_dataframe(frame)
        checklist = report.data_quality_checklist()
        assert len(checklist) > 0
        for row in checklist:
            assert row["check"]
            assert row["status"] in ("pass", "fail")
            assert isinstance(row["issues"], int)

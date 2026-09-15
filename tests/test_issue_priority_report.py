import pytest

from dataset_audit_kit.core import AuditIssue, AuditReport


def _report(*issues):
    return AuditReport(rows=1, columns=1, duplicate_rows=0, missing_cells=0, issues=list(issues))


def test_groups_repeated_findings_and_prioritizes_errors():
    report = _report(
        AuditIssue(check="missingness", severity="warning", message="missing", column="age"),
        AuditIssue(check="missingness", severity="warning", message="missing again", column="age"),
        AuditIssue(check="schema", severity="error", message="bad type", column="age"),
    )
    rows = report.issue_priority_report()
    assert rows[0] == {"check": "schema", "column": "age", "severity": "error", "count": 1, "message": "bad type"}
    assert rows[1]["count"] == 2
    assert rows[1]["message"] == "missing"


def test_empty_report_and_top_validation():
    assert _report().issue_priority_report() == []
    with pytest.raises(ValueError, match="top"):
        _report().issue_priority_report(top=True)

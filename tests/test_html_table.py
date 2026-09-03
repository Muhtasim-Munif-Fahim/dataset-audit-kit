"""Tests for AuditReport.to_html_table."""

from __future__ import annotations

import re

from dataset_audit_kit.core import AuditReport


def _report(rows=10, columns=2, **profiles):
    report = AuditReport(rows=rows, columns=columns, duplicate_rows=0, missing_cells=0)
    report.column_profiles = dict(profiles)
    return report


class TestToHtmlTable:
    def test_contains_report_headline(self) -> None:
        report = _report(
            a={"dtype": "numeric", "count": 100, "missing": 10, "outlier_ratio": 0.20},
        )
        report.audit_id = "abc"
        report.created_utc = "2026-01-01T00:00:00Z"
        report.risk_score = 0.5
        html = report.to_html_table()
        assert "abc" in html
        assert "2026-01-01T00:00:00Z" in html
        assert "dataset-audit-kit report" in html
        assert "0.5000" in html

    def test_lists_numeric_and_categorical_columns(self) -> None:
        report = _report(
            a={"dtype": "numeric"},
            b={"dtype": "categorical", "top_5": {"x": 1}},
        )
        html = report.to_html_table()
        assert re.search(r"<td>a</td>", html)
        assert re.search(r"<td>b</td>", html)
        assert "Column types" in html

    def test_escapes_user_data(self) -> None:
        report = _report(
            a={"dtype": "categorical", "top_5": {"<script>": 1}},
        )
        report.audit_id = "<x>"
        html = report.to_html_table()
        assert "<x>" not in html
        assert "&lt;x&gt;" in html

    def test_handles_empty_report(self) -> None:
        report = _report()
        html = report.to_html_table()
        assert "<!DOCTYPE html>" in html
        assert "dataset-audit-kit report" in html
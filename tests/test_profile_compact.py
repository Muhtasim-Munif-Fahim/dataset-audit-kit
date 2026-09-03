"""Tests for AuditReport.profile_to_dict_compact."""

from __future__ import annotations

from dataset_audit_kit.core import AuditReport


def _report(rows=10, columns=2, **profiles):
    report = AuditReport(rows=rows, columns=columns, duplicate_rows=0, missing_cells=0)
    report.column_profiles = dict(profiles)
    return report


class TestProfileToDictCompact:
    def test_returns_expected_keys(self) -> None:
        report = _report(
            a={"dtype": "numeric", "count": 100, "missing": 30, "outlier_ratio": 0.20},
            b={"dtype": "categorical", "top_5": {"x": 1}},
            c={"dtype": "other"},
        )
        payload = report.profile_to_dict_compact()
        assert set(payload) == {
            "audit_id", "created_utc", "rows", "columns", "duplicate_rows",
            "missing_cells", "risk_score", "status", "column_types",
            "high_missingness_columns", "high_outlier_columns",
            "blocking_issue_count", "checks_seen",
        }
        assert payload["column_types"]["numeric"] == ["a"]
        assert payload["column_types"]["categorical"] == ["b"]
        assert payload["column_types"]["other"] == ["c"]

    def test_high_missingness_listing(self) -> None:
        report = _report(
            a={"dtype": "numeric", "count": 100, "missing": 30},
            b={"dtype": "numeric", "count": 100, "missing": 1},
        )
        payload = report.profile_to_dict_compact()
        assert payload["high_missingness_columns"][0] == {"column": "a", "rate": 0.3}

    def test_high_outlier_listing(self) -> None:
        report = _report(
            a={"dtype": "numeric", "outlier_ratio": 0.40, "count": 100, "missing": 0},
            b={"dtype": "numeric", "outlier_ratio": 0.01, "count": 100, "missing": 0},
        )
        payload = report.profile_to_dict_compact()
        assert payload["high_outlier_columns"] == [{"column": "a", "ratio": 0.4}]

    def test_handles_empty_profiles(self) -> None:
        report = _report()
        payload = report.profile_to_dict_compact()
        assert payload["column_types"] == {"numeric": [], "categorical": [], "other": []}
        assert payload["high_missingness_columns"] == []
        assert payload["high_outlier_columns"] == []

    def test_status_is_pass_when_no_blocking_issues(self) -> None:
        report = _report()
        assert report.profile_to_dict_compact()["status"] in {"pass", "warn"}
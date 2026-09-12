"""Tests for AuditReport.dtype_summary."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from dataset_audit_kit.cli import main
from dataset_audit_kit.core import AuditReport, DatasetAuditor


def _report(rows=10, columns=2, **profiles):
    report = AuditReport(rows=rows, columns=columns, duplicate_rows=0, missing_cells=0)
    report.column_profiles = dict(profiles)
    return report


class TestDtypeSummaryBasic:
    def test_groups_columns_by_dtype(self) -> None:
        report = _report(
            a={"dtype": "numeric", "count": 10, "missing": 0, "unique": 5},
            b={"dtype": "categorical", "count": 10, "missing": 0, "unique": 3},
            c={"dtype": "numeric", "count": 10, "missing": 0, "unique": 8},
        )
        rows = report.dtype_summary()
        dtypes = {row["dtype"]: row for row in rows}
        assert dtypes["numeric"]["column_count"] == 2
        assert dtypes["categorical"]["column_count"] == 1

    def test_columns_sorted_alphabetically_within_group(self) -> None:
        report = _report(
            z={"dtype": "numeric", "count": 10, "missing": 0, "unique": 5},
            a={"dtype": "numeric", "count": 10, "missing": 0, "unique": 3},
            m={"dtype": "numeric", "count": 10, "missing": 0, "unique": 8},
        )
        row = report.dtype_summary()[0]
        assert row["columns"] == ["a", "m", "z"]

    def test_groups_sorted_by_count_desc_then_dtype_alpha(self) -> None:
        report = _report(
            a={"dtype": "categorical", "count": 10, "missing": 0, "unique": 3},
            b={"dtype": "numeric", "count": 10, "missing": 0, "unique": 5},
            c={"dtype": "numeric", "count": 10, "missing": 0, "unique": 8},
            d={"dtype": "categorical", "count": 10, "missing": 0, "unique": 2},
        )
        rows = report.dtype_summary()
        # Both groups have 2 columns; tied on count, so alphabetical by dtype.
        assert rows[0]["dtype"] == "categorical"
        assert rows[1]["dtype"] == "numeric"


class TestDtypeSummaryShape:
    def test_row_carries_documented_keys(self) -> None:
        report = _report(a={"dtype": "numeric", "count": 8, "missing": 2, "unique": 3})
        row = report.dtype_summary()[0]
        assert row == {
            "dtype": "numeric",
            "column_count": 1,
            "columns": ["a"],
        }

    def test_top_truncates_after_sorting(self) -> None:
        report = _report(
            a={"dtype": "numeric", "count": 100, "missing": 0, "unique": 10},
            b={"dtype": "categorical", "count": 100, "missing": 0, "unique": 20},
            c={"dtype": "datetime", "count": 100, "missing": 0, "unique": 50},
        )
        rows = report.dtype_summary(top=2)
        assert len(rows) == 2
        # All tied at count=1; alphabetical: categorical, datetime, numeric
        assert [row["dtype"] for row in rows] == ["categorical", "datetime"]

    def test_columns_without_a_profile_are_absent(self) -> None:
        report = _report()
        assert report.dtype_summary() == []


class TestDtypeSummaryValidation:
    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3"])
    def test_top_must_be_a_positive_integer(self, bad: object) -> None:
        with pytest.raises(ValueError, match="top must be a positive integer"):
            _report().dtype_summary(top=bad)


class TestDtypeSummaryEndToEnd:
    def test_groups_a_real_audited_frame(self) -> None:
        frame = pd.DataFrame(
            {
                "id": range(20),
                "flag": ["y", "n"] * 10,
                "constant": ["same"] * 20,
                "ts": pd.date_range("2024-01-01", periods=20, freq="D"),
            }
        )
        report = DatasetAuditor().audit_dataframe(frame)
        summary = {row["dtype"]: row for row in report.dtype_summary()}
        assert summary["numeric"]["column_count"] == 1
        assert summary["categorical"]["column_count"] == 2


class TestDtypeSummaryCLI:
    @staticmethod
    def _csv(tmp_path) -> str:
        frame = pd.DataFrame(
            {
                "id": range(20),
                "flag": ["y", "n"] * 10,
                "constant": ["same"] * 20,
            }
        )
        path = tmp_path / "dtype_summary.csv"
        frame.to_csv(path, index=False)
        return str(path)

    def test_table_output_names_dtypes(self, tmp_path, capsys) -> None:
        assert main(["dtype-summary", self._csv(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "numeric" in out
        assert "categorical" in out

    def test_json_output_is_parseable(self, tmp_path, capsys) -> None:
        assert main(["dtype-summary", self._csv(tmp_path), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert {row["dtype"] for row in payload} == {"numeric", "categorical"}

    def test_top_limits_groups(self, tmp_path, capsys) -> None:
        assert main(["dtype-summary", self._csv(tmp_path), "--top", "1", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload) == 1

    def test_invalid_top_exits_nonzero(self, tmp_path, capsys) -> None:
        assert main(["dtype-summary", self._csv(tmp_path), "--top", "0"]) == 1
        assert "--top must be a positive integer" in capsys.readouterr().out

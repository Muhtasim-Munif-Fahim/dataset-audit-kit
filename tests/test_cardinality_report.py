"""Tests for AuditReport.cardinality_report."""

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


class TestCardinalityLabels:
    def test_constant_column_is_labelled_constant(self) -> None:
        report = _report(a={"dtype": "categorical", "count": 10, "missing": 0, "unique": 1})
        assert report.cardinality_report()[0]["cardinality"] == "constant"

    def test_all_null_column_is_constant_and_does_not_divide_by_zero(self) -> None:
        report = _report(a={"dtype": "other", "count": 10, "missing": 10, "unique": 0})
        row = report.cardinality_report()[0]
        assert row["cardinality"] == "constant"
        assert row["unique_ratio"] == 0.0
        assert row["non_null"] == 0

    def test_two_distinct_values_is_binary(self) -> None:
        report = _report(a={"dtype": "categorical", "count": 10, "missing": 0, "unique": 2})
        assert report.cardinality_report()[0]["cardinality"] == "binary"

    def test_all_distinct_is_identifier_and_wins_over_high(self) -> None:
        report = _report(a={"dtype": "numeric", "count": 10, "missing": 0, "unique": 10})
        row = report.cardinality_report()[0]
        assert row["cardinality"] == "identifier"
        assert row["unique_ratio"] == 1.0

    def test_identifier_ignores_missing_values(self) -> None:
        # 6 distinct values across 6 non-null rows is still a key even though
        # the column has holes.
        report = _report(a={"dtype": "categorical", "count": 10, "missing": 4, "unique": 6})
        assert report.cardinality_report()[0]["cardinality"] == "identifier"

    @pytest.mark.parametrize(
        "unique, expected",
        [(3, "low"), (10, "medium"), (60, "high")],
    )
    def test_ratio_buckets(self, unique: int, expected: str) -> None:
        report = _report(
            a={"dtype": "categorical", "count": 100, "missing": 0, "unique": unique}
        )
        assert report.cardinality_report()[0]["cardinality"] == expected

    def test_bucket_boundaries_are_inclusive(self) -> None:
        low_edge = _report(
            a={"dtype": "categorical", "count": 100, "missing": 0, "unique": 5}
        )
        high_edge = _report(
            a={"dtype": "categorical", "count": 100, "missing": 0, "unique": 50}
        )
        assert low_edge.cardinality_report()[0]["cardinality"] == "medium"
        assert high_edge.cardinality_report()[0]["cardinality"] == "high"


class TestCardinalityReportShape:
    def test_row_carries_the_documented_keys(self) -> None:
        report = _report(a={"dtype": "numeric", "count": 8, "missing": 2, "unique": 3})
        assert report.cardinality_report()[0] == {
            "column": "a",
            "cardinality": "high",
            "unique": 3,
            "unique_ratio": 0.5,
            "non_null": 6,
            "total_rows": 8,
            "dtype": "numeric",
        }

    def test_sorted_by_ratio_then_column_name(self) -> None:
        report = _report(
            b={"dtype": "categorical", "count": 100, "missing": 0, "unique": 20},
            a={"dtype": "categorical", "count": 100, "missing": 0, "unique": 20},
            c={"dtype": "categorical", "count": 100, "missing": 0, "unique": 40},
        )
        assert [row["column"] for row in report.cardinality_report()] == ["c", "a", "b"]

    def test_top_truncates_after_sorting(self) -> None:
        report = _report(
            a={"dtype": "categorical", "count": 100, "missing": 0, "unique": 10},
            b={"dtype": "categorical", "count": 100, "missing": 0, "unique": 40},
        )
        assert [row["column"] for row in report.cardinality_report(top=1)] == ["b"]

    def test_columns_without_a_profile_are_absent(self) -> None:
        report = _report()
        assert report.cardinality_report() == []


class TestCardinalityReportFilters:
    def test_min_unique_ratio_drops_lower_columns(self) -> None:
        report = _report(
            a={"dtype": "categorical", "count": 100, "missing": 0, "unique": 2},
            b={"dtype": "categorical", "count": 100, "missing": 0, "unique": 90},
        )
        rows = report.cardinality_report(min_unique_ratio=0.5)
        assert [row["column"] for row in rows] == ["b"]

    def test_include_restricts_to_named_labels(self) -> None:
        report = _report(
            a={"dtype": "categorical", "count": 100, "missing": 0, "unique": 1},
            b={"dtype": "categorical", "count": 100, "missing": 0, "unique": 100},
            c={"dtype": "categorical", "count": 100, "missing": 0, "unique": 10},
        )
        rows = report.cardinality_report(include=("constant", "identifier"))
        assert sorted(row["column"] for row in rows) == ["a", "b"]

    def test_include_applies_before_min_unique_ratio(self) -> None:
        report = _report(
            a={"dtype": "categorical", "count": 100, "missing": 0, "unique": 1},
        )
        assert report.cardinality_report(include=("constant",), min_unique_ratio=0.5) == []


class TestCardinalityReportValidation:
    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3"])
    def test_top_must_be_a_positive_integer(self, bad: object) -> None:
        with pytest.raises(ValueError, match="top must be a positive integer"):
            _report().cardinality_report(top=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.1, True, "0.5"])
    def test_min_unique_ratio_must_be_a_fraction(self, bad: object) -> None:
        with pytest.raises(ValueError, match="min_unique_ratio must be a number"):
            _report().cardinality_report(min_unique_ratio=bad)

    def test_empty_include_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one cardinality label"):
            _report().cardinality_report(include=())

    def test_unknown_include_label_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown cardinality label\\(s\\): huge"):
            _report().cardinality_report(include=("huge", "low"))


class TestCardinalityReportEndToEnd:
    def test_labels_a_real_audited_frame(self) -> None:
        frame = pd.DataFrame(
            {
                "id": range(20),
                "flag": ["y", "n"] * 10,
                "constant": ["same"] * 20,
                "bucket": [f"g{i % 4}" for i in range(20)],
            }
        )
        report = DatasetAuditor().audit_dataframe(frame)
        labels = {row["column"]: row["cardinality"] for row in report.cardinality_report()}
        assert labels["id"] == "identifier"
        assert labels["flag"] == "binary"
        assert labels["constant"] == "constant"
        assert labels["bucket"] == "medium"


class TestCardinalityCli:
    @staticmethod
    def _csv(tmp_path) -> str:
        frame = pd.DataFrame(
            {
                "id": range(20),
                "flag": ["y", "n"] * 10,
                "constant": ["same"] * 20,
                "bucket": [f"g{i % 4}" for i in range(20)],
            }
        )
        path = tmp_path / "cardinality.csv"
        frame.to_csv(path, index=False)
        return str(path)

    def test_table_output_names_droppable_columns(self, tmp_path, capsys) -> None:
        assert main(["cardinality", self._csv(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "identifier" in out
        assert "constant or identifier-like" in out

    def test_json_output_is_parseable(self, tmp_path, capsys) -> None:
        assert main(["cardinality", self._csv(tmp_path), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert {row["column"] for row in payload} == {"id", "flag", "constant", "bucket"}

    def test_include_filter_narrows_the_table(self, tmp_path, capsys) -> None:
        assert main(["cardinality", self._csv(tmp_path), "--include", "binary", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert [row["column"] for row in payload] == ["flag"]

    def test_no_match_reports_cleanly(self, tmp_path, capsys) -> None:
        assert main(["cardinality", self._csv(tmp_path), "--min-unique-ratio", "1.0",
                     "--include", "binary"]) == 0
        assert "(no columns matched)" in capsys.readouterr().out

    def test_invalid_top_exits_nonzero(self, tmp_path, capsys) -> None:
        assert main(["cardinality", self._csv(tmp_path), "--top", "0"]) == 1
        assert "--top must be a positive integer" in capsys.readouterr().out

    def test_invalid_ratio_exits_nonzero(self, tmp_path, capsys) -> None:
        assert main(["cardinality", self._csv(tmp_path), "--min-unique-ratio", "2"]) == 1
        assert "--min-unique-ratio must be between 0 and 1" in capsys.readouterr().out

"""Tests for the profile-diff subcommand."""

from __future__ import annotations

import json
import textwrap

import pandas as pd
import pytest

from dataset_audit_kit.cli import main


@pytest.fixture()
def two_reports(tmp_path):
    """Create two saved audit reports with shared and unique columns."""
    first_csv = tmp_path / "before.csv"
    second_csv = tmp_path / "after.csv"
    first_csv.write_text(
        "x,y,label\n1.0,A,0\n2.0,B,1\n3.0,A,0\n,,1\n",
        encoding="utf-8",
    )
    second_csv.write_text(
        "x,y,label\n1.5,A,0\n2.5,C,1\n3.5,A,0\n4.5,B,1\n",
        encoding="utf-8",
    )
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    assert main(["audit", str(first_csv), "--save-json", str(before)]) in (0, 1)
    assert main(["audit", str(second_csv), "--save-json", str(after)]) in (0, 1)
    return before, after


class TestProfileDiff:
    def test_table_output_lists_shared_columns(self, two_reports, capsys):
        before, after = two_reports
        assert main(["profile-diff", str(before), str(after)]) == 0
        out = capsys.readouterr().out
        for name in ("x", "y", "mean_delta"):
            assert name in out

    def test_json_output_is_a_json_array(self, two_reports, capsys):
        before, after = two_reports
        assert main(["profile-diff", str(before), str(after), "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, list)
        assert all(isinstance(row, dict) for row in data)

    def test_csv_output_has_header_and_rows(self, two_reports, capsys):
        before, after = two_reports
        assert main(["profile-diff", str(before), str(after), "--csv"]) == 0
        out = capsys.readouterr().out
        lines = out.strip().splitlines()
        assert len(lines) >= 2
        assert "column" in lines[0]

    def test_columns_argument_restricts_output(self, two_reports, capsys):
        before, after = two_reports
        assert main(["profile-diff", str(before), str(after), "--columns", "x", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        column_names = [row["column"] for row in data]
        assert column_names == ["x"]

    def test_numeric_only_omits_categorical(self, two_reports, capsys):
        before, after = two_reports
        assert main(["profile-diff", str(before), str(after), "--numeric-only", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        # Categorical columns like 'y' should not appear when --numeric-only is set
        column_names = [row["column"] for row in data]
        assert "y" not in column_names

    def test_missing_baseline_file_returns_2(self, tmp_path, capsys):
        assert main(["profile-diff", str(tmp_path / "nope.json"), str(tmp_path / "nope.json")]) == 2

    def test_bad_json_baseline_returns_2(self, tmp_path, capsys):
        bad = tmp_path / "bad.json"
        bad.write_text("not json", encoding="utf-8")
        other = tmp_path / "other.json"
        other.write_text("{}", encoding="utf-8")
        assert main(["profile-diff", str(bad), str(other)]) == 2

    def test_no_shared_columns_prints_message(self, tmp_path, capsys):
        first = tmp_path / "a.json"
        second = tmp_path / "b.json"
        first.write_text(json.dumps({
            "rows": 1, "columns": 1, "duplicate_rows": 0, "missing_cells": 0,
            "column_profiles": {"only_a": {"dtype": "numeric", "count": 1}},
        }), encoding="utf-8")
        second.write_text(json.dumps({
            "rows": 1, "columns": 1, "duplicate_rows": 0, "missing_cells": 0,
            "column_profiles": {"only_b": {"dtype": "numeric", "count": 1}},
        }), encoding="utf-8")
        assert main(["profile-diff", str(first), str(second)]) == 0
        out = capsys.readouterr().out
        assert "No shared columns" in out

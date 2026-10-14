"""Tests for Cramér's V and categorical_association_report / associate CLI."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from dataset_audit_kit.cli import main
from dataset_audit_kit.core import DatasetAuditor


def _perfect_copy(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    city = rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2])
    # Exact copy of city under a different name.
    return pd.DataFrame({"city": city, "city_copy": city.copy(), "noise": rng.integers(0, 5, n)})


def _independent(n: int = 2000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.choice(["a", "b", "c"], n),
            "y": rng.choice(["p", "q", "r"], n),
        }
    )


class TestCramersV:
    def test_identical_series_is_one(self) -> None:
        s = pd.Series(["a", "b", "a", "c", "b", "a"] * 10)
        assert DatasetAuditor.cramers_v(s, s) == pytest.approx(1.0, abs=1e-9)

    def test_independent_is_near_zero(self) -> None:
        frame = _independent()
        v = DatasetAuditor.cramers_v(frame["x"], frame["y"])
        assert v == pytest.approx(0.0, abs=0.05)

    def test_single_level_is_zero(self) -> None:
        left = pd.Series(["a"] * 20)
        right = pd.Series(["x", "y"] * 10)
        assert DatasetAuditor.cramers_v(left, right) == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            DatasetAuditor.cramers_v(pd.Series([1, 2]), pd.Series([1]))

    def test_all_missing_is_nan(self) -> None:
        left = pd.Series([None, None, None])
        right = pd.Series([None, "a", None])
        assert math.isnan(DatasetAuditor.cramers_v(left, right))

    def test_bias_correction_reduces_small_table(self) -> None:
        # Tiny table where uncorrected V is inflated.
        left = pd.Series(["a", "a", "b", "b", "c"])
        right = pd.Series(["x", "x", "y", "y", "z"])
        raw = DatasetAuditor.cramers_v(left, right, bias_correction=False)
        corr = DatasetAuditor.cramers_v(left, right, bias_correction=True)
        assert corr <= raw + 1e-12


class TestCategoricalAssociationReport:
    def test_perfect_copy_is_flagged_and_sorted_first(self) -> None:
        frame = _perfect_copy()
        rows = DatasetAuditor().categorical_association_report(frame, threshold=0.8)
        assert rows[0]["column_a"] == "city"
        assert rows[0]["column_b"] == "city_copy"
        assert rows[0]["cramers_v"] == pytest.approx(1.0, abs=1e-9)
        assert rows[0]["associated"] is True

    def test_min_v_filters(self) -> None:
        frame = _independent()
        rows = DatasetAuditor().categorical_association_report(frame, min_v=0.5)
        assert rows == []

    def test_identifier_columns_skipped(self) -> None:
        frame = pd.DataFrame(
            {
                "id": [f"u{i}" for i in range(30)],
                "group": ["a", "b"] * 15,
            }
        )
        rows = DatasetAuditor().categorical_association_report(frame)
        assert rows == []

    def test_unknown_column_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown columns"):
            DatasetAuditor().categorical_association_report(
                _perfect_copy(), columns=["missing"]
            )

    def test_rejects_bad_top(self) -> None:
        with pytest.raises(ValueError, match="top"):
            DatasetAuditor().categorical_association_report(_perfect_copy(), top=0)

    def test_numeric_codes_are_eligible(self) -> None:
        frame = pd.DataFrame(
            {
                "code": [0, 1, 0, 1, 0, 1] * 20,
                "label": ["n", "y", "n", "y", "n", "y"] * 20,
            }
        )
        rows = DatasetAuditor().categorical_association_report(frame)
        assert len(rows) == 1
        assert rows[0]["cramers_v"] == pytest.approx(1.0, abs=1e-9)


class TestAssociateCLI:
    def test_json_lists_perfect_pair(self, tmp_path) -> None:
        path = tmp_path / "data.csv"
        _perfect_copy().to_csv(path, index=False)
        rc = main(["associate", str(path), "--json", "--top", "5"])
        assert rc == 0
        # main prints to stdout; capture via subprocess in a leaner way:
        # re-run through the report API already covered; here just exercise CLI path.
        # Use the library call for assertions; CLI smoke is exit code + no crash.

    def test_fail_on_associated(self, tmp_path, capsys) -> None:
        path = tmp_path / "data.csv"
        _perfect_copy().to_csv(path, index=False)
        rc = main(["associate", str(path), "--fail-on-associated", "--threshold", "0.8"])
        assert rc == 1
        out = capsys.readouterr().out
        assert "city" in out
        assert "city_copy" in out

    def test_json_output(self, tmp_path, capsys) -> None:
        path = tmp_path / "data.csv"
        _perfect_copy().to_csv(path, index=False)
        rc = main(["associate", str(path), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload[0]["column_a"] == "city"
        assert payload[0]["associated"] is True

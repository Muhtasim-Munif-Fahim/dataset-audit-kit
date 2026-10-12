"""Tests for DatasetAuditor.distribution_distance_report and the distance CLI."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance as scipy_wasserstein

from dataset_audit_kit.cli import main
from dataset_audit_kit.core import DatasetAuditor


def _frames(shift: float = 0.0, n: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    reference = pd.DataFrame(
        {
            "x": rng.normal(0.0, 1.0, n),
            "city": rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2]),
        }
    )
    current = pd.DataFrame(
        {
            "x": rng.normal(shift, 1.0, n),
            "city": rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2]),
        }
    )
    return current, reference


class TestWasserstein:
    def test_matches_scipy(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.exponential(2.0, 137)
        b = rng.normal(1.0, 3.0, 211)
        assert DatasetAuditor.wasserstein_distance(a, b) == pytest.approx(
            scipy_wasserstein(a, b), rel=1e-12
        )

    def test_pure_shift_equals_shift(self) -> None:
        a = np.arange(50, dtype=float)
        assert DatasetAuditor.wasserstein_distance(a, a + 2.5) == pytest.approx(2.5)

    def test_identical_is_zero_and_symmetric(self) -> None:
        a = np.array([1.0, 2.0, 2.0, 7.0])
        b = np.array([0.0, 5.0])
        assert DatasetAuditor.wasserstein_distance(a, a) == 0.0
        assert DatasetAuditor.wasserstein_distance(a, b) == pytest.approx(
            DatasetAuditor.wasserstein_distance(b, a)
        )

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            DatasetAuditor.wasserstein_distance([], [1.0])


class TestDistributionDistanceReport:
    def test_identical_frames_have_zero_distance(self) -> None:
        current, _ = _frames()
        rows = DatasetAuditor().distribution_distance_report(current, current.copy())
        assert {r["column"] for r in rows} == {"x", "city"}
        for row in rows:
            assert row["js_distance"] == pytest.approx(0.0, abs=1e-12)
            assert row["hellinger"] == pytest.approx(0.0, abs=1e-7)
            assert row["total_variation"] == pytest.approx(0.0, abs=1e-12)
            assert row["drifted"] is False

    def test_shifted_numeric_column_is_flagged_and_sorted_first(self) -> None:
        current, reference = _frames(shift=1.5)
        rows = DatasetAuditor().distribution_distance_report(current, reference)
        assert rows[0]["column"] == "x"
        assert rows[0]["kind"] == "numeric"
        assert rows[0]["drifted"] is True
        # W1 between N(0,1) and N(1.5,1) is the shift itself.
        assert rows[0]["wasserstein"] == pytest.approx(1.5, abs=0.1)
        assert rows[0]["wasserstein_scaled"] == pytest.approx(1.5, abs=0.15)
        city = next(r for r in rows if r["column"] == "city")
        assert city["drifted"] is False
        assert city["wasserstein"] is None

    def test_categorical_matches_closed_forms(self) -> None:
        reference = pd.DataFrame({"c": ["a"] * 50 + ["b"] * 50})
        current = pd.DataFrame({"c": ["a"] * 80 + ["b"] * 10 + ["z"] * 10})
        row = DatasetAuditor().distribution_distance_report(current, reference)[0]
        p = np.array([0.5, 0.5, 0.0])
        q = np.array([0.8, 0.1, 0.1])
        assert row["js_distance"] == pytest.approx(jensenshannon(p, q, base=2))
        expected_h = math.sqrt(1.0 - np.sum(np.sqrt(p * q)))
        assert row["hellinger"] == pytest.approx(expected_h)
        assert row["total_variation"] == pytest.approx(0.4)

    def test_disjoint_support_hits_upper_bounds(self) -> None:
        reference = pd.DataFrame({"c": ["a", "b"] * 20})
        current = pd.DataFrame({"c": ["x", "y"] * 20})
        row = DatasetAuditor().distribution_distance_report(current, reference)[0]
        assert row["js_distance"] == pytest.approx(1.0)
        assert row["hellinger"] == pytest.approx(1.0)
        assert row["total_variation"] == pytest.approx(1.0)

    def test_out_of_range_values_fall_into_edge_buckets(self) -> None:
        reference = pd.DataFrame({"x": np.linspace(0.0, 1.0, 100)})
        current = pd.DataFrame({"x": np.full(100, 50.0)})
        row = DatasetAuditor().distribution_distance_report(
            current, reference, bins=4
        )[0]
        # All current mass lands in the top bucket, which holds 1/4 of the
        # reference: TV = 1 - 1/4.
        assert row["total_variation"] == pytest.approx(0.75)
        assert row["wasserstein"] == pytest.approx(49.5)

    def test_columns_and_threshold(self) -> None:
        current, reference = _frames(shift=0.3)
        rows = DatasetAuditor().distribution_distance_report(
            current, reference, columns=["x"], js_threshold=1.0
        )
        assert [r["column"] for r in rows] == ["x"]
        assert rows[0]["drifted"] is False

    def test_validation(self) -> None:
        current, reference = _frames()
        auditor = DatasetAuditor()
        with pytest.raises(ValueError, match="top"):
            auditor.distribution_distance_report(current, reference, top=0)
        with pytest.raises(ValueError, match="bins"):
            auditor.distribution_distance_report(current, reference, bins=1)
        with pytest.raises(ValueError, match="js_threshold"):
            auditor.distribution_distance_report(current, reference, js_threshold=2)
        with pytest.raises(ValueError, match="both frames"):
            auditor.distribution_distance_report(
                current, reference.drop(columns="city"), columns=["city"]
            )


class TestDistanceCli:
    def test_json_and_fail_on_drift(self, tmp_path, capsys) -> None:
        current, reference = _frames(shift=2.0, n=400)
        cur_path = tmp_path / "cur.csv"
        ref_path = tmp_path / "ref.csv"
        current.to_csv(cur_path, index=False)
        reference.to_csv(ref_path, index=False)

        code = main(
            ["distance", str(cur_path), "--reference", str(ref_path), "--json", "--fail-on-drift"]
        )
        rows = json.loads(capsys.readouterr().out)
        assert code == 1
        assert rows[0]["column"] == "x" and rows[0]["drifted"] is True

        code = main(["distance", str(cur_path), "--reference", str(ref_path), "--columns", "city"])
        out = capsys.readouterr().out
        assert code == 0
        assert "city" in out and "hellinger" in out

    def test_missing_reference_returns_2(self, tmp_path, capsys) -> None:
        cur_path = tmp_path / "cur.csv"
        pd.DataFrame({"x": [1, 2, 3]}).to_csv(cur_path, index=False)
        code = main(["distance", str(cur_path), "--reference", str(tmp_path / "nope.csv")])
        assert code == 2

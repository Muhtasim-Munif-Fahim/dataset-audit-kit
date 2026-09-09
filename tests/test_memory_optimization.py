"""Tests for dtype-narrowing memory optimization."""

from __future__ import annotations

import json
from unittest import mock

import pandas as pd
import pytest

from dataset_audit_kit.cli import main
from dataset_audit_kit.core import DatasetAuditor


@pytest.fixture
def mixed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "small": [1, 2, 3] * 200,
            "wide": [10**12] * 600,
            "cat": ["aa", "bb", "cc"] * 200,
            "ratio": [1.25, 2.5, 3.75] * 200,
            "holes": [1.0, None, 3.0] * 200,
        }
    )


class TestInferOptimalDtypes:
    def test_values_wider_than_uint32_stay_64_bit(self) -> None:
        # int32 tops out near 2.1e9, so narrowing 1e12 would wrap it.
        frame = pd.DataFrame({"wide": [10**12] * 4})
        assert DatasetAuditor.infer_optimal_dtypes(frame) == {}

    def test_values_inside_uint32_still_narrow(self) -> None:
        frame = pd.DataFrame({"mid": [4_000_000_000] * 4})
        suggestion = DatasetAuditor.infer_optimal_dtypes(frame)
        assert suggestion["mid"]["suggested_dtype"] == "uint32"


class TestApplyOptimalDtypes:
    def test_narrows_without_changing_values(self, mixed_frame) -> None:
        converted, skipped = DatasetAuditor.apply_optimal_dtypes(mixed_frame)
        assert skipped == {}
        assert str(converted["small"].dtype) == "uint8"
        assert str(converted["cat"].dtype) == "category"
        assert converted["small"].tolist() == mixed_frame["small"].tolist()
        assert converted["wide"].tolist() == mixed_frame["wide"].tolist()

    def test_integer_target_with_nulls_uses_the_nullable_dtype(self, mixed_frame) -> None:
        converted, skipped = DatasetAuditor.apply_optimal_dtypes(mixed_frame)
        assert str(converted["holes"].dtype) == "Int32"
        assert "holes" not in skipped
        assert converted["holes"].isna().sum() == 200

    def test_the_original_frame_is_not_mutated(self, mixed_frame) -> None:
        before = {c: str(mixed_frame[c].dtype) for c in mixed_frame.columns}
        DatasetAuditor.apply_optimal_dtypes(mixed_frame)
        assert {c: str(mixed_frame[c].dtype) for c in mixed_frame.columns} == before

    def test_a_lossy_integer_cast_is_refused_not_wrapped(self) -> None:
        frame = pd.DataFrame({"wide": [10**12] * 4})
        bad = {"wide": {"current_dtype": "int64", "suggested_dtype": "int32"}}
        with mock.patch.object(
            DatasetAuditor, "infer_optimal_dtypes", staticmethod(lambda _: bad)
        ):
            converted, skipped = DatasetAuditor.apply_optimal_dtypes(frame)
        assert converted["wide"].tolist() == [10**12] * 4
        assert "round-trip" in skipped["wide"]

    def test_float_narrowing_is_allowed_despite_precision_loss(self) -> None:
        frame = pd.DataFrame({"f": [0.1, 0.2, 0.3] * 4})
        converted, skipped = DatasetAuditor.apply_optimal_dtypes(frame)
        assert str(converted["f"].dtype) == "float32"
        assert skipped == {}

    def test_a_failing_cast_keeps_the_original_column(self) -> None:
        frame = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        bad = {"a": {"current_dtype": "int64", "suggested_dtype": "not-a-dtype"}}
        with mock.patch.object(
            DatasetAuditor, "infer_optimal_dtypes", staticmethod(lambda _: bad)
        ):
            converted, skipped = DatasetAuditor.apply_optimal_dtypes(frame)
        assert converted["a"].tolist() == [1, 2, 3]
        assert converted["b"].tolist() == [4, 5, 6]
        assert "a" in skipped


class TestMemoryOptimizationReport:
    def test_totals_add_up(self, mixed_frame) -> None:
        report = DatasetAuditor.memory_optimization_report(mixed_frame)
        assert report["saved_bytes"] == report["current_bytes"] - report["optimized_bytes"]
        assert report["saved_bytes"] > 0
        assert 0.0 < report["saved_ratio"] <= 1.0

    def test_rows_sorted_by_bytes_saved(self, mixed_frame) -> None:
        rows = DatasetAuditor.memory_optimization_report(mixed_frame)["columns"]
        saved = [int(row["saved_bytes"]) for row in rows]
        assert saved == sorted(saved, reverse=True)

    def test_row_carries_the_documented_keys(self, mixed_frame) -> None:
        row = DatasetAuditor.memory_optimization_report(mixed_frame)["columns"][0]
        assert set(row) == {
            "column",
            "current_dtype",
            "optimized_dtype",
            "current_bytes",
            "optimized_bytes",
            "saved_bytes",
            "saved_ratio",
        }

    def test_min_saved_bytes_filters_rows_but_not_totals(self, mixed_frame) -> None:
        full = DatasetAuditor.memory_optimization_report(mixed_frame)
        filtered = DatasetAuditor.memory_optimization_report(
            mixed_frame, min_saved_bytes=10**9
        )
        assert filtered["columns"] == []
        assert filtered["saved_bytes"] == full["saved_bytes"]
        assert filtered["current_bytes"] == full["current_bytes"]

    def test_a_frame_with_nothing_to_gain_reports_zero(self) -> None:
        frame = pd.DataFrame({"a": pd.Series([1, 2, 3], dtype="uint8")})
        report = DatasetAuditor.memory_optimization_report(frame)
        assert report["saved_bytes"] == 0
        assert report["saved_ratio"] == 0.0

    def test_an_empty_frame_does_not_divide_by_zero(self) -> None:
        report = DatasetAuditor.memory_optimization_report(pd.DataFrame())
        assert report["saved_ratio"] == 0.0
        assert report["columns"] == []

    @pytest.mark.parametrize("bad", [-1, 1.5, True, "10"])
    def test_min_saved_bytes_validation(self, bad: object) -> None:
        with pytest.raises(ValueError, match="min_saved_bytes must be a non-negative integer"):
            DatasetAuditor.memory_optimization_report(pd.DataFrame(), min_saved_bytes=bad)


class TestOptimizeCli:
    @staticmethod
    def _csv(tmp_path, frame) -> str:
        path = tmp_path / "opt.csv"
        frame.to_csv(path, index=False)
        return str(path)

    def test_table_output_reports_a_saving(self, tmp_path, mixed_frame, capsys) -> None:
        assert main(["optimize", self._csv(tmp_path, mixed_frame)]) == 0
        out = capsys.readouterr().out
        assert "saved," in out
        assert "->" in out

    def test_json_output_is_parseable(self, tmp_path, mixed_frame, capsys) -> None:
        assert main(["optimize", self._csv(tmp_path, mixed_frame), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["saved_bytes"] == payload["current_bytes"] - payload["optimized_bytes"]

    def test_out_writes_a_readable_dataset_with_the_same_values(
        self, tmp_path, mixed_frame, capsys
    ) -> None:
        source = self._csv(tmp_path, mixed_frame)
        destination = tmp_path / "optimized.csv"
        assert main(["optimize", source, "--out", str(destination)]) == 0
        assert "Wrote optimized dataset" in capsys.readouterr().out
        written = pd.read_csv(destination)
        assert written["wide"].tolist() == mixed_frame["wide"].tolist()
        assert written["small"].tolist() == mixed_frame["small"].tolist()

    def test_threshold_above_every_column_reports_cleanly(
        self, tmp_path, mixed_frame, capsys
    ) -> None:
        assert main(
            ["optimize", self._csv(tmp_path, mixed_frame), "--min-saved-bytes", "999999999"]
        ) == 0
        assert "(no columns above the saving threshold)" in capsys.readouterr().out

    def test_negative_threshold_exits_nonzero(self, tmp_path, mixed_frame, capsys) -> None:
        assert main(["optimize", self._csv(tmp_path, mixed_frame), "--min-saved-bytes", "-1"]) == 1
        assert "--min-saved-bytes must be a non-negative integer" in capsys.readouterr().out

"""Tests for functional_dependency_g3 / functional_dependency_report / fd CLI."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from dataset_audit_kit.cli import main
from dataset_audit_kit.core import DatasetAuditor


def _geo(n: int = 400, seed: int = 0, typos: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    zips = rng.choice([f"z{i}" for i in range(20)], n)
    city_of = {f"z{i}": f"city{i // 4}" for i in range(20)}
    city = np.array([city_of[z] for z in zips], dtype=object)
    if typos:
        idx = rng.choice(n, typos, replace=False)
        city[idx] = "TYPO"
    city_code = np.array([c.upper() for c in city], dtype=object)
    return pd.DataFrame(
        {
            "row_id": np.arange(n),
            "zip": zips,
            "city": city,
            "city_code": city_code,
            "noise": rng.integers(0, 3, n),
        }
    )


class TestG3:
    def test_exact_dependency_is_zero(self) -> None:
        frame = _geo()
        g3, violations, n = DatasetAuditor.functional_dependency_g3(frame["zip"], frame["city"])
        assert (g3, violations, n) == (0.0, 0, 400)

    def test_known_violation_count(self) -> None:
        a = pd.Series(["x", "x", "x", "y", "y"])
        b = pd.Series([1, 1, 2, 3, 3])
        g3, violations, n = DatasetAuditor.functional_dependency_g3(a, b)
        assert violations == 1 and n == 5
        assert g3 == pytest.approx(0.2)

    def test_missing_rows_ignored_and_all_missing_nan(self) -> None:
        a = pd.Series(["x", None, "x"])
        b = pd.Series([1, 2, 1])
        assert DatasetAuditor.functional_dependency_g3(a, b) == (0.0, 0, 2)
        g3, _, n = DatasetAuditor.functional_dependency_g3(pd.Series([None]), pd.Series([1]))
        assert math.isnan(g3) and n == 0

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            DatasetAuditor.functional_dependency_g3(pd.Series([1, 2]), pd.Series([1]))


class TestReport:
    def test_finds_zip_to_city_and_bijective_copy(self) -> None:
        rows = DatasetAuditor().functional_dependency_report(_geo())
        pairs = {(r["determinant"], r["dependent"]): r for r in rows}
        assert ("zip", "city") in pairs and pairs[("zip", "city")]["exact"]
        assert not pairs[("zip", "city")]["bijective"]
        assert ("city", "zip") not in pairs  # many zips per city
        assert pairs[("city", "city_code")]["bijective"]
        assert pairs[("city_code", "city")]["bijective"]
        # identifier-like row_id is never a determinant
        assert all(r["determinant"] != "row_id" for r in rows)
        assert all(r["dependency"] for r in rows)

    def test_approximate_dependency_with_typos(self) -> None:
        frame = _geo(typos=2)
        strict = DatasetAuditor().functional_dependency_report(frame, max_g3=0.0)
        assert ("zip", "city") not in {(r["determinant"], r["dependent"]) for r in strict}
        loose = DatasetAuditor().functional_dependency_report(frame, max_g3=0.01)
        row = next(r for r in loose if (r["determinant"], r["dependent"]) == ("zip", "city"))
        assert not row["exact"]
        assert 0 < row["violating_rows"] <= 2

    def test_include_all_and_ordering(self) -> None:
        rows = DatasetAuditor().functional_dependency_report(
            _geo(), include_all=True, top=100
        )
        flags = [r["dependency"] for r in rows]
        assert flags == sorted(flags, reverse=True)
        assert any(not f for f in flags)

    def test_columns_subset_and_validation(self) -> None:
        frame = _geo()
        rows = DatasetAuditor().functional_dependency_report(frame, columns=["zip", "noise"])
        assert rows == []
        with pytest.raises(ValueError, match="unknown columns"):
            DatasetAuditor().functional_dependency_report(frame, columns=["nope"])
        with pytest.raises(ValueError, match="max_g3"):
            DatasetAuditor().functional_dependency_report(frame, max_g3=1.5)
        with pytest.raises(ValueError, match="top"):
            DatasetAuditor().functional_dependency_report(frame, top=0)


class TestCli:
    def test_json_output(self, tmp_path, capsys) -> None:
        path = tmp_path / "geo.csv"
        _geo().to_csv(path, index=False)
        rc = main(["fd", str(path), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert any(r["determinant"] == "zip" and r["dependent"] == "city" for r in payload)

    def test_table_and_fail_flag(self, tmp_path, capsys) -> None:
        path = tmp_path / "geo.csv"
        _geo().to_csv(path, index=False)
        rc = main(["fd", str(path), "--fail-on-dependency"])
        assert rc == 1
        out = capsys.readouterr().out
        assert "zip -> city" in out

    def test_no_dependencies_message(self, tmp_path, capsys) -> None:
        rng = np.random.default_rng(3)
        path = tmp_path / "rand.csv"
        pd.DataFrame({"a": rng.integers(0, 3, 300), "b": rng.integers(0, 3, 300)}).to_csv(path, index=False)
        assert main(["fd", str(path)]) == 0
        assert "No functional dependencies found." in capsys.readouterr().out

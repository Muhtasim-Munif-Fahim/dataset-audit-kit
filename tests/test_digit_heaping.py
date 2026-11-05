"""Tests for DatasetAuditor.digit_heaping_report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataset_audit_kit.core import DatasetAuditor


def _frame(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    honest_age = rng.integers(20, 90, size=n)  # 70 values: every digit equally likely
    heaped_age = honest_age.copy()
    mask = rng.random(n) < 0.5
    heaped_age[mask] = (np.round(heaped_age[mask] / 5) * 5).astype(int)
    weight = np.round(rng.normal(70, 12, size=n), 1)  # one decimal, honest
    continuous = rng.normal(size=n)  # unrounded floats
    return pd.DataFrame(
        {"honest_age": honest_age, "heaped_age": heaped_age, "weight": weight, "z": continuous}
    )


def test_flags_heaped_column_and_not_honest_ones() -> None:
    rows = DatasetAuditor().digit_heaping_report(_frame())
    by = {r["column"]: r for r in rows}
    assert set(by) == {"honest_age", "heaped_age", "weight"}  # z skipped (no resolution)
    assert rows[0]["column"] == "heaped_age"
    assert by["heaped_age"]["suspect"] is True
    assert by["heaped_age"]["whipple_index"] > 2.5
    assert by["heaped_age"]["preferred_digit"] in (0, 5)
    assert by["honest_age"]["suspect"] is False
    assert by["honest_age"]["whipple_index"] == pytest.approx(1.0, abs=0.2)
    assert by["weight"]["decimals"] == 1
    assert by["weight"]["p_value"] > 0.001
    assert by["weight"]["whipple_index"] == pytest.approx(1.0, abs=0.2)
    assert sum(by["weight"]["digit_shares"].values()) == pytest.approx(1.0)


def test_explicit_decimals_and_columns() -> None:
    df = _frame()
    rows = DatasetAuditor().digit_heaping_report(df, columns=["weight"], decimals=0)
    assert len(rows) == 1 and rows[0]["decimals"] == 0
    with pytest.raises(ValueError, match="unknown columns"):
        DatasetAuditor().digit_heaping_report(df, columns=["nope"])


def test_negative_values_use_absolute_digits() -> None:
    rng = np.random.default_rng(3)
    vals = rng.integers(-500, 500, size=2000) * 10  # every value ends in 0
    rows = DatasetAuditor().digit_heaping_report(pd.DataFrame({"v": vals}))
    assert rows[0]["preferred_digit"] == 0
    assert rows[0]["preferred_share"] == pytest.approx(1.0)
    assert rows[0]["whipple_index"] == pytest.approx(5.0)


def test_skips_small_and_narrow_columns() -> None:
    df = pd.DataFrame({"few": [1.0] * 20 + [np.nan] * 180, "narrow": np.tile(np.arange(5), 40), "flag": [True, False] * 100})
    rows = DatasetAuditor().digit_heaping_report(df, min_samples=50)
    assert rows == []


def test_top_ordering_and_validation() -> None:
    df = _frame()
    assert len(DatasetAuditor().digit_heaping_report(df, top=1)) == 1
    auditor = DatasetAuditor()
    with pytest.raises(ValueError, match="top"):
        auditor.digit_heaping_report(df, top=0)
    with pytest.raises(ValueError, match="min_samples"):
        auditor.digit_heaping_report(df, min_samples=5)
    with pytest.raises(ValueError, match="decimals"):
        auditor.digit_heaping_report(df, decimals=9)
    with pytest.raises(ValueError, match="min_range_units"):
        auditor.digit_heaping_report(df, min_range_units=0)
    with pytest.raises(ValueError, match="alpha"):
        auditor.digit_heaping_report(df, alpha=1.5)

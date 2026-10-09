"""Tests for DatasetAuditor.benford_law_report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataset_audit_kit.core import BENFORD_EXPECTED, DatasetAuditor


def _benford_samples(n: int = 2000, seed: int = 0) -> np.ndarray:
    """Draw magnitudes whose leading digits follow Benford's law."""
    rng = np.random.default_rng(seed)
    # Uniform in log10 space over several orders of magnitude.
    log_v = rng.uniform(0.0, 6.0, size=n)
    return np.power(10.0, log_v)


def _uniform_samples(n: int = 2000, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(100.0, 999.0, size=n)


class TestBenfordLawReport:
    def test_benford_like_column_not_suspect(self) -> None:
        data = pd.DataFrame({"good": _benford_samples()})
        rows = DatasetAuditor().benford_law_report(data, min_samples=50)
        assert len(rows) == 1
        assert rows[0]["column"] == "good"
        assert rows[0]["p_value"] > 0.05
        assert rows[0]["suspect"] is False

    def test_uniform_magnitudes_are_suspect(self) -> None:
        data = pd.DataFrame({"bad": _uniform_samples(n=3000)})
        rows = DatasetAuditor().benford_law_report(data, min_samples=50)
        assert len(rows) == 1
        assert rows[0]["suspect"] is True
        assert rows[0]["p_value"] < 0.05

    def test_skips_short_columns(self) -> None:
        data = pd.DataFrame({"tiny": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        rows = DatasetAuditor().benford_law_report(data, min_samples=50)
        assert rows == []

    def test_expected_probs_sum_to_one(self) -> None:
        assert sum(BENFORD_EXPECTED) == pytest.approx(1.0)

    def test_digit_shares_present(self) -> None:
        data = pd.DataFrame({"good": _benford_samples(n=500)})
        row = DatasetAuditor().benford_law_report(data, min_samples=50)[0]
        assert set(row["digit_shares"]) == {str(d) for d in range(1, 10)}
        assert sum(row["digit_shares"].values()) == pytest.approx(1.0)

    def test_top_validation(self) -> None:
        data = pd.DataFrame({"good": _benford_samples()})
        with pytest.raises(ValueError, match="top"):
            DatasetAuditor().benford_law_report(data, top=0)

    def test_column_subset(self) -> None:
        data = pd.DataFrame(
            {
                "good": _benford_samples(),
                "bad": _uniform_samples(),
                "label": ["a"] * 2000,
            }
        )
        rows = DatasetAuditor().benford_law_report(
            data, columns=["bad"], min_samples=50
        )
        assert len(rows) == 1
        assert rows[0]["column"] == "bad"

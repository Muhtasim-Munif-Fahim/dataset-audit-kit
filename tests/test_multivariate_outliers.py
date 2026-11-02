"""Tests for FastMCD robust Mahalanobis outliers (minimum_covariance_determinant / multivariate_outlier_report / mv-outliers CLI)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dataset_audit_kit.cli import main
from dataset_audit_kit.core import DatasetAuditor

COV = np.array([[1.0, 0.9], [0.9, 1.0]])


def _contaminated(n: int = 300, n_out: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    clean = rng.multivariate_normal([0.0, 0.0], COV, size=n)
    # Outliers break the correlation: each coordinate is modest on its own.
    out = rng.multivariate_normal([2.0, -2.0], 0.05 * np.eye(2), size=n_out)
    X = np.vstack([clean, out])
    return pd.DataFrame(X, columns=["a", "b"])


class TestChiSquareQuantile:
    def test_known_quantiles(self) -> None:
        q = DatasetAuditor.chi_square_quantile
        assert q(0.95, 1) == pytest.approx(3.841458820694124, rel=1e-8)
        assert q(0.975, 2) == pytest.approx(7.377758908227871, rel=1e-8)
        assert q(0.5, 10) == pytest.approx(9.341817765591966, rel=1e-8)

    def test_round_trip(self) -> None:
        for df in (1, 3, 7.5):
            x = DatasetAuditor.chi_square_quantile(0.3, df)
            assert 1 - DatasetAuditor.chi_square_sf(x, df) == pytest.approx(0.3, abs=1e-10)

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            DatasetAuditor.chi_square_quantile(1.0, 2)
        with pytest.raises(ValueError):
            DatasetAuditor.chi_square_quantile(0.5, 0)


class TestMCD:
    def test_clean_gaussian_recovers_parameters(self) -> None:
        rng = np.random.default_rng(3)
        X = rng.multivariate_normal([1.0, -2.0, 0.5], np.diag([1.0, 4.0, 0.25]), size=2000)
        mu, cov, support = DatasetAuditor.minimum_covariance_determinant(X)
        assert np.allclose(mu, [1.0, -2.0, 0.5], atol=0.1)
        assert np.allclose(np.diag(cov), [1.0, 4.0, 0.25], rtol=0.15)
        assert support.sum() == (2000 + 3 + 1) // 2

    def test_resists_contamination(self) -> None:
        X = _contaminated().to_numpy()
        mu, cov, _ = DatasetAuditor.minimum_covariance_determinant(X)
        assert np.allclose(mu, [0.0, 0.0], atol=0.2)
        corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        assert corr > 0.8  # classical correlation is dragged far lower
        assert np.corrcoef(X, rowvar=False)[0, 1] < 0.6

    def test_deterministic_with_seed(self) -> None:
        X = _contaminated().to_numpy()
        a = DatasetAuditor.minimum_covariance_determinant(X, random_state=5)
        b = DatasetAuditor.minimum_covariance_determinant(X, random_state=5)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            DatasetAuditor.minimum_covariance_determinant(np.ones((3, 2)))
        with pytest.raises(ValueError):
            DatasetAuditor.minimum_covariance_determinant(np.ones(5))
        X = np.random.default_rng(0).normal(size=(20, 2))
        with pytest.raises(ValueError):
            DatasetAuditor.minimum_covariance_determinant(X, support_fraction=0.2)
        X[0, 0] = np.nan
        with pytest.raises(ValueError):
            DatasetAuditor.minimum_covariance_determinant(X)


class TestReport:
    def test_flags_planted_outliers_and_masking(self) -> None:
        df = _contaminated()
        res = DatasetAuditor().multivariate_outlier_report(df, top=100)
        flagged = {r["index"] for r in res["outliers"]}
        planted = set(range(300, 330))
        assert planted <= flagged
        assert res["n_outliers"] < 50
        assert len(res["masked"]) > 0  # classical distance misses some planted rows
        assert set(res["masked"]) & planted
        dists = [r["robust_distance"] for r in res["outliers"]]
        assert dists == sorted(dists, reverse=True)
        assert all(0 <= r["p_value"] <= 1 for r in res["outliers"])
        assert set(res["outliers"][0]["values"]) == {"a", "b"}

    def test_drops_missing_and_non_numeric(self) -> None:
        df = _contaminated()
        df["label"] = "x"
        df["const"] = 1.0
        df.loc[[0, 1, 2], "a"] = np.nan
        res = DatasetAuditor().multivariate_outlier_report(df)
        assert res["columns"] == ["a", "b"]
        assert res["n_dropped_rows"] == 3
        assert res["n_rows"] == 327

    def test_top_limits_table(self) -> None:
        res = DatasetAuditor().multivariate_outlier_report(_contaminated(), top=5)
        assert len(res["outliers"]) == 5
        assert res["n_outliers"] >= 30

    def test_validation(self) -> None:
        df = _contaminated()
        df["s"] = "z"
        with pytest.raises(ValueError):
            DatasetAuditor().multivariate_outlier_report(df, columns=["nope"])
        with pytest.raises(ValueError):
            DatasetAuditor().multivariate_outlier_report(df, columns=["s"])
        with pytest.raises(ValueError):
            DatasetAuditor().multivariate_outlier_report(df, alpha=1.5)
        with pytest.raises(ValueError):
            DatasetAuditor().multivariate_outlier_report(df, top=0)


class TestCLI:
    def test_text_and_json(self, tmp_path, capsys) -> None:
        path = tmp_path / "d.csv"
        _contaminated().to_csv(path, index=False)
        assert main(["mv-outliers", str(path)]) == 0
        out = capsys.readouterr().out
        assert "FastMCD" in out and "outlier(s)" in out
        assert main(["mv-outliers", str(path), "--json", "--top", "3"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["outliers"]) == 3
        assert main(["mv-outliers", str(path), "--fail-on-outliers"]) == 1
        capsys.readouterr()
        assert main(["mv-outliers", str(path), "--columns", "zzz"]) == 2

"""Tests for Little's MCAR test (chi_square_sf / em_mean_covariance / mcar_test_report / mcar CLI)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from dataset_audit_kit.cli import main
from dataset_audit_kit.core import DatasetAuditor

COV = [[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]


def _gaussian(n: int = 300, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal([0.0, 1.0, 2.0], COV, size=n)


class TestChiSquare:
    def test_known_values(self) -> None:
        sf = DatasetAuditor.chi_square_sf
        assert sf(3.841458820694124, 1) == pytest.approx(0.05, abs=1e-10)
        assert sf(5.991464547107979, 2) == pytest.approx(0.05, abs=1e-10)
        assert sf(2.0, 2) == pytest.approx(math.exp(-1.0), rel=1e-12)  # df=2 is exponential
        assert sf(18.307038053275146, 10) == pytest.approx(0.05, abs=1e-10)
        assert sf(0.0, 4) == 1.0
        assert sf(1e6, 3) == pytest.approx(0.0, abs=1e-300)
        with pytest.raises(ValueError):
            sf(1.0, 0)


class TestEM:
    def test_complete_data_gives_sample_ml_estimates(self) -> None:
        X = _gaussian(200, seed=1)
        mu, sigma, _ = DatasetAuditor.em_mean_covariance(X)
        assert np.allclose(mu, X.mean(axis=0))
        assert np.allclose(sigma, np.cov(X.T, bias=True))

    def test_monotone_bivariate_matches_closed_form(self) -> None:
        # x fully observed, y missing for some rows: Anderson (1957) closed-form MLE
        rng = np.random.default_rng(2)
        x = rng.normal(size=400)
        y = 2.0 + 1.5 * x + rng.normal(scale=0.5, size=400)
        y[x > 0.8] = np.nan  # MAR on x
        X = np.c_[x, y]
        mu, sigma, _ = DatasetAuditor.em_mean_covariance(X, tol=1e-12, max_iter=5000)
        cc = ~np.isnan(y)
        beta = np.cov(x[cc], y[cc], bias=True)[0, 1] / np.var(x[cc])
        mu_x = x.mean()
        mu_y = y[cc].mean() + beta * (mu_x - x[cc].mean())
        assert mu[0] == pytest.approx(mu_x)
        assert mu[1] == pytest.approx(mu_y, abs=1e-6)
        # the complete-case mean of y is biased; EM recovers the true mean (2.0)
        assert abs(mu[1] - 2.0) < abs(y[cc].mean() - 2.0)


class TestReport:
    def test_mcar_data_rejection_rate_near_alpha(self) -> None:
        auditor = DatasetAuditor()
        rng = np.random.default_rng(3)
        pvalues = []
        for seed in range(60):
            X = _gaussian(250, seed=100 + seed)
            X[rng.uniform(size=X.shape) < 0.15] = np.nan
            res = auditor.mcar_test_report(pd.DataFrame(X, columns=list("abc")))
            pvalues.append(res["p_value"])
        rate = float(np.mean(np.array(pvalues) < 0.05))
        assert rate <= 0.15
        assert 0.3 < float(np.mean(pvalues)) < 0.7  # roughly uniform under H0

    def test_mar_missingness_is_rejected_and_pattern_flagged(self) -> None:
        X = _gaussian(300, seed=4)
        X[X[:, 0] > 0.5, 1] = np.nan  # b missing when a is large
        res = DatasetAuditor().mcar_test_report(pd.DataFrame(X, columns=list("abc")))
        assert res["mcar_rejected"] and res["p_value"] < 1e-10
        assert res["df"] == (3 + 2) - 3
        assert res["n_patterns"] == 2
        flagged = max(res["patterns"], key=lambda r: r["contribution"])
        assert flagged["missing_columns"] in (["b"], [])
        assert sum(r["n"] for r in res["patterns"]) == 300
        assert res["missing_share"]["b"] == pytest.approx(np.isnan(X[:, 1]).mean())

    def test_statistic_matches_formula_from_em_estimates(self) -> None:
        X = _gaussian(200, seed=5)
        rng = np.random.default_rng(5)
        X[rng.uniform(size=X.shape) < 0.2] = np.nan
        X = X[~np.isnan(X).all(axis=1)]
        res = DatasetAuditor().mcar_test_report(pd.DataFrame(X, columns=list("abc")))
        mu, sigma, _ = DatasetAuditor.em_mean_covariance(X)
        obs = ~np.isnan(X)
        d2, df = 0.0, 0
        for key in {tuple(r) for r in obs}:
            rows = np.all(obs == np.array(key), axis=1)
            idx = np.flatnonzero(key)
            diff = X[np.ix_(rows, idx)].mean(axis=0) - mu[idx]
            d2 += rows.sum() * diff @ np.linalg.inv(sigma[np.ix_(idx, idx)]) @ diff
            df += idx.size
        assert res["statistic"] == pytest.approx(d2, rel=1e-6)
        assert res["df"] == df - 3
        assert res["p_value"] == pytest.approx(DatasetAuditor.chi_square_sf(d2, df - 3), rel=1e-5)

    def test_column_selection_dropped_rows_and_no_missing(self) -> None:
        X = _gaussian(50, seed=6)
        frame = pd.DataFrame(X, columns=list("abc"))
        frame["label"] = "x"  # non-numeric ignored by default
        frame["const"] = 1.0  # zero variance ignored
        frame.loc[0, ["a", "b", "c", "const"]] = np.nan
        res = DatasetAuditor().mcar_test_report(frame)
        assert res["columns"] == ["a", "b", "c"]
        assert res["n_dropped_rows"] == 1 and res["n_rows"] == 49
        assert res["p_value"] is None and "no missing" in res["note"]
        with pytest.raises(ValueError, match="numeric"):
            DatasetAuditor().mcar_test_report(frame, columns=["a", "label"])
        with pytest.raises(ValueError, match="unknown"):
            DatasetAuditor().mcar_test_report(frame, columns=["zzz"])
        with pytest.raises(ValueError, match="alpha"):
            DatasetAuditor().mcar_test_report(frame, alpha=1.5)


class TestCLI:
    def _write(self, tmp_path) -> str:
        X = _gaussian(300, seed=7)
        X[X[:, 0] > 0.5, 2] = np.nan
        path = tmp_path / "mar.csv"
        pd.DataFrame(X, columns=["age", "income", "score"]).to_csv(path, index=False)
        return str(path)

    def test_table_and_fail_flag(self, tmp_path, capsys) -> None:
        path = self._write(tmp_path)
        assert main(["mcar", path]) == 0
        out = capsys.readouterr().out
        assert "MCAR REJECTED" in out and "score" in out
        assert main(["mcar", path, "--fail-on-reject"]) == 1

    def test_json_and_columns(self, tmp_path, capsys) -> None:
        path = self._write(tmp_path)
        assert main(["mcar", path, "--columns", "age,income", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["columns"] == ["age", "income"]
        assert payload["p_value"] is None  # no missing values in those columns

    def test_bad_column_exit_code(self, tmp_path, capsys) -> None:
        path = self._write(tmp_path)
        assert main(["mcar", path, "--columns", "nope"]) == 2
        assert "unknown columns" in capsys.readouterr().err

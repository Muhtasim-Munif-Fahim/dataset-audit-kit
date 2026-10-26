"""Tests for DatasetAuditor.adversarial_validation and the adversarial CLI."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from dataset_audit_kit.cli import main
from dataset_audit_kit.core import DatasetAuditor


def _frames(shift: float = 0.0, n: int = 800, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    reference = pd.DataFrame(
        {
            "x": rng.normal(0.0, 1.0, n),
            "y": rng.normal(0.0, 1.0, n),
            "city": rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2]),
        }
    )
    current = pd.DataFrame(
        {
            "x": rng.normal(shift, 1.0, n),
            "y": rng.normal(0.0, 1.0, n),
            "city": rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2]),
        }
    )
    return current, reference


class TestRankAuc:
    def test_matches_sklearn_with_ties(self) -> None:
        rng = np.random.default_rng(1)
        labels = rng.integers(0, 2, 400)
        scores = rng.integers(0, 6, 400).astype(float)
        assert DatasetAuditor.rank_auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))

    def test_readme_example(self) -> None:
        assert DatasetAuditor.rank_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)

    def test_needs_both_classes(self) -> None:
        with pytest.raises(ValueError):
            DatasetAuditor.rank_auc([1, 1], [0.2, 0.3])


class TestLogistic:
    def test_matches_unpenalised_mle_score_equations(self) -> None:
        rng = np.random.default_rng(2)
        X = rng.normal(size=(500, 3))
        y = (rng.random(500) < 1 / (1 + np.exp(-(0.3 + X @ [1.0, -0.5, 0.0])))).astype(float)
        w = DatasetAuditor._fit_logistic(X, y, l2=0.0)
        p = 1 / (1 + np.exp(-(w[0] + X @ w[1:])))
        Xb = np.hstack([np.ones((500, 1)), X])
        assert np.abs(Xb.T @ (p - y)).max() < 1e-6

    def test_penalty_shrinks(self) -> None:
        rng = np.random.default_rng(3)
        X = rng.normal(size=(300, 2))
        y = (X[:, 0] + rng.normal(size=300) > 0).astype(float)
        free = DatasetAuditor._fit_logistic(X, y, l2=0.0)
        heavy = DatasetAuditor._fit_logistic(X, y, l2=100.0)
        assert np.abs(heavy[1:]).sum() < np.abs(free[1:]).sum()


class TestAdversarialValidation:
    def test_same_distribution_is_near_half(self) -> None:
        current, reference = _frames(0.0)
        result = DatasetAuditor().adversarial_validation(current, reference, n_permutations=19)
        assert abs(result["auc"] - 0.5) < 0.06
        assert result["drifted"] is False
        assert result["p_value"] > 0.05

    def test_location_shift_is_detected_and_attributed(self) -> None:
        current, reference = _frames(0.8)
        result = DatasetAuditor().adversarial_validation(current, reference, n_permutations=9)
        assert result["auc"] > 0.65
        assert result["drifted"] is True
        assert result["p_value"] == pytest.approx(0.1)
        assert result["columns"][0]["column"] == "x"
        top = result["columns"][0]
        assert top["importance"] == max(r["importance"] for r in result["columns"])

    def test_categorical_shift(self) -> None:
        rng = np.random.default_rng(4)
        n = 800
        reference = pd.DataFrame({"city": rng.choice(["a", "b"], n, p=[0.8, 0.2])})
        current = pd.DataFrame({"city": rng.choice(["a", "b"], n, p=[0.3, 0.7])})
        result = DatasetAuditor().adversarial_validation(current, reference)
        assert result["auc"] > 0.7

    def test_correlation_shift_needs_interactions(self) -> None:
        rng = np.random.default_rng(5)
        n = 1200
        reference = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})
        x = rng.normal(size=n)
        current = pd.DataFrame({"x": x, "y": 0.9 * x + np.sqrt(0.19) * rng.normal(size=n)})
        linear = DatasetAuditor().adversarial_validation(current, reference)
        joint = DatasetAuditor().adversarial_validation(current, reference, interactions=True)
        assert linear["auc"] < 0.58
        assert joint["auc"] > 0.75
        # Neither column has shifted on its own.
        assert all(r["solo_auc"] < 0.58 for r in joint["columns"])

    def test_leaky_timestamp_and_exclude(self) -> None:
        current, reference = _frames(0.0)
        reference["t"] = pd.date_range("2020-01-01", periods=len(reference), freq="h")
        current["t"] = pd.date_range("2022-01-01", periods=len(current), freq="h", tz="UTC")
        result = DatasetAuditor().adversarial_validation(current, reference)
        assert result["auc"] == pytest.approx(1.0)
        assert result["columns"][0]["column"] == "t"
        assert result["columns"][0]["solo_auc"] == pytest.approx(1.0)
        cleaned = DatasetAuditor().adversarial_validation(current, reference, exclude=["t"])
        assert cleaned["drifted"] is False

    def test_missing_values_and_constant_columns(self) -> None:
        current, reference = _frames(0.0)
        current.loc[current.index[:400], "x"] = np.nan
        current["k"] = 1
        reference["k"] = 1
        result = DatasetAuditor().adversarial_validation(current, reference)
        assert result["auc"] > 0.7  # missingness pattern differs
        assert result["ignored_columns"] == ["k"]

    def test_reference_most_like_current(self) -> None:
        current, reference = _frames(1.5)
        reference.index = [f"r{i}" for i in range(len(reference))]
        result = DatasetAuditor().adversarial_validation(current, reference, top=5)
        rows = result["reference_most_like_current"]
        assert len(rows) == 5
        probs = [r["p_current"] for r in rows]
        assert probs == sorted(probs, reverse=True)
        picked = reference.loc[[r["index"] for r in rows], "x"]
        assert picked.mean() > reference["x"].quantile(0.8)

    def test_deterministic_with_seed(self) -> None:
        current, reference = _frames(0.3)
        a = DatasetAuditor().adversarial_validation(current, reference, seed=7)
        b = DatasetAuditor().adversarial_validation(current, reference, seed=7)
        assert a == b

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"folds": 1},
            {"l2": -1.0},
            {"max_levels": 0},
            {"n_permutations": -1},
            {"columns": ["nope"]},
            {"exclude": ["x", "y", "city"]},
        ],
    )
    def test_invalid_arguments(self, kwargs: dict) -> None:
        current, reference = _frames(0.0, n=50)
        with pytest.raises(ValueError):
            DatasetAuditor().adversarial_validation(current, reference, **kwargs)


class TestAdversarialCli:
    def _write(self, tmp_path, shift: float) -> tuple[str, str]:
        current, reference = _frames(shift)
        cur = tmp_path / "current.csv"
        ref = tmp_path / "reference.csv"
        current.to_csv(cur, index=False)
        reference.to_csv(ref, index=False)
        return str(cur), str(ref)

    def test_table_output(self, tmp_path, capsys) -> None:
        cur, ref = self._write(tmp_path, 0.0)
        assert main(["adversarial", cur, "--reference", ref, "--fail-on-shift"]) == 0
        out = capsys.readouterr().out
        assert "Domain-classifier AUC" in out
        assert "no shift" in out
        assert "solo_auc" in out

    def test_json_and_fail_on_shift(self, tmp_path, capsys) -> None:
        cur, ref = self._write(tmp_path, 1.0)
        code = main(
            ["adversarial", cur, "--reference", ref, "--json", "--fail-on-shift", "--permutations", "4", "--interactions"]
        )
        assert code == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["drifted"] is True
        assert payload["interactions"] is True
        assert payload["n_permutations"] == 4

    def test_bad_column_is_usage_error(self, tmp_path, capsys) -> None:
        cur, ref = self._write(tmp_path, 0.0)
        assert main(["adversarial", cur, "--reference", ref, "--columns", "nope"]) == 2
        assert "Error" in capsys.readouterr().err

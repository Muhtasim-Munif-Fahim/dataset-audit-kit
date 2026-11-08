"""Tests for the kernel MMD multivariate drift test (mmd_drift_report / mmd-drift CLI)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dataset_audit_kit.cli import main
from dataset_audit_kit.core import DatasetAuditor


def _gauss(n, cov, seed, shift=(0.0, 0.0)):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.multivariate_normal(shift, cov, size=n), columns=["x", "y"])


POS = np.array([[1.0, 0.9], [0.9, 1.0]])
NEG = np.array([[1.0, -0.9], [-0.9, 1.0]])


def test_unbiased_statistic_matches_explicit_formula():
    rng = np.random.default_rng(0)
    K = rng.random((7, 7))
    K = (K + K.T) / 2
    lab = np.array([1, 1, 1, 0, 0, 0, 0], dtype=bool)
    xx = K[:3, :3]; yy = K[3:, 3:]; xy = K[:3, 3:]
    expected = (xx.sum() - np.trace(xx)) / 6 + (yy.sum() - np.trace(yy)) / 12 - 2 * xy.mean()
    assert DatasetAuditor._mmd_unbiased_stats(K, [lab])[0] == pytest.approx(expected)


def test_same_distribution_not_flagged():
    res = DatasetAuditor().mmd_drift_report(_gauss(300, POS, 1), _gauss(300, POS, 2), n_permutations=99)
    assert not res["drift_detected"]
    assert res["p_value"] > 0.05
    assert abs(res["mmd2"]) < 0.02


def test_mean_shift_flagged_jointly_and_marginally():
    res = DatasetAuditor().mmd_drift_report(_gauss(200, POS, 3, shift=(1.0, 0.0)), _gauss(200, POS, 4), n_permutations=99)
    assert res["drift_detected"]
    assert res["p_value"] == pytest.approx(1 / 100)
    assert res["per_column"][0]["column"] == "x"
    assert res["per_column"][0]["drift"]


def test_correlation_flip_detected_only_jointly():
    res = DatasetAuditor().mmd_drift_report(_gauss(300, NEG, 5), _gauss(300, POS, 6), n_permutations=99)
    assert res["drift_detected"]
    assert not any(r["drift"] for r in res["per_column"])
    assert "no single column" in res["note"]


def test_missing_rows_dropped_and_columns_selected():
    cur = _gauss(50, POS, 7)
    cur.loc[:4, "x"] = np.nan
    cur["label"] = "a"
    res = DatasetAuditor().mmd_drift_report(cur, _gauss(60, POS, 8), n_permutations=19, per_column=False)
    assert res["columns"] == ["x", "y"]
    assert res["n_dropped_current"] == 5
    assert res["n_current"] == 45
    assert res["per_column"] == []


def test_validation():
    a = _gauss(30, POS, 9)
    with pytest.raises(ValueError):
        DatasetAuditor().mmd_drift_report(a, a, n_permutations=0)
    with pytest.raises(ValueError):
        DatasetAuditor().mmd_drift_report(a, a, columns=["nope"])
    with pytest.raises(ValueError):
        DatasetAuditor().mmd_drift_report(a.head(3), a)


def test_cli(tmp_path, capsys):
    cur = tmp_path / "cur.csv"
    ref = tmp_path / "ref.csv"
    _gauss(150, NEG, 10).to_csv(cur, index=False)
    _gauss(150, POS, 11).to_csv(ref, index=False)
    code = main(["mmd-drift", str(cur), "--reference", str(ref), "--permutations", "49", "--fail-on-drift"])
    out = capsys.readouterr().out
    assert code == 1
    assert "MMD^2_u" in out and "DRIFT" in out
    code = main(["mmd-drift", str(cur), "--reference", str(ref), "--permutations", "49", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["drift_detected"] is True

"""Tests for the Ho & Basu data-complexity report (data_complexity_report / complexity CLI)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dataset_audit_kit.cli import main
from dataset_audit_kit.core import DatasetAuditor


def _blobs(n=100, gap=10.0, seed=0, k=2, d=2):
    rng = np.random.default_rng(seed)
    parts = []
    for c in range(k):
        X = rng.normal(size=(n, d))
        X[:, 0] += c * gap
        parts.append(pd.DataFrame(X, columns=[f"f{j}" for j in range(d)]).assign(label=f"c{c}"))
    return pd.concat(parts, ignore_index=True)


def test_separated_classes_are_easy():
    res = DatasetAuditor().data_complexity_report(_blobs(), "label")
    m = res["measures"]
    assert m["F1"] < 0.05
    assert m["F2"] == 0.0 and m["F3"] == 0.0
    assert m["N3"] == 0.0
    assert m["N1"] == pytest.approx(2 / 200)  # a single MST edge bridges the blobs
    assert m["N2"] < 0.2
    assert res["best_fisher_feature"] == "f0"
    assert not res["hard_to_separate"]


def test_identical_distributions_are_hard():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(400, 3)), columns=list("abc"))
    df["label"] = rng.integers(0, 2, size=400)
    m = DatasetAuditor().data_complexity_report(df, "label")["measures"]
    assert m["F1"] > 0.95
    assert m["F2"] > 0.5
    assert m["F3"] > 0.8
    assert 0.35 < m["N3"] < 0.65
    assert m["N1"] > 0.5
    assert m["N2"] > 0.4


def test_hand_computed_one_dimensional_example():
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 10.0, 11.0], "y": ["a", "a", "b", "a", "b", "b"]})
    res = DatasetAuditor().data_complexity_report(df, "y")
    m = res["measures"]
    # Fisher: mu_a=4/3, mu_b=23/3, mu=4.5
    mu_a, mu_b, mu = 4 / 3, 23 / 3, 4.5
    between = 3 * (mu_a - mu) ** 2 + 3 * (mu_b - mu) ** 2
    within = sum((v - mu_a) ** 2 for v in [0, 1, 3]) + sum((v - mu_b) ** 2 for v in [2, 10, 11])
    assert res["max_fisher_ratio"] == pytest.approx(between / within)
    assert m["F1"] == pytest.approx(1 / (1 + between / within))
    # Overlap of [0,3] and [2,11] over [0,11]: 1/11; rows in [2,3]: 2 of 6.
    assert m["F2"] == pytest.approx(1 / 11)
    assert m["F3"] == pytest.approx(2 / 6)
    # MST on a line joins neighbours; cross-class edges 1-2, 2-3, 3-10 touch rows 1, 2, 3, 10
    assert res["borderline_rows"] == 4 and m["N1"] == pytest.approx(4 / 6)
    # 1-NN (scaled by 1/11 — ordering unchanged): 0->1 ok, 1->0 ok (tie 1/2 -> first),
    # 2->1 or 3 (both a) wrong, 3->2 wrong, 10->11 ok, 11->10 ok
    assert m["N3"] == pytest.approx(2 / 6)
    assert m["T2"] == pytest.approx(1 / 6)


def test_multiclass_missing_and_feature_selection():
    df = _blobs(n=40, k=3, d=3)
    df.loc[0, "f1"] = np.nan
    df["name"] = "x"
    res = DatasetAuditor().data_complexity_report(df, "label", features=["f0", "f1"])
    assert res["n_classes"] == 3
    assert res["n_rows_dropped_missing"] == 1
    assert res["features"] == ["f0", "f1"]
    assert "name" in res["excluded_columns"] and "f2" in res["excluded_columns"]
    assert all(0.0 <= v <= 1.0 for k, v in res["measures"].items() if k != "T2")


def test_sampling_cap():
    res = DatasetAuditor().data_complexity_report(_blobs(n=300), "label", max_rows=100, seed=3)
    assert res["n_rows_used"] == 100 and res["sampled"]


def test_validation():
    df = _blobs(n=10)
    aud = DatasetAuditor()
    with pytest.raises(ValueError):
        aud.data_complexity_report(df, "nope")
    with pytest.raises(ValueError):
        aud.data_complexity_report(df.assign(label="one"), "label")
    with pytest.raises(ValueError):
        aud.data_complexity_report(df, "label", features=["label"])
    with pytest.raises(ValueError):
        aud.data_complexity_report(df, "label", features=["zzz"])
    with pytest.raises(ValueError):
        aud.data_complexity_report(df[["label"]], "label")
    with pytest.raises(ValueError):
        aud.data_complexity_report(df, "label", max_rows=2)
    with pytest.raises(ValueError):
        aud.data_complexity_report(df, "label", n3_threshold=2.0)


def test_cli(tmp_path, capsys):
    easy = tmp_path / "easy.csv"
    _blobs(n=50).to_csv(easy, index=False)
    assert main(["complexity", str(easy), "--target", "label"]) == 0
    out = capsys.readouterr().out
    assert "N3" in out and "reasonably separable" in out
    assert main(["complexity", str(easy), "--target", "label", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["measures"]["N3"] == 0.0
    rng = np.random.default_rng(0)
    hard = tmp_path / "hard.csv"
    pd.DataFrame({"a": rng.normal(size=200), "label": rng.integers(0, 2, 200)}).to_csv(hard, index=False)
    assert main(["complexity", str(hard), "--target", "label", "--fail-on-hard"]) == 1
    assert main(["complexity", str(hard), "--target", "missing"]) == 2

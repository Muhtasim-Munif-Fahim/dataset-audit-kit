"""Tests for DatasetAuditor.class_imbalance_report and the imbalance CLI."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from dataset_audit_kit.cli import main
from dataset_audit_kit.core import DatasetAuditor


def _frame() -> pd.DataFrame:
    labels = ["a"] * 800 + ["b"] * 150 + ["c"] * 45 + ["d"] * 5 + [None] * 7
    site = (["x", "y"] * 600)[: len(labels)]
    return pd.DataFrame({"label": labels, "site": site})


def test_counts_ratio_and_evenness() -> None:
    res = DatasetAuditor().class_imbalance_report(_frame(), "label")
    assert res["n_rows"] == 1000 and res["n_missing"] == 7 and res["n_classes"] == 4
    assert [c["class"] for c in res["classes"]] == ["a", "b", "c", "d"]
    assert res["majority_class"] == "a" and res["minority_class"] == "d"
    assert res["imbalance_ratio"] == pytest.approx(160.0)
    assert res["severity"] == "extreme" and res["imbalanced"] is True
    p = np.array([0.8, 0.15, 0.045, 0.005])
    assert res["evenness"] == pytest.approx(-np.sum(p * np.log(p)) / math.log(4))
    assert res["gini_simpson"] == pytest.approx((1 - np.sum(p**2)) / 0.75)
    # minority: share < 0.5 / 4 = 0.125; too_few: count < 10
    assert res["minority_classes"] == ["c", "d"]
    assert res["too_few"] == ["d"]


def test_effective_numbers_and_weights() -> None:
    res = DatasetAuditor().class_imbalance_report(_frame(), "label", beta=0.99)
    counts = np.array([800, 150, 45, 5], dtype=float)
    eff = (1 - 0.99**counts) / (1 - 0.99)
    got = np.array([c["effective_number"] for c in res["classes"]])
    np.testing.assert_allclose(got, eff)
    cb = np.array([c["class_balanced_weight"] for c in res["classes"]])
    assert cb.sum() == pytest.approx(4.0)
    assert np.all(np.diff(cb) > 0)  # rarer classes get larger weights
    inv = np.array([c["inverse_frequency_weight"] for c in res["classes"]])
    np.testing.assert_allclose(inv, 1000 / (4 * counts))
    assert np.sum(inv * counts) / 1000 == pytest.approx(1.0)
    # beta -> 0 makes every effective number 1 (no reweighting)
    flat = DatasetAuditor().class_imbalance_report(_frame(), "label", beta=0.0)
    np.testing.assert_allclose([c["class_balanced_weight"] for c in flat["classes"]], 1.0)


def test_balanced_and_single_class_targets() -> None:
    df = pd.DataFrame({"y": [0, 1, 2] * 50})
    res = DatasetAuditor().class_imbalance_report(df, "y")
    assert res["imbalance_ratio"] == 1.0 and res["severity"] == "balanced"
    assert res["evenness"] == pytest.approx(1.0) and res["gini_simpson"] == pytest.approx(1.0)
    assert res["imbalanced"] is False and res["minority_classes"] == []
    assert isinstance(res["classes"][0]["class"], int)
    one = DatasetAuditor().class_imbalance_report(pd.DataFrame({"y": ["z"] * 20}), "y")
    assert one["n_classes"] == 1 and one["imbalance_ratio"] == 1.0


def test_label_shift_by_segment_matches_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    site = np.repeat(["x", "y", "z"], 400)
    p_pos = np.repeat([0.1, 0.1, 0.4], 400)
    df = pd.DataFrame({"site": site, "y": np.where(rng.random(1200) < p_pos, "pos", "neg")})
    res = DatasetAuditor().class_imbalance_report(df, "y", by="site")
    seg = res["by"]
    table = pd.crosstab(df["site"], df["y"])
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    assert seg["chi_square"] == pytest.approx(chi2) and seg["p_value"] == pytest.approx(p)
    assert seg["dof"] == dof == 2 and seg["p_value"] < 1e-6
    assert 0 < seg["cramers_v"] < 1
    z = next(g for g in seg["groups"] if g["level"] == "z")
    assert z["n_rows"] == 400 and sum(z["shares"].values()) == pytest.approx(1.0)
    assert z["shares"]["pos"] > 0.3


def test_segment_missing_classes() -> None:
    df = pd.DataFrame({"y": ["a"] * 30 + ["b"] * 10, "g": ["u"] * 20 + ["v"] * 20})
    res = DatasetAuditor().class_imbalance_report(df, "y", by="g")
    u = next(g for g in res["by"]["groups"] if g["level"] == "u")
    assert u["missing_classes"] == ["b"] and u["imbalance_ratio"] == 1.0


def test_validation() -> None:
    df = _frame()
    auditor = DatasetAuditor()
    with pytest.raises(ValueError, match="unknown column"):
        auditor.class_imbalance_report(df, "nope")
    with pytest.raises(ValueError, match="unknown column"):
        auditor.class_imbalance_report(df, "label", by="nope")
    with pytest.raises(ValueError, match="by must differ"):
        auditor.class_imbalance_report(df, "label", by="label")
    with pytest.raises(ValueError, match="beta"):
        auditor.class_imbalance_report(df, "label", beta=1.0)
    with pytest.raises(ValueError, match="max_classes"):
        auditor.class_imbalance_report(pd.DataFrame({"id": range(50)}), "id", max_classes=10)
    with pytest.raises(ValueError, match="no non-missing"):
        auditor.class_imbalance_report(pd.DataFrame({"y": [None, None]}), "y")


def test_cli(tmp_path, capsys) -> None:
    path = tmp_path / "d.csv"
    _frame().to_csv(path, index=False)
    assert main(["imbalance", str(path), "--target", "label", "--by", "site"]) == 0
    out = capsys.readouterr().out
    assert "Imbalance ratio 160.00 (extreme)" in out and "Label shift across 'site'" in out
    assert "minority,too_few" in out
    assert main(["imbalance", str(path), "--target", "label", "--fail-on-imbalance"]) == 1
    capsys.readouterr()
    assert main(["imbalance", str(path), "--target", "label", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_classes"] == 4
    assert main(["imbalance", str(path), "--target", "missing"]) == 2

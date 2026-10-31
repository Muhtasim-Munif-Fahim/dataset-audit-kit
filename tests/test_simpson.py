"""Tests for the Simpson's paradox scan (simpsons_paradox_report / simpson CLI)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dataset_audit_kit.cli import main
from dataset_audit_kit.core import DatasetAuditor
from dataset_audit_kit.simpson import simpsons_paradox_report


def _kidney_stones() -> pd.DataFrame:
    """Charig et al. (1986) kidney-stone data: treatment A wins in each stratum, B overall."""
    cells = [
        ("A", "small", 81, 87), ("B", "small", 234, 270),
        ("A", "large", 192, 263), ("B", "large", 55, 80),
    ]
    rows = []
    for treat, size, ok, n in cells:
        rows += [{"treatment": treat, "size": size, "success": 1}] * ok
        rows += [{"treatment": treat, "size": size, "success": 0}] * (n - ok)
    return pd.DataFrame(rows)


def test_kidney_stone_reversal_detected() -> None:
    res = simpsons_paradox_report(_kidney_stones(), "success")
    assert res["n_reversals"] == 1
    top = res["pairs"][0]
    assert top["factor"] == "treatment" and top["stratum"] == "size"
    assert (top["level_a"], top["level_b"]) == ("A", "B")
    # overall B looks better: 289/350 - 273/350
    assert top["aggregate_effect"] == pytest.approx(289 / 350 - 273 / 350)
    assert top["adjusted_effect"] < 0  # A better within strata
    assert top["reversed_strata"] == 2 and top["n_strata"] == 2
    assert top["reversal"] is True


def test_no_reversal_without_confounding() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({"x": rng.choice(["a", "b"], n), "z": rng.choice(["p", "q", "r"], n)})
    df["y"] = (df.x == "b") * 1.0 + rng.normal(0, 1, n)
    res = DatasetAuditor().simpsons_paradox_report(df, "y", factors=["x"], strata=["z"])
    assert res["n_pairs"] == 1 and res["n_reversals"] == 0
    assert res["pairs"][0]["adjusted_effect"] == pytest.approx(1.0, abs=0.15)


def test_min_cell_and_min_effect_filters() -> None:
    df = _kidney_stones()
    assert simpsons_paradox_report(df, "success", min_cell=90)["n_pairs"] == 0
    assert simpsons_paradox_report(df, "success", min_effect=0.1)["n_reversals"] == 0


def test_string_target_and_validation() -> None:
    df = _kidney_stones()
    df["outcome"] = np.where(df.success == 1, "yes", "no")
    res = simpsons_paradox_report(df.drop(columns="success"), "outcome")
    assert res["n_reversals"] == 1
    with pytest.raises(ValueError):
        simpsons_paradox_report(df, "missing_col")
    with pytest.raises(ValueError):
        simpsons_paradox_report(df, "success", factors=["size", "outcome"], strata=["nope"])
    with pytest.raises(ValueError):
        simpsons_paradox_report(df.assign(t=list("abc") * (len(df) // 3) + ["a"] * (len(df) % 3)), "t")


def test_cli_simpson(tmp_path, capsys) -> None:
    path = tmp_path / "stones.csv"
    _kidney_stones().to_csv(path, index=False)
    assert main(["simpson", str(path), "--target", "success"]) == 0
    out = capsys.readouterr().out
    assert "1 reversal(s)" in out and "YES" in out
    assert main(["simpson", str(path), "--target", "success", "--fail-on-reversal"]) == 1
    capsys.readouterr()
    assert main(["simpson", str(path), "--target", "success", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_reversals"] == 1
    assert main(["simpson", str(path), "--target", "nope"]) == 2

"""Tests for confident-learning label issues (confident_joint / find_label_issues / label_issues_report / label-issues CLI)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dataset_audit_kit.cli import main
from dataset_audit_kit.core import DatasetAuditor


def _small(seed: int = 7, n: int = 60, k: int = 3, shift: float = 1.8, flip: float = 0.2):
    rng = np.random.default_rng(seed)
    true = rng.integers(0, k, n)
    logits = rng.normal(size=(n, k))
    logits[np.arange(n), true] += shift
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    y = true.copy()
    flipped = rng.random(n) < flip
    y[flipped] = (y[flipped] + 1) % k
    return y, P, true, flipped


# Reference values computed offline with cleanlab 2.x on _small():
#   cleanlab.count.compute_confident_joint(y, P, calibrate=False / True)
#   cleanlab.filter.find_label_issues(y, P, filter_by=...)
CLEANLAB_RAW = [[7, 0, 3], [6, 11, 2], [0, 6, 14]]
CLEANLAB_CAL = [[10, 0, 4], [7, 13, 3], [0, 7, 16]]
CLEANLAB_ISSUES = {
    "prune_by_noise_rate": [5, 8, 21, 22, 24, 31, 33, 38, 43, 45, 47, 48, 49, 50, 51, 52, 56, 58],
    "prune_by_class": [5, 8, 12, 21, 22, 24, 31, 33, 38, 43, 45, 47, 48, 49, 50, 51, 52, 53, 56, 58],
    "both": [5, 8, 21, 22, 24, 31, 33, 38, 43, 45, 47, 48, 49, 50, 51, 52, 56, 58],
    "confident_joint": [5, 8, 21, 22, 24, 31, 33, 38, 43, 47, 48, 49, 50, 51, 52, 56, 58],
}


class TestConfidentJoint:
    def test_matches_cleanlab_reference(self) -> None:
        y, P, _, _ = _small()
        raw = DatasetAuditor.confident_joint(y, P, calibrate=False)
        cal = DatasetAuditor.confident_joint(y, P)
        assert raw["raw_joint"].tolist() == CLEANLAB_RAW
        assert cal["joint"].tolist() == CLEANLAB_CAL

    def test_brute_force_definition(self) -> None:
        y, P, _, _ = _small(seed=3, n=120, k=4)
        res = DatasetAuditor.confident_joint(y, P, calibrate=False)
        t = np.array([P[y == j, j].mean() for j in range(4)])
        assert np.allclose(res["thresholds"], t)
        C = np.zeros((4, 4))
        for i in range(len(y)):
            conf = np.where(P[i] >= t - 1e-6)[0]
            if len(conf) == 1:
                C[y[i], conf[0]] += 1
            elif len(conf) > 1:
                C[y[i], P[i].argmax()] += 1
        np.fill_diagonal(C, np.maximum(np.diag(C), 1))
        assert np.array_equal(res["raw_joint"], C)

    def test_calibrated_rows_match_label_counts(self) -> None:
        y, P, _, _ = _small(seed=11, n=300, k=5)
        res = DatasetAuditor.confident_joint(y, P)
        J = res["joint"]
        assert np.array_equal(J.sum(axis=1), np.bincount(y, minlength=5))
        assert J.sum() == len(y)
        assert np.allclose(J, np.round(J))
        assert np.allclose(res["noise_matrix"].sum(axis=0), 1.0)
        assert np.allclose(res["inverse_noise_matrix"].sum(axis=1), 1.0)

    def test_noise_rate_recovers_flip_rate(self) -> None:
        y, P, _, flipped = _small(seed=5, n=3000, k=3, shift=3.0, flip=0.15)
        res = DatasetAuditor.confident_joint(y, P)
        assert res["noise_rate"] == pytest.approx(flipped.mean(), abs=0.03)

    def test_clean_labels_have_little_noise(self) -> None:
        y, P, _, _ = _small(seed=6, n=1000, k=3, shift=4.0, flip=0.0)
        assert DatasetAuditor.confident_joint(y, P)["noise_rate"] < 0.03


class TestFindLabelIssues:
    @pytest.mark.parametrize("method", list(CLEANLAB_ISSUES))
    def test_matches_cleanlab_reference(self, method: str) -> None:
        y, P, _, _ = _small()
        mask = DatasetAuditor.find_label_issues(y, P, method=method)
        assert np.where(mask)[0].tolist() == CLEANLAB_ISSUES[method]

    def test_precision_recall_on_synthetic_noise(self) -> None:
        y, P, _, flipped = _small(seed=9, n=2000, k=4, shift=3.0, flip=0.1)
        mask = DatasetAuditor.find_label_issues(y, P)
        precision = (mask & flipped).sum() / mask.sum()
        recall = (mask & flipped).sum() / flipped.sum()
        assert precision > 0.8 and recall > 0.6

    def test_both_is_intersection_and_agreeing_rows_never_flagged(self) -> None:
        y, P, _, _ = _small(seed=12, n=400, k=3)
        a = DatasetAuditor.find_label_issues(y, P, method="prune_by_noise_rate")
        b = DatasetAuditor.find_label_issues(y, P, method="prune_by_class")
        both = DatasetAuditor.find_label_issues(y, P, method="both")
        assert np.array_equal(both, a & b)
        for m in (a, b, both):
            assert not np.any(m & (P.argmax(axis=1) == y))

    def test_validation(self) -> None:
        P = np.full((4, 2), 0.5)
        with pytest.raises(ValueError, match="method"):
            DatasetAuditor.find_label_issues([0, 1, 0, 1], P, method="nope")
        with pytest.raises(ValueError, match="class indices"):
            DatasetAuditor.find_label_issues([0, 1, 2, 1], P)
        with pytest.raises(ValueError, match="one entry"):
            DatasetAuditor.find_label_issues([0, 1], P)
        with pytest.raises(ValueError, match="2-D"):
            DatasetAuditor.find_label_issues([0, 1], np.ones(2))
        with pytest.raises(ValueError, match="non-negative"):
            DatasetAuditor.find_label_issues([0, 1], np.array([[1.0, -0.1], [0.5, 0.5]]))


class TestReport:
    def _frame(self):
        y, P, _, flipped = _small(seed=21, n=500, k=3, shift=3.0, flip=0.1)
        names = np.array(["cat", "dog", "fox"])
        df = pd.DataFrame({"x": np.arange(len(y)), "animal": names[y]}, index=np.arange(len(y)) + 1000)
        probs = pd.DataFrame(P, columns=list(names))
        return df, probs, flipped

    def test_report_with_named_columns(self) -> None:
        df, probs, flipped = self._frame()
        res = DatasetAuditor().label_issues_report(df, "animal", probs, top=5)
        assert res["classes"] == ["cat", "dog", "fox"]
        assert res["n_rows"] == 500
        assert len(res["issues"]) == 5
        margins = [r["normalized_margin"] for r in res["issues"]]
        assert margins == sorted(margins) and margins[0] < 0
        top = res["issues"][0]
        assert top["index"] >= 1000 and top["suggested_label"] != top["given_label"]
        assert sum(r["n_issues"] for r in res["per_class"]) == res["n_issues"]
        assert res["noise_rate"] == pytest.approx(flipped.mean(), abs=0.04)
        json.dumps(res)

    def test_array_probs_use_sorted_classes(self) -> None:
        df, probs, _ = self._frame()
        a = DatasetAuditor().label_issues_report(df, "animal", probs.to_numpy())
        b = DatasetAuditor().label_issues_report(df, "animal", probs)
        assert a["n_issues"] == b["n_issues"] and a["confident_joint"] == b["confident_joint"]

    def test_report_validation(self) -> None:
        df, probs, _ = self._frame()
        with pytest.raises(ValueError, match="not found"):
            DatasetAuditor().label_issues_report(df, "nope", probs)
        with pytest.raises(ValueError, match="one row per"):
            DatasetAuditor().label_issues_report(df.iloc[:10], "animal", probs)
        with pytest.raises(ValueError, match="not among classes"):
            DatasetAuditor().label_issues_report(df, "animal", probs[["cat", "dog"]].assign(emu=0.1))


def test_cli_label_issues(tmp_path, capsys) -> None:
    y, P, _, _ = _small(seed=21, n=300, k=3, shift=3.0, flip=0.1)
    data = tmp_path / "data.csv"
    probs = tmp_path / "probs.csv"
    pd.DataFrame({"x": np.arange(len(y)), "label": y}).to_csv(data, index=False)
    pd.DataFrame(P, columns=["0", "1", "2"]).to_csv(probs, index=False)
    assert main(["label-issues", str(data), "--label", "label", "--probs", str(probs)]) == 0
    out = capsys.readouterr().out
    assert "Estimated label noise" in out and "Most suspicious rows" in out
    code = main(
        ["label-issues", str(data), "--label", "label", "--probs", str(probs), "--json", "--fail-on-issues", "--method", "both"]
    )
    res = json.loads(capsys.readouterr().out)
    assert code == 1 and res["method"] == "both" and res["n_issues"] > 0
    assert main(["label-issues", str(data), "--label", "missing", "--probs", str(probs)]) == 2

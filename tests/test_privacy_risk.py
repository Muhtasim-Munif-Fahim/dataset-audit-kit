"""Tests for the privacy audit (t_closeness_distance / privacy_risk_report / privacy CLI)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from dataset_audit_kit.cli import main
from dataset_audit_kit.core import DatasetAuditor


def _patients() -> pd.DataFrame:
    # Li, Li & Venkatasubramanian (2007), Table 3: a 3-diverse table whose
    # first class has only low salaries (3K, 4K, 5K).
    return pd.DataFrame(
        {
            "zip": ["4767*"] * 3 + ["4790*"] * 3 + ["4760*"] * 3,
            "age": ["<=40"] * 3 + [">=40"] * 3 + ["<=40"] * 3,
            "salary": [3, 4, 5, 6, 11, 8, 7, 9, 10],
            "disease": [
                "gastric ulcer", "gastritis", "stomach cancer",
                "gastritis", "flu", "bronchitis",
                "bronchitis", "pneumonia", "stomach cancer",
            ],
        }
    )


class TestTClosenessDistance:
    def test_matches_paper_examples(self) -> None:
        q = np.ones(9)
        p1 = np.r_[1, 1, 1, np.zeros(6)]
        p2 = np.zeros(9)
        p2[[3, 5, 8]] = 1  # {6K, 8K, 11K}
        assert DatasetAuditor.t_closeness_distance(p1, q, ordered=True) == pytest.approx(0.375)
        assert DatasetAuditor.t_closeness_distance(p2, q, ordered=True) == pytest.approx(1 / 6)

    def test_ordered_emd_equals_wasserstein_over_rank_gaps(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            m = int(rng.integers(2, 9))
            p, q = rng.random(m), rng.random(m)
            p, q = p / p.sum(), q / q.sum()
            # 1-Wasserstein on the support {0, 1/(m-1), ..., 1} via quantile functions.
            grid = np.linspace(0, 1, 2001)[1:-1]
            pos = np.arange(m) / (m - 1)
            qp = pos[np.searchsorted(np.cumsum(p), grid)]
            qq = pos[np.searchsorted(np.cumsum(q), grid)]
            ref = float(np.mean(np.abs(qp - qq)))
            got = DatasetAuditor.t_closeness_distance(p, q, ordered=True)
            assert got == pytest.approx(ref, abs=5e-3)

    def test_equal_distance_is_total_variation(self) -> None:
        p, q = np.array([0.5, 0.5, 0.0]), np.array([0.2, 0.3, 0.5])
        assert DatasetAuditor.t_closeness_distance(p, q) == pytest.approx(0.5)
        assert DatasetAuditor.t_closeness_distance(q, q) == 0.0
        assert DatasetAuditor.t_closeness_distance([2, 0], [0, 1]) == pytest.approx(1.0)

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            DatasetAuditor.t_closeness_distance([1, 0], [1, 0, 0])
        with pytest.raises(ValueError):
            DatasetAuditor.t_closeness_distance([-1, 2], [1, 1])
        with pytest.raises(ValueError):
            DatasetAuditor.t_closeness_distance([0, 0], [1, 1])


class TestPrivacyRiskReport:
    def test_paper_table(self) -> None:
        res = DatasetAuditor().privacy_risk_report(
            _patients(), ["zip", "age"], sensitive=["salary", "disease"], k=3, l=3, t=0.2
        )
        assert res["n_classes"] == 3 and res["k_anonymity"] == 3 and res["satisfies_k"]
        salary, disease = res["sensitive"]
        assert salary["kind"] == "numeric" and disease["kind"] == "categorical"
        assert salary["distinct_l"] == 3 and salary["satisfies_l"]
        assert salary["entropy_l"] == pytest.approx(3.0)
        # The low-salary class is 0.375 away from the overall distribution.
        assert salary["t_closeness"] == pytest.approx(0.375)
        # {7K, 9K, 10K} is 0.2361 away and also breaks t = 0.2; {6K, 8K, 11K} is 1/6.
        assert not salary["satisfies_t"] and salary["classes_violating_t"] == 2
        assert disease["distinct_l"] == 3 and disease["satisfies_l"]
        assert not res["satisfies_all"]
        assert res["prosecutor_risk_max"] == pytest.approx(1 / 3)

    def test_brute_force_k_and_l(self) -> None:
        rng = np.random.default_rng(4)
        df = pd.DataFrame(
            {
                "a": rng.integers(0, 4, 300),
                "b": rng.choice(["x", "y", "z"], 300),
                "s": rng.choice(["p", "q", "r", "s"], 300, p=[0.7, 0.1, 0.1, 0.1]),
            }
        )
        res = DatasetAuditor().privacy_risk_report(df, ["a", "b"], sensitive="s", k=30, l=3, top=100)
        groups = df.groupby(["a", "b"])
        sizes = groups.size()
        assert res["k_anonymity"] == sizes.min()
        assert res["n_classes"] == len(sizes)
        assert res["n_violating_rows"] == int(sizes[sizes < 30].sum())
        assert res["prosecutor_risk_mean"] == pytest.approx(len(sizes) / 300, abs=1e-6)
        assert res["sensitive"][0]["distinct_l"] == groups["s"].nunique().min()
        ent = groups["s"].apply(
            lambda v: math.exp(-sum(p * math.log(p) for p in v.value_counts(normalize=True)))
        )
        assert res["sensitive"][0]["entropy_l"] == pytest.approx(ent.min(), abs=1e-6)
        overall = df["s"].value_counts(normalize=True)
        tv = groups["s"].apply(
            lambda v: 0.5 * (v.value_counts(normalize=True).reindex(overall.index, fill_value=0) - overall).abs().sum()
        )
        assert res["sensitive"][0]["t_closeness"] == pytest.approx(tv.max(), abs=1e-6)
        assert [c["size"] for c in res["classes"]] == sorted(sizes.tolist())

    def test_unique_rows_and_missing_qi(self) -> None:
        df = pd.DataFrame({"zip": ["1", "1", None, None, "2"], "s": ["a", "b", "a", None, "c"]})
        res = DatasetAuditor().privacy_risk_report(df, ["zip"], sensitive="s", k=2, top=5)
        assert res["n_classes"] == 3  # missing zip forms its own class
        assert res["k_anonymity"] == 1 and res["n_unique_rows"] == 1
        worst = res["classes"][0]
        assert worst["quasi_identifiers"] == {"zip": "2"} and worst["violates_k"]
        missing_cls = [c for c in res["classes"] if c["quasi_identifiers"]["zip"] is None][0]
        assert missing_cls["size"] == 2 and missing_cls["s"]["distinct_l"] == 1

    def test_well_mixed_table_satisfies_everything(self) -> None:
        df = pd.DataFrame({"g": ["a", "b"] * 48, "s": np.tile(np.repeat(np.arange(4), 2), 12)})
        res = DatasetAuditor().privacy_risk_report(df, "g", sensitive="s", k=10, l=2, t=0.2)
        assert res["satisfies_all"] and res["sensitive"][0]["t_closeness"] == pytest.approx(0.0, abs=1e-12)

    def test_validation(self) -> None:
        df = _patients()
        a = DatasetAuditor()
        with pytest.raises(ValueError):
            a.privacy_risk_report(df, [])
        with pytest.raises(ValueError):
            a.privacy_risk_report(df, ["nope"])
        with pytest.raises(ValueError):
            a.privacy_risk_report(df, ["zip"], sensitive="zip")
        with pytest.raises(ValueError):
            a.privacy_risk_report(df, ["zip"], k=0)
        with pytest.raises(ValueError):
            a.privacy_risk_report(df, ["zip"], l=0.5)
        with pytest.raises(ValueError):
            a.privacy_risk_report(df, ["zip"], t=1.5)


class TestPrivacyCli:
    def test_json_and_exit_codes(self, tmp_path, capsys) -> None:
        path = tmp_path / "p.csv"
        _patients().to_csv(path, index=False)
        code = main(["privacy", str(path), "--qi", "zip,age", "--sensitive", "salary", "--k", "3", "--l", "3", "--json"])
        assert code == 0
        out = json.loads(capsys.readouterr().out)
        assert out["k_anonymity"] == 3 and out["sensitive"][0]["t_closeness"] == pytest.approx(0.375)
        code = main(["privacy", str(path), "--qi", "zip,age", "--sensitive", "salary", "--k", "3", "--fail-on-violation"])
        assert code == 1
        text = capsys.readouterr().out
        assert "k-anonymity: 3" in text and "Riskiest equivalence classes" in text
        assert main(["privacy", str(path), "--qi", "zip", "--k", "2", "--fail-on-violation"]) == 0

    def test_bad_column(self, tmp_path, capsys) -> None:
        path = tmp_path / "p.csv"
        _patients().to_csv(path, index=False)
        assert main(["privacy", str(path), "--qi", "missing"]) == 2
        assert "unknown columns" in capsys.readouterr().err

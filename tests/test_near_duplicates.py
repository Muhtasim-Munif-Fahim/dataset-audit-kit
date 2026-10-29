"""Tests for MinHash-LSH near-duplicate detection (near_duplicates_report / near-dups CLI)."""

from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from dataset_audit_kit.cli import main
from dataset_audit_kit.core import DatasetAuditor


def _customers(n=400, n_dups=30, seed=0):
    rng = np.random.default_rng(seed)
    cities = ["dhaka", "rajshahi", "khulna", "sylhet", "chittagong", "barisal"]
    df = pd.DataFrame(
        {
            "id_name": [f"user{i}" for i in range(n)],
            "email": [f"user{i}@mail.com" for i in range(n)],
            "city": rng.choice(cities, n),
            "age": rng.integers(18, 80, n),
            "plan": rng.choice(["free", "pro", "team"], n),
            "zip": rng.integers(1000, 9999, n),
            "phone": rng.integers(10**9, 10**10 - 1, n),
            "segment": rng.choice(list("ABCDEFGH"), n),
            "score": rng.integers(0, 1000, n),
            "ref": rng.integers(0, 10**6, n),
        }
    )
    src = rng.choice(n, n_dups, replace=False)
    dups = df.iloc[src].copy()
    # change exactly one of 10 fields -> Jaccard 9/11 ~= 0.818
    dups["score"] = dups["score"] + 1
    dups.index = range(n, n + n_dups)
    return pd.concat([df, dups]), {(int(s), n + k) for k, s in enumerate(src)}


def _brute_force(sets, threshold):
    out = set()
    for i, j in combinations(range(len(sets)), 2):
        a, b = sets[i], sets[j]
        if a and b and len(a & b) / len(a | b) >= threshold:
            out.add((i, j))
    return out


class TestLshParameters:
    def test_layout_fits_budget_and_s_curve_threshold(self) -> None:
        for t in (0.5, 0.7, 0.8, 0.9):
            b, r = DatasetAuditor.lsh_parameters(t, 128)
            assert b * r <= 128
            # the S-curve inflection (1/b)**(1/r) sits near the threshold
            assert (1 / b) ** (1 / r) == pytest.approx(t, abs=0.12)
        b1, r1 = DatasetAuditor.lsh_parameters(0.5, 128)
        b2, r2 = DatasetAuditor.lsh_parameters(0.9, 128)
        assert r2 > r1

    def test_recall_weight_raises_candidate_probability_at_threshold(self) -> None:
        prob = lambda b, r, s: 1 - (1 - s**r) ** b  # noqa: E731
        b0, r0 = DatasetAuditor.lsh_parameters(0.8, 128)
        b1, r1 = DatasetAuditor.lsh_parameters(0.8, 128, false_negative_weight=0.9)
        assert prob(b1, r1, 0.8) > prob(b0, r0, 0.8)

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            DatasetAuditor.lsh_parameters(0.5, 128, false_negative_weight=1.5)
        with pytest.raises(ValueError):
            DatasetAuditor.lsh_parameters(1.0, 128)
        with pytest.raises(ValueError):
            DatasetAuditor.lsh_parameters(0.5, 1)


class TestMinHash:
    def test_signature_agreement_estimates_jaccard(self) -> None:
        a = {f"t{i}" for i in range(100)}
        b = {f"t{i}" for i in range(50, 150)}  # J = 50/150
        sig = DatasetAuditor.minhash_signatures([a, b, a, set()], num_perm=2048, seed=1)
        assert sig.shape == (4, 2048)
        assert np.mean(sig[0] == sig[1]) == pytest.approx(1 / 3, abs=0.04)
        assert np.array_equal(sig[0], sig[2])
        assert np.all(sig[3] == np.iinfo(np.uint64).max)
        again = DatasetAuditor.minhash_signatures([a], num_perm=16, seed=1)
        assert np.array_equal(again[0], DatasetAuditor.minhash_signatures([a], num_perm=16, seed=1)[0])


class TestNearDuplicatesReport:
    def test_finds_planted_one_field_edits(self) -> None:
        df, planted = _customers()
        res = DatasetAuditor().near_duplicates_report(df, threshold=0.75)
        res_all = DatasetAuditor().near_duplicates_report(df, threshold=0.75, top=1000)
        found = {tuple(sorted((p["left"], p["right"]))) for p in res_all["pairs"]}
        assert len(found & planted) >= 0.95 * len(planted)
        assert found <= planted  # verified with exact Jaccard: no false positives
        top = res_all["pairs"][0]
        assert top["jaccard"] == pytest.approx(9 / 11, abs=1e-6)
        assert top["differing_columns"] == ["score"]
        assert res_all["n_candidate_pairs"] < 0.01 * len(df) ** 2 / 2
        assert res_all["n_rows_in_clusters"] == 2 * res_all["n_clusters"]
        assert len(res["pairs"]) == 20

    def test_lsh_matches_exact_method(self) -> None:
        df, _ = _customers(150, 15, seed=2)
        exact = DatasetAuditor().near_duplicates_report(df, threshold=0.75, method="exact", top=10_000)
        lsh = DatasetAuditor().near_duplicates_report(df, threshold=0.75, num_perm=256, top=10_000)
        to_set = lambda r: {tuple(sorted((p["left"], p["right"]))) for p in r["pairs"]}  # noqa: E731
        assert to_set(lsh) <= to_set(exact)
        assert len(to_set(lsh)) >= 0.9 * len(to_set(exact))
        sets = DatasetAuditor._row_token_sets(df.reset_index(drop=True), list(df.columns), [], 4)
        assert exact["n_near_duplicate_pairs"] == len(_brute_force(sets, 0.75))

    def test_text_columns_are_shingled(self) -> None:
        base = "the quick brown fox jumps over the lazy dog near the river bank"
        df = pd.DataFrame(
            {
                "review": [base, base.replace("lazy", "lazzy"), "completely different words about databases and audits",
                           "another unrelated sentence on survival models and kits"],
                "stars": [5, 5, 1, 3],
            }
        )
        res = DatasetAuditor().near_duplicates_report(df, threshold=0.7, method="exact")
        assert res["text_columns"] == ["review"]
        assert res["n_near_duplicate_pairs"] == 1
        assert {res["pairs"][0]["left"], res["pairs"][0]["right"]} == {0, 1}
        assert res["pairs"][0]["differing_columns"] == ["review"]
        none = DatasetAuditor().near_duplicates_report(df, threshold=0.7, text_columns=[], method="exact")
        assert none["n_near_duplicate_pairs"] == 0

    def test_exact_copies_collapse_into_clusters(self) -> None:
        df, _ = _customers(60, 0, seed=3)
        row = df.iloc[[0]]
        edited = row.copy()
        edited["plan"] = "zzz"
        big = pd.concat([df] + [row] * 50 + [edited], ignore_index=True)
        res = DatasetAuditor().near_duplicates_report(big, threshold=0.75)
        assert res["n_distinct_rows"] == 61
        assert res["n_near_duplicate_pairs"] == 1
        assert res["clusters"][0]["size"] == 52
        assert res["clusters"][0]["n_distinct"] == 2

    def test_missing_values_and_column_subset(self) -> None:
        df = pd.DataFrame({"a": [1, 1, None, None], "b": ["x", "x", None, None], "c": [1, 2, 3, 4]})
        res = DatasetAuditor().near_duplicates_report(df, columns=["a", "b"], threshold=0.5, method="exact")
        assert res["n_distinct_rows"] == 2
        assert res["n_near_duplicate_pairs"] == 0  # identical rows and all-missing rows are not "near"

    def test_validation(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        auditor = DatasetAuditor()
        for kwargs in ({"threshold": 1.0}, {"threshold": 0}, {"method": "x"}, {"columns": ["zz"]},
                       {"shingle_size": 0}, {"top": -1}, {"columns": []}, {"text_columns": ["q"]}):
            with pytest.raises(ValueError):
                auditor.near_duplicates_report(df, **kwargs)


class TestNearDupsCli:
    def test_text_json_and_exit_codes(self, tmp_path, capsys) -> None:
        df, _ = _customers(120, 5, seed=4)
        path = tmp_path / "d.csv"
        df.to_csv(path, index=False)
        assert main(["near-dups", str(path), "--threshold", "0.75"]) == 0
        out = capsys.readouterr().out
        assert "near-duplicate pair(s)" in out and "score" in out
        assert main(["near-dups", str(path), "--threshold", "0.75", "--json", "--exact"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["method"] == "exact" and payload["n_near_duplicate_pairs"] == 5
        assert main(["near-dups", str(path), "--threshold", "0.75", "--fail-on-duplicates"]) == 1
        capsys.readouterr()
        assert main(["near-dups", str(path), "--threshold", "1.5"]) == 2

"""Tests for the command-line interface."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from dataset_audit_kit import __version__
from dataset_audit_kit.cli import main


class TestVersion:
    def test_version_without_a_subcommand(self, capsys) -> None:
        assert main(["--version"]) == 0
        assert __version__ in capsys.readouterr().out

    def test_bare_invocation_explains_itself(self, capsys) -> None:
        assert main([]) == 2
        assert "a command is required" in capsys.readouterr().err


class TestAuditOutput:
    def test_json_output_stays_parseable(self, tmp_path, clean_csv, capsys) -> None:
        saved = tmp_path / "report.json"
        code = main(
            [
                "audit",
                clean_csv,
                "--json",
                "--fix-suggestions",
                "--save-json",
                str(saved),
                "--html-out",
                str(tmp_path / "report.html"),
            ]
        )
        assert code == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["status"] == "pass"
        assert "fix_suggestions" in payload
        # The chatter about written files belongs on stderr in this mode.
        assert "report saved to" in captured.err

    def test_markdown_mode_keeps_its_notices_on_stdout(self, tmp_path, clean_csv, capsys) -> None:
        main(["audit", clean_csv, "--save-json", str(tmp_path / "r.json")])
        assert "JSON report saved to" in capsys.readouterr().out

    def test_save_json_writes_json_whatever_the_extension(self, tmp_path, clean_csv) -> None:
        destination = tmp_path / "report.txt"
        assert main(["audit", clean_csv, "--save-json", str(destination)]) == 0
        assert json.loads(destination.read_text())["status"] == "pass"

    def test_save_creates_missing_directories(self, tmp_path, clean_csv) -> None:
        destination = tmp_path / "out" / "nested" / "report.md"
        assert main(["audit", clean_csv, "--save-markdown", str(destination)]) == 0
        assert destination.read_text().startswith("# Dataset Audit Report")

    def test_sarif_output_is_machine_readable(self, tmp_path, clean_csv) -> None:
        destination = tmp_path / "reports" / "audit.sarif"
        assert main(["audit", clean_csv, "--sarif-out", str(destination)]) == 0
        payload = json.loads(destination.read_text(encoding="utf-8"))
        assert payload["version"] == "2.1.0"
        assert payload["runs"][0]["tool"]["driver"]["name"] == "dataset-audit-kit"

    def test_csv_output_is_machine_readable(self, tmp_path, clean_csv) -> None:
        destination = tmp_path / "reports" / "audit.csv"
        assert main(["audit", clean_csv, "--csv-out", str(destination)]) == 0
        assert destination.read_text(encoding="utf-8").splitlines()[0] == (
            "severity,check,column,message,observed,threshold"
        )

    def test_unwritable_destination_reports_cleanly(self, tmp_path, clean_csv, capsys) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        code = main(["audit", clean_csv, "--save-json", str(blocker / "r.json")])
        assert code == 2
        assert "Cannot write report" in capsys.readouterr().err


class TestWhitespaceFlag:
    def test_padding_fails_only_when_enabled(self, tmp_path, capsys) -> None:
        path = tmp_path / "padded.csv"
        pd.DataFrame({"name": [" ann", "bo"]}).to_csv(path, index=False)
        assert main(["audit", str(path), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert not [i for i in payload["issues"] if i["check"] == "whitespace"]
        assert main(["audit", str(path), "--json", "--check-whitespace"]) == 1
        payload = json.loads(capsys.readouterr().out)
        assert [i for i in payload["issues"] if i["check"] == "whitespace"]

    def test_check_gate_counts_whitespace_findings(self, tmp_path, capsys) -> None:
        path = tmp_path / "invisible.csv"
        pd.DataFrame({"name": ["\u200bann", "bo"]}).to_csv(path, index=False)
        assert main(["check", str(path)]) == 0
        assert main(["check", str(path), "--check-whitespace"]) == 1
        assert "[FAIL]" in capsys.readouterr().out


class TestSensitiveFlag:
    def test_emails_fail_only_when_enabled(self, tmp_path, capsys) -> None:
        path = tmp_path / "contacts.csv"
        pd.DataFrame({"contact": ["alice@example.com", "bo"]}).to_csv(path, index=False)
        assert main(["audit", str(path), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert not [i for i in payload["issues"] if i["check"] == "sensitive"]
        assert main(["audit", str(path), "--json", "--check-sensitive"]) == 1
        payload = json.loads(capsys.readouterr().out)
        sensitive = [i for i in payload["issues"] if i["check"] == "sensitive"]
        assert sensitive and sensitive[0]["message"].startswith("1 value(s) match email")

    def test_check_gate_counts_sensitive_findings(self, tmp_path, capsys) -> None:
        path = tmp_path / "contacts.csv"
        pd.DataFrame({"contact": ["123-45-6789", "bo"]}).to_csv(path, index=False)
        assert main(["check", str(path)]) == 0
        assert main(["check", str(path), "--check-sensitive"]) == 1
        assert "[FAIL]" in capsys.readouterr().out

    def test_sensitive_flag_is_part_of_the_report_fingerprint(
        self, tmp_path, capsys
    ) -> None:
        path = tmp_path / "contacts.csv"
        pd.DataFrame({"contact": ["alice@example.com"]}).to_csv(path, index=False)
        first = tmp_path / "first.json"
        second = tmp_path / "second.json"
        assert main(["audit", str(path), "--save-json", str(first)]) == 0
        assert main(
            ["audit", str(path), "--check-sensitive", "--save-json", str(second)]
        ) == 1
        baseline = json.loads(first.read_text(encoding="utf-8"))
        enabled = json.loads(second.read_text(encoding="utf-8"))
        assert baseline["meta"]["config_hash"] != enabled["meta"]["config_hash"]


class TestCategoryShareCli:
    @pytest.fixture
    def dominant_csv(self, tmp_path):
        # A distinct id column keeps duplicate-row warnings out of the picture.
        path = tmp_path / "dominant.csv"
        pd.DataFrame(
            {"id": list(range(10)), "status": ["ok"] * 9 + ["no"]}
        ).to_csv(path, index=False)
        return str(path)

    def test_audit_flags_dominance_only_when_configured(
        self, dominant_csv, capsys
    ) -> None:
        assert main(["audit", dominant_csv, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert not [i for i in payload["issues"] if i["check"] == "category_share"]

        assert main(
            ["audit", dominant_csv, "--json", "--max-category-share", "0.8"]
        ) == 1
        payload = json.loads(capsys.readouterr().out)
        assert [i for i in payload["issues"] if i["check"] == "category_share"]

    def test_rare_category_share_flag_is_wired_through(self, tmp_path, capsys) -> None:
        path = tmp_path / "sparse.csv"
        pd.DataFrame(
            {"id": list(range(10)), "status": ["a"] * 8 + ["b"] + ["c"]}
        ).to_csv(path, index=False)

        assert main(
            ["audit", str(path), "--json", "--rare-category-share", "0.15"]
        ) == 1
        payload = json.loads(capsys.readouterr().out)
        assert [i for i in payload["issues"] if i["check"] == "category_share"]

    def test_check_gate_counts_category_share_findings(
        self, dominant_csv, capsys
    ) -> None:
        assert main(
            ["check", dominant_csv, "--max-category-share", "0.8", "--minimal"]
        ) == 1
        assert "[FAIL]" in capsys.readouterr().out

    def test_out_of_range_share_is_a_usage_error(self, tmp_path, capsys) -> None:
        path = tmp_path / "x.csv"
        pd.DataFrame({"status": ["ok"]}).to_csv(path, index=False)

        with pytest.raises(SystemExit) as exc:
            main(["audit", str(path), "--max-category-share", "1.5"])
        assert exc.value.code == 2
        assert "between 0 and 1" in capsys.readouterr().err


class TestExitCodes:
    def test_informational_findings_do_not_fail_the_audit(self, tmp_path, capsys) -> None:
        current = tmp_path / "cur.csv"
        reference = tmp_path / "ref.csv"
        pd.DataFrame({"a": [1, 2, 3], "note": ["x", "y", "z"]}).to_csv(current, index=False)
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(reference, index=False)

        code = main(["audit", str(current), "--reference", str(reference), "--minimal"])
        out = capsys.readouterr().out
        assert "[PASS]" in out
        assert code == 0

    def test_real_problems_still_fail(self, tmp_path, capsys) -> None:
        path = tmp_path / "dupes.csv"
        pd.DataFrame({"a": [1, 1], "b": [2, 2]}).to_csv(path, index=False)
        assert main(["audit", str(path), "--minimal"]) == 1
        assert "[FAIL]" in capsys.readouterr().out

    def test_error_gate_allows_warning_only_audit(self, tmp_path, capsys) -> None:
        path = tmp_path / "missing.csv"
        pd.DataFrame({"a": [1, None]}).to_csv(path, index=False)
        assert main(["audit", str(path), "--fail-on", "error", "--minimal"]) == 0
        assert "below threshold" in capsys.readouterr().out

    def test_check_error_gate_still_fails_errors(self, tmp_path, capsys) -> None:
        path = tmp_path / "dupes.csv"
        rules = tmp_path / "rules.json"
        pd.DataFrame({"a": ["one", "two"]}).to_csv(path, index=False)
        rules.write_text('{"a": {"dtype": "numeric"}}', encoding="utf-8")
        assert main(
            ["check", str(path), "--rules", str(rules), "--fail-on", "error", "--minimal"]
        ) == 1
        assert "[FAIL]" in capsys.readouterr().out

    def test_check_ignores_informational_findings(self, tmp_path, clean_csv, capsys) -> None:
        assert main(["check", clean_csv]) == 0
        assert "[PASS]" in capsys.readouterr().out


class TestReaderOptions:
    def test_check_honours_encoding(self, tmp_path, capsys) -> None:
        path = tmp_path / "latin.csv"
        pd.DataFrame({"n": ["café", "b"], "v": [1, 2]}).to_csv(
            path, index=False, encoding="latin-1"
        )
        assert main(["check", str(path), "--encoding", "latin-1"]) == 0
        assert "2 rows" in capsys.readouterr().out

    def test_audit_honours_delimiter(self, tmp_path, capsys) -> None:
        path = tmp_path / "semi.csv"
        path.write_text("a;b\n1;2\n3;4\n", encoding="utf-8")
        main(["audit", str(path), "--delimiter", ";", "--json"])
        assert json.loads(capsys.readouterr().out)["columns"] == 2


class TestHist:
    def test_zero_bins_is_a_clean_error(self, clean_csv, capsys) -> None:
        assert main(["hist", clean_csv, "--column", "id", "--bins", "0"]) == 2
        assert "--bins must be at least 1" in capsys.readouterr().err

    def test_all_missing_column_is_a_clean_error(self, tmp_path, capsys) -> None:
        path = tmp_path / "empty.csv"
        pd.DataFrame({"n": [None, None]}).to_csv(path, index=False)
        assert main(["hist", str(path), "--column", "n", "--bins", "4"]) == 1
        assert "no non-missing values" in capsys.readouterr().err

    def test_histogram_renders(self, clean_csv, capsys) -> None:
        assert main(["hist", clean_csv, "--column", "id", "--bins", "2"]) == 0
        assert "Histogram for 'id'" in capsys.readouterr().out


class TestOtherSubcommands:
    def test_infer_rules_writes_a_reusable_contract(self, clean_csv, capsys) -> None:
        assert main(["infer-rules", clean_csv, "--max-categories", "5"]) == 0
        rules = json.loads(capsys.readouterr().out)
        assert rules["id"]["dtype"] == "numeric"
        assert rules["label"]["allowed_values"] == ["no", "yes"]

    @pytest.mark.parametrize(
        "argv",
        [
            ["columns"],
            ["shape"],
            ["info"],
            ["stats"],
            ["missing"],
            ["describe"],
            ["head", "--rows", "2"],
            ["tail", "--rows", "2"],
            ["schema"],
        ],
    )
    def test_subcommand_runs(self, clean_csv, argv, capsys) -> None:
        assert main([argv[0], clean_csv, *argv[1:]]) == 0
        assert capsys.readouterr().out
class TestSampledCliAudit:
    def test_sample_flag_reports_the_sampled_row_count(
        self, tmp_path, clean_frame, capsys
    ) -> None:
        big = clean_frame.iloc[[i % 4 for i in range(40)]].reset_index(drop=True)
        big["id"] = range(40)
        path = tmp_path / "big.csv"
        big.to_csv(path, index=False)
        code = main(["audit", str(path), "--json", "--sample-rows", "10", "--seed", "5"])
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["rows"] == 10
class TestGlobRollup:
    @pytest.fixture
    def mixed_dir(self, tmp_path, clean_frame):
        clean = tmp_path / "good.csv"
        clean_frame.to_csv(clean, index=False)
        dirty = tmp_path / "bad.csv"
        pd.DataFrame(
            {
                "a": [1.0, None, None],
                "b": [4.0, 5.0, 6.0],
            }
        ).to_csv(dirty, index=False)
        return tmp_path

    def test_rollup_lists_each_file_and_the_worst_checks(self, mixed_dir, capsys) -> None:
        code = main(["audit-glob", str(mixed_dir / "*.csv")])
        assert code == 1
        out = capsys.readouterr().out
        assert "[PASS] 0 issue(s)" in out
        assert "[FAIL]" in out
        assert "rollup: 2 file(s), 1 failed" in out
        assert "worst checks: missingness=1" in out

    def test_a_clean_directory_exits_zero(self, tmp_path, clean_frame, capsys) -> None:
        (tmp_path / "only.csv").write_text("x\ny\n")
        assert main(["audit-glob", str(tmp_path / "*.csv")]) == 0
        assert "rollup: 1 file(s), 0 failed" in capsys.readouterr().out

    def test_no_matches_report_cleanly(self, tmp_path, capsys) -> None:
        assert main(["audit-glob", str(tmp_path / "*.parquet")]) == 2
        assert "No files match" in capsys.readouterr().err

    def test_json_mode_stays_parseable(self, mixed_dir, capsys) -> None:
        assert main(["audit-glob", str(mixed_dir / "*.csv"), "--json"]) == 1
        payload = json.loads(capsys.readouterr().out)
        assert set(payload["files"]) == {str(mixed_dir / "bad.csv"), str(mixed_dir / "good.csv")}


class TestRiskGate:
    def test_check_tolerates_warnings_until_they_pile_up(self, tmp_path, capsys) -> None:
        path = tmp_path / "gaps.csv"
        pd.DataFrame({"a": [1.0, None], "b": [2.0, None]}).to_csv(path, index=False)
        base = ["check", str(path), "--fail-on", "error"]
        assert main(base + ["--minimal"]) == 0
        assert main(base + ["--minimal", "--max-risk", "9"]) == 1
        assert "risk score" in capsys.readouterr().out
        assert main(base + ["--minimal", "--max-risk", "10"]) == 0

    def test_audit_applies_the_same_ceiling(self, tmp_path) -> None:
        path = tmp_path / "gaps.csv"
        pd.DataFrame({"a": [1.0, None]}).to_csv(path, index=False)
        flags = ["--fail-on", "error", "--minimal"]
        assert main(["audit", str(path), *flags, "--max-risk", "4"]) == 1
        assert main(["audit", str(path), *flags, "--max-risk", "5"]) == 0

    def test_audit_glob_marks_files_over_the_ceiling(self, tmp_path, capsys) -> None:
        pd.DataFrame({"a": [1.0, None]}).to_csv(tmp_path / "a.csv", index=False)
        pd.DataFrame({"a": [1.0, 2.0]}).to_csv(tmp_path / "b.csv", index=False)
        code = main(
            [
                "audit-glob",
                str(tmp_path / "*.csv"),
                "--fail-on",
                "error",
                "--max-risk",
                "4",
            ]
        )
        out = capsys.readouterr().out
        assert "[FAIL]" in out and "[PASS]" in out and "risk" in out
        assert code == 1

    def test_a_negative_ceiling_is_refused_as_a_bad_argument(self, tmp_path, capsys) -> None:
        path = tmp_path / "gaps.csv"
        pd.DataFrame({"a": [1.0, 2.0]}).to_csv(path, index=False)
        with pytest.raises(SystemExit) as excinfo:
            main(["check", str(path), "--max-risk", "-1"])
        assert excinfo.value.code == 2
        assert "non-negative" in capsys.readouterr().err
class TestNamedProfilesCli:
    @pytest.fixture
    def profiled_rules(self, tmp_path):
        path = tmp_path / "rules.json"
        path.write_text(
            json.dumps(
                {
                    "profiles": {
                        "uppercase_only": {"name": {"allowed_values": ["ANN", "BO"]}},
                        "any_case": {
                            "name": {"allowed_values": ["ann", "bo"], "ignore_case": True}
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        return str(path)

    @pytest.fixture
    def mixed_csv(self, tmp_path):
        path = tmp_path / "people.csv"
        pd.DataFrame({"name": ["ann", "BO"]}).to_csv(path, index=False)
        return str(path)

    def test_audit_enforces_whichever_profile_is_selected(
        self, mixed_csv, profiled_rules, capsys
    ) -> None:
        assert main(
            ["audit", mixed_csv, "--json", "--rules", profiled_rules, "--profile", "uppercase_only"]
        ) == 1
        payload = json.loads(capsys.readouterr().out)
        assert [i for i in payload["issues"] if i["check"] == "rule"]

        assert main(
            ["audit", mixed_csv, "--json", "--rules", profiled_rules, "--profile", "any_case"]
        ) == 0

    def test_omitting_the_profile_on_a_profiles_file_is_a_usage_error(
        self, mixed_csv, profiled_rules, capsys
    ) -> None:
        assert main(["audit", mixed_csv, "--rules", profiled_rules]) == 2
        assert "any_case, uppercase_only" in capsys.readouterr().err

    def test_unknown_profile_is_a_usage_error(
        self, mixed_csv, profiled_rules, capsys
    ) -> None:
        assert main(
            ["audit", mixed_csv, "--rules", profiled_rules, "--profile", "nope"]
        ) == 2
        assert "'nope' not found" in capsys.readouterr().err

    def test_check_honours_profiles_too(
        self, mixed_csv, profiled_rules, capsys
    ) -> None:
        assert main(["check", mixed_csv, "--rules", profiled_rules, "--profile", "any_case"]) == 0
        assert main(["check", mixed_csv, "--rules", profiled_rules, "--profile", "uppercase_only"]) == 1
        assert "[FAIL]" in capsys.readouterr().out

    def test_audit_glob_rejects_an_unknown_profile(
        self, tmp_path, clean_frame, profiled_rules, capsys
    ) -> None:
        clean_frame.to_csv(tmp_path / "one.csv", index=False)
        assert main(
            [
                "audit-glob",
                str(tmp_path / "*.csv"),
                "--rules",
                profiled_rules,
                "--profile",
                "nope",
            ]
        ) == 2
        assert "'nope' not found" in capsys.readouterr().err
class TestNumericBoundModesCli:
    def test_exclusive_bounds_from_a_rules_file_change_the_gate(self, tmp_path, capsys) -> None:
        path = tmp_path / "scores.csv"
        pd.DataFrame({"score": [0.0, 1.0, 2.0]}).to_csv(path, index=False)

        inclusive = tmp_path / "inclusive.json"
        inclusive.write_text('{"score": {"min_value": 0.0}}', encoding="utf-8")
        assert main(["audit", str(path), "--json", "--rules", str(inclusive)]) == 0
        capsys.readouterr()

        exclusive = tmp_path / "exclusive.json"
        exclusive.write_text(
            '{"score": {"min_value": 0.0, "min_inclusive": false}}',
            encoding="utf-8",
        )
        assert main(["audit", str(path), "--json", "--rules", str(exclusive)]) == 1
        payload = json.loads(capsys.readouterr().out)
        rule_issues = [i for i in payload["issues"] if i["check"] == "rule"]
        assert any("exclusive bound" in i["message"] for i in rule_issues)

    def test_validate_config_accepts_the_new_options(self, tmp_path, capsys) -> None:
        rules = tmp_path / "bounds.json"
        rules.write_text(
            json.dumps(
                {
                    "score": {
                        "min_value": 0.0,
                        "max_value": 100.0,
                        "min_inclusive": False,
                        "max_inclusive": False,
                        "value_tolerance": 0.5,
                    }
                }
            ),
            encoding="utf-8",
        )
        assert main(["validate-config", str(rules)]) == 0
        assert "1 column rule(s)" in capsys.readouterr().out


class TestDateRangeRulesCli:
    def test_date_bounds_from_a_rules_file_change_the_gate(self, tmp_path, capsys) -> None:
        path = tmp_path / "events.csv"
        pd.DataFrame({"when": ["2021-06-01", "2022-06-01", "2019-06-01"]}).to_csv(
            path, index=False
        )

        rules = tmp_path / "dates.json"
        rules.write_text(
            json.dumps(
                {"when": {"min_date": "2020-01-01", "max_date": "2020-12-31"}}
            ),
            encoding="utf-8",
        )
        assert main(["check", str(path), "--rules", str(rules), "--minimal"]) == 1
        out = capsys.readouterr()
        assert "[FAIL]" in out.out

    def test_future_dates_rule_flags_out_of_range_values(self, tmp_path, capsys) -> None:
        path = tmp_path / "events.csv"
        pd.DataFrame({"when": ["2020-06-01", "2050-06-01"]}).to_csv(path, index=False)

        rules = tmp_path / "future.json"
        rules.write_text(json.dumps({"when": {"no_future_dates": True}}), encoding="utf-8")
        assert main(["check", str(path), "--rules", str(rules), "--minimal"]) == 1
        out = capsys.readouterr()
        assert "[FAIL]" in out.out

    def test_validate_config_accepts_date_bounds(self, tmp_path, capsys) -> None:
        rules = tmp_path / "dates.json"
        rules.write_text(
            json.dumps(
                {
                    "when": {
                        "min_date": "2020-01-01",
                        "max_date": "2020-12-31",
                        "no_future_dates": True,
                    }
                }
            ),
            encoding="utf-8",
        )
        assert main(["validate-config", str(rules)]) == 0
        assert "1 column rule(s)" in capsys.readouterr().out


class TestValidateConfig:

    @pytest.fixture
    def rules_path(self, tmp_path):
        return tmp_path / "rules.json"

    def test_a_valid_file_reports_its_rule_counts(self, rules_path, capsys) -> None:
        rules_path.write_text(
            json.dumps(
                {
                    "age": {"dtype": "numeric", "min_value": 0},
                    "cross": [{"left": "a", "op": "le", "right": "b"}],
                }
            ),
            encoding="utf-8",
        )
        assert main(["validate-config", str(rules_path)]) == 0
        out = capsys.readouterr().out
        assert "OK" in out
        assert "1 column rule(s)" in out
        assert "1 cross-column rule(s)" in out

    def test_structural_errors_name_the_offending_column(
        self, rules_path, capsys
    ) -> None:
        rules_path.write_text('{"age": {"min_unique": -2}}', encoding="utf-8")
        assert main(["validate-config", str(rules_path)]) == 1
        assert "min_unique for column 'age'" in capsys.readouterr().err

    def test_malformed_json_is_reported_with_its_position(
        self, rules_path, capsys
    ) -> None:
        rules_path.write_text('{"age": ', encoding="utf-8")
        assert main(["validate-config", str(rules_path)]) == 1
        assert "not valid JSON" in capsys.readouterr().err

    def test_a_missing_file_is_a_usage_error(self, tmp_path, capsys) -> None:
        missing = tmp_path / "nope.json"
        assert main(["validate-config", str(missing)]) == 2
        assert "Cannot read rules file" in capsys.readouterr().err

    def test_an_uncompilable_pattern_is_caught_before_any_audit(
        self, rules_path, capsys
    ) -> None:
        rules_path.write_text(
            '{"code": {"pattern": "([a-z]+"}}', encoding="utf-8"
        )
        assert main(["validate-config", str(rules_path)]) == 1
        err = capsys.readouterr().err
        assert "code" in err and "pattern does not compile" in err

    def test_an_invalid_date_format_is_caught(self, rules_path, capsys) -> None:
        rules_path.write_text(
            '{"when": {"date_format": "%Y-%Q"}}', encoding="utf-8"
        )
        assert main(["validate-config", str(rules_path)]) == 1
        assert "invalid date_format" in capsys.readouterr().err

    def test_an_unknown_dtype_is_flagged(self, rules_path, capsys) -> None:
        rules_path.write_text('{"age": {"dtype": "integer"}}', encoding="utf-8")
        assert main(["validate-config", str(rules_path)]) == 1
        err = capsys.readouterr().err
        assert "dtype 'integer'" in err and "numeric, categorical, string" in err

    def test_profiles_are_linted_through_the_same_door(
        self, rules_path, capsys
    ) -> None:
        rules_path.write_text(
            json.dumps(
                {
                    "profiles": {
                        "strict": {"age": {"dtype": "numeric"}},
                        "sloppy": {"name": {"pattern": "("}},
                    }
                }
            ),
            encoding="utf-8",
        )
        assert main(
            ["validate-config", str(rules_path), "--profile", "strict"]
        ) == 0
        assert "profile 'strict'" in capsys.readouterr().out

        assert main(["validate-config", str(rules_path), "--profile", "sloppy"]) == 1
        assert "pattern does not compile" in capsys.readouterr().err

        assert main(["validate-config", str(rules_path)]) == 1
        assert "sloppy, strict" in capsys.readouterr().err

    def test_an_unknown_profile_name_exits_like_a_bad_contract(
        self, rules_path, capsys
    ) -> None:
        rules_path.write_text(
            '{"profiles": {"strict": {"age": {"dtype": "numeric"}}}}',
            encoding="utf-8",
        )
        assert main(
            ["validate-config", str(rules_path), "--profile", "wide"]
        ) == 1
        assert "'wide' not found" in capsys.readouterr().err


class TestEmitConfig:
    def test_emit_config_default_output(self, capsys) -> None:
        assert main(["emit-config"]) == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert "age" in payload
        assert "category" in payload
        assert "code" in payload
        assert "event_date" in payload
        assert "score" in payload
        assert "cross" in payload
        assert len(payload["cross"]) == 2

    def test_emit_config_minimal(self, capsys) -> None:
        assert main(["emit-config", "--minimal"]) == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert "column_name" in payload
        assert payload["column_name"]["dtype"] == "numeric|categorical|string"
        assert "cross" in payload

    def test_emit_config_with_profiles(self, capsys) -> None:
        assert main(["emit-config", "--with-profiles"]) == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert "profiles" in payload
        assert "strict" in payload["profiles"]
        assert "lenient" in payload["profiles"]
        assert "age" in payload["profiles"]["strict"]

    def test_emit_config_writes_to_file(self, tmp_path) -> None:
        dest = tmp_path / "template.json"
        assert main(["emit-config", "--output", str(dest)]) == 0
        assert dest.exists()
        payload = json.loads(dest.read_text(encoding="utf-8"))
        assert "age" in payload
        assert "cross" in payload

    def test_emit_config_output_validates(self, tmp_path) -> None:
        dest = tmp_path / "template.json"
        assert main(["emit-config", "--output", str(dest)]) == 0
        # The emitted template should pass validate-config
        assert main(["validate-config", str(dest)]) == 0

    def test_emit_config_with_profiles_validates(self, tmp_path) -> None:
        dest = tmp_path / "profiles.json"
        assert main(["emit-config", "--with-profiles", "--output", str(dest)]) == 0
        # Each profile should validate
        assert main(["validate-config", str(dest), "--profile", "strict"]) == 0
        assert main(["validate-config", str(dest), "--profile", "lenient"]) == 0

    def test_emit_config_creates_parent_directories(self, tmp_path) -> None:
        dest = tmp_path / "nested" / "dir" / "template.json"
        assert main(["emit-config", "--output", str(dest)]) == 0
        assert dest.exists()

class TestStampedCliReports:
    def test_saved_json_carries_run_metadata(self, tmp_path, clean_csv) -> None:
        destination = tmp_path / "report.json"
        assert main(["audit", clean_csv, "--save-json", str(destination)]) == 0
        meta = json.loads(destination.read_text(encoding="utf-8"))["meta"]
        assert len(meta["audit_id"]) == 32
        assert meta["created_utc"].endswith("+00:00")
        assert len(meta["config_hash"]) == 64

    def test_runs_of_one_config_share_a_hash_but_not_an_id(
        self, tmp_path, clean_csv
    ) -> None:
        hashes = []
        ids = []
        for name in ("first.json", "second.json"):
            destination = tmp_path / name
            assert main(["audit", clean_csv, "--save-json", str(destination)]) == 0
            meta = json.loads(destination.read_text(encoding="utf-8"))["meta"]
            hashes.append(meta["config_hash"])
            ids.append(meta["audit_id"])
        assert hashes[0] == hashes[1]
        assert ids[0] != ids[1]

    def test_changing_a_threshold_changes_the_config_hash(
        self, tmp_path, clean_csv
    ) -> None:
        first = tmp_path / "a.json"
        second = tmp_path / "b.json"
        assert main(["audit", clean_csv, "--save-json", str(first)]) == 0
        assert main(
            ["audit", clean_csv, "--missing-threshold", "0.9", "--save-json", str(second)]
        ) == 0
        meta_a = json.loads(first.read_text(encoding="utf-8"))["meta"]
        meta_b = json.loads(second.read_text(encoding="utf-8"))["meta"]
        assert meta_a["config_hash"] != meta_b["config_hash"]

    def test_rules_file_content_is_part_of_the_fingerprint(
        self, tmp_path, clean_csv
    ) -> None:
        rules = tmp_path / "rules.json"
        rules.write_text('{"id": {"dtype": "numeric"}}', encoding="utf-8")
        first = tmp_path / "one.json"
        second = tmp_path / "two.json"
        assert main(["audit", clean_csv, "--rules", str(rules), "--save-json", str(first)]) == 0
        rules.write_text('{"name": {"min_length": 2}}', encoding="utf-8")
        assert main(["audit", clean_csv, "--rules", str(rules), "--save-json", str(second)]) == 0
        hash_one = json.loads(first.read_text(encoding="utf-8"))["meta"]["config_hash"]
        hash_two = json.loads(second.read_text(encoding="utf-8"))["meta"]["config_hash"]
        assert hash_one != hash_two

    def test_sarif_and_html_outputs_are_stamped(self, tmp_path, clean_csv) -> None:
        sarif = tmp_path / "out" / "report.sarif"
        page = tmp_path / "out" / "report.html"
        assert main(
            ["audit", clean_csv, "--sarif-out", str(sarif), "--html-out", str(page)]
        ) == 0
        properties = json.loads(sarif.read_text(encoding="utf-8"))["runs"][0]["properties"]
        assert properties["auditId"]
        assert properties["configHash"]
        html = page.read_text(encoding="utf-8")
        assert "Audit ID" in html and properties["auditId"] in html

    def test_plain_json_stdout_mode_includes_meta_too(self, clean_csv, capsys) -> None:
        assert main(["audit", clean_csv, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert "meta" in payload

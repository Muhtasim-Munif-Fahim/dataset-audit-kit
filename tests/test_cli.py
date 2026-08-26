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

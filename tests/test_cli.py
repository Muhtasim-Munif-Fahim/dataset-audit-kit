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

    def test_unwritable_destination_reports_cleanly(self, tmp_path, clean_csv, capsys) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        code = main(["audit", clean_csv, "--save-json", str(blocker / "r.json")])
        assert code == 2
        assert "Cannot write report" in capsys.readouterr().err


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

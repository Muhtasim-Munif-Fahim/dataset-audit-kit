"""Command-line entry point for dataset-audit-kit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .core import DatasetAuditor, ValidationRules


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dataset-audit-kit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Audit a dataset file")
    audit.add_argument("data", help="Path to the dataset (.csv, .jsonl, .ndjson, .parquet)")
    audit.add_argument(
        "--reference",
        help="Path to the reference dataset (.csv, .jsonl, .ndjson, .parquet)",
        default=None,
    )
    audit.add_argument("--label-column", help="Name of the label column", default=None)
    audit.add_argument(
        "--expected-columns",
        help="Comma-separated list of expected columns",
        default=None,
    )
    audit.add_argument(
        "--missing-threshold",
        type=float,
        default=0.05,
        help="Fraction of missing values allowed before warning",
    )
    audit.add_argument(
        "--drift-threshold",
        type=float,
        default=0.20,
        help="Drift score threshold before warning",
    )
    audit.add_argument(
        "--rules",
        help="Path to a JSON file with per-column validation rules",
        default=None,
    )
    audit.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of Markdown",
    )
    audit.add_argument(
        "--html-out",
        help="Write an HTML report to the given path",
        default=None,
    )

    check = subparsers.add_parser("check", help="Audit a dataset and exit with code 1 on issues (for CI)")
    check.add_argument("data", help="Path to the dataset (.csv, .jsonl, .ndjson, .parquet)")
    check.add_argument("--rules", help="Path to a JSON file with per-column validation rules", default=None)
    check.add_argument(
        "--missing-threshold",
        type=float,
        default=0.05,
        help="Fraction of missing values allowed before warning",
    )
    check.add_argument(
        "--drift-threshold",
        type=float,
        default=0.20,
        help="Drift score threshold before warning",
    )
    return parser


def _parse_columns(raw: str | None) -> Sequence[str] | None:
    if raw is None:
        return None
    columns = [part.strip() for part in raw.split(",") if part.strip()]
    return columns or None


def _cmd_audit(args: argparse.Namespace) -> int:
    """Handle the audit subcommand."""
    auditor = DatasetAuditor(
        missing_threshold=args.missing_threshold,
        drift_threshold=args.drift_threshold,
        rules=ValidationRules.from_json(args.rules) if args.rules else None,
    )
    report = auditor.audit_file(
        args.data,
        reference_path=args.reference,
        label_column=args.label_column,
        expected_columns=_parse_columns(args.expected_columns),
    )

    if args.html_out:
        html_path = Path(args.html_out)
        html_path.write_text(report.to_html(), encoding="utf-8")

    if args.json:
        print(report.to_json())
    else:
        print(report.to_markdown())

    if args.html_out:
        print(f"HTML report written to {args.html_out}")

    return 0 if report.status == "pass" else 1


def _cmd_check(args: argparse.Namespace) -> int:
    """Handle the check subcommand (CI-friendly)."""
    auditor = DatasetAuditor(
        missing_threshold=args.missing_threshold,
        drift_threshold=args.drift_threshold,
        rules=ValidationRules.from_json(args.rules) if args.rules else None,
    )
    report = auditor.audit_file(
        args.data,
        reference_path=None,
        label_column=None,
        expected_columns=None,
    )

    if report.issues:
        print(report.to_markdown())
        print(f"\n[FAIL] Found {len(report.issues)} issue(s) - check failed.", flush=True)
        return 1

    print(f"[PASS] Dataset '{args.data}' passed all checks ({report.rows} rows, {report.columns} columns).")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "audit":
        return _cmd_audit(args)
    elif args.command == "check":
        return _cmd_check(args)
    else:
        parser.error("unsupported command")

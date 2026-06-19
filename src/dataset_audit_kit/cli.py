"""Command-line entry point for dataset-audit-kit."""

from __future__ import annotations

import argparse
import sys

import numpy as np
from pathlib import Path
from typing import Sequence

from .core import DatasetAuditor, ValidationRules


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dataset-audit-kit")
    parser.add_argument(
        "--version", action="store_true",
        help="Show version and exit",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
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
    audit.add_argument(
        "--unique-columns",
        help="Comma-separated list of columns that should contain unique values",
        default=None,
    )
    audit.add_argument(
        "--fix-suggestions",
        action="store_true",
        help="Print fix suggestions for each issue",
    )
    audit.add_argument(
        "--save-json",
        help="Write JSON report to the specified path",
        default=None,
    )
    audit.add_argument(
        "--save-markdown",
        help="Write Markdown report to the specified path",
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
    check.add_argument(
        "--save-json",
        help="Write JSON report to the specified path",
        default=None,
    )
    check.add_argument(
        "--save-markdown",
        help="Write Markdown report to the specified path",
        default=None,
    )

    columns_parser = subparsers.add_parser("columns", help="List columns with their data types")
    columns_parser.add_argument("data", help="Path to the dataset (.csv, .jsonl, .ndjson, .parquet)")

    head_parser = subparsers.add_parser("head", help="Preview the first N rows of a dataset")
    head_parser.add_argument("data", help="Path to the dataset")
    head_parser.add_argument("--rows", type=int, default=10, help="Number of rows (default: 10)")

    info_parser = subparsers.add_parser("info", help="Show dataset shape, memory usage, and dtypes")
    info_parser.add_argument("data", help="Path to the dataset")

    tail_parser = subparsers.add_parser("tail", help="Show the last N rows of a dataset")
    tail_parser.add_argument("data", help="Path to the dataset")
    tail_parser.add_argument("--rows", type=int, default=10, help="Number of rows (default: 10)")

    unique_parser = subparsers.add_parser("unique", help="Show unique values in a column")
    unique_parser.add_argument("data", help="Path to the dataset")
    unique_parser.add_argument("--column", required=True, help="Column name")
    unique_parser.add_argument("--top", type=int, default=20, help="Show top N values (default: 20)")

    dtype_parser = subparsers.add_parser("dtype", help="Show column dtypes with inferred optimal types")
    dtype_parser.add_argument("data", help="Path to the dataset")

    correlate_parser = subparsers.add_parser("correlate", help="Show pairwise correlation matrix")
    correlate_parser.add_argument("data", help="Path to the dataset")
    correlate_parser.add_argument("--method", default="pearson", choices=["pearson", "spearman", "kendall"], help="Correlation method")

    shape_parser = subparsers.add_parser("shape", help="Show dataset shape (rows x columns)")
    shape_parser.add_argument("data", help="Path to the dataset")

    hist_parser = subparsers.add_parser("hist", help="Show ASCII histogram for a numeric column")
    hist_parser.add_argument("data", help="Path to the dataset")
    hist_parser.add_argument("--column", required=True, help="Numeric column name")
    hist_parser.add_argument("--bins", type=int, default=10, help="Number of bins")

    return parser


def _cmd_head(args: argparse.Namespace) -> int:
    data = DatasetAuditor.load_dataframe(args.data)
    print(data.head(args.rows).to_csv(index=False))
    return 0


def _cmd_columns(args: argparse.Namespace) -> int:
    """Handle the columns subcommand."""
    data = DatasetAuditor.load_dataframe(args.data)
    print(f"{'Column':<30} {'Dtype':<15} {'Non-null':<10} {'Missing':<10}")
    print("-" * 65)
    for col in data.columns:
        non_null = data[col].count()
        missing = int(data[col].isna().sum())
        print(f"{col:<30} {str(data[col].dtype):<15} {non_null:<10} {missing:<10}")
    return 0


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
        unique_columns=_parse_columns(args.unique_columns),
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

    if args.fix_suggestions:
        suggestions = report.fix_suggestions
        if suggestions:
            print()
            print("## Fix suggestions")
            for s in suggestions:
                print("- **" + s["action"] + "**: " + s["description"])
                print("  ```python")
                print("  " + s["code"])
                print("  ```")
        else:
            print()
            print("_No fix suggestions -- dataset is clean._")

    if args.save_json:
        saved = report.to_file(args.save_json)
        print(f"JSON report saved to {saved}")

    if args.save_markdown:
        saved = report.to_file(args.save_markdown)
        print(f"Markdown report saved to {saved}")

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
        unique_columns=None,
    )

    if report.issues:
        print(report.to_markdown())
        if args.save_json:
            report.to_file(args.save_json)
        if args.save_markdown:
            report.to_file(args.save_markdown)
        print(f"\n[FAIL] Found {len(report.issues)} issue(s) - check failed.", flush=True)
        return 1

    if args.save_json:
        report.to_file(args.save_json)
    if args.save_markdown:
        report.to_file(args.save_markdown)
    print(f"[PASS] Dataset '{args.data}' passed all checks ({report.rows} rows, {report.columns} columns).")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    data = DatasetAuditor.load_dataframe(args.data)
    mem = data.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"Shape:  {data.shape[0]} rows x {data.shape[1]} columns")
    print(f"Memory: {mem:.2f} MB")
    print(f"Dtypes:")
    for dt, cnt in data.dtypes.value_counts().items():
        print(f"  {dt}: {cnt}")
    return 0


def _cmd_tail(args: argparse.Namespace) -> int:
    import pandas as pd
    data = DatasetAuditor.load_dataframe(args.data)
    print(data.tail(args.rows).to_csv(index=False))
    return 0


def _cmd_unique(args: argparse.Namespace) -> int:
    data = DatasetAuditor.load_dataframe(args.data)
    if args.column not in data.columns:
        print(f"Column '{args.column}' not found.", file=sys.stderr)
        return 1
    counts = data[args.column].astype(str).value_counts().head(args.top)
    print(f"{'Value':<40} {'Count':<10}")
    print("-" * 50)
    for val, cnt in counts.items():
        print(f"{str(val):<40} {cnt:<10}")
    return 0


def _cmd_dtype(args: argparse.Namespace) -> int:
    data = DatasetAuditor.load_dataframe(args.data)
    suggested = DatasetAuditor.infer_optimal_dtypes(data)
    print(f"{'Column':<30} {'Current':<15} {'Suggested':<15}")
    print("-" * 60)
    for col in data.columns:
        cur = str(data[col].dtype)
        sug = suggested.get(col, {}).get("suggested_dtype", cur)
        print(f"{col:<30} {cur:<15} {sug:<15}")
    return 0


def _cmd_shape(args: argparse.Namespace) -> int:
    data = DatasetAuditor.load_dataframe(args.data)
    print(f"{data.shape[0]} rows x {data.shape[1]} columns")
    return 0


def _cmd_hist(args: argparse.Namespace) -> int:
    import pandas as pd
    data = DatasetAuditor.load_dataframe(args.data)
    if args.column not in data.columns:
        print(f"Column '{args.column}' not found.", file=sys.stderr)
        return 1
    col = data[args.column].dropna()
    if not pd.api.types.is_numeric_dtype(col):
        print(f"Column '{args.column}' is not numeric.", file=sys.stderr)
        return 1
    counts, edges = np.histogram(col, bins=args.bins)
    max_count = max(counts) if len(counts) > 0 else 1
    bar_width = 40
    print(f"Histogram for '{args.column}' ({len(col)} values, {args.bins} bins):")
    for i in range(len(counts)):
        pct = counts[i] / max_count
        bar = "█" * int(pct * bar_width)
        print(f"{edges[i]:>8.2f}-{edges[i+1]:<8.2f} │{bar} {counts[i]}")
    return 0


def _cmd_correlate(args: argparse.Namespace) -> int:
    data = DatasetAuditor.load_dataframe(args.data)
    numeric = data.select_dtypes(include="number")
    if numeric.empty:
        print("No numeric columns found.", file=sys.stderr)
        return 1
    corr = numeric.corr(method=args.method)
    print(corr.to_csv(float_format="%.4f"))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__
        print(f"dataset-audit-kit v{__version__}")
        return 0

    if args.command == "audit":
        return _cmd_audit(args)
    elif args.command == "check":
        return _cmd_check(args)
    elif args.command == "columns":
        return _cmd_columns(args)
    elif args.command == "head":
        return _cmd_head(args)
    elif args.command == "info":
        return _cmd_info(args)
    elif args.command == "tail":
        return _cmd_tail(args)
    elif args.command == "unique":
        return _cmd_unique(args)
    elif args.command == "dtype":
        return _cmd_dtype(args)
    elif args.command == "correlate":
        return _cmd_correlate(args)
    elif args.command == "shape":
        return _cmd_shape(args)
    elif args.command == "hist":
        return _cmd_hist(args)
    else:
        parser.error("unsupported command")
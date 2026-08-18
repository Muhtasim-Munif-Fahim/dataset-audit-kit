"""Command-line entry point for dataset-audit-kit."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
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
    # Not required: `dataset-audit-kit --version` is a complete command line,
    # and argparse would otherwise reject it for naming no subcommand.
    subparsers = parser.add_subparsers(dest="command", required=False)

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
        "--fail-on",
        choices=["warning", "error"],
        default="warning",
        help="Severity at or above which the command exits with code 1",
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
        "--unique-together",
        action="append",
        metavar="COL1,COL2",
        help="Columns forming a composite unique key; repeatable",
        default=None,
    )
    audit.add_argument(
        "--select-columns",
        help="Comma-separated columns to audit",
        default=None,
    )
    audit.add_argument(
        "--exclude-columns",
        help="Comma-separated columns to leave out of the audit",
        default=None,
    )
    audit.add_argument(
        "--fix-suggestions",
        action="store_true",
        help="Print fix suggestions for each issue",
    )
    audit.add_argument(
        "--date-column",
        help="Column to apply --from-date/--to-date to",
        default=None,
    )
    audit.add_argument(
        "--from-date",
        help="Keep rows on or after this date (YYYY-MM-DD)",
        default=None,
    )
    audit.add_argument(
        "--to-date",
        help="Keep rows on or before this date (YYYY-MM-DD)",
        default=None,
    )
    audit.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=None,
        help="Always show audit progress (default: only above 100k rows)",
    )
    audit.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Never show audit progress",
    )
    audit.add_argument(
        "--minimal",
        action="store_true",
        help="Print only the status line and issue count",
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
    audit.add_argument(
        "--sarif-out",
        help="Write SARIF 2.1.0 findings for code-scanning integrations",
        default=None,
    )
    audit.add_argument(
        "--csv-out",
        help="Write flat CSV findings for CI or spreadsheet consumers",
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
        "--fail-on",
        choices=["warning", "error"],
        default="warning",
        help="Severity at or above which the command exits with code 1",
    )
    check.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=None,
        help="Always show audit progress (default: only above 100k rows)",
    )
    check.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Never show audit progress",
    )
    check.add_argument(
        "--minimal",
        action="store_true",
        help="Print only the status line and issue count",
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
    check.add_argument(
        "--csv-out",
        help="Write flat CSV findings for CI or spreadsheet consumers",
        default=None,
    )

    columns_parser = subparsers.add_parser("columns", help="List columns with their data types")
    columns_parser.add_argument("data", help="Path to the dataset (.csv, .jsonl, .ndjson, .parquet)")
    columns_parser.add_argument("--format", choices=["table", "csv", "markdown"], default="table", help="Output format (default: table)")
    columns_parser.add_argument("--sort", choices=["name", "dtype", "missing"], default=None, help="Sort columns by the given criterion")

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
    unique_parser.add_argument("--format", choices=["table", "csv", "markdown"], default="table", help="Output format (default: table)")
    unique_parser.add_argument("--top", type=int, default=20, help="Show top N values (default: 20)")

    dtype_parser = subparsers.add_parser("dtype", help="Show column dtypes with inferred optimal types")
    dtype_parser.add_argument("data", help="Path to the dataset")

    correlate_parser = subparsers.add_parser("correlate", help="Show pairwise correlation matrix")
    correlate_parser.add_argument("data", help="Path to the dataset")
    correlate_parser.add_argument("--format", choices=["table", "csv", "markdown"], default="table", help="Output format (default: table)")
    correlate_parser.add_argument("--method", default="pearson", choices=["pearson", "spearman", "kendall"], help="Correlation method")

    shape_parser = subparsers.add_parser("shape", help="Show dataset shape (rows x columns)")
    shape_parser.add_argument("data", help="Path to the dataset")
    shape_parser.add_argument("--csv", action="store_true", help="CSV output (rows,columns)")

    refs_parser = subparsers.add_parser("refs", help="Check referential integrity against a parent table")
    refs_parser.add_argument("data", help="Path to the child dataset (the one holding the foreign key)")
    refs_parser.add_argument("--key", required=True, help="Foreign key column in the child dataset")
    refs_parser.add_argument("--parent", required=True, help="Path to the parent dataset")
    refs_parser.add_argument("--parent-key", default=None, help="Key column in the parent (defaults to --key)")
    refs_parser.add_argument("--show", type=int, default=10, help="Number of orphan values to list (default: 10)")
    refs_parser.add_argument("--fail-on-orphans", action="store_true", help="Exit 1 if any orphaned keys are found")

    diff_parser = subparsers.add_parser("diff", help="Compare two audit reports saved as JSON")
    diff_parser.add_argument("baseline", help="Path to the earlier report (--save-json output)")
    diff_parser.add_argument("current", help="Path to the later report")
    diff_parser.add_argument("--fail-on-regression", action="store_true", help="Exit 1 if quality dropped or issues were added")

    schema_parser = subparsers.add_parser("schema", help="Export the dataset schema as JSON Schema")
    schema_parser.add_argument("data", help="Path to the dataset")
    schema_parser.add_argument("--title", default=None, help="Schema title (defaults to the file stem)")
    schema_parser.add_argument("--indent", type=int, default=2, help="JSON indent (default: 2)")

    infer_rules_parser = subparsers.add_parser(
        "infer-rules",
        help="Infer a JSON validation contract from a baseline dataset",
    )
    infer_rules_parser.add_argument("data", help="Path to the baseline dataset")
    infer_rules_parser.add_argument(
        "--max-categories",
        type=int,
        default=20,
        help="Maximum distinct values to encode as allowed values (default: 20)",
    )
    infer_rules_parser.add_argument(
        "--missing-tolerance",
        type=float,
        default=0.0,
        help="Extra missing fraction allowed above the baseline (default: 0)",
    )

    rename_parser = subparsers.add_parser("rename", help="Rename columns and write the result to a new file")
    rename_parser.add_argument("data", help="Path to the dataset")
    rename_parser.add_argument("--map", required=True, action="append", metavar="OLD=NEW", help="Rename OLD to NEW; repeatable")
    rename_parser.add_argument("--output", required=True, help="Path to write the renamed dataset to")
    rename_parser.add_argument("--force", action="store_true", help="Overwrite the output file if it exists")

    profile_parser = subparsers.add_parser("profile", help="Deep-dive a single column")
    profile_parser.add_argument("data", help="Path to the dataset")
    profile_parser.add_argument("--column", required=True, help="Column name")
    profile_parser.add_argument("--top", type=int, default=10, help="Number of frequent values to show (default: 10)")

    stats_parser = subparsers.add_parser("stats", help="Show dataset-level statistics")
    stats_parser.add_argument("data", help="Path to the dataset")

    missing_parser = subparsers.add_parser("missing", help="Report missing values per column")
    missing_parser.add_argument("data", help="Path to the dataset")
    missing_parser.add_argument("--threshold", type=float, default=0.0, help="Only show columns missing more than this fraction (default: 0.0)")
    missing_parser.add_argument("--all", action="store_true", help="Include columns with no missing values")

    describe_parser = subparsers.add_parser("describe", help="Show summary statistics for every column")
    describe_parser.add_argument("data", help="Path to the dataset")
    describe_parser.add_argument("--include", choices=["numeric", "categorical", "all"], default="all", help="Which columns to describe (default: all)")

    hist_parser = subparsers.add_parser("hist", help="Show ASCII histogram for a numeric column")
    hist_parser.add_argument("data", help="Path to the dataset")
    hist_parser.add_argument("--column", required=True, help="Numeric column name")
    hist_parser.add_argument("--bins", type=int, default=10, help="Number of bins")

    # Every subcommand reads a dataset, so the flag belongs on all of them.
    for subparser in subparsers.choices.values():
        subparser.add_argument(
            "--encoding",
            default=None,
            help="Text encoding of the input file (e.g. latin-1, cp1252). "
                 "Ignored for .parquet and Excel inputs.",
        )
        # `rename` already defines --output as its destination file.
        if "output" not in {action.dest for action in subparser._actions}:
            subparser.add_argument(
                "--output", "-o",
                default=None,
                help="Write this command's output to a file instead of stdout.",
            )
        subparser.add_argument(
            "--delimiter",
            default=None,
            help="Field separator for delimited text. Defaults to the "
                 "convention for .csv/.tsv and is sniffed for .txt.",
        )

    return parser


def _load(args: argparse.Namespace) -> "pd.DataFrame":
    """Load the dataset named by args, honouring --encoding when given."""
    return DatasetAuditor.load_dataframe(
        args.data,
        encoding=getattr(args, "encoding", None),
        delimiter=getattr(args, "delimiter", None),
    )


def _cmd_head(args: argparse.Namespace) -> int:
    data = _load(args)
    print(data.head(args.rows).to_csv(index=False))
    return 0


def _render_table(frame: "pd.DataFrame", fmt: str) -> None:
    """Print a frame as a fixed-width table, CSV, or Markdown."""

    if fmt == "csv":
        print(frame.to_csv(index=False).rstrip("\r\n"))
        return

    columns = [str(c) for c in frame.columns]
    cells = [[("" if value is None else str(value)) for value in row] for row in frame.itertuples(index=False)]
    widths = [
        max(len(columns[i]), *(len(row[i]) for row in cells)) if cells else len(columns[i])
        for i in range(len(columns))
    ]

    if fmt == "markdown":
        print("| " + " | ".join(c.ljust(w) for c, w in zip(columns, widths)) + " |")
        print("| " + " | ".join("-" * w for w in widths) + " |")
        for row in cells:
            print("| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |")
        return

    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)).rstrip())
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in cells:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)).rstrip())


def _cmd_columns(args: argparse.Namespace) -> int:
    """Handle the columns subcommand."""
    data = _load(args)
    cols = list(data.columns)
    if args.sort == "name":
        cols.sort()
    elif args.sort == "dtype":
        cols.sort(key=lambda c: str(data[c].dtype))
    elif args.sort == "missing":
        cols.sort(key=lambda c: int(data[c].isna().sum()), reverse=True)
    table = pd.DataFrame(
        {
            "Column": [str(col) for col in cols],
            "Dtype": [str(data[col].dtype) for col in cols],
            "Non-null": [int(data[col].count()) for col in cols],
            "Missing": [int(data[col].isna().sum()) for col in cols],
        }
    )
    _render_table(table, args.format)
    return 0


def _parse_columns(raw: str | None) -> Sequence[str] | None:
    if raw is None:
        return None
    columns = [part.strip() for part in raw.split(",") if part.strip()]
    return columns or None


def _parse_unique_groups(raw: Sequence[str] | None) -> list[list[str]] | None:
    if raw is None:
        return None
    return [
        [column.strip() for column in group.split(",") if column.strip()]
        for group in raw
    ]


def _apply_date_filter(
    data: "pd.DataFrame", args: argparse.Namespace
) -> "tuple[pd.DataFrame | None, str | None]":
    """Filter rows by a date range. Returns (frame, error message)."""

    if not args.from_date and not args.to_date:
        return data, None

    column = args.date_column
    if column is None:
        candidates = [
            name for name in data.columns
            if pd.api.types.is_datetime64_any_dtype(data[name])
        ]
        if len(candidates) != 1:
            return None, (
                "Cannot infer which column to filter on "
                f"({len(candidates)} datetime column(s) found). "
                "Pass --date-column."
            )
        column = candidates[0]
    elif column not in data.columns:
        return None, f"Column '{column}' not found in dataset."

    parsed = pd.to_datetime(data[column], errors="coerce")
    if parsed.isna().all():
        return None, f"Column '{column}' holds no parseable dates."

    mask = pd.Series(True, index=data.index)
    for bound, comparison in ((args.from_date, "ge"), (args.to_date, "le")):
        if not bound:
            continue
        try:
            edge = pd.Timestamp(bound)
        except ValueError:
            return None, f"Cannot parse date '{bound}'; expected YYYY-MM-DD."
        mask &= getattr(parsed, comparison)(edge)

    # Rows whose date failed to parse cannot be placed in the range.
    mask &= parsed.notna()
    return data[mask], None


def _cmd_audit(args: argparse.Namespace) -> int:
    """Handle the audit subcommand."""
    auditor = DatasetAuditor(
        missing_threshold=args.missing_threshold,
        drift_threshold=args.drift_threshold,
        rules=ValidationRules.from_json(args.rules) if args.rules else None,
        progress=getattr(args, "progress", None),
    )

    select_columns = _parse_columns(args.select_columns)
    exclude_columns = _parse_columns(getattr(args, "exclude_columns", None))
    if select_columns and exclude_columns:
        print(
            "Error: --select-columns and --exclude-columns are mutually exclusive.",
            file=sys.stderr,
        )
        return 2
    date_filtered = bool(args.from_date or args.to_date)
    if select_columns or exclude_columns or date_filtered:
        data = _load(args)
        if date_filtered:
            before = len(data)
            data, error = _apply_date_filter(data, args)
            if error:
                print(f"Error: {error}", file=sys.stderr)
                return 2
            if data.empty:
                print("Error: no rows fall within the requested date range.", file=sys.stderr)
                return 2
            print(
                f"Date filter kept {len(data)} of {before} row(s).",
                file=sys.stderr,
            )
        present: list[str] = []
        missing: list[str] = []
        if select_columns:
            present = [c for c in select_columns if c in data.columns]
            missing = [c for c in select_columns if c not in data.columns]
        elif exclude_columns:
            missing = [c for c in exclude_columns if c not in data.columns]
            present = [c for c in data.columns if c not in set(exclude_columns)]
        if missing:
            print(f"Warning: requested columns not found in dataset: {missing}", file=sys.stderr)
        if select_columns or exclude_columns:
            if not present:
                print("Error: no columns left to audit after filtering.", file=sys.stderr)
                return 2
            data = data[present]
        reference = (
            DatasetAuditor.load_dataframe(
                args.reference,
                encoding=getattr(args, "encoding", None),
                delimiter=getattr(args, "delimiter", None),
            )
            if args.reference
            else None
        )
        report = auditor.audit_dataframe(
            data,
            reference=reference,
            label_column=args.label_column,
            expected_columns=_parse_columns(args.expected_columns),
            unique_columns=_parse_columns(args.unique_columns),
            unique_together=_parse_unique_groups(args.unique_together),
        )
    else:
        report = auditor.audit_file(
            args.data,
            reference_path=args.reference,
            label_column=args.label_column,
            expected_columns=_parse_columns(args.expected_columns),
            unique_columns=_parse_columns(args.unique_columns),
            unique_together=_parse_unique_groups(args.unique_together),
            encoding=getattr(args, "encoding", None),
            delimiter=getattr(args, "delimiter", None),
        )

    # With --json or --minimal the output is meant to be consumed by another
    # program, so progress notes go to stderr and leave stdout parseable.
    machine_readable = bool(args.json or args.minimal)

    def notice(message: str) -> None:
        print(message, file=sys.stderr if machine_readable else sys.stdout)

    if args.html_out:
        try:
            html_path = Path(args.html_out)
            if html_path.parent != Path(""):
                html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(report.to_html(), encoding="utf-8")
        except OSError as exc:
            print(f"Cannot write HTML report to '{args.html_out}': {exc}", file=sys.stderr)
            return 2

    if args.minimal:
        gated = report.gated_issues(args.fail_on)
        errors = sum(1 for issue in gated if issue.severity == "error")
        status = "PASS" if report.exit_code(args.fail_on) == 0 else "FAIL"
        ignored = len(report.blocking_issues) - len(gated)
        suffix = f" {ignored} warning(s) below threshold." if ignored else ""
        print(f"[{status}] {len(gated)} issue(s), {errors} error(s).{suffix}")
    elif args.json:
        import json

        payload = report.to_dict()
        if args.fix_suggestions:
            payload["fix_suggestions"] = report.fix_suggestions
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(report.to_markdown())

    if args.html_out:
        notice(f"HTML report written to {args.html_out}")

    if args.fix_suggestions and not args.json:
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

    for path, content, label in (
        (args.save_json, report.to_json(), "JSON"),
        (args.save_markdown, report.to_markdown(), "Markdown"),
        (
            args.sarif_out,
            report.to_sarif(artifact_uri=str(Path(args.data).as_posix())),
            "SARIF",
        ),
        (args.csv_out, report.to_csv(), "CSV"),
    ):
        if not path:
            continue
        error = _write_text(path, content)
        if error:
            print(error, file=sys.stderr)
            return 2
        notice(f"{label} report saved to {path}")

    return report.exit_code(args.fail_on)


def _write_text(path: str, content: str) -> str | None:
    """Write text to a path, creating parent directories.

    Returns an error message instead of raising, so a bad destination ends in a
    one-line diagnostic rather than a traceback.
    """

    destination = Path(path)
    try:
        if destination.parent != Path(""):
            destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"Cannot write report to '{path}': {exc}"
    return None


def _cmd_check(args: argparse.Namespace) -> int:
    """Handle the check subcommand (CI-friendly)."""
    auditor = DatasetAuditor(
        missing_threshold=args.missing_threshold,
        drift_threshold=args.drift_threshold,
        rules=ValidationRules.from_json(args.rules) if args.rules else None,
        progress=getattr(args, "progress", None),
    )
    report = auditor.audit_file(
        args.data,
        reference_path=None,
        label_column=None,
        expected_columns=None,
        unique_columns=None,
        encoding=getattr(args, "encoding", None),
        delimiter=getattr(args, "delimiter", None),
    )

    for path, content in (
        (args.save_json, report.to_json()),
        (args.save_markdown, report.to_markdown()),
        (args.csv_out, report.to_csv()),
    ):
        if not path:
            continue
        error = _write_text(path, content)
        if error:
            print(error, file=sys.stderr)
            return 2

    gated = report.gated_issues(args.fail_on)
    if gated:
        if not args.minimal:
            print(report.to_markdown())
        errors = sum(1 for issue in gated if issue.severity == "error")
        summary = f"[FAIL] {len(gated)} issue(s), {errors} error(s) - check failed."
        print(summary if args.minimal else "\n" + summary, flush=True)
        return 1

    ignored = len(report.blocking_issues) - len(gated)
    warning_note = f" ({ignored} warning(s) below threshold)" if ignored else ""
    if args.minimal:
        print(f"[PASS] 0 issue(s).{warning_note}")
    else:
        print(
            f"[PASS] Dataset '{args.data}' passed the {args.fail_on} gate "
            f"({report.rows} rows, {report.columns} columns).{warning_note}"
        )
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    data = _load(args)
    mem = data.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"Shape:  {data.shape[0]} rows x {data.shape[1]} columns")
    print(f"Memory: {mem:.2f} MB")
    print(f"Dtypes:")
    for dt, cnt in data.dtypes.value_counts().items():
        print(f"  {dt}: {cnt}")
    return 0


def _cmd_tail(args: argparse.Namespace) -> int:
    import pandas as pd
    data = _load(args)
    print(data.tail(args.rows).to_csv(index=False))
    return 0


def _cmd_unique(args: argparse.Namespace) -> int:
    data = _load(args)
    if args.column not in data.columns:
        print(f"Column '{args.column}' not found.", file=sys.stderr)
        return 1
    counts = data[args.column].astype(str).value_counts().head(args.top)
    table = pd.DataFrame(
        {"Value": [str(v) for v in counts.index], "Count": [int(c) for c in counts.values]}
    )
    _render_table(table, args.format)
    return 0


def _cmd_dtype(args: argparse.Namespace) -> int:
    data = _load(args)
    suggested = DatasetAuditor.infer_optimal_dtypes(data)
    print(f"{'Column':<30} {'Current':<15} {'Suggested':<15}")
    print("-" * 60)
    for col in data.columns:
        cur = str(data[col].dtype)
        sug = suggested.get(col, {}).get("suggested_dtype", cur)
        print(f"{col:<30} {cur:<15} {sug:<15}")
    return 0


def _cmd_shape(args: argparse.Namespace) -> int:
    data = _load(args)
    if getattr(args, "csv", False):
        print(f"{data.shape[0]},{data.shape[1]}")
    else:
        print(f"{data.shape[0]} rows x {data.shape[1]} columns")
    return 0


def _cmd_hist(args: argparse.Namespace) -> int:
    import pandas as pd
    data = _load(args)
    if args.column not in data.columns:
        print(f"Column '{args.column}' not found.", file=sys.stderr)
        return 1
    col = data[args.column].dropna()
    if not pd.api.types.is_numeric_dtype(col):
        print(f"Column '{args.column}' is not numeric.", file=sys.stderr)
        return 1
    if args.bins < 1:
        print(f"--bins must be at least 1 (got {args.bins}).", file=sys.stderr)
        return 2
    if col.empty:
        print(f"Column '{args.column}' has no non-missing values to plot.", file=sys.stderr)
        return 1
    counts, edges = np.histogram(col, bins=args.bins)
    max_count = int(counts.max()) if len(counts) else 0
    max_count = max_count or 1
    bar_width = 40
    block, divider = _bar_glyphs()
    print(f"Histogram for '{args.column}' ({len(col)} values, {args.bins} bins):")
    for i in range(len(counts)):
        pct = counts[i] / max_count
        bar = block * int(pct * bar_width)
        print(f"{edges[i]:>8.2f}-{edges[i+1]:<8.2f} {divider}{bar} {counts[i]}")
    return 0


def _bar_glyphs() -> tuple[str, str]:
    """Return (bar, divider) glyphs the current stdout can actually encode.

    The Windows console defaults to cp1252, which cannot encode the block and
    box-drawing characters, so printing them raised UnicodeEncodeError and the
    histogram died halfway through its header.
    """

    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    for candidate in (("█", "│"), ("#", "|")):
        try:
            "".join(candidate).encode(encoding)
        except (UnicodeEncodeError, LookupError):
            continue
        return candidate
    return ("#", "|")


def _cmd_correlate(args: argparse.Namespace) -> int:
    data = _load(args)
    numeric = data.select_dtypes(include="number")
    if numeric.empty:
        print("No numeric columns found.", file=sys.stderr)
        return 1
    corr = numeric.corr(method=args.method).round(4)
    # The index carries the row labels, so promote it to a real column.
    table = corr.reset_index().rename(columns={"index": ""})
    _render_table(table, args.format)
    return 0


#: pandas dtype kind -> (JSON Schema type, optional format)
_JSON_SCHEMA_TYPES: dict[str, tuple[str, str | None]] = {
    "b": ("boolean", None),
    "i": ("integer", None),
    "u": ("integer", None),
    "f": ("number", None),
    "M": ("string", "date-time"),
    "m": ("string", "duration"),
    "O": ("string", None),
    "S": ("string", None),
    "U": ("string", None),
}


def _key_strings(series: "pd.Series") -> "pd.Series":
    """Render a key column as strings that compare across dtypes.

    A key column containing a null is read as float64, so a plain astype(str)
    turns 1 into "1.0" and it stops matching a parent key of "1". Integral
    floats are therefore narrowed back to integers before stringifying.
    """

    values = series.dropna()
    if pd.api.types.is_float_dtype(values) and (values == values.round()).all():
        return values.astype("int64").astype(str)
    return values.astype(str)


def _cmd_refs(args: argparse.Namespace) -> int:
    """Handle the refs subcommand."""
    child = _load(args)
    parent_key = args.parent_key or args.key

    if args.key not in child.columns:
        print(f"Column '{args.key}' not found in child dataset.", file=sys.stderr)
        return 2
    try:
        parent = DatasetAuditor.load_dataframe(
            args.parent,
            encoding=getattr(args, "encoding", None),
            delimiter=getattr(args, "delimiter", None),
        )
    except (OSError, ValueError) as exc:
        print(f"Cannot read parent dataset '{args.parent}': {exc}", file=sys.stderr)
        return 2
    if parent_key not in parent.columns:
        print(f"Column '{parent_key}' not found in parent dataset.", file=sys.stderr)
        return 2

    child_values = _key_strings(child[args.key])
    parent_values = set(_key_strings(parent[parent_key]))

    orphan_mask = ~child_values.isin(parent_values)
    orphan_rows = int(orphan_mask.sum())
    orphan_values = sorted(set(child_values[orphan_mask]))

    null_keys = int(child[args.key].isna().sum())
    duplicate_parent = int(parent[parent_key].dropna().duplicated().sum())

    print(f"Child   : {args.data} ({len(child)} rows, key '{args.key}')")
    print(f"Parent  : {args.parent} ({len(parent)} rows, key '{parent_key}')")
    print("-" * 60)
    print(f"{'Rows with a key':<28}{len(child_values):>10}")
    print(f"{'Null keys':<28}{null_keys:>10}")
    print(f"{'Orphaned rows':<28}{orphan_rows:>10}")
    print(f"{'Distinct orphaned values':<28}{len(orphan_values):>10}")
    if duplicate_parent:
        print(f"{'Duplicate parent keys':<28}{duplicate_parent:>10}  <-- parent key is not unique")

    if orphan_values:
        print()
        print(f"Orphaned values (first {min(args.show, len(orphan_values))}):")
        for value in orphan_values[: args.show]:
            print(f"  {value}")
        if len(orphan_values) > args.show:
            print(f"  ... and {len(orphan_values) - args.show} more")
    else:
        print()
        print("No orphaned keys: every child key is present in the parent.")

    if args.fail_on_orphans and (orphan_rows or duplicate_parent):
        return 1
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    """Handle the diff subcommand."""
    import json

    reports = {}
    for label, path in (("baseline", args.baseline), ("current", args.current)):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                reports[label] = json.load(handle)
        except OSError as exc:
            print(f"Cannot read {label} report '{path}': {exc}", file=sys.stderr)
            return 2
        except json.JSONDecodeError as exc:
            print(
                f"{label.capitalize()} report '{path}' is not valid JSON: {exc}. "
                "Reports come from `audit --save-json`.",
                file=sys.stderr,
            )
            return 2

    before, after = reports["baseline"], reports["current"]

    def _num(report: dict, key: str) -> float:
        value = report.get(key, 0)
        return float(value) if isinstance(value, (int, float)) else 0.0

    print(f"{'Metric':<20}{'baseline':>12}{'current':>12}{'change':>12}")
    print("-" * 56)
    regressed = False
    for key in ("quality_score", "rows", "columns", "duplicate_rows", "missing_cells"):
        old, new = _num(before, key), _num(after, key)
        delta = new - old
        # Higher is better for quality_score; for the rest, higher is worse.
        if key == "quality_score":
            worse = delta < 0
        else:
            worse = delta > 0 and key in {"duplicate_rows", "missing_cells"}
        regressed = regressed or worse
        marker = "  <-- worse" if worse else ""
        print(f"{key:<20}{old:>12.0f}{new:>12.0f}{delta:>+12.0f}{marker}")

    old_issues = {_issue_key(i) for i in before.get("issues", [])}
    new_issues = {_issue_key(i) for i in after.get("issues", [])}

    added = sorted(new_issues - old_issues)
    resolved = sorted(old_issues - new_issues)
    regressed = regressed or bool(added)

    print()
    print(f"Issues: {len(old_issues)} -> {len(new_issues)} "
          f"({len(added)} added, {len(resolved)} resolved)")
    for key in added:
        print(f"  + {key}")
    for key in resolved:
        print(f"  - {key}")

    before_columns = set(before.get("column_profiles", {}))
    after_columns = set(after.get("column_profiles", {}))
    dropped, gained = sorted(before_columns - after_columns), sorted(after_columns - before_columns)
    if dropped or gained:
        print()
        print("Schema changes:")
        for column in dropped:
            print(f"  - {column} (dropped)")
        for column in gained:
            print(f"  + {column} (new)")
        regressed = regressed or bool(dropped)

    if args.fail_on_regression and regressed:
        return 1
    return 0


def _issue_key(issue: dict) -> str:
    """Identify an issue by what it is about, not by its wording."""
    check = issue.get("check", "?")
    column = issue.get("column")
    severity = issue.get("severity", "?")
    return f"[{severity}] {check}" + (f" on '{column}'" if column else "")


def _cmd_schema(args: argparse.Namespace) -> int:
    """Handle the schema subcommand."""
    import json

    data = _load(args)

    properties: dict[str, dict[str, object]] = {}
    required: list[str] = []

    for column in data.columns:
        series = data[column]
        json_type, json_format = _JSON_SCHEMA_TYPES.get(series.dtype.kind, ("string", None))

        has_missing = bool(series.isna().any())
        # A column with gaps must admit null; one without becomes required.
        entry: dict[str, object] = {
            "type": [json_type, "null"] if has_missing else json_type
        }
        if json_format:
            entry["format"] = json_format
        if not has_missing:
            required.append(str(column))

        # Enumerate genuinely low-cardinality string columns; anything wider is
        # data rather than a closed set.
        if json_type == "string" and json_format is None:
            values = series.dropna().unique()
            if 0 < len(values) <= 20 and len(values) < len(series):
                entry["enum"] = sorted(str(value) for value in values)

        properties[str(column)] = entry

    schema: dict[str, object] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": args.title or Path(args.data).stem,
        "description": f"Schema inferred from {Path(args.data).name} ({len(data)} rows).",
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    print(json.dumps(schema, indent=args.indent, sort_keys=False))
    return 0


def _cmd_infer_rules(args: argparse.Namespace) -> int:
    """Infer per-column validation rules and print them as JSON."""

    import json

    try:
        rules = ValidationRules.infer(
            _load(args),
            max_categories=args.max_categories,
            missing_tolerance=args.missing_tolerance,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(rules.to_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_rename(args: argparse.Namespace) -> int:
    """Handle the rename subcommand."""
    mapping: dict[str, str] = {}
    for pair in args.map:
        if "=" not in pair:
            print(f"Invalid --map value '{pair}'; expected OLD=NEW.", file=sys.stderr)
            return 2
        old, new = pair.split("=", 1)
        old, new = old.strip(), new.strip()
        if not old or not new:
            print(f"Invalid --map value '{pair}'; both sides must be non-empty.", file=sys.stderr)
            return 2
        mapping[old] = new

    output = Path(args.output)
    if output.exists() and not args.force:
        print(f"Refusing to overwrite existing file '{output}'; pass --force.", file=sys.stderr)
        return 2

    data = _load(args)

    unknown = [old for old in mapping if old not in data.columns]
    if unknown:
        print(f"Column(s) not found in dataset: {unknown}", file=sys.stderr)
        return 2

    renamed = data.rename(columns=mapping)
    duplicates = renamed.columns[renamed.columns.duplicated()].tolist()
    if duplicates:
        print(f"Refusing to write: rename would create duplicate column(s) {duplicates}.", file=sys.stderr)
        return 2

    suffix = DatasetAuditor._data_suffix(output)
    if suffix == ".csv":
        renamed.to_csv(output, index=False)
    elif suffix == ".tsv":
        renamed.to_csv(output, sep="\t", index=False)
    elif suffix in {".jsonl", ".ndjson"}:
        renamed.to_json(output, orient="records", lines=True)
    elif suffix == ".parquet":
        renamed.to_parquet(output, index=False)
    else:
        print(f"Unsupported output format '{suffix}'. Use .csv, .tsv, .jsonl or .parquet.", file=sys.stderr)
        return 2

    for old, new in mapping.items():
        print(f"{old} -> {new}")
    print(f"Wrote {len(renamed)} rows to {output}")
    return 0


def _cmd_profile(args: argparse.Namespace) -> int:
    """Handle the profile subcommand."""
    data = _load(args)
    if args.column not in data.columns:
        available = ", ".join(map(str, list(data.columns)[:10]))
        print(f"Column '{args.column}' not found. Available: {available}", file=sys.stderr)
        return 2

    series = data[args.column]
    total = len(series)
    non_null = series.dropna()

    def _row(label: str, value: object) -> None:
        print(f"{label:<24}{value}")

    print(f"Column: {args.column}")
    print("=" * 56)
    _row("Dtype", series.dtype)
    _row("Rows", f"{total:,}")
    _row("Non-null", f"{len(non_null):,}")
    missing = total - len(non_null)
    _row("Missing", f"{missing:,} ({missing / total:.1%})" if total else "0")
    _row("Unique", f"{int(series.nunique(dropna=True)):,}")
    if total:
        _row("Cardinality ratio", f"{series.nunique(dropna=True) / total:.3f}")
    _row("Memory", _human_bytes(int(series.memory_usage(deep=True))))

    if non_null.empty:
        print()
        print("Column is entirely missing.")
        return 0

    if pd.api.types.is_numeric_dtype(series):
        print()
        print("Distribution")
        print("-" * 56)
        described = non_null.describe()
        for key in ("mean", "std", "min", "25%", "50%", "75%", "max"):
            if key in described:
                _row(key, f"{described[key]:.6g}")
        _row("zeros", f"{int((non_null == 0).sum()):,}")
        _row("negatives", f"{int((non_null < 0).sum()):,}")
    else:
        print()
        print(f"Top {args.top} values")
        print("-" * 56)
        counts = non_null.value_counts().head(args.top)
        for value, count in counts.items():
            share = count / len(non_null)
            print(f"{str(value)[:29]:<30}{int(count):>8}{share:>9.1%}")

    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    """Handle the stats subcommand."""
    data = _load(args)
    rows, cols = data.shape
    cells = rows * cols

    numeric = data.select_dtypes(include="number").shape[1]
    datetime_cols = data.select_dtypes(include="datetime").shape[1]
    boolean = data.select_dtypes(include="bool").shape[1]
    other = cols - numeric - datetime_cols - boolean

    missing = int(data.isna().sum().sum())
    duplicates = int(data.duplicated().sum())
    constant = [c for c in data.columns if data[c].nunique(dropna=False) <= 1]
    memory = int(data.memory_usage(deep=True).sum())

    def _row(label: str, value: object) -> None:
        print(f"{label:<26}{value}")

    _row("Rows", f"{rows:,}")
    _row("Columns", f"{cols:,}")
    _row("Cells", f"{cells:,}")
    _row("Memory", _human_bytes(memory))
    print()
    _row("Numeric columns", numeric)
    _row("Datetime columns", datetime_cols)
    _row("Boolean columns", boolean)
    _row("Other columns", other)
    print()
    _row("Missing cells", f"{missing:,} ({missing / cells:.1%})" if cells else "0")
    _row("Rows with any missing", f"{int(data.isna().any(axis=1).sum()):,}")
    _row("Duplicate rows", f"{duplicates:,}")
    _row("Constant columns", f"{len(constant)}" + (f" ({', '.join(map(str, constant[:5]))})" if constant else ""))
    return 0


def _human_bytes(size: int) -> str:
    """Render a byte count in the largest unit that keeps it above 1."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def _cmd_missing(args: argparse.Namespace) -> int:
    """Handle the missing subcommand."""
    data = _load(args)
    rows = len(data)
    if rows == 0:
        print("Dataset has no rows.")
        return 0

    counts = data.isna().sum().sort_values(ascending=False)
    total_missing = int(counts.sum())

    print(f"{'Column':<30}{'Missing':>10}{'Percent':>10}")
    print("-" * 50)
    shown = 0
    for column, count in counts.items():
        count = int(count)
        ratio = count / rows
        if not args.all and (count == 0 or ratio <= args.threshold):
            continue
        print(f"{str(column)[:29]:<30}{count:>10}{ratio:>9.1%}")
        shown += 1

    if shown == 0:
        print("(no columns above the threshold)")

    print("-" * 50)
    cells = rows * len(data.columns)
    overall = total_missing / cells if cells else 0.0
    print(f"{'TOTAL':<30}{total_missing:>10}{overall:>9.1%}")
    print()
    print(
        f"{rows} rows x {len(data.columns)} columns; "
        f"{int((data.isna().any(axis=1)).sum())} row(s) have at least one missing value."
    )
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    """Handle the describe subcommand."""
    data = _load(args)

    numeric = data.select_dtypes(include="number")
    categorical = data.select_dtypes(exclude="number")

    if args.include == "numeric":
        categorical = categorical.iloc[:, :0]
    elif args.include == "categorical":
        numeric = numeric.iloc[:, :0]

    if numeric.empty and categorical.empty:
        print("No columns to describe.")
        return 0

    if not numeric.empty:
        print("Numeric columns")
        print("-" * 78)
        print(f"{'Column':<22}{'count':>8}{'mean':>12}{'std':>12}{'min':>12}{'max':>12}")
        for column in numeric.columns:
            series = numeric[column]
            print(
                f"{str(column)[:21]:<22}{int(series.count()):>8}"
                f"{series.mean():>12.4g}{series.std():>12.4g}"
                f"{series.min():>12.4g}{series.max():>12.4g}"
            )

    if not categorical.empty:
        if not numeric.empty:
            print()
        print("Categorical columns")
        print("-" * 78)
        print(f"{'Column':<22}{'count':>8}{'unique':>10}{'top':>20}{'freq':>10}")
        for column in categorical.columns:
            series = categorical[column].dropna()
            if series.empty:
                print(f"{str(column)[:21]:<22}{0:>8}{0:>10}{'-':>20}{'-':>10}")
                continue
            counts = series.value_counts()
            print(
                f"{str(column)[:21]:<22}{int(series.count()):>8}"
                f"{int(series.nunique()):>10}{str(counts.index[0])[:19]:>20}"
                f"{int(counts.iloc[0]):>10}"
            )

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # `rename` takes --output as its destination file, so it is not a redirect.
    redirect = getattr(args, "output", None) if args.command != "rename" else None
    if redirect:
        import contextlib

        try:
            handle = open(redirect, "w", encoding="utf-8", newline="")
        except OSError as exc:
            print(f"Cannot write to '{redirect}': {exc}", file=sys.stderr)
            return 2
        with handle, contextlib.redirect_stdout(handle):
            return _dispatch(args, parser)
    return _dispatch(args, parser)


def _dispatch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:

    if getattr(args, "version", False):
        from . import __version__
        print(f"dataset-audit-kit v{__version__} (Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})")
        return 0

    if args.command is None:
        parser.print_help(sys.stderr)
        print("\nError: a command is required.", file=sys.stderr)
        return 2

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
    elif args.command == "describe":
        return _cmd_describe(args)
    elif args.command == "missing":
        return _cmd_missing(args)
    elif args.command == "stats":
        return _cmd_stats(args)
    elif args.command == "profile":
        return _cmd_profile(args)
    elif args.command == "rename":
        return _cmd_rename(args)
    elif args.command == "schema":
        return _cmd_schema(args)
    elif args.command == "infer-rules":
        return _cmd_infer_rules(args)
    elif args.command == "diff":
        return _cmd_diff(args)
    elif args.command == "refs":
        return _cmd_refs(args)
    else:
        parser.error("unsupported command")

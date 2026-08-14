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
    unique_parser.add_argument("--top", type=int, default=20, help="Show top N values (default: 20)")

    dtype_parser = subparsers.add_parser("dtype", help="Show column dtypes with inferred optimal types")
    dtype_parser.add_argument("data", help="Path to the dataset")

    correlate_parser = subparsers.add_parser("correlate", help="Show pairwise correlation matrix")
    correlate_parser.add_argument("data", help="Path to the dataset")
    correlate_parser.add_argument("--method", default="pearson", choices=["pearson", "spearman", "kendall"], help="Correlation method")

    shape_parser = subparsers.add_parser("shape", help="Show dataset shape (rows x columns)")
    shape_parser.add_argument("data", help="Path to the dataset")
    shape_parser.add_argument("--csv", action="store_true", help="CSV output (rows,columns)")

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

    return parser


def _load(args: argparse.Namespace) -> "pd.DataFrame":
    """Load the dataset named by args, honouring --encoding when given."""
    return DatasetAuditor.load_dataframe(args.data, encoding=getattr(args, "encoding", None))


def _cmd_head(args: argparse.Namespace) -> int:
    data = _load(args)
    print(data.head(args.rows).to_csv(index=False))
    return 0


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
    print(f"{'Column':<30} {'Dtype':<15} {'Non-null':<10} {'Missing':<10}")
    print("-" * 65)
    for col in cols:
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

    select_columns = _parse_columns(args.select_columns)
    exclude_columns = _parse_columns(getattr(args, "exclude_columns", None))
    if select_columns and exclude_columns:
        print(
            "Error: --select-columns and --exclude-columns are mutually exclusive.",
            file=sys.stderr,
        )
        return 2
    if select_columns or exclude_columns:
        data = _load(args)
        if select_columns:
            present = [c for c in select_columns if c in data.columns]
            missing = [c for c in select_columns if c not in data.columns]
        else:
            missing = [c for c in exclude_columns if c not in data.columns]
            present = [c for c in data.columns if c not in set(exclude_columns)]
        if missing:
            print(f"Warning: requested columns not found in dataset: {missing}", file=sys.stderr)
        if not present:
            print("Error: no columns left to audit after filtering.", file=sys.stderr)
            return 2
        data = data[present]
        reference = DatasetAuditor.load_dataframe(args.reference) if args.reference else None
        report = auditor.audit_dataframe(
            data,
            reference=reference,
            label_column=args.label_column,
            expected_columns=_parse_columns(args.expected_columns),
            unique_columns=_parse_columns(args.unique_columns),
        )
    else:
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
    print(f"{'Value':<40} {'Count':<10}")
    print("-" * 50)
    for val, cnt in counts.items():
        print(f"{str(val):<40} {cnt:<10}")
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
    data = _load(args)
    numeric = data.select_dtypes(include="number")
    if numeric.empty:
        print("No numeric columns found.", file=sys.stderr)
        return 1
    corr = numeric.corr(method=args.method)
    print(corr.to_csv(float_format="%.4f"))
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

    if getattr(args, "version", False):
        from . import __version__
        print(f"dataset-audit-kit v{__version__} (Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})")
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
    else:
        parser.error("unsupported command")
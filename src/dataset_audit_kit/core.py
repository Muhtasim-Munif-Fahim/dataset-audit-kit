"""Core dataset validation logic."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Sequence

import pandas as pd

#: Compression suffixes pandas can infer from a filename, for text formats.
COMPRESSION_SUFFIXES = frozenset({".gz", ".bz2", ".zip", ".xz", ".zst"})

#: URL schemes pandas hands to fsspec rather than opening from the filesystem.
REMOTE_SCHEMES = (
    "s3://", "gs://", "gcs://", "az://", "abfs://", "adl://",
    "http://", "https://", "ftp://", "sftp://", "hdfs://",
)


def _format_stat(value: object, places: int = 3) -> str:
    """Render a summary statistic, tolerating a missing or non-numeric value.

    A numeric column that is entirely missing has no mean or standard
    deviation, so the profile carries None (or NaN) for them.
    """

    if value is None:
        return "?"
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "?"
    return f"{number:.{places}f}"


@dataclass(frozen=True)
class AuditIssue:
    """A single dataset-quality issue."""

    check: str
    severity: str
    message: str
    column: str | None = None
    observed: float | int | None = None
    threshold: float | int | None = None


@dataclass(frozen=True)
class ColumnRule:
    """Per-column validation rule.

    Attributes
    ----------
    name : str
        Column name to which this rule applies.
    dtype : str | None
        Expected data type: ``"numeric"``, ``"categorical"``, or ``"string"``.
        If set, the auditor checks whether the column's inferred type matches.
    min_value : float | None
        Minimum allowed value (numeric columns only).
    max_value : float | None
        Maximum allowed value (numeric columns only).
    allowed_values : list[str] | None
        Set of allowed values (categorical/string columns only).
    max_missing_ratio : float | None
        Maximum allowed fraction of missing values for this column.
        Overrides the global ``missing_threshold`` when set.
    """

    name: str
    dtype: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[str] | None = None
    max_missing_ratio: float | None = None


@dataclass(frozen=True)
class ValidationRules:
    """A collection of per-column validation rules.

    Parameters
    ----------
    columns : dict[str, ColumnRule]
        Mapping from column name to its validation rule.
    """

    columns: dict[str, ColumnRule] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, rules: dict[str, dict[str, object]]) -> "ValidationRules":
        """Build rules from a plain dictionary (e.g. loaded from JSON)."""
        column_rules: dict[str, ColumnRule] = {}
        for col_name, col_config in rules.items():
            column_rules[col_name] = ColumnRule(
                name=col_name,
                dtype=str(col_config.get("dtype") or "").lower() or None if col_config.get("dtype") else None,
                min_value=col_config.get("min_value"),
                max_value=col_config.get("max_value"),
                allowed_values=col_config.get("allowed_values"),
                max_missing_ratio=col_config.get("max_missing_ratio"),
            )
        return cls(columns=column_rules)

    @classmethod
    def from_json(cls, path: str) -> "ValidationRules":
        """Load rules from a JSON file."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, dict[str, object]] = json.load(f)
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, dict[str, object]]:
        """Serialize to a plain dictionary for JSON export."""
        result: dict[str, dict[str, object]] = {}
        for name, rule in self.columns.items():
            entry: dict[str, object] = {}
            for attr in ("dtype", "min_value", "max_value", "allowed_values", "max_missing_ratio"):
                value = getattr(rule, attr)
                if value is not None:
                    entry[attr] = value
            result[name] = entry
        return result


@dataclass
class AuditReport:
    """Structured output from a dataset audit."""

    rows: int
    columns: int
    duplicate_rows: int
    missing_cells: int
    missingness: dict[str, float] = field(default_factory=dict)
    label_distribution: dict[str, int] = field(default_factory=dict)
    drift_scores: dict[str, float] = field(default_factory=dict)
    column_profiles: dict[str, dict[str, object]] = field(default_factory=dict)
    issues: list[AuditIssue] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "pass" if not self.issues else "warn"

    @property
    def quality_score(self) -> int:
        score = 100
        for issue in self.issues:
            if issue.severity == "error":
                score -= 15
            elif issue.severity == "warning":
                score -= 5
        return max(0, score)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "quality_score": self.quality_score,
            "rows": self.rows,
            "columns": self.columns,
            "duplicate_rows": self.duplicate_rows,
            "missing_cells": self.missing_cells,
            "missingness": self.missingness,
            "label_distribution": self.label_distribution,
            "drift_scores": self.drift_scores,
            "column_profiles": self.column_profiles,
            "issues": [issue.__dict__ for issue in self.issues],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_markdown(self) -> str:
        lines = [
            "# Dataset Audit Report",
            "",
            f"- Status: **{self.status}**",
            f"- Quality score: **{self.quality_score}/100**",
            f"- Rows: **{self.rows}**",
            f"- Columns: **{self.columns}**",
            f"- Duplicate rows: **{self.duplicate_rows}**",
            f"- Missing cells: **{self.missing_cells}**",
        ]

        if self.column_profiles:
            lines.extend(["", "## Column profiles"])
            for col, profile in self.column_profiles.items():
                dtype = profile.get("dtype", "?")
                count = profile.get("count", "?")
                missing_pct = f"{float(profile.get('missing', 0)) / max(int(count), 1) * 100:.1f}%" if count else "?"
                lines.append(f"\n### {col} ({dtype})")
                lines.append(f"- Count: {count}, Missing: {profile.get('missing', 0)} ({missing_pct}), Unique: {profile.get('unique', '?')}")
                if dtype == "numeric":
                    lines.append(f"- Range: {profile.get('min', '?')} - {profile.get('max', '?')}")
                    lines.append(
                        f"- Mean: {_format_stat(profile.get('mean'))}, "
                        f"Std: {_format_stat(profile.get('std'))}"
                    )
                    lines.append(f"- Quartiles: Q1={profile.get('q25', '?')} Q2={profile.get('q50', '?')} Q3={profile.get('q75', '?')}")
                elif dtype == "categorical":
                    top_val = profile.get("top", "?")
                    top_freq = profile.get("freq", "?")
                    lines.append(f"- Top value: {top_val} ({top_freq})")

        if self.label_distribution:
            lines.extend(["", "## Label distribution"])
            for label, count in self.label_distribution.items():
                lines.append(f"- `{label}`: {count}")

        if self.missingness:
            lines.extend(["", "## Missingness"])
            for column, ratio in self.missingness.items():
                lines.append(f"- `{column}`: {ratio:.1%}")

        if self.drift_scores:
            lines.extend(["", "## Drift scores"])
            for column, score in self.drift_scores.items():
                lines.append(f"- `{column}`: {score:.3f}")

        if self.issues:
            lines.extend(["", "## Issues"])
            for issue in self.issues:
                parts = [f"- **{issue.severity.upper()}**", f"`{issue.check}`", issue.message]
                if issue.column:
                    parts.insert(2, f"column `{issue.column}`")
                lines.append(" ".join(parts))

            # Summary counts by check type
            from collections import Counter
            check_counts = Counter(i.check for i in self.issues)
            lines.append("")
            lines.append("*Issue summary:*")
            for check, count in check_counts.most_common():
                lines.append(f"  - `{check}`: {count} issue(s)")
        else:
            lines.extend(["", "_No issues found._"])

        return "\n".join(lines)

    def to_html(self) -> str:
        """Render the report as a standalone HTML document."""

        def esc(value: object) -> str:
            return html.escape(str(value), quote=True)

        issue_rows = []
        if self.issues:
            for issue in self.issues:
                issue_rows.append(
                    "<tr>"
                    f"<td>{esc(issue.severity.upper())}</td>"
                    f"<td>{esc(issue.check)}</td>"
                    f"<td>{esc(issue.column or '')}</td>"
                    f"<td>{esc(issue.message)}</td>"
                    f"<td>{esc(issue.observed if issue.observed is not None else '')}</td>"
                    f"<td>{esc(issue.threshold if issue.threshold is not None else '')}</td>"
                    "</tr>"
                )
        else:
            issue_rows.append('<tr><td colspan="6"><em>No issues found.</em></td></tr>')

        missingness_rows = []
        for column, ratio in self.missingness.items():
            missingness_rows.append(f"<tr><td>{esc(column)}</td><td>{ratio:.1%}</td></tr>")

        drift_rows = []
        for column, score in self.drift_scores.items():
            drift_rows.append(f"<tr><td>{esc(column)}</td><td>{score:.3f}</td></tr>")

        label_rows = []
        for label, count in self.label_distribution.items():
            label_rows.append(f"<tr><td>{esc(label)}</td><td>{count}</td></tr>")

        sections = [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>Dataset Audit Report</title>",
            "<style>",
            "body{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:32px;line-height:1.5;color:#1f2937;background:#f8fafc;}",
            "main{max-width:960px;margin:0 auto;background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:32px;box-shadow:0 8px 24px rgba(15,23,42,.06);}",
            "h1,h2{line-height:1.2;}",
            "table{width:100%;border-collapse:collapse;margin:12px 0 24px;}",
            "th,td{border:1px solid #e5e7eb;padding:10px 12px;text-align:left;vertical-align:top;}",
            "th{background:#f9fafb;}",
            ".metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:16px 0 24px;}",
            ".metric{border:1px solid #e5e7eb;border-radius:12px;padding:12px;background:#f9fafb;}",
            ".metric strong{display:block;font-size:1.4rem;margin-top:4px;}",
            ".pass{color:#166534;}",
            ".warn{color:#b45309;}",
            ".muted{color:#6b7280;}",
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            "<h1>Dataset Audit Report</h1>",
            f'<p class="muted">Status: <strong class="{esc(self.status)}">{esc(self.status)}</strong></p>',
            '<div class="metrics">',
            f'<div class="metric"><span>Rows</span><strong>{self.rows}</strong></div>',
            f'<div class="metric"><span>Columns</span><strong>{self.columns}</strong></div>',
            f'<div class="metric"><span>Duplicate rows</span><strong>{self.duplicate_rows}</strong></div>',
            f'<div class="metric"><span>Missing cells</span><strong>{self.missing_cells}</strong></div>',
            f'<div class="metric"><span>Quality score</span><strong>{self.quality_score}/100</strong></div>',
            "</div>",
        ]

        if label_rows:
            sections.extend([
                "<h2>Label distribution</h2>",
                "<table><thead><tr><th>Label</th><th>Count</th></tr></thead><tbody>",
                *label_rows,
                "</tbody></table>",
            ])

        if missingness_rows:
            sections.extend([
                "<h2>Missingness</h2>",
                "<table><thead><tr><th>Column</th><th>Missing share</th></tr></thead><tbody>",
                *missingness_rows,
                "</tbody></table>",
            ])

        if self.column_profiles:
            profile_sections = ["<h2>Column profiles</h2>"]
            for col, profile in self.column_profiles.items():
                dtype = profile.get("dtype", "?")
                profile_sections.append(
                    f"<details><summary>{esc(col)} ({dtype})</summary>"
                    f"<table>"
                    f"<tr><td>Count</td><td>{profile.get('count', '?')}</td></tr>"
                    f"<tr><td>Missing</td><td>{profile.get('missing', 0)}</td></tr>"
                    f"<tr><td>Unique</td><td>{profile.get('unique', '?')}</td></tr>"
                )
                if dtype == "numeric":
                    profile_sections.extend([
                        f"<tr><td>Min</td><td>{profile.get('min', '?')}</td></tr>",
                        f"<tr><td>Max</td><td>{profile.get('max', '?')}</td></tr>",
                        f"<tr><td>Mean</td><td>{profile.get('mean', '?'):.3f}</td></tr>",
                        f"<tr><td>Std</td><td>{profile.get('std', '?'):.3f}</td></tr>",
                        f"<tr><td>Q1</td><td>{profile.get('q25', '?')}</td></tr>",
                        f"<tr><td>Q2 (median)</td><td>{profile.get('q50', '?')}</td></tr>",
                        f"<tr><td>Q3</td><td>{profile.get('q75', '?')}</td></tr>",
                    ])
                elif dtype == "categorical":
                    top_val = profile.get("top", "?")
                    top_freq = profile.get("freq", "?")
                    profile_sections.extend([
                        f"<tr><td>Top value</td><td>{esc(str(top_val))} ({top_freq})</td></tr>",
                    ])
                profile_sections.append("</table></details>")
            sections.extend(profile_sections)

        if drift_rows:
            sections.extend([
                "<h2>Drift scores</h2>",
                "<table><thead><tr><th>Column</th><th>Score</th></tr></thead><tbody>",
                *drift_rows,
                "</tbody></table>",
            ])

        sections.extend([
            "<h2>Issues</h2>",
            '<table><thead><tr><th>Severity</th><th>Check</th><th>Column</th><th>Message</th><th>Observed</th><th>Threshold</th></tr></thead><tbody>',
            *issue_rows,
            "</tbody></table>",
            "</main>",
            "</body>",
            "</html>",
        ])
        return "\n".join(sections)


    @property
    def fix_suggestions(self) -> list[dict[str, str]]:
        """Generate actionable fix suggestions for each issue in the report."""
        suggestions: list[dict[str, str]] = []
        seen: set[str] = set()

        for issue in self.issues:
            key = f"{issue.check}:{issue.column or ''}"
            if key in seen:
                continue
            seen.add(key)

            suggestion: dict[str, str] = {
                "issue": issue.message,
                "severity": issue.severity,
            }

            if issue.check == "duplicates":
                suggestion["action"] = "drop_duplicates"
                suggestion["code"] = "df = df.drop_duplicates()"
                suggestion["description"] = "Remove duplicate rows from the dataset."
            elif issue.check == "missingness":
                col = issue.column or ""
                suggestion["action"] = "impute_median"
                suggestion["code"] = f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())"
                suggestion["description"] = f"Impute missing values in '{col}' with the median."
            elif issue.check == "schema" and "Missing expected columns" in issue.message:
                suggestion["action"] = "add_columns"
                suggestion["code"] = "df = df.reindex(columns=expected_columns)"
                suggestion["description"] = "Add missing columns with NaN values."
            elif issue.check == "schema" and "Unexpected columns" in issue.message:
                suggestion["action"] = "drop_columns"
                suggestion["code"] = "df = df[expected_columns]"
                suggestion["description"] = "Drop columns not in the expected schema."
            elif issue.check == "rule" and "above maximum" in issue.message:
                suggestion["action"] = "clip_values"
                col = issue.column or ""
                suggestion["code"] = f"df['{col}'] = df['{col}'].clip(upper=max_value)"
                suggestion["description"] = f"Clip values in '{col}' to the allowed maximum."
            elif issue.check == "rule" and "below minimum" in issue.message:
                suggestion["action"] = "clip_values"
                col = issue.column or ""
                suggestion["code"] = f"df['{col}'] = df['{col}'].clip(lower=min_value)"
                suggestion["description"] = f"Clip values in '{col}' to the allowed minimum."
            elif issue.check == "rule" and "Unexpected values" in issue.message:
                suggestion["action"] = "replace_values"
                col = issue.column or ""
                suggestion["code"] = f"df['{col}'] = df['{col}'].where(df['{col}'].isin(allowed_values), other=default)"
                suggestion["description"] = f"Replace unexpected values in '{col}' with a default."
            elif issue.check == "rule" and "missing column" in issue.message:
                suggestion["action"] = "remove_rule"
                col = issue.column or ""
                suggestion["code"] = f"# Remove rule for '{col}' from rules config"
                suggestion["description"] = f"Column '{col}' defined in rules but not in dataset."
            elif issue.check == "labels" and "missing label" in issue.message:
                suggestion["action"] = "drop_missing_labels"
                col = issue.column or ""
                suggestion["code"] = f"df = df.dropna(subset=['{col}'])"
                suggestion["description"] = f"Drop rows with missing label values in '{col}'."
            elif issue.check == "labels" and "imbalanced" in issue.message:
                suggestion["action"] = "resample"
                col = issue.column or ""
                suggestion["code"] = "# Consider stratified sampling or class weighting"
                suggestion["description"] = f"Address label imbalance in '{col}' via resampling or weighting."
            elif issue.check == "drift":
                suggestion["action"] = "investigate_drift"
                suggestion["code"] = "# Review data pipeline for distribution shift source"
                suggestion["description"] = "Investigate root cause of distribution drift."
            elif issue.check == "correlation_drift":
                suggestion["action"] = "investigate_correlation_shift"
                suggestion["code"] = "# Compare feature generation logic between reference and current"
                suggestion["description"] = "Investigate pairwise correlation structure change."
            elif issue.check == "uniqueness":
                col = issue.column or ""
                suggestion["action"] = "deduplicate_column"
                suggestion["code"] = f"df = df.drop_duplicates(subset=['{col}'])"
                suggestion["description"] = f"Remove rows with duplicate values in '{col}'."
            else:
                suggestion["action"] = "manual_review"
                suggestion["code"] = "# No automated fix available"
                suggestion["description"] = "Manual review required."

            suggestions.append(suggestion)

        return suggestions

    def to_file(self, path: str) -> str:
        """Write the report to a file, auto-detecting format from extension.

        Supported extensions:
        - ``.json`` — JSON format
        - ``.md`` — Markdown format
        - ``.html`` — HTML format

        Parameters
        ----------
        path : str
            Output file path.

        Returns
        -------
        str
            The path that was written to.
        """
        path_obj = Path(path)
        suffix = path_obj.suffix.lower()

        if suffix == ".json":
            content = self.to_json()
        elif suffix == ".md":
            content = self.to_markdown()
        elif suffix == ".html":
            content = self.to_html()
        else:
            raise ValueError(
                f"Unsupported report format '{suffix}'. "
                "Supported formats are .json, .md, .html."
            )

        path_obj.write_text(content, encoding="utf-8")
        return path





    @classmethod
    def diff(cls, before: 'AuditReport', after: 'AuditReport') -> 'AuditReport':
        issues_diff = []
        before_keys = {(i.check, i.column or '') for i in before.issues}
        after_keys = {(i.check, i.column or '') for i in after.issues}
        for issue in after.issues:
            key = (issue.check, issue.column or '')
            if key not in before_keys:
                issues_diff.append(issue)
        for issue in before.issues:
            key = (issue.check, issue.column or '')
            if key not in after_keys:
                issues_diff.append(
                    AuditIssue(
                        check=issue.check,
                        severity='info',
                        message=f'Resolved: {issue.message}',
                        column=issue.column,
                    )
                )
        return cls(
            rows=after.rows - before.rows,
            columns=after.columns - before.columns,
            duplicate_rows=after.duplicate_rows - before.duplicate_rows,
            missing_cells=after.missing_cells - before.missing_cells,
            issues=issues_diff,
        )

class DatasetAuditor:
    """Run a small battery of quality checks over tabular data."""

    def __init__(
        self,
        *,
        missing_threshold: float = 0.05,
        drift_threshold: float = 0.20,
        label_min_share: float = 0.05,
        rules: ValidationRules | None = None,
    ) -> None:
        self.missing_threshold = missing_threshold
        self.drift_threshold = drift_threshold
        self.label_min_share = label_min_share
        self.rules = rules

    def audit_dataframe(
        self,
        data: pd.DataFrame,
        *,
        reference: pd.DataFrame | None = None,
        label_column: str | None = None,
        expected_columns: Sequence[str] | None = None,
        unique_columns: Sequence[str] | None = None,
    ) -> AuditReport:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas.DataFrame")

        issues: list[AuditIssue] = []
        missingness = self._missingness(data, issues)
        duplicate_rows = int(data.duplicated().sum())
        if duplicate_rows:
            issues.append(
                AuditIssue(
                    check="duplicates",
                    severity="warning",
                    message=f"{duplicate_rows} duplicate row(s) detected.",
                    observed=duplicate_rows,
                )
            )

        self._check_column_names(data, issues)

        if expected_columns is not None:
            self._check_schema(data, expected_columns, issues)

        if unique_columns is not None:
            self._check_uniqueness(data, unique_columns, issues)

        label_distribution: dict[str, int] = {}
        if label_column is not None and label_column in data.columns:
            label_distribution = self._check_label_balance(data, label_column, issues)

        drift_scores: dict[str, float] = {}
        correlation_drift_scores: dict[str, float] = {}
        if reference is not None:
            drift_scores = self._check_drift(data, reference, issues, label_column=label_column)
            correlation_drift_scores = self._correlation_drift(
                data, reference, issues, drift_threshold=self.drift_threshold
            )

        # Schema diff between reference and current
        schema_diff_summary: dict[str, dict[str, object]] = {}
        if reference is not None:
            schema_diff_summary = self._schema_diff(data, reference, issues)

        # Per-column validation rules
        self._apply_rules(data, issues)

        column_profiles = self._profile_columns(data)

        # Redundancy / collinearity check
        self._check_redundancy(data, issues, correlation_threshold=0.95)

        all_drift_scores = {**drift_scores, **correlation_drift_scores}

        return AuditReport(
            rows=int(len(data)),
            columns=int(len(data.columns)),
            duplicate_rows=duplicate_rows,
            missing_cells=int(data.isna().sum().sum()),
            missingness=missingness,
            column_profiles=column_profiles,
            label_distribution=label_distribution,
            drift_scores=all_drift_scores,
            issues=issues,
        )

    def audit_csv(
        self,
        data_path: str,
        *,
        reference_path: str | None = None,
        label_column: str | None = None,
        expected_columns: Sequence[str] | None = None,
    ) -> AuditReport:
        return self.audit_file(
            data_path,
            reference_path=reference_path,
            label_column=label_column,
            expected_columns=expected_columns,
        )

    def audit_file(
        self,
        data_path: str,
        *,
        reference_path: str | None = None,
        label_column: str | None = None,
        expected_columns: Sequence[str] | None = None,
        unique_columns: Sequence[str] | None = None,
    ) -> AuditReport:
        data = self.load_dataframe(data_path)
        reference = self.load_dataframe(reference_path) if reference_path else None
        return self.audit_dataframe(
            data,
            reference=reference,
            label_column=label_column,
            expected_columns=expected_columns,
            unique_columns=unique_columns,
        )

    @staticmethod
    def load_dataframe(
        path: str | Path,
        encoding: str | None = None,
        delimiter: str | None = None,
    ) -> pd.DataFrame:
        """Load a tabular dataset from CSV, TSV, JSONL/NDJSON, Parquet, or Excel.

        Text formats may additionally carry a compression suffix, for example
        ``.csv.gz``. pandas infers the codec from the filename, so the suffix is
        only stripped here to work out which reader to dispatch to.

        ``encoding`` applies to the text formats only; Parquet and Excel carry
        their own encoding information and ignore it.

        ``delimiter`` overrides the separator for delimited text. Without it,
        ``.csv`` and ``.tsv`` use their conventional separators and ``.txt`` is
        sniffed, since the extension says nothing about the format.
        """

        if DatasetAuditor._is_remote(path):
            return DatasetAuditor._load_remote(str(path), encoding, delimiter)

        dataset_path = Path(path)
        suffix = DatasetAuditor._data_suffix(dataset_path)

        if suffix in {".csv", ".tsv", ".txt"}:
            sep = delimiter or DatasetAuditor._default_delimiter(
                suffix, dataset_path, encoding
            )
            return pd.read_csv(dataset_path, sep=sep, encoding=encoding)
        if suffix in {".jsonl", ".ndjson"}:
            return pd.read_json(dataset_path, lines=True, encoding=encoding)
        if suffix == ".parquet":
            return pd.read_parquet(dataset_path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(dataset_path, engine="openpyxl" if suffix == ".xlsx" else "xlrd")

        raise ValueError(
            f"Unsupported dataset format for `{dataset_path}`. "
            "Supported formats are .csv, .tsv, .txt, .jsonl, .ndjson, .parquet, "
            "and .xlsx/.xls, optionally compressed with "
            f"{', '.join(sorted(COMPRESSION_SUFFIXES))}."
        )

    @staticmethod
    def _default_delimiter(
        suffix: str, dataset_path: Path, encoding: str | None
    ) -> str:
        """Pick a separator for a delimited text file.

        ``.csv`` and ``.tsv`` name their separator by convention. ``.txt`` does
        not, so the first non-empty line is sniffed against the delimiters that
        actually occur in exported data. Sniffing failures fall back to a comma
        rather than raising, so a one-column ``.txt`` still loads.
        """

        if suffix == ".csv":
            return ","
        if suffix == ".tsv":
            return "\t"

        try:
            with open(
                dataset_path, "r", encoding=encoding or "utf-8", newline=""
            ) as handle:
                sample = handle.readline()
                while sample.strip() == "" and sample != "":
                    sample = handle.readline()
        except (OSError, UnicodeDecodeError):
            return ","

        if not sample.strip():
            return ","

        # csv.Sniffer misreads text containing prose, so count instead: the
        # candidate appearing most often on the header line wins.
        counts = {candidate: sample.count(candidate) for candidate in ("\t", "|", ";", ",")}
        best = max(counts, key=lambda candidate: counts[candidate])
        return best if counts[best] else ","

    @staticmethod
    def _is_remote(path: str | Path) -> bool:
        """True when the path is a URL pandas should fetch rather than open."""

        return str(path).startswith(REMOTE_SCHEMES)

    @staticmethod
    def _load_remote(
        url: str, encoding: str | None, delimiter: str | None
    ) -> pd.DataFrame:
        """Read a dataset from a remote URL.

        pandas delegates object-store URLs to fsspec, so the URL is handed over
        untouched — normalizing it through Path would collapse the `//` after
        the scheme. Only the suffix is inspected here, to pick a reader.
        """

        # Strip any query string before looking at the extension.
        without_query = url.split("?", 1)[0].split("#", 1)[0]
        suffix = DatasetAuditor._data_suffix(PurePosixPath(without_query))

        if suffix in {".csv", ".tsv", ".txt"}:
            sep = delimiter or (",", "\t")[suffix == ".tsv"]
            return pd.read_csv(url, sep=sep, encoding=encoding)
        if suffix in {".jsonl", ".ndjson"}:
            return pd.read_json(url, lines=True, encoding=encoding)
        if suffix == ".parquet":
            return pd.read_parquet(url)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(url, engine="openpyxl" if suffix == ".xlsx" else "xlrd")

        raise ValueError(
            f"Unsupported dataset format for remote path `{url}`. "
            "Supported formats are .csv, .tsv, .txt, .jsonl, .ndjson, .parquet, "
            "and .xlsx/.xls."
        )

    @staticmethod
    def _data_suffix(dataset_path: Path | PurePosixPath) -> str:
        """Return the format suffix, ignoring a trailing compression suffix."""

        suffixes = [suffix.lower() for suffix in dataset_path.suffixes]
        if suffixes and suffixes[-1] in COMPRESSION_SUFFIXES:
            # `.csv.gz` -> `.csv`; a bare `.gz` leaves nothing to dispatch on.
            return suffixes[-2] if len(suffixes) > 1 else ""
        return dataset_path.suffix.lower()

    def _missingness(self, data: pd.DataFrame, issues: list[AuditIssue]) -> dict[str, float]:
        ratios = (data.isna().mean()).sort_values(ascending=False)
        missingness: dict[str, float] = {}
        for column, ratio in ratios.items():
            ratio = float(ratio)
            if ratio <= 0:
                continue
            missingness[column] = ratio
            if ratio >= self.missing_threshold:
                issues.append(
                    AuditIssue(
                        check="missingness",
                        severity="warning",
                        message=(
                            f"{ratio:.1%} missing values exceed the {self.missing_threshold:.1%} threshold."
                        ),
                        column=column,
                        observed=ratio,
                        threshold=self.missing_threshold,
                    )
                )
        return missingness

    def _check_column_names(self, data: pd.DataFrame, issues: list[AuditIssue]) -> None:
        """Flag column names that tend to break downstream tooling.

        These are warnings, not errors: pandas is happy with any of them, but
        they routinely cause trouble once the frame reaches SQL, Parquet
        round-trips, ``df.query`` or attribute access.
        """

        seen: dict[str, str] = {}
        for column in data.columns:
            name = str(column)

            if name != name.strip():
                issues.append(
                    AuditIssue(
                        check="column_names",
                        severity="warning",
                        message="Column name has leading or trailing whitespace.",
                        column=name,
                        observed=repr(name),
                    )
                )

            if not name.strip():
                issues.append(
                    AuditIssue(
                        check="column_names",
                        severity="error",
                        message="Column name is empty or whitespace only.",
                        column=name,
                        observed=repr(name),
                    )
                )
                continue

            if any(char.isspace() for char in name.strip()):
                issues.append(
                    AuditIssue(
                        check="column_names",
                        severity="warning",
                        message=(
                            "Column name contains whitespace, which blocks "
                            "attribute access and needs quoting in df.query()."
                        ),
                        column=name,
                        observed=repr(name),
                    )
                )

            lowered = name.strip().lower()
            if lowered in seen and seen[lowered] != name:
                issues.append(
                    AuditIssue(
                        check="column_names",
                        severity="error",
                        message=(
                            f"Column name differs from `{seen[lowered]}` only by case "
                            "or surrounding whitespace, which collides in "
                            "case-insensitive stores such as SQL Server."
                        ),
                        column=name,
                        observed=repr(name),
                    )
                )
            else:
                seen.setdefault(lowered, name)

    def _check_schema(
        self,
        data: pd.DataFrame,
        expected_columns: Sequence[str],
        issues: list[AuditIssue],
    ) -> None:
        expected = list(expected_columns)
        observed = list(data.columns)
        missing = [column for column in expected if column not in observed]
        extra = [column for column in observed if column not in expected]

        if missing:
            issues.append(
                AuditIssue(
                    check="schema",
                    severity="error",
                    message=f"Missing expected columns: {', '.join(missing)}.",
                )
            )

        if extra:
            issues.append(
                AuditIssue(
                    check="schema",
                    severity="warning",
                    message=f"Unexpected columns present: {', '.join(extra)}.",
                )
            )

    @staticmethod
    def _check_uniqueness(
        data: pd.DataFrame,
        unique_columns: Sequence[str],
        issues: list[AuditIssue],
    ) -> None:
        """Check that specified columns contain only unique values."""
        for col in unique_columns:
            if col not in data.columns:
                issues.append(
                    AuditIssue(
                        check="uniqueness",
                        severity="error",
                        message=f"Unique column '{col}' not found in dataset.",
                        column=col,
                    )
                )
                continue
            total = len(data[col])
            duplicates = int(data[col].duplicated(keep=False).sum()) // 2
            if duplicates > 0:
                issues.append(
                    AuditIssue(
                        check="uniqueness",
                        severity="warning",
                        message=f"{duplicates} duplicate value(s) in unique column '{col}'.",
                        column=col,
                        observed=duplicates,
                    )
                )

    def _check_label_balance(
        self,
        data: pd.DataFrame,
        label_column: str,
        issues: list[AuditIssue],
    ) -> dict[str, int]:
        counts = data[label_column].fillna("<missing>").astype(str).value_counts()
        total = int(counts.sum()) or 1
        missing_count = int(data[label_column].isna().sum())
        if missing_count:
            issues.append(
                AuditIssue(
                    check="labels",
                    severity="warning",
                    message=f"{missing_count} missing label value(s) in `{label_column}`.",
                    column=label_column,
                    observed=missing_count,
                )
            )

        if counts.empty:
            issues.append(
                AuditIssue(
                    check="labels",
                    severity="warning",
                    message=f"No label values found in `{label_column}`.",
                    column=label_column,
                    observed=0,
                )
            )
        elif counts.size < 2:
            issues.append(
                AuditIssue(
                    check="labels",
                    severity="warning",
                    message="Only one label value is present.",
                    column=label_column,
                    observed=int(counts.iloc[0]),
                )
            )
        else:
            minority_share = counts.min() / total
            if minority_share < self.label_min_share:
                issues.append(
                    AuditIssue(
                        check="labels",
                        severity="warning",
                        message=f"Label distribution is imbalanced (minority share {minority_share:.1%}).",
                        column=label_column,
                        observed=minority_share,
                        threshold=self.label_min_share,
                    )
                )

        return {str(label): int(count) for label, count in counts.items()}

    def _apply_rules(
        self,
        data: pd.DataFrame,
        issues: list[AuditIssue],
    ) -> None:
        """Evaluate per-column validation rules against the dataset."""
        if self.rules is None:
            return

        for column_name, rule in self.rules.columns.items():
            if column_name not in data.columns:
                issues.append(
                    AuditIssue(
                        check="rule",
                        severity="error",
                        message=f"Rule defined for missing column '{column_name}'.",
                        column=column_name,
                    )
                )
                continue

            col_data = data[column_name]

            # --- dtype check ---
            if rule.dtype is not None:
                inferred = self._infer_dtype(col_data)
                if inferred != rule.dtype:
                    issues.append(
                        AuditIssue(
                            check="rule",
                            severity="error",
                            message=f"Expected dtype '{rule.dtype}', inferred '{inferred}'.",
                            column=column_name,
                        )
                    )

            # --- missingness check (per-column override) ---
            if rule.max_missing_ratio is not None:
                missing_ratio = float(col_data.isna().mean())
                if missing_ratio > rule.max_missing_ratio:
                    issues.append(
                        AuditIssue(
                            check="rule",
                            severity="warning",
                            message=f"Missing ratio {missing_ratio:.1%} exceeds allowed {rule.max_missing_ratio:.1%}.",
                            column=column_name,
                            observed=missing_ratio,
                            threshold=rule.max_missing_ratio,
                        )
                    )

            # --- numeric bounds ---
            if rule.min_value is not None or rule.max_value is not None:
                numeric = pd.to_numeric(col_data.dropna(), errors="coerce")
                if rule.min_value is not None:
                    violations = (numeric < rule.min_value).sum()
                    if violations:
                        issues.append(
                            AuditIssue(
                                check="rule",
                                severity="warning",
                                message=f"{int(violations)} value(s) below minimum {rule.min_value}.",
                                column=column_name,
                                observed=float(violations),
                                threshold=rule.min_value,
                            )
                        )
                if rule.max_value is not None:
                    violations = (numeric > rule.max_value).sum()
                    if violations:
                        issues.append(
                            AuditIssue(
                                check="rule",
                                severity="warning",
                                message=f"{int(violations)} value(s) above maximum {rule.max_value}.",
                                column=column_name,
                                observed=float(violations),
                                threshold=rule.max_value,
                            )
                        )

            # --- IQR outlier detection ---
            if rule.min_value is not None or rule.max_value is not None:
                numeric = pd.to_numeric(col_data.dropna(), errors="coerce")
                if len(numeric) >= 4:
                    q1 = float(numeric.quantile(0.25))
                    q3 = float(numeric.quantile(0.75))
                    iqr = q3 - q1
                    if iqr > 0:
                        lower_fence = q1 - 1.5 * iqr
                        upper_fence = q3 + 1.5 * iqr
                        low_outliers = int((numeric < max(lower_fence, rule.min_value if rule.min_value is not None else lower_fence)).sum())
                        high_outliers = int((numeric > min(upper_fence, rule.max_value if rule.max_value is not None else upper_fence)).sum())
                        total_outliers = low_outliers + high_outliers
                        total = len(numeric)
                        if total_outliers > 0 and total_outliers / max(total, 1) > 0.01:
                            issues.append(
                                AuditIssue(
                                    check="rule",
                                    severity="info",
                                    message=f"{total_outliers} IQR outlier(s) detected ({total_outliers / max(total, 1) * 100:.1f}% of values).",
                                    column=column_name,
                                    observed=total_outliers,
                                )
                            )

            # --- allowed values ---
            if rule.allowed_values is not None:
                allowed_set = set(str(v) for v in rule.allowed_values)
                actual_values = col_data.dropna().astype(str).unique()
                unexpected = [str(v) for v in actual_values if str(v) not in allowed_set]
                if unexpected:
                    issues.append(
                        AuditIssue(
                            check="rule",
                            severity="warning",
                            message=f"Unexpected values found: {', '.join(sorted(unexpected)[:10])}.",
                            column=column_name,
                            observed=len(unexpected),
                        )
                    )

    @staticmethod
    def _infer_dtype(series: pd.Series) -> str:
        """Infer a human-readable type for a series."""
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if pd.api.types.is_categorical_dtype(series) or series.dtype == object or pd.api.types.is_string_dtype(series):
            # Check if mostly numeric
            numeric_count = pd.to_numeric(series.dropna(), errors="coerce").notna().sum()
            total_non_null = series.dropna().shape[0]
            if total_non_null > 0 and numeric_count / total_non_null > 0.8:
                return "numeric"
            return "categorical"
        return "string"

    def _check_drift(
        self,
        data: pd.DataFrame,
        reference: pd.DataFrame,
        issues: list[AuditIssue],
        *,
        label_column: str | None = None,
    ) -> dict[str, float]:
        drift_scores: dict[str, float] = {}
        shared_columns = [column for column in data.columns if column in reference.columns]

        for column in shared_columns:
            if label_column is not None and column == label_column:
                continue

            current = data[column]
            baseline = reference[column]

            if pd.api.types.is_numeric_dtype(current) and pd.api.types.is_numeric_dtype(baseline):
                score = self._numeric_drift(current, baseline)
            else:
                score = self._categorical_drift(current.astype(str), baseline.astype(str))

            drift_scores[column] = score
            if score >= self.drift_threshold:
                issues.append(
                    AuditIssue(
                        check="drift",
                        severity="warning",
                        message=f"Drift score {score:.3f} exceeds the {self.drift_threshold:.3f} threshold.",
                        column=column,
                        observed=score,
                        threshold=self.drift_threshold,
                    )
                )

        return drift_scores

    @staticmethod
    def _numeric_drift(current: pd.Series, baseline: pd.Series) -> float:
        current_mean = float(current.mean())
        baseline_mean = float(baseline.mean())
        baseline_std = float(baseline.std(ddof=0)) or 1.0
        return abs(current_mean - baseline_mean) / max(abs(baseline_mean), baseline_std, 1e-9)

    @staticmethod
    def _correlation_drift(
        data: pd.DataFrame,
        reference: pd.DataFrame,
        issues: list[AuditIssue],
        *,
        drift_threshold: float = 0.20,
    ) -> dict[str, float]:
        """Compare pairwise Pearson correlations between shared numeric columns."""
        drift_scores: dict[str, float] = {}
        numeric_cols = [
            col for col in data.columns
            if col in reference.columns
            and pd.api.types.is_numeric_dtype(data[col])
            and pd.api.types.is_numeric_dtype(reference[col])
        ]
        if len(numeric_cols) < 2:
            return drift_scores

        data_corr = data[numeric_cols].corr()
        ref_corr = reference[numeric_cols].corr()

        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col_i = numeric_cols[i]
                col_j = numeric_cols[j]
                pair_name = f"{col_i}~{col_j}"
                data_val = float(data_corr.loc[col_i, col_j]) if col_i in data_corr.index and col_j in data_corr.columns else 0.0
                ref_val = float(ref_corr.loc[col_i, col_j]) if col_i in ref_corr.index and col_j in ref_corr.columns else 0.0
                drift = abs(data_val - ref_val)
                drift_scores[pair_name] = drift
                if drift >= drift_threshold:
                    issues.append(
                        AuditIssue(
                            check="correlation_drift",
                            severity="warning",
                            message=(
                                f"Correlation between '{col_i}' and '{col_j}' "
                                f"shifted from {ref_val:.3f} to {data_val:.3f} "
                                f"(drift={drift:.3f})."
                            ),
                            observed=drift,
                            threshold=drift_threshold,
                        )
                    )
        return drift_scores

    @staticmethod
    def _profile_columns(data: pd.DataFrame) -> dict[str, dict[str, object]]:
        """Build statistical profiles for all columns in the dataset."""
        profiles: dict[str, dict[str, object]] = {}

        for column in data.columns:
            col = data[column]
            non_null = col.dropna()
            profile: dict[str, object] = {
                "count": int(len(col)),
                "missing": int(col.isna().sum()),
                "unique": int(non_null.nunique()) if len(non_null) > 0 else 0,
            }

            if pd.api.types.is_numeric_dtype(col):
                profile["dtype"] = "numeric"
                if len(non_null) > 0:
                    vals = non_null.astype(float)
                    q1 = float(vals.quantile(0.25))
                    q3 = float(vals.quantile(0.75))
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers = int(((vals < lower) | (vals > upper)).sum())
                    profile["min"] = float(vals.min())
                    profile["max"] = float(vals.max())
                    profile["mean"] = float(vals.mean())
                    profile["median"] = float(vals.median())
                    profile["std"] = float(vals.std(ddof=0))
                    profile["q25"] = q1
                    profile["q50"] = float(vals.quantile(0.50))
                    profile["q75"] = q3
                    profile["skewness"] = float(vals.skew())
                    profile["kurtosis"] = float(vals.kurtosis())
                    profile["outliers_iqr"] = outliers
                    profile["outlier_ratio"] = round(outliers / max(len(vals), 1), 4)
            elif pd.api.types.is_categorical_dtype(col) or col.dtype == object:
                profile["dtype"] = "categorical"
                if len(non_null) > 0:
                    value_counts = non_null.astype(str).value_counts()
                    profile["top"] = value_counts.index[0]
                    profile["freq"] = int(value_counts.iloc[0])
                    profile["top_5"] = {
                        str(k): int(v) for k, v in value_counts.head(5).items()
                    }
            else:
                profile["dtype"] = "other"

            profiles[column] = profile

        return profiles

    @staticmethod
    def _categorical_drift(current: pd.Series, baseline: pd.Series) -> float:
        current_dist = current.value_counts(normalize=True, dropna=False)
        baseline_dist = baseline.value_counts(normalize=True, dropna=False)
        categories = set(current_dist.index).union(baseline_dist.index)
        divergence = 0.0
        for category in categories:
            divergence += abs(float(current_dist.get(category, 0.0)) - float(baseline_dist.get(category, 0.0)))
        return divergence / 2.0

    @staticmethod
    def _check_redundancy(
        data: pd.DataFrame,
        issues: list[AuditIssue],
        *,
        correlation_threshold: float = 0.95,
    ) -> None:
        """Detect highly correlated numeric column pairs (redundancy)."""
        numeric_cols = [
            col for col in data.columns
            if pd.api.types.is_numeric_dtype(data[col])
        ]
        if len(numeric_cols) < 2:
            return

        corr_matrix = data[numeric_cols].corr().abs()
        seen_pairs: set[tuple[str, str]] = set()

        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col_i = numeric_cols[i]
                col_j = numeric_cols[j]
                r_val = float(corr_matrix.loc[col_i, col_j])
                if r_val >= correlation_threshold:
                    pair_key = tuple(sorted([col_i, col_j]))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        issues.append(
                            AuditIssue(
                                check="redundancy",
                                severity="warning",
                                message=(
                                    f"Columns '{col_i}' and '{col_j}' are highly correlated "
                                    f"(|r| = {r_val:.3f}), suggesting redundancy."
                                ),
                                observed=round(r_val, 4),
                                threshold=correlation_threshold,
                            )
                        )

    @staticmethod
    def _schema_diff(
        data: pd.DataFrame,
        reference: pd.DataFrame,
        issues: list[AuditIssue],
    ) -> dict[str, dict[str, object]]:
        """Compare column schemas between current and reference datasets.

        Returns a dict mapping column names to a diff description:
        ``{"status": "added"|"removed"|"dtype_changed"|"same", "details": ...}``
        """
        diff: dict[str, dict[str, object]] = {}
        current_cols = set(data.columns)
        reference_cols = set(reference.columns)

        # Columns added in current data
        added_cols = current_cols - reference_cols
        for col in sorted(added_cols):
            diff[col] = {
                "status": "added",
                "dtype": str(data[col].dtype),
            }
            issues.append(
                AuditIssue(
                    check="schema_diff",
                    severity="info",
                    message=f"Column '{col}' added (dtype: {data[col].dtype}).",
                    column=col,
                )
            )

        # Columns removed from current data
        removed_cols = reference_cols - current_cols
        for col in sorted(removed_cols):
            diff[col] = {
                "status": "removed",
                "dtype": str(reference[col].dtype),
            }
            issues.append(
                AuditIssue(
                    check="schema_diff",
                    severity="warning",
                    message=f"Column '{col}' removed from dataset.",
                    column=col,
                )
            )

        # Columns with changed dtype
        shared_cols = current_cols & reference_cols
        for col in sorted(shared_cols):
            cur_dtype = str(data[col].dtype)
            ref_dtype = str(reference[col].dtype)
            if cur_dtype != ref_dtype:
                diff[col] = {
                    "status": "dtype_changed",
                    "from_dtype": ref_dtype,
                    "to_dtype": cur_dtype,
                }
                issues.append(
                    AuditIssue(
                        check="schema_diff",
                        severity="warning",
                        message=f"Column '{col}' dtype changed from '{ref_dtype}' to '{cur_dtype}'.",
                        column=col,
                        observed=cur_dtype,
                        threshold=ref_dtype,
                    )
                )
            else:
                diff[col] = {"status": "same", "dtype": cur_dtype}

        return diff

    @staticmethod
    def dataset_summary(data: pd.DataFrame) -> str:
        lines = [
            f'Shape: {data.shape[0]} rows x {data.shape[1]} columns',
            f'Memory usage: {data.memory_usage(deep=True).sum() / 1024:.1f} KB',
            '',
            'Column dtypes:',
        ]
        for dtype, count in data.dtypes.value_counts().items():
            lines.append(f'  {dtype}: {count}')
        lines.append('')
        lines.append('Missing values:')
        total_missing = int(data.isna().sum().sum())
        lines.append(f'  Total: {total_missing} ({total_missing / max(data.size, 1) * 100:.1f}%)')
        for col in data.columns[data.isna().any()]:
            n = int(data[col].isna().sum())
            lines.append(f'  {col}: {n} ({n / max(len(data), 1) * 100:.1f}%)')
        return chr(10).join(lines)

    @staticmethod
    def sample_dataset(
        data: pd.DataFrame,
        n: int = 5,
        method: str = 'head',
        seed: int = None,
    ) -> pd.DataFrame:
        if method == 'head':
            return data.head(n)
        if method == 'tail':
            return data.tail(n)
        if method in ('random', 'stratified'):
            rng = None if seed is None else seed
            if 'random' == method:
                return data.sample(n=n, random_state=rng)
            # stratified: use last column as strata
            strata_col = data.columns[-1]
            result = data.groupby(strata_col, group_keys=False).apply(
                lambda g: g.sample(min(len(g), max(1, n // data[strata_col].nunique())), random_state=rng)
            )
            return result
        raise ValueError(f"Unknown sampling method: {method}")

    @staticmethod
    def infer_optimal_dtypes(data: pd.DataFrame) -> dict[str, dict[str, str]]:
        suggestions = {}
        for col in data.columns:
            series = data[col]
            current = str(series.dtype)
            suggested = current
            if pd.api.types.is_float_dtype(series):
                if (series.dropna() % 1 == 0).all():
                    suggested = 'int32'
                else:
                    suggested = 'float32'
            elif pd.api.types.is_integer_dtype(series):
                if series.min() >= 0:
                    if series.max() <= 255:
                        suggested = 'uint8'
                    elif series.max() <= 65535:
                        suggested = 'uint16'
                    elif series.max() <= 4294967295:
                        suggested = 'uint32'
                    else:
                        suggested = 'int32'
                else:
                    if series.min() >= -128 and series.max() <= 127:
                        suggested = 'int8'
                    elif series.min() >= -32768 and series.max() <= 32767:
                        suggested = 'int16'
                    elif series.min() >= -2147483648 and series.max() <= 2147483647:
                        suggested = 'int32'
                    else:
                        suggested = 'int64'
            elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                nunique = series.nunique()
                if nunique > 0 and nunique / max(len(series), 1) < 0.5 and nunique < 100:
                    suggested = 'category'
            if suggested != current:
                suggestions[col] = {
                    'current_dtype': current,
                    'suggested_dtype': suggested,
                }
        return suggestions


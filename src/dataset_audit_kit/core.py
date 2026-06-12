"""Core dataset validation logic."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd


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
                    lines.append(f"- Mean: {profile.get('mean', '?'):.3f}, Std: {profile.get('std', '?'):.3f}")
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

        if expected_columns is not None:
            self._check_schema(data, expected_columns, issues)

        label_distribution: dict[str, int] = {}
        if label_column is not None and label_column in data.columns:
            label_distribution = self._check_label_balance(data, label_column, issues)

        drift_scores: dict[str, float] = {}
        if reference is not None:
            drift_scores = self._check_drift(data, reference, issues, label_column=label_column)

        # Per-column validation rules
        self._apply_rules(data, issues)

        column_profiles = self._profile_columns(data)

        return AuditReport(
            rows=int(len(data)),
            columns=int(len(data.columns)),
            duplicate_rows=duplicate_rows,
            missing_cells=int(data.isna().sum().sum()),
            missingness=missingness,
            column_profiles=column_profiles,
            label_distribution=label_distribution,
            drift_scores=drift_scores,
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
    ) -> AuditReport:
        data = self.load_dataframe(data_path)
        reference = self.load_dataframe(reference_path) if reference_path else None
        return self.audit_dataframe(
            data,
            reference=reference,
            label_column=label_column,
            expected_columns=expected_columns,
        )

    @staticmethod
    def load_dataframe(path: str | Path) -> pd.DataFrame:
        """Load a tabular dataset from CSV, JSONL/NDJSON, or Parquet."""

        dataset_path = Path(path)
        suffix = dataset_path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(dataset_path)
        if suffix in {".jsonl", ".ndjson"}:
            return pd.read_json(dataset_path, lines=True)
        if suffix == ".parquet":
            return pd.read_parquet(dataset_path)

        raise ValueError(
            f"Unsupported dataset format for `{dataset_path}`. "
            "Supported formats are .csv, .jsonl, .ndjson, and .parquet."
        )

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
        if pd.api.types.is_categorical_dtype(series) or series.dtype == object:
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
    def _profile_columns(data: pd.DataFrame) -> dict[str, dict[str, object]]:
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
                    profile["min"] = float(non_null.min())
                    profile["max"] = float(non_null.max())
                    profile["mean"] = float(non_null.mean())
                    profile["std"] = float(non_null.std(ddof=0))
                    profile["q25"] = float(non_null.quantile(0.25))
                    profile["q50"] = float(non_null.quantile(0.50))
                    profile["q75"] = float(non_null.quantile(0.75))
            elif pd.api.types.is_categorical_dtype(col) or col.dtype == object:
                profile["dtype"] = "categorical"
                if len(non_null) > 0:
                    value_counts = non_null.astype(str).value_counts()
                    profile["top"] = value_counts.index[0]
                    profile["freq"] = int(value_counts.iloc[0])
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

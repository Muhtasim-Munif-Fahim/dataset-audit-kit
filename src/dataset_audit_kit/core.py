"""Core dataset validation logic."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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
    issues: list[AuditIssue] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "pass" if not self.issues else "warn"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "rows": self.rows,
            "columns": self.columns,
            "duplicate_rows": self.duplicate_rows,
            "missing_cells": self.missing_cells,
            "missingness": self.missingness,
            "label_distribution": self.label_distribution,
            "drift_scores": self.drift_scores,
            "issues": [issue.__dict__ for issue in self.issues],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_markdown(self) -> str:
        lines = [
            "# Dataset Audit Report",
            "",
            f"- Status: **{self.status}**",
            f"- Rows: **{self.rows}**",
            f"- Columns: **{self.columns}**",
            f"- Duplicate rows: **{self.duplicate_rows}**",
            f"- Missing cells: **{self.missing_cells}**",
        ]

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


class DatasetAuditor:
    """Run a small battery of quality checks over tabular data."""

    def __init__(
        self,
        *,
        missing_threshold: float = 0.05,
        drift_threshold: float = 0.20,
        label_min_share: float = 0.05,
    ) -> None:
        self.missing_threshold = missing_threshold
        self.drift_threshold = drift_threshold
        self.label_min_share = label_min_share

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

        return AuditReport(
            rows=int(len(data)),
            columns=int(len(data.columns)),
            duplicate_rows=duplicate_rows,
            missing_cells=int(data.isna().sum().sum()),
            missingness=missingness,
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
        data = pd.read_csv(data_path)
        reference = pd.read_csv(reference_path) if reference_path else None
        return self.audit_dataframe(
            data,
            reference=reference,
            label_column=label_column,
            expected_columns=expected_columns,
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
    def _categorical_drift(current: pd.Series, baseline: pd.Series) -> float:
        current_dist = current.value_counts(normalize=True, dropna=False)
        baseline_dist = baseline.value_counts(normalize=True, dropna=False)
        categories = set(current_dist.index).union(baseline_dist.index)
        divergence = 0.0
        for category in categories:
            divergence += abs(float(current_dist.get(category, 0.0)) - float(baseline_dist.get(category, 0.0)))
        return divergence / 2.0

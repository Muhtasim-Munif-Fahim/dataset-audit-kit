"""Core dataset validation logic."""

from __future__ import annotations

import csv
import html
import io
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Sequence
from xml.etree import ElementTree

import pandas as pd
import numpy as np

#: Compression suffixes pandas can infer from a filename, for text formats.
COMPRESSION_SUFFIXES = frozenset({".gz", ".bz2", ".zip", ".xz", ".zst"})

#: Row count above which an audit reports progress unless told otherwise.
PROGRESS_ROW_THRESHOLD = 100_000

#: Weight a check carries in the risk score when not configured explicitly.
DEFAULT_RISK_WEIGHT = 10.0

#: Default fraction of null-pattern values allowed before warning.
DEFAULT_NULL_PATTERN_THRESHOLD = 0.05

#: Fraction of its weight a warning contributes, relative to an error.
RISK_WARNING_FACTOR = 0.5

#: Upper bound of the aggregate risk score.
MAX_RISK_SCORE = 100.0

#: Unicode characters that are invisible on screen yet split groupby keys
#: and join matches when they ride along inside text values.
INVISIBLE_CHARACTERS = (
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u2060",  # word joiner
    "\ufeff",  # byte-order mark
    "\u00ad",  # soft hyphen
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
)

#: Pattern matching any single invisible character.
INVISIBLE_CHAR_PATTERN = "[" + "".join(INVISIBLE_CHARACTERS) + "]"

#: Literal strings commonly used to encode missing values in text columns.
NULL_PATTERNS = frozenset({
    "na", "n/a", "null", "none", "nan", "nil", "", "-", "unknown", "missing", "na", "n.a."
})

#: Lowercased version for case-insensitive matching.
NULL_PATTERNS_LOWER = {p.lower() for p in NULL_PATTERNS}

#: (kind, compiled regex) pairs scanned by the opt-in sensitive-data check.
#: The patterns are heuristics for obvious PII shapes in text columns, not
#: guarantees: matches should be confirmed before acting on them.
SENSITIVE_PATTERNS = (
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("phone", re.compile(r"(?<!\d)\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)")),
    ("ssn", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
)

#: Ordered names of the audit phases, used for progress reporting.
_AUDIT_PHASES = (
    "missingness",
    "missing_cooccurrence",
    "duplicates",
    "column names",
    "schema",
    "uniqueness",
    "label balance",
    "drift",
    "rules",
    "whitespace",
    "null_patterns",
    "sensitive",
    "profiles",
    "category_share",
    "redundancy",
)


class _Progress:
    """Minimal stderr progress reporter for a fixed number of phases.

    Deliberately not tqdm: a progress indicator is not worth a runtime
    dependency, and writing to stderr keeps it clear of a report redirected to
    stdout. Each update rewrites one line, and the line is erased on close so
    it leaves no residue in a captured log.
    """

    def __init__(self, enabled: bool, total: int) -> None:
        self.enabled = enabled
        self.total = max(total, 1)
        self.done = 0
        self._width = 0

    def advance(self, label: str) -> None:
        if not self.enabled:
            return
        self.done += 1
        filled = int(self.done / self.total * 20)
        bar = "#" * filled + "-" * (20 - filled)
        line = f"\rauditing [{bar}] {self.done}/{self.total} {label}"
        self._width = max(self._width, len(line))
        sys.stderr.write(line.ljust(self._width))
        sys.stderr.flush()

    def close(self) -> None:
        if not self.enabled:
            return
        sys.stderr.write("\r" + " " * self._width + "\r")
        sys.stderr.flush()

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
    min_inclusive : bool
        When True, a value equal to ``min_value`` satisfies the bound;
        when False, only values strictly above it do.
    max_inclusive : bool
        When True, a value equal to ``max_value`` satisfies the bound;
        when False, only values strictly below it do.
    value_tolerance : float
        Slack applied to both bounds: a value within this distance of a bound
        is accepted. Useful for floating-point pipelines where rounding
        nudges values across an exact boundary.
    allowed_values : list[str] | None
        Set of allowed values (categorical/string columns only).
    ignore_case : bool
        When True, allowed-value membership ignores letter case.
    max_missing_ratio : float | None
        Maximum allowed fraction of missing values for this column.
        Overrides the global ``missing_threshold`` when set.
    pattern : str | None
        Regular expression that every non-missing string value must fully match.
    date_format : str | None
        strptime format that every non-missing string value must parse as a
        datetime. Useful for guarding date columns stored as text.
    date_formats : list[str] | None
        strptime formats tried in order, for columns that legitimately mix
        representations (for example ``%Y-%m-%d`` and ``%d/%m/%Y``). A value
        passes as soon as any format parses it. When both this and
        ``date_format`` are set, this list wins.
    min_length : int | None
        Minimum number of characters allowed for each non-missing value.
    max_length : int | None
        Maximum number of characters allowed for each non-missing value.
    min_date : str | None
        Earliest allowed date, as a parseable date string such as
        ``"2020-01-01"``. Values that parse to a date strictly before this
        bound are flagged.
    max_date : str | None
        Latest allowed date, as a parseable date string such as
        ``"2030-12-31"``. Values that parse to a date strictly after this
        bound are flagged.
    no_future_dates : bool
        When True, values that parse to a date after the current date are
        flagged.
    max_outlier_ratio : float | None
        Maximum fraction of numeric values allowed outside the outlier fences.
        Without ``percentile_fences`` the fences are the standard IQR fences;
        with them they are the configured quantiles of the observed values.
    percentile_fences : tuple[float, float] | None
        Lower and upper quantiles used as outlier fences, for example
        ``[0.01, 0.99]``. When set, values below the lower quantile or above
        the upper quantile count as outliers instead of the IQR rule.
    max_drift : float | None
        Maximum drift score allowed for this column when a reference dataset is
        supplied. Overrides the auditor-wide ``drift_threshold``.
    """

    name: str
    dtype: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    min_inclusive: bool = True
    max_inclusive: bool = True
    value_tolerance: float = 0.0
    allowed_values: list[str] | None = None
    ignore_case: bool = False
    max_missing_ratio: float | None = None
    pattern: str | None = None
    min_unique: int | None = None
    max_unique: int | None = None
    date_format: str | None = None
    date_formats: tuple[str, ...] | None = None
    min_length: int | None = None
    max_length: int | None = None
    min_date: str | None = None
    max_date: str | None = None
    no_future_dates: bool = False
    max_outlier_ratio: float | None = None
    percentile_fences: tuple[float, float] | None = None
    max_drift: float | None = None


@dataclass(frozen=True)
class CrossColumnRule:
    """A relational constraint between two columns.

    Attributes
    ----------
    left : str
        First column name in the comparison.
    op : str
        One of ``"le"``, ``"lt"``, ``"ge"``, ``"gt"``, ``"eq"``, ``"ne"``.
    right : str
        Second column name in the comparison.
    missing_ok : bool
        When True, rows where either side is missing are skipped instead of
        being reported as violations.
    """

    left: str
    op: str
    right: str
    missing_ok: bool = False

    _OPERATORS = frozenset({"le", "lt", "ge", "gt", "eq", "ne"})

    def __post_init__(self) -> None:
        if self.op not in self._OPERATORS:
            raise ValueError(
                f"op must be one of {sorted(self._OPERATORS)}, got '{self.op}'"
            )


@dataclass(frozen=True)
class ValidationRules:
    """A collection of per-column validation rules.

    Parameters
    ----------
    columns : dict[str, ColumnRule]
        Mapping from column name to its validation rule.
    cross : tuple[CrossColumnRule, ...]
        Relational constraints between pairs of columns.
    """

    columns: dict[str, ColumnRule] = field(default_factory=dict)
    cross: tuple[CrossColumnRule, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, rules: dict[str, dict[str, object]]) -> "ValidationRules":
        """Build rules from a plain dictionary (e.g. loaded from JSON).

        Per-column contracts live under each column name. Relational
        constraints are read from an optional ``"cross"`` key holding a list
        of ``{"left": ..., "op": ..., "right": ...}`` entries.
        """
        column_rules: dict[str, ColumnRule] = {}
        raw_cross = rules.get("cross", [])
        for col_name, col_config in rules.items():
            if col_name == "cross":
                continue
            integer_bounds: dict[str, int | None] = {}
            for bound in ("min_unique", "max_unique", "min_length", "max_length"):
                value = col_config.get(bound)
                if value is not None:
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise ValueError(
                            f"{bound} for column '{col_name}' must be a non-negative integer"
                        )
                integer_bounds[bound] = value
            for lower, upper in (("min_unique", "max_unique"), ("min_length", "max_length")):
                if (
                    integer_bounds[lower] is not None
                    and integer_bounds[upper] is not None
                    and integer_bounds[lower] > integer_bounds[upper]
                ):
                    raise ValueError(
                        f"{lower} cannot exceed {upper} for column '{col_name}'"
                    )
            max_outlier_ratio = col_config.get("max_outlier_ratio")
            if max_outlier_ratio is not None:
                if (
                    isinstance(max_outlier_ratio, bool)
                    or not isinstance(max_outlier_ratio, (int, float))
                    or not 0.0 <= float(max_outlier_ratio) <= 1.0
                ):
                    raise ValueError(
                        f"max_outlier_ratio for column '{col_name}' must be between 0 and 1"
                    )
            percentile_fences = col_config.get("percentile_fences")
            if percentile_fences is not None:
                if (
                    isinstance(percentile_fences, (bool, str))
                    or not isinstance(percentile_fences, (list, tuple))
                    or len(percentile_fences) != 2
                ):
                    raise ValueError(
                        f"percentile_fences for column '{col_name}' must be a two-element list"
                    )
                parsed_fences: list[float] = []
                for quantile in percentile_fences:
                    if (
                        isinstance(quantile, bool)
                        or not isinstance(quantile, (int, float))
                        or not 0.0 <= float(quantile) <= 1.0
                    ):
                        raise ValueError(
                            f"percentile_fences for column '{col_name}' must "
                            "contain fractions between 0 and 1"
                        )
                    parsed_fences.append(float(quantile))
                if parsed_fences[0] >= parsed_fences[1]:
                    raise ValueError(
                        f"percentile_fences lower bound must be below the upper "
                        f"bound for column '{col_name}'"
                    )
                percentile_fences = tuple(parsed_fences)
            value_tolerance = col_config.get("value_tolerance", 0.0)
            if value_tolerance is not None:
                if (
                    isinstance(value_tolerance, bool)
                    or not isinstance(value_tolerance, (int, float))
                    or not math.isfinite(float(value_tolerance))
                    or float(value_tolerance) < 0.0
                ):
                    raise ValueError(
                        f"value_tolerance for column '{col_name}' must be a non-negative number"
                    )
            min_inclusive = col_config.get("min_inclusive", True)
            if not isinstance(min_inclusive, bool):
                raise ValueError(
                    f"min_inclusive for column '{col_name}' must be a boolean"
                )
            max_inclusive = col_config.get("max_inclusive", True)
            if not isinstance(max_inclusive, bool):
                raise ValueError(
                    f"max_inclusive for column '{col_name}' must be a boolean"
                )
            ignore_case = col_config.get("ignore_case", False)
            if not isinstance(ignore_case, bool):
                raise ValueError(
                    f"ignore_case for column '{col_name}' must be a boolean"
                )
            max_drift = col_config.get("max_drift")
            if max_drift is not None:
                if (
                    isinstance(max_drift, bool)
                    or not isinstance(max_drift, (int, float))
                    or float(max_drift) < 0.0
                ):
                    raise ValueError(
                        f"max_drift for column '{col_name}' must be a non-negative number"
                    )
            min_date = col_config.get("min_date")
            max_date = col_config.get("max_date")
            for bound, label in ((min_date, "min_date"), (max_date, "max_date")):
                if bound is None:
                    continue
                if isinstance(bound, bool) or not isinstance(bound, str):
                    raise ValueError(
                        f"{label} for column '{col_name}' must be a date string"
                    )
                try:
                    pd.Timestamp(bound)
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"{label} for column '{col_name}' must be a parseable date: {exc}"
                    ) from exc
            if (
                min_date is not None
                and max_date is not None
                and pd.Timestamp(min_date) > pd.Timestamp(max_date)
            ):
                raise ValueError(
                    f"min_date cannot exceed max_date for column '{col_name}'"
                )
            no_future_dates = col_config.get("no_future_dates", False)
            if not isinstance(no_future_dates, bool):
                raise ValueError(
                    f"no_future_dates for column '{col_name}' must be a boolean"
                )
            raw_date_formats = col_config.get("date_formats")
            if raw_date_formats is not None:
                if (
                    isinstance(raw_date_formats, (bool, str))
                    or not isinstance(raw_date_formats, (list, tuple))
                    or not raw_date_formats
                    or not all(isinstance(fmt, str) for fmt in raw_date_formats)
                ):
                    raise ValueError(
                        f"date_formats for column '{col_name}' must be a "
                        "non-empty list of format strings"
                    )
                # The list supersedes the single-format key so an audit never
                # has to reconcile two conflicting contracts.
                parsed_date_formats = tuple(str(fmt) for fmt in raw_date_formats)
                parsed_date_format = None
            elif col_config.get("date_format") is not None:
                parsed_date_format = str(col_config["date_format"])
                parsed_date_formats = None
            else:
                parsed_date_format = None
                parsed_date_formats = None
            column_rules[col_name] = ColumnRule(
                name=col_name,
                dtype=str(col_config.get("dtype") or "").lower() or None if col_config.get("dtype") else None,
                min_value=col_config.get("min_value"),
                max_value=col_config.get("max_value"),
                min_inclusive=min_inclusive,
                max_inclusive=max_inclusive,
                value_tolerance=float(value_tolerance) if value_tolerance is not None else 0.0,
                allowed_values=col_config.get("allowed_values"),
                ignore_case=ignore_case,
                max_missing_ratio=col_config.get("max_missing_ratio"),
                pattern=str(col_config["pattern"]) if col_config.get("pattern") is not None else None,
                min_unique=integer_bounds["min_unique"],
                max_unique=integer_bounds["max_unique"],
                date_format=parsed_date_format,
                date_formats=parsed_date_formats,
                min_length=integer_bounds["min_length"],
                max_length=integer_bounds["max_length"],
                min_date=min_date,
                max_date=max_date,
                no_future_dates=no_future_dates,
                max_outlier_ratio=(
                    float(max_outlier_ratio)
                    if max_outlier_ratio is not None
                    else None
                ),
                percentile_fences=percentile_fences,
                max_drift=float(max_drift) if max_drift is not None else None,
            )
        cross_rules: list[CrossColumnRule] = []
        for entry in raw_cross:
            if not isinstance(entry, dict):
                raise ValueError("each cross-column rule must be an object")
            left = entry.get("left")
            right = entry.get("right")
            if not isinstance(left, str) or not isinstance(right, str):
                raise ValueError("cross-column rules need string 'left' and 'right' columns")
            missing_ok = entry.get("missing_ok", False)
            if not isinstance(missing_ok, bool):
                raise ValueError("missing_ok must be a boolean")
            cross_rules.append(
                CrossColumnRule(
                    left=left,
                    op=str(entry.get("op", "le")),
                    right=right,
                    missing_ok=missing_ok,
                )
            )
        return cls(columns=column_rules, cross=tuple(cross_rules))

    @classmethod
    def from_json(cls, path: str, *, profile: str | None = None) -> "ValidationRules":
        """Load rules from a JSON file.

        A file may hold several named rule sets under a top-level
        ``"profiles"`` object; ``profile`` picks one of them. A file without
        that section is one flat rule set and must not name a profile.
        """
        import json
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("rules file must contain a JSON object")
        profiles = raw.get("profiles")
        if profiles is None:
            if profile is not None:
                raise ValueError(
                    f"profile '{profile}' was requested, but the rules file "
                    "has no profiles section"
                )
            return cls.from_dict(raw)
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError(
                "'profiles' must be an object mapping profile names to rule sets"
            )
        available = ", ".join(sorted(str(name) for name in profiles))
        if profile is None:
            raise ValueError(
                f"rules file defines named profiles ({available}); "
                "pass --profile to choose one"
            )
        if profile not in profiles:
            raise ValueError(
                f"profile '{profile}' not found; available profiles: {available}"
            )
        selected = profiles[profile]
        if not isinstance(selected, dict):
            raise ValueError(f"profile '{profile}' must contain rule objects")
        return cls.from_dict(selected)

    @classmethod
    def infer(
        cls,
        data: pd.DataFrame,
        *,
        max_categories: int = 20,
        missing_tolerance: float = 0.0,
    ) -> "ValidationRules":
        """Infer a reusable validation contract from a baseline dataframe.

        Numeric columns receive observed bounds. Low-cardinality categorical
        columns receive an allowed-value set, while high-cardinality text
        columns retain only their inferred type and missingness allowance.
        """

        if max_categories < 1:
            raise ValueError("max_categories must be at least 1")
        if not 0.0 <= missing_tolerance <= 1.0:
            raise ValueError("missing_tolerance must be between 0 and 1")
        if data.columns.duplicated().any():
            raise ValueError("cannot infer rules from duplicate column names")

        inferred: dict[str, ColumnRule] = {}
        for name in data.columns:
            series = data[name]
            dtype = DatasetAuditor._infer_dtype(series)
            non_missing = series.dropna()
            min_value: float | None = None
            max_value: float | None = None
            allowed_values: list[str] | None = None

            if dtype == "numeric" and not non_missing.empty:
                numeric = pd.to_numeric(non_missing, errors="coerce").dropna()
                if not numeric.empty:
                    min_value = float(numeric.min())
                    max_value = float(numeric.max())
            elif non_missing.nunique() <= max_categories:
                allowed_values = sorted({str(value) for value in non_missing})

            inferred[str(name)] = ColumnRule(
                name=str(name),
                dtype=dtype,
                min_value=min_value,
                max_value=max_value,
                allowed_values=allowed_values,
                max_missing_ratio=min(
                    1.0, float(series.isna().mean()) + missing_tolerance
                ),
            )
        return cls(columns=inferred)

    def to_dict(self) -> dict[str, dict[str, object]]:
        """Serialize to a plain dictionary for JSON export."""
        result: dict[str, dict[str, object]] = {}
        for name, rule in self.columns.items():
            entry: dict[str, object] = {}
            for attr in (
                "dtype",
                "min_value",
                "max_value",
                "allowed_values",
                "max_missing_ratio",
                "pattern",
                "min_unique",
                "max_unique",
                "date_format",
                "min_length",
                "max_length",
                "min_date",
                "max_date",
                "max_outlier_ratio",
                "max_drift",
            ):
                value = getattr(rule, attr)
                if value is not None:
                    entry[attr] = value
            if rule.percentile_fences is not None:
                entry["percentile_fences"] = list(rule.percentile_fences)
            if rule.date_formats is not None:
                entry["date_formats"] = list(rule.date_formats)
            if rule.ignore_case:
                entry["ignore_case"] = True
            if rule.no_future_dates:
                entry["no_future_dates"] = True
            if not rule.min_inclusive:
                entry["min_inclusive"] = False
            if not rule.max_inclusive:
                entry["max_inclusive"] = False
            if rule.value_tolerance:
                entry["value_tolerance"] = rule.value_tolerance
            result[name] = entry
        if self.cross:
            result["cross"] = [
                {"left": r.left, "op": r.op, "right": r.right}
                | ({"missing_ok": True} if r.missing_ok else {})
                for r in self.cross
            ]
        return result


@dataclass(frozen=True)
class DatasetBaseline:
    """Privacy-preserving column profile for monitoring future datasets."""

    rows: int
    column_profiles: dict[str, dict[str, object]]

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame) -> "DatasetBaseline":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas.DataFrame")
        return cls(
            rows=int(len(data)),
            column_profiles=DatasetAuditor._profile_columns(data),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "DatasetBaseline":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read dataset baseline '{path}': {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(
            payload.get("column_profiles"), dict
        ):
            raise ValueError("dataset baseline must contain column_profiles")
        return cls(
            rows=int(payload.get("rows", 0)),
            column_profiles=payload["column_profiles"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "rows": self.rows,
            "column_profiles": self.column_profiles,
        }

    def to_file(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    def compare(
        self,
        data: pd.DataFrame,
        *,
        missing_ratio_delta: float = 0.10,
        mean_shift_std: float = 3.0,
    ) -> list[AuditIssue]:
        """Compare new data without retaining the baseline's raw records."""

        if missing_ratio_delta < 0 or mean_shift_std <= 0:
            raise ValueError("comparison thresholds must be positive")
        current = DatasetAuditor._profile_columns(data)
        issues: list[AuditIssue] = []
        baseline_names = set(self.column_profiles)
        current_names = set(current)
        for name in sorted(baseline_names - current_names):
            issues.append(
                AuditIssue(
                    check="baseline_schema",
                    severity="error",
                    message=f"Baseline column '{name}' is missing.",
                    column=name,
                )
            )
        for name in sorted(current_names - baseline_names):
            issues.append(
                AuditIssue(
                    check="baseline_schema",
                    severity="info",
                    message=f"New column '{name}' was added.",
                    column=name,
                )
            )
        for name in sorted(baseline_names & current_names):
            before = self.column_profiles[name]
            after = current[name]
            if before.get("dtype") != after.get("dtype"):
                issues.append(
                    AuditIssue(
                        check="baseline_dtype",
                        severity="error",
                        message=(
                            f"Column dtype changed from {before.get('dtype')} "
                            f"to {after.get('dtype')}."
                        ),
                        column=name,
                    )
                )
                continue
            before_count = max(int(before.get("count", 0)), 1)
            after_count = max(int(after.get("count", 0)), 1)
            before_missing = float(before.get("missing", 0)) / before_count
            after_missing = float(after.get("missing", 0)) / after_count
            delta = after_missing - before_missing
            if delta > missing_ratio_delta:
                issues.append(
                    AuditIssue(
                        check="baseline_missingness",
                        severity="warning",
                        message=f"Missing ratio increased by {delta:.1%}.",
                        column=name,
                        observed=after_missing,
                        threshold=before_missing + missing_ratio_delta,
                    )
                )
            if before.get("dtype") == "numeric":
                try:
                    baseline_mean = float(before["mean"])
                    current_mean = float(after["mean"])
                    baseline_std = float(before["std"])
                except (KeyError, TypeError, ValueError):
                    continue
                if baseline_std > 0:
                    shift = abs(current_mean - baseline_mean) / baseline_std
                    if shift > mean_shift_std:
                        issues.append(
                            AuditIssue(
                                check="baseline_mean_shift",
                                severity="warning",
                                message=f"Mean shifted by {shift:.2f} baseline standard deviations.",
                                column=name,
                                observed=shift,
                                threshold=mean_shift_std,
                            )
                        )
        return issues


def _checked_max_risk(max_risk: float | None) -> float | None:
    """Validate a risk-score ceiling handed to an exit-code gate."""

    if max_risk is None:
        return None
    if (
        isinstance(max_risk, bool)
        or not isinstance(max_risk, (int, float))
        or max_risk < 0
    ):
        raise ValueError("max_risk must be a non-negative number")
    return float(max_risk)


def _sarif_run_properties(report: "AuditReport") -> dict[str, object]:
    """Fold status, score, and any run stamps into SARIF run properties."""

    properties: dict[str, object] = {
        "auditStatus": report.status,
        "qualityScore": report.quality_score,
    }
    if report.audit_id:
        properties["auditId"] = report.audit_id
    if report.created_utc:
        properties["createdUtc"] = report.created_utc
    if report.config_hash:
        properties["configHash"] = report.config_hash
    return properties


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
    risk_score: float = 0.0
    #: Provenance identifying the run that produced this report. All three
    #: stay unset until a caller stamps them, keeping library output stable.
    audit_id: str | None = None
    created_utc: str | None = None
    config_hash: str | None = None

    @property
    def status(self) -> str:
        # `info` findings (a new column, an outlier note) describe the dataset
        # rather than fault it, so they must not turn the status — and with it
        # the CLI exit code — into a failure.
        return "warn" if self.blocking_issues else "pass"

    @property
    def blocking_issues(self) -> list[AuditIssue]:
        """Issues that count against the dataset: errors and warnings."""

        return [issue for issue in self.issues if issue.severity in {"error", "warning"}]

    def gated_issues(self, fail_on: str = "warning") -> list[AuditIssue]:
        """Return findings that should fail a quality gate.

        ``warning`` preserves the default behavior where warnings and errors
        fail.  ``error`` is useful for exploratory or staged pipelines that
        want to surface warnings without stopping execution.
        """

        if fail_on not in {"warning", "error"}:
            raise ValueError("fail_on must be 'warning' or 'error'")
        severities = {"warning", "error"} if fail_on == "warning" else {"error"}
        return [issue for issue in self.issues if issue.severity in severities]

    def exit_code(
        self, fail_on: str = "warning", *, max_risk: float | None = None
    ) -> int:
        """Return the process exit code for a configurable quality gate.

        ``max_risk`` adds a weighted-score ceiling: the run fails when the
        risk score exceeds it even though every individual finding sits
        below ``fail_on``. Staged pipelines use it to tolerate scattered
        warnings but stop once they accumulate.
        """

        ceiling = _checked_max_risk(max_risk)
        if ceiling is not None and self.risk_score > ceiling:
            return 1
        return 1 if self.gated_issues(fail_on) else 0

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
        payload: dict[str, object] = {
            "status": self.status,
            "quality_score": self.quality_score,
            "risk_score": self.risk_score,
            "rows": self.rows,
            "columns": self.columns,
            "duplicate_rows": self.duplicate_rows,
            "missing_cells": self.missing_cells,
            "missingness": self.missingness,
            "label_distribution": self.label_distribution,
            "drift_scores": self.drift_scores,
            "column_profiles": self.column_profiles,
            "issues": [issue.__dict__ for issue in self.issues],
            "rule_cooccurrence": self.rule_cooccurrence(),
            "outlier_summary": self.outlier_summary(),
        }
        if any((self.audit_id, self.created_utc, self.config_hash)):
            payload["meta"] = {
                "audit_id": self.audit_id,
                "created_utc": self.created_utc,
                "config_hash": self.config_hash,
            }
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_sarif(self, *, artifact_uri: str = "dataset") -> str:
        """Serialize findings as a SARIF 2.1.0 log for CI integrations."""

        checks = sorted({issue.check for issue in self.issues})
        rules = [
            {
                "id": check,
                "name": check.replace("_", " ").title(),
                "shortDescription": {"text": f"Dataset audit check: {check}"},
            }
            for check in checks
        ]
        levels = {"error": "error", "warning": "warning", "info": "note"}
        results: list[dict[str, object]] = []
        for issue in self.issues:
            properties: dict[str, object] = {}
            if issue.column is not None:
                properties["column"] = issue.column
            if issue.observed is not None:
                properties["observed"] = issue.observed
            if issue.threshold is not None:
                properties["threshold"] = issue.threshold
            result: dict[str, object] = {
                "ruleId": issue.check,
                "level": levels.get(issue.severity, "warning"),
                "message": {"text": issue.message},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": artifact_uri}
                        }
                    }
                ],
            }
            if properties:
                result["properties"] = properties
            results.append(result)

        payload = {
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "dataset-audit-kit",
                            "informationUri": (
                                "https://github.com/Muhtasim-Munif-Fahim/"
                                "dataset-audit-kit"
                            ),
                            "rules": rules,
                        }
                    },
                    "results": results,
                    "properties": _sarif_run_properties(self),
                }
            ],
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def to_junit_xml(self, *, suite_name: str = "dataset-audit") -> str:
        """Serialize audit findings as JUnit XML for CI test reporters."""

        blocking = self.blocking_issues
        cases = self.issues or [
            AuditIssue(check="audit", severity="info", message="No issues found.")
        ]
        suite = ElementTree.Element(
            "testsuite",
            {
                "name": suite_name,
                "tests": str(len(cases)),
                "failures": str(len(blocking)),
                "skipped": str(sum(issue.severity == "info" for issue in self.issues)),
            },
        )
        properties = ElementTree.SubElement(suite, "properties")
        ElementTree.SubElement(
            properties, "property", {"name": "status", "value": self.status}
        )
        ElementTree.SubElement(
            properties,
            "property",
            {"name": "quality_score", "value": str(self.quality_score)},
        )
        for issue in cases:
            name = issue.check if issue.column is None else f"{issue.check}[{issue.column}]"
            case = ElementTree.SubElement(
                suite, "testcase", {"classname": "dataset_audit", "name": name}
            )
            if issue.severity in {"error", "warning"}:
                failure = ElementTree.SubElement(
                    case, "failure", {"type": issue.severity, "message": issue.message}
                )
                failure.text = issue.message
            elif self.issues:
                ElementTree.SubElement(case, "skipped", {"message": issue.message})
        return ElementTree.tostring(suite, encoding="unicode")

    def to_csv(self) -> str:
        """Serialize each finding as one flat CSV row.

        JSON and HTML are useful for people, while a flat findings table is
        easier for CI jobs and spreadsheet tools to consume. A clean audit
        still emits the header so downstream jobs can rely on a stable schema.
        """

        fields = ["severity", "check", "column", "message", "observed", "threshold"]
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for issue in self.issues:
            writer.writerow(
                {
                    "severity": issue.severity,
                    "check": issue.check,
                    "column": issue.column or "",
                    "message": issue.message,
                    "observed": "" if issue.observed is None else issue.observed,
                    "threshold": "" if issue.threshold is None else issue.threshold,
                }
            )
        return output.getvalue()

    def to_jsonl(self) -> str:
        """Serialize each finding as one JSON object per line (NDJSON).

        Unlike ``to_json``, which nests findings inside the full report, every
        line here is a standalone finding in the same shape as the ``issues``
        array of the JSON report, so a downstream job can stream the file line
        by line instead of parsing a nested document. A clean audit emits an
        empty file, which is the JSONL convention for zero records.
        """

        return "".join(
            json.dumps(issue.__dict__) + "\n" for issue in self.issues
        )

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

        rule_cooccurrence = self.rule_cooccurrence()
        if rule_cooccurrence:
            lines.extend(["", "## Rule co-occurrence"])
            lines.append(
                "Columns flagged by more than one check, and the counts behind each pair:"
            )
            for entry in rule_cooccurrence:
                checks = ", ".join(f"`{check}`" for check in entry["checks"])
                lines.append(
                    f"- **{entry['column']}**: {entry['findings']} finding(s) across {checks}"
                )
                for pair in entry["pairs"]:
                    left, right = pair["checks"]
                    lines.append(
                        f"  - `{left}` ({pair['counts'][left]}) & `{right}` ({pair['counts'][right]})"
                    )

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
        label_total = sum(self.label_distribution.values())
        label_peak = max(self.label_distribution.values(), default=0)
        for label, count in self.label_distribution.items():
            share = count / label_total if label_total else 0.0
            # Bars are scaled against the largest class, so the smallest class
            # stays visible instead of collapsing to nothing.
            width = (count / label_peak * 100) if label_peak else 0.0
            label_rows.append(
                f"<tr><td>{esc(label)}</td><td>{count}</td>"
                f"<td>{share:.1%}</td>"
                f'<td class="bar-cell">'
                f'<span class="bar" style="width:{width:.2f}%"'
                f' title="{esc(label)}: {count} ({share:.1%})"></span>'
                f"</td></tr>"
            )

        meta_parts = []
        if self.audit_id:
            meta_parts.append(f"Audit ID <code>{esc(self.audit_id)}</code>")
        if self.created_utc:
            meta_parts.append(f"Generated {esc(self.created_utc)}")
        if self.config_hash:
            meta_parts.append(f"Config hash <code>{esc(self.config_hash)}</code>")
        meta_line = (
            [f'<p class="muted">{" &middot; ".join(meta_parts)}</p>']
            if meta_parts
            else []
        )

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
            # Label distribution bars: pure CSS, no scripts or images, so the
            # report stays a single self-contained file that opens offline.
            ".label-table td:nth-child(2),.label-table td:nth-child(3){text-align:right;white-space:nowrap;}",
            ".label-table .bar-cell{width:55%;padding:10px 12px;}",
            ".bar{display:block;height:14px;min-width:2px;border-radius:7px;background:linear-gradient(90deg,#2563eb,#60a5fa);}",
            "@media print{.bar{background:#2563eb;-webkit-print-color-adjust:exact;print-color-adjust:exact;}}",
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            "<h1>Dataset Audit Report</h1>",
            f'<p class="muted">Status: <strong class="{esc(self.status)}">{esc(self.status)}</strong></p>',
            *meta_line,
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
                '<table class="label-table"><thead><tr><th>Label</th><th>Count</th>'
                '<th>Share</th><th>Distribution</th></tr></thead><tbody>',
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
                    # A column that is entirely missing has no statistics at
                    # all, so every value goes through _format_stat rather than
                    # a bare float format that would raise on the placeholder.
                    profile_sections.extend([
                        f"<tr><td>Min</td><td>{esc(_format_stat(profile.get('min')))}</td></tr>",
                        f"<tr><td>Max</td><td>{esc(_format_stat(profile.get('max')))}</td></tr>",
                        f"<tr><td>Mean</td><td>{esc(_format_stat(profile.get('mean')))}</td></tr>",
                        f"<tr><td>Std</td><td>{esc(_format_stat(profile.get('std')))}</td></tr>",
                        f"<tr><td>Q1</td><td>{esc(_format_stat(profile.get('q25')))}</td></tr>",
                        f"<tr><td>Q2 (median)</td><td>{esc(_format_stat(profile.get('q50')))}</td></tr>",
                        f"<tr><td>Q3</td><td>{esc(_format_stat(profile.get('q75')))}</td></tr>",
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

        rule_cooccurrence = self.rule_cooccurrence()
        if rule_cooccurrence:
            cooccurrence_rows: list[str] = []
            for entry in rule_cooccurrence:
                for pair in entry["pairs"]:
                    left, right = pair["checks"]
                    cooccurrence_rows.append(
                        "<tr>"
                        f"<td>{esc(entry['column'])}</td>"
                        f"<td>{esc(left)}</td>"
                        f"<td>{pair['counts'][left]}</td>"
                        f"<td>{esc(right)}</td>"
                        f"<td>{pair['counts'][right]}</td>"
                        "</tr>"
                    )
            sections.extend([
                "<h2>Rule co-occurrence</h2>",
                "<table><thead><tr><th>Column</th><th>Check</th>"
                "<th>Findings</th><th>Check</th><th>Findings</th></tr></thead><tbody>",
                *cooccurrence_rows,
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

    def rule_cooccurrence(self) -> list[dict[str, object]]:
        """Surface check pairs that fire together on the same column.

        A column flagged by several checks at once often means the checks are
        redundant or coupled: a ``rule`` bound that sits inside the missingness
        threshold fires on the same rows, so the user is warned twice about one
        underlying problem. Grouping findings by column and listing every check
        pair makes that coupling visible so the contract can be simplified.

        Only columns flagged by two or more distinct checks are included. The
        output is fully sorted, so it is stable across runs.
        """
        from collections import Counter

        by_column: dict[str, "Counter[str]"] = {}
        for issue in self.issues:
            if issue.column is None:
                continue
            if issue.column not in by_column:
                by_column[issue.column] = Counter()
            by_column[issue.column][issue.check] += 1

        result: list[dict[str, object]] = []
        for column in sorted(by_column):
            counts = by_column[column]
            if len(counts) < 2:
                continue
            checks = sorted(counts)
            pairs: list[dict[str, object]] = []
            for i, left in enumerate(checks):
                for right in checks[i + 1 :]:
                    pairs.append(
                        {
                            "checks": [left, right],
                            "counts": {left: counts[left], right: counts[right]},
                        }
                    )
            result.append(
                {
                    "column": column,
                    "checks": checks,
                    "findings": int(sum(counts.values())),
                    "pairs": pairs,
                }
            )
        return result

    def outlier_summary(
        self,
        *,
        top: int = 5,
        threshold: float = 0.05,
    ) -> list[dict[str, object]]:
        """Rank numeric columns by their IQR-based outlier ratio.

        Surfaces the columns most contaminated by IQR-flagged outliers so the
        audit consumer can see which fields would benefit from outlier
        treatment before modeling. Columns with no IQR outlier ratio in their
        profile (non-numeric or absent) are skipped, as are columns whose
        outlier ratio falls below ``threshold``. Returned rows are sorted by
        ``outlier_ratio`` descending then by ``outliers_iqr`` descending so
        ties are stable. ``top`` is clamped to a positive integer; ``top=0``
        raises ``ValueError``.
        """
        if not isinstance(top, int) or isinstance(top, bool) or top <= 0:
            raise ValueError("top must be a positive integer")
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise ValueError("threshold must be a number between 0 and 1")

        rows: list[dict[str, object]] = []
        for column, profile in self.column_profiles.items():
            ratio = profile.get("outlier_ratio")
            count = profile.get("outliers_iqr")
            if ratio is None or count is None:
                continue
            if profile.get("dtype") != "numeric":
                continue
            ratio_f = float(ratio)
            if ratio_f < float(threshold):
                continue
            rows.append(
                {
                    "column": column,
                    "outliers_iqr": int(count),
                    "outlier_ratio": ratio_f,
                    "non_null": int(profile.get("count", 0)) - int(profile.get("missing", 0)),
                    "q1": profile.get("q25"),
                    "q3": profile.get("q75"),
                    "mean": profile.get("mean"),
                    "std": profile.get("std"),
                }
            )
        rows.sort(key=lambda row: (-float(row["outlier_ratio"]), -int(row["outliers_iqr"]), row["column"]))
        return rows[:top]

    def profile_diff(
        self,
        other: "AuditReport",
        *,
        columns: list[str] | None = None,
        include_categorical: bool = True,
    ) -> list[dict[str, object]]:
        """Compare two reports' column profiles side by side.

        Reports are compared at the profile level rather than as issue lists:
        the consumer gets one row per shared column with the captured summary
        statistics from both sides (counts, missing rates, numeric means/std
        and quantiles, categorical top-key frequencies, and IQR outlier
        ratios). Columns are sorted alphabetically for stability, and only
        columns present in both reports are included. ``columns`` restricts
        the comparison to a chosen subset of those shared columns; raising
        ``ValueError`` for an unknown name so callers fail fast. The
        ``include_categorical`` flag keeps the table numeric-only when set to
        ``False`` so consumers can avoid mixing scales.

        Unlike :class:`DatasetBaseline.compare`, this method does not raise
        an issue list: the goal is a stable table that downstream code can
        render as-is.
    """
        if other is None:
            raise ValueError("other report must not be None")
        if not isinstance(columns, list) and columns is not None:
            raise ValueError("columns must be a list of column names")

        before = self.column_profiles
        after = other.column_profiles
        shared = sorted(set(before).intersection(after))
        if columns is not None:
            requested = [str(name) for name in columns]
            missing = [name for name in requested if name not in shared]
            if missing:
                raise ValueError(
                    "unknown columns not present in both reports: " + ", ".join(missing)
                )
            selected = requested
        else:
            selected = shared

        rows: list[dict[str, object]] = []
        for column in selected:
            left = before[column]
            right = after[column]
            left_dtype = str(left.get("dtype", "other"))
            right_dtype = str(right.get("dtype", "other"))
            if not include_categorical and (left_dtype != "numeric" or right_dtype != "numeric"):
                continue

            def _num(value: object) -> float | None:
                if isinstance(value, bool):
                    return None
                if isinstance(value, (int, float)):
                    return float(value)
                return None

            def _missing_rate(profile: dict[str, object]) -> float | None:
                count = profile.get("count")
                missing = profile.get("missing")
                num_count = _num(count)
                num_missing = _num(missing)
                if num_count is None or num_missing is None or num_count <= 0:
                    return None
                return float(num_missing) / float(num_count)

            row: dict[str, object] = {
                "column": column,
                "dtype_before": left_dtype,
                "dtype_after": right_dtype,
                "count_before": left.get("count"),
                "count_after": right.get("count"),
                "missing_rate_before": _missing_rate(left),
                "missing_rate_after": _missing_rate(right),
            }

            for stat in ("mean", "std", "min", "max", "median", "q25", "q75"):
                before_val = _num(left.get(stat))
                after_val = _num(right.get(stat))
                row[f"{stat}_before"] = before_val
                row[f"{stat}_after"] = after_val
                if before_val is not None and after_val is not None:
                    row[f"{stat}_delta"] = round(after_val - before_val, 6)
                else:
                    row[f"{stat}_delta"] = None

            for stat in ("unique",):
                before_val = left.get(stat)
                after_val = right.get(stat)
                if isinstance(before_val, (int, float)) and not isinstance(before_val, bool) \
                    and isinstance(after_val, (int, float)) and not isinstance(after_val, bool):
                    row[f"{stat}_delta"] = int(after_val) - int(before_val)
                else:
                    row[f"{stat}_delta"] = None

            before_ratio = _num(left.get("outlier_ratio"))
            after_ratio = _num(right.get("outlier_ratio"))
            row["outlier_ratio_before"] = before_ratio
            row["outlier_ratio_after"] = after_ratio
            if before_ratio is not None and after_ratio is not None:
                row["outlier_ratio_delta"] = round(after_ratio - before_ratio, 6)
            else:
                row["outlier_ratio_delta"] = None

            if left_dtype == "categorical" and right_dtype == "categorical":
                row["top_before"] = left.get("top")
                row["top_after"] = right.get("top")

            rows.append(row)
        return rows

    def to_file(self, path: str) -> str:
        """Write the report to a file, auto-detecting format from extension.

        Supported extensions:
        - ``.json`` — JSON format
        - ``.md`` — Markdown format
        - ``.html`` — HTML format
        - ``.xml`` — JUnit XML format

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
        elif suffix == ".xml":
            content = self.to_junit_xml()
        elif suffix == ".csv":
            content = self.to_csv()
        elif suffix == ".jsonl":
            content = self.to_jsonl()
        else:
            raise ValueError(
                f"Unsupported report format '{suffix}'. "
                "Supported formats are .json, .md, .html, .xml, .csv, .jsonl."
            )

        # `--save-json reports/today.json` should not fail because `reports/`
        # does not exist yet; the caller named a destination, not a directory.
        if path_obj.parent != Path(""):
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(content, encoding="utf-8")
        return path

    def to_html_table(self) -> str:
        """Render a single-file HTML page with the report's headline numbers.

        The page is intentionally minimal (no external CSS/JS) so the
        output renders in a notebook, on GitHub, or in a static-file
        preview. It is the same compact view as
        :meth:`profile_to_dict_compact` rendered as an HTML table.
        """
        payload = self.profile_to_dict_compact()
        column_types = payload["column_types"]

        def _rows(items: list[str]) -> str:
            return "".join(
                f"<tr><td>{html.escape(name)}</td></tr>" for name in items
            )

        def _list_rows(items: list[dict[str, object]], value_key: str) -> str:
            return "".join(
                f"<tr><td>{html.escape(str(item['column']))}</td>"
                f"<td>{html.escape(str(item[value_key]))}</td></tr>"
                for item in items
            )

        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>dataset-audit-kit report - {html.escape(str(payload['audit_id']) or 'report')}</title>"
            "<style>body{font-family:sans-serif;margin:2em auto;max-width:60em}"
            "table{border-collapse:collapse;margin:1em 0}th,td{border:1px solid #ccc;padding:0.3em 0.6em}"
            "th{background:#f4f4f4}</style></head><body>"
            f"<h1>dataset-audit-kit report</h1>"
            f"<p><strong>audit_id:</strong> {html.escape(str(payload['audit_id']) or '-')}<br>"
            f"<strong>created_utc:</strong> {html.escape(str(payload['created_utc']) or '-')}<br>"
            f"<strong>status:</strong> {html.escape(str(payload['status']))}<br>"
            f"<strong>rows:</strong> {int(payload['rows'])} "
            f"<strong>columns:</strong> {int(payload['columns'])} "
            f"<strong>missing_cells:</strong> {int(payload['missing_cells'])} "
            f"<strong>duplicate_rows:</strong> {int(payload['duplicate_rows'])}<br>"
            f"<strong>blocking_issues:</strong> {int(payload['blocking_issue_count'])} "
            f"<strong>risk_score:</strong> {float(payload['risk_score']):.4f}</p>"
            "<h2>Column types</h2><table><thead><tr><th>Numeric</th></tr></thead>"
            f"<tbody>{_rows(column_types['numeric'])}</tbody></table>"
            "<table><thead><tr><th>Categorical</th></tr></thead>"
            f"<tbody>{_rows(column_types['categorical'])}</tbody></table>"
            "<table><thead><tr><th>Other</th></tr></thead>"
            f"<tbody>{_rows(column_types['other'])}</tbody></table>"
            "<h2>High-missingness columns (rate &gt; 5%)</h2>"
            "<table><thead><tr><th>Column</th><th>Missing rate</th></tr></thead>"
            f"<tbody>{_list_rows(payload['high_missingness_columns'], 'rate')}</tbody></table>"
            "<h2>High-outlier columns (ratio &gt; 5%)</h2>"
            "<table><thead><tr><th>Column</th><th>Outlier ratio</th></tr></thead>"
            f"<tbody>{_list_rows(payload['high_outlier_columns'], 'ratio')}</tbody></table>"
            "<h2>Checks seen</h2><ul>"
            + "".join(
                f"<li>{html.escape(check)}</li>"
                for check in payload["checks_seen"]
            )
            + "</ul></body></html>"
        )

    def profile_to_dict_compact(self) -> dict[str, object]:
        """Return a view-friendly dict of the report's headline numbers.

        Intended for API/UI consumers that do not need the full per-column
        profile payload. The shape is::

            {
              "audit_id": str | None,
              "created_utc": str | None,
              "rows": int,
              "columns": int,
              "duplicate_rows": int,
              "missing_cells": int,
              "risk_score": float,
              "status": str,
              "column_types": {"numeric": [...], "categorical": [...], "other": [...]},
              "high_missingness_columns": [{"column": str, "rate": float}, ...],
              "high_outlier_columns": [{"column": str, "ratio": float}, ...],
              "blocking_issue_count": int,
              "checks_seen": [str, ...],
            }

        The two ``*_columns`` lists are sorted descending by their rate /
        ratio and clipped at 10 rows so the response stays compact.
        """
        column_types: dict[str, list[str]] = {"numeric": [], "categorical": [], "other": []}
        high_missing: list[dict[str, float]] = []
        high_outliers: list[dict[str, float]] = []
        for name, profile in self.column_profiles.items():
            dtype = str(profile.get("dtype", "other"))
            if dtype in column_types:
                bucket = column_types[dtype]
            else:
                bucket = column_types["other"]
            bucket.append(name)
            missing_rate = float(profile.get("missing", 0) or 0)
            count = float(profile.get("count", 0) or 0)
            if count > 0:
                rate = missing_rate / count
            else:
                rate = 0.0
            if rate > 0.05:
                high_missing.append({"column": name, "rate": round(rate, 4)})
            outlier_ratio = profile.get("outlier_ratio")
            if isinstance(outlier_ratio, (int, float)) and outlier_ratio > 0.05:
                high_outliers.append({
                    "column": name,
                    "ratio": round(float(outlier_ratio), 4),
                })
        high_missing.sort(key=lambda row: row["rate"], reverse=True)
        high_outliers.sort(key=lambda row: row["ratio"], reverse=True)
        return {
            "audit_id": self.audit_id,
            "created_utc": self.created_utc,
            "rows": self.rows,
            "columns": self.columns,
            "duplicate_rows": self.duplicate_rows,
            "missing_cells": self.missing_cells,
            "risk_score": round(float(self.risk_score), 4),
            "status": self.status,
            "column_types": {k: sorted(v) for k, v in column_types.items()},
            "high_missingness_columns": high_missing[:10],
            "high_outlier_columns": high_outliers[:10],
            "blocking_issue_count": len(self.blocking_issues),
            "checks_seen": sorted({issue.check for issue in self.issues}),
        }

    def batch_summary_csv(
        self,
        path: str,
        reports: list["AuditReport"] | None = None,
    ) -> str:
        """Write a per-report summary CSV (one row per report) to ``path``.

        ``reports`` defaults to ``[self]`` so the call doubles as a
        single-report summary writer. Each row carries the report's
        ``audit_id`` (when stamped), ``created_utc`` (when stamped), row
        and column counts, duplicate rows, missing cells, label
        distribution length, drift-score count, blocking-issue count, the
        list of failing checks, and the risk score. ``path`` is created
        with parents as needed. The CSV header is always written even when
        the input is empty so downstream parsers do not have to special-
        case zero reports.
        """
        target = Path(path)
        if target.parent != Path(""):
            target.parent.mkdir(parents=True, exist_ok=True)
        rows = reports if reports is not None else [self]
        fieldnames = [
            "audit_id", "created_utc", "rows", "columns", "duplicate_rows",
            "missing_cells", "label_distribution_size", "drift_scores",
            "blocking_issues", "failing_checks", "risk_score",
        ]
        with open(target, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for report in rows:
                failing = sorted({
                    f"{issue.check}:{issue.column or ''}"
                    for issue in report.blocking_issues
                })
                writer.writerow({
                    "audit_id": report.audit_id or "",
                    "created_utc": report.created_utc or "",
                    "rows": report.rows,
                    "columns": report.columns,
                    "duplicate_rows": report.duplicate_rows,
                    "missing_cells": report.missing_cells,
                    "label_distribution_size": len(report.label_distribution),
                    "drift_scores": len(report.drift_scores),
                    "blocking_issues": len(report.blocking_issues),
                    "failing_checks": "|".join(failing),
                    "risk_score": round(float(report.risk_score), 4),
                })
        return str(target)

    def column_overlap_table(
        self,
        other: "AuditReport",
        *,
        columns: list[str] | None = None,
        levels_left: dict[str, list[str]] | None = None,
        levels_right: dict[str, list[str]] | None = None,
    ) -> list[dict[str, object]]:
        """Quantify how categorical levels overlap with another report.

        When ``levels_left`` / ``levels_right`` are provided, they are used
        to compute Jaccard and overlap counts per column. Otherwise the
        report falls back to the ``top_5`` field captured in the report's
        column_profiles, which is enough to flag large mismatches between
        the most-frequent categories of two related datasets. Only columns
        present in both reports are considered; ``columns`` restricts the
        comparison to a chosen subset. ``columns`` referencing a name that
        is not in both reports raises ``ValueError`` so callers fail fast.

        Returned rows are sorted alphabetically by column name and carry
        ``column``, ``dtype_left``, ``dtype_right``, ``levels_left``,
        ``levels_right``, ``intersection``, ``union_size``, ``jaccard``,
        ``only_left``, and ``only_right``. ``only_left`` and ``only_right``
        are empty strings when no caller-supplied level sets were given.
        """

        def _coerce_levels(values: object) -> set[str]:
            if values is None:
                return set()
            if isinstance(values, (list, tuple, set)):
                return {str(item) for item in values}
            if isinstance(values, dict):
                return {str(k) for k in values.keys()}
            return {str(values)}

        left = self.column_profiles
        right = other.column_profiles
        shared = sorted(set(left).intersection(right))
        if columns is not None:
            requested = [str(name) for name in columns]
            missing = [name for name in requested if name not in shared]
            if missing:
                raise ValueError(
                    "unknown columns not present in both reports: " + ", ".join(missing)
                )
            selected = requested
        else:
            selected = shared

        rows: list[dict[str, object]] = []
        for column in selected:
            left_levels = (
                _coerce_levels(levels_left.get(column)) if levels_left else set()
            )
            right_levels = (
                _coerce_levels(levels_right.get(column)) if levels_right else set()
            )
            if not left_levels:
                left_levels = set(str(k) for k in (left[column].get("top_5") or {}).keys())
            if not right_levels:
                right_levels = set(str(k) for k in (right[column].get("top_5") or {}).keys())

            intersection = sorted(left_levels & right_levels)
            union = left_levels | right_levels
            jaccard = (len(intersection) / len(union)) if union else 1.0
            rows.append({
                "column": column,
                "dtype_left": str(left[column].get("dtype", "other")),
                "dtype_right": str(right[column].get("dtype", "other")),
                "levels_left": sorted(left_levels),
                "levels_right": sorted(right_levels),
                "intersection": intersection,
                "union_size": len(union),
                "jaccard": round(float(jaccard), 4),
                "only_left": sorted(left_levels - right_levels),
                "only_right": sorted(right_levels - left_levels),
            })
        return rows

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


@dataclass
class BatchAuditReport:
    """Deterministic collection of reports produced by :meth:`audit_many`."""

    reports: dict[str, AuditReport] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "warn" if any(report.status == "warn" for report in self.reports.values()) else "pass"

    @property
    def failed_paths(self) -> list[str]:
        """Return input paths with at least one warning or error."""

        return [path for path, report in self.reports.items() if report.blocking_issues]

    def gated_paths(self, fail_on: str = "warning") -> list[str]:
        """Return input paths that fail the requested severity gate."""

        return [path for path, report in self.reports.items() if report.gated_issues(fail_on)]

    def exit_code(
        self, fail_on: str = "warning", *, max_risk: float | None = None
    ) -> int:
        ceiling = _checked_max_risk(max_risk)
        if ceiling is not None and any(
            report.risk_score > ceiling for report in self.reports.values()
        ):
            return 1
        return 1 if self.gated_paths(fail_on) else 0

    def issue_counts(self) -> dict[str, int]:
        """Count blocking findings by check across every file in the batch.

        The rollup is what a caller scanning twenty files actually wants
        first: which checks fire at all, and how loudly.
        """

        counts: dict[str, int] = {}
        for report in self.reports.values():
            for issue in report.blocking_issues:
                counts[issue.check] = counts.get(issue.check, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def consistency(self) -> dict[str, object]:
        """Compare column sets and order across every audited file.

        The first file in the batch is the reference schema; every later file
        is checked against it for columns that are missing, columns that were
        added, and ordering changes. A batch of daily exports that silently
        loses or reorders a column slips past per-file audits, because each
        file alone looks fine, but the consistency report surfaces it.

        Returns a summary with the reference path, whether the batch is
        consistent, and per-file status plus a human-readable detail list.
        """

        paths = list(self.reports)
        summary: dict[str, object] = {
            "reference": paths[0] if paths else None,
            "consistent": True,
            "files": {},
        }
        if not paths:
            return summary
        reference_columns = list(self.reports[paths[0]].column_profiles)
        for path in paths:
            columns = list(self.reports[path].column_profiles)
            missing = [column for column in reference_columns if column not in columns]
            extra = [column for column in columns if column not in reference_columns]
            reordered = not missing and not extra and columns != reference_columns
            if path != paths[0] and (missing or extra or reordered):
                summary["consistent"] = False
            details: list[str] = []
            if missing:
                details.append(f"missing {', '.join(missing)}")
            if extra:
                details.append(f"extra {', '.join(extra)}")
            if reordered:
                details.append("same columns in a different order")
            summary["files"][path] = {
                "columns": columns,
                "status": "ok" if not (missing or extra or reordered) else "deviates",
                "details": details,
            }
        return summary

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "files": {path: report.to_dict() for path, report in self.reports.items()},
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

class DatasetAuditor:
    """Run a small battery of quality checks over tabular data."""

    def __init__(
        self,
        *,
        missing_threshold: float = 0.05,
        drift_threshold: float = 0.20,
        redundancy_threshold: float = 0.95,
        label_min_share: float = 0.05,
        max_duplicate_ratio: float = 0.0,
        min_rows: int | None = None,
        max_rows: int | None = None,
        max_missing_cells: int | None = None,
        max_columns: int | None = None,
        rules: ValidationRules | None = None,
        severity_weights: dict[str, float] | None = None,
        progress: bool | None = None,
        whitespace_check: bool = False,
        null_pattern_check: bool = False,
        null_pattern_threshold: float = DEFAULT_NULL_PATTERN_THRESHOLD,
        sensitive_check: bool = False,
        max_category_share: float | None = None,
        rare_category_share: float | None = None,
        missing_cooccurrence_check: bool = False,
        missing_cooccurrence_min_count: int = 1,
        missing_cooccurrence_top: int | None = 10,
    ) -> None:
        if not 0.0 <= redundancy_threshold <= 1.0:
            raise ValueError("redundancy_threshold must be between 0 and 1")
        if not 0.0 <= max_duplicate_ratio <= 1.0:
            raise ValueError("max_duplicate_ratio must be between 0 and 1")
        if not 0.0 <= null_pattern_threshold <= 1.0:
            raise ValueError("null_pattern_threshold must be between 0 and 1")
        if (
            isinstance(missing_cooccurrence_min_count, bool)
            or not isinstance(missing_cooccurrence_min_count, int)
            or missing_cooccurrence_min_count < 1
        ):
            raise ValueError("missing_cooccurrence_min_count must be a positive integer")
        if missing_cooccurrence_top is not None and (
            isinstance(missing_cooccurrence_top, bool)
            or not isinstance(missing_cooccurrence_top, int)
            or missing_cooccurrence_top < 1
        ):
            raise ValueError("missing_cooccurrence_top must be a positive integer or None")
        for name, value in (
            ("max_category_share", max_category_share),
            ("rare_category_share", rare_category_share),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 < float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be a fraction between 0 and 1")
        for name, value in (
            ("min_rows", min_rows),
            ("max_rows", max_rows),
            ("max_missing_cells", max_missing_cells),
            ("max_columns", max_columns),
        ):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if min_rows is not None and max_rows is not None and min_rows > max_rows:
            raise ValueError("min_rows must not exceed max_rows")
        validated_weights: dict[str, float] = {}
        for check, weight in (severity_weights or {}).items():
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight))
                or float(weight) < 0.0
            ):
                raise ValueError(
                    f"severity weight for '{check}' must be a non-negative finite number"
                )
            validated_weights[str(check)] = float(weight)
        self.missing_threshold = missing_threshold
        self.drift_threshold = drift_threshold
        self.redundancy_threshold = redundancy_threshold
        self.label_min_share = label_min_share
        self.max_duplicate_ratio = max_duplicate_ratio
        self.min_rows = min_rows
        self.max_rows = max_rows
        self.max_missing_cells = max_missing_cells
        self.max_columns = max_columns
        self.rules = rules
        self.severity_weights = validated_weights
        #: None means "decide from the row count"; True/False force it.
        self.progress = progress
        self.whitespace_check = whitespace_check
        self.null_pattern_check = null_pattern_check
        self.null_pattern_threshold = null_pattern_threshold
        self.sensitive_check = sensitive_check
        self.max_category_share = (
            float(max_category_share) if max_category_share is not None else None
        )
        self.rare_category_share = (
            float(rare_category_share) if rare_category_share is not None else None
        )
        self.missing_cooccurrence_check = missing_cooccurrence_check
        self.missing_cooccurrence_min_count = missing_cooccurrence_min_count
        self.missing_cooccurrence_top = missing_cooccurrence_top

    def _progress_reporter(self, data: pd.DataFrame) -> "_Progress":
        enabled = (
            self.progress
            if self.progress is not None
            else len(data) >= PROGRESS_ROW_THRESHOLD
        )
        return _Progress(enabled, total=len(_AUDIT_PHASES))

    def audit_dataframe(
        self,
        data: pd.DataFrame,
        *,
        reference: pd.DataFrame | None = None,
        label_column: str | None = None,
        expected_columns: Sequence[str] | None = None,
        require_column_order: bool = False,
        unique_columns: Sequence[str] | None = None,
        unique_together: Sequence[Sequence[str]] | None = None,
        sample_rows: int | None = None,
        sample_seed: int | None = None,
    ) -> AuditReport:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas.DataFrame")
        if sample_rows is not None and (
            isinstance(sample_rows, bool)
            or not isinstance(sample_rows, int)
            or sample_rows < 1
        ):
            raise ValueError("sample_rows must be a positive integer")
        if sample_seed is not None and (
            isinstance(sample_seed, bool) or not isinstance(sample_seed, int)
        ):
            raise ValueError("sample_seed must be an integer")
        if sample_seed is not None and sample_rows is None:
            raise ValueError("sample_seed requires sample_rows")

        issues: list[AuditIssue] = []

        original_rows = len(data)
        if sample_rows is not None and original_rows > sample_rows:
            data = data.sample(n=sample_rows, random_state=sample_seed)
            issues.append(
                AuditIssue(
                    check="sampling",
                    severity="info",
                    message=(
                        f"Audited a random sample of {sample_rows} of "
                        f"{original_rows} row(s)"
                        + (
                            f" (seed {sample_seed})."
                            if sample_seed is not None
                            else "."
                        )
                    ),
                )
            )

        row_count = len(data)
        if self.min_rows is not None and row_count < self.min_rows:
            issues.append(
                AuditIssue(
                    check="rows",
                    severity="error",
                    message=f"Dataset has {row_count} row(s); expected at least {self.min_rows}.",
                    observed=row_count,
                    threshold=self.min_rows,
                )
            )
        if self.max_rows is not None and row_count > self.max_rows:
            issues.append(
                AuditIssue(
                    check="rows",
                    severity="error",
                    message=f"Dataset has {row_count} row(s); expected at most {self.max_rows}.",
                    observed=row_count,
                    threshold=self.max_rows,
                )
            )
        column_count = len(data.columns)
        if self.max_columns is not None and column_count > self.max_columns:
            issues.append(
                AuditIssue(
                    check="columns",
                    severity="warning",
                    message=(
                        f"Dataset has {column_count} column(s); allowed at most "
                        f"{self.max_columns}."
                    ),
                    observed=column_count,
                    threshold=self.max_columns,
                )
            )
        progress = self._progress_reporter(data)
        progress.advance("missingness")
        missingness = self._missingness(data, issues)
        progress.advance("missing_cooccurrence")
        self._check_missing_cooccurrence(data, issues)
        missing_cells = int(data.isna().sum().sum())
        if (
            self.max_missing_cells is not None
            and missing_cells > self.max_missing_cells
        ):
            issues.append(
                AuditIssue(
                    check="missing_cells",
                    severity="warning",
                    message=(
                        f"Dataset has {missing_cells} missing cell(s); allowed at most "
                        f"{self.max_missing_cells}."
                    ),
                    observed=missing_cells,
                    threshold=self.max_missing_cells,
                )
            )
        progress.advance("duplicates")
        duplicate_rows = int(data.duplicated().sum())
        duplicate_ratio = duplicate_rows / max(len(data), 1)
        if duplicate_rows and duplicate_ratio > self.max_duplicate_ratio:
            issues.append(
                AuditIssue(
                    check="duplicates",
                    severity="warning",
                    message=(
                        f"Duplicate ratio {duplicate_ratio:.1%} exceeds allowed "
                        f"{self.max_duplicate_ratio:.1%} ({duplicate_rows} row(s))."
                    ),
                    observed=duplicate_rows,
                    threshold=self.max_duplicate_ratio,
                )
            )

        progress.advance("column names")
        self._check_column_names(data, issues)

        progress.advance("schema")
        if expected_columns is not None:
            self._check_schema(
                data, expected_columns, issues, require_column_order=require_column_order
            )

        progress.advance("uniqueness")
        if unique_columns is not None:
            self._check_uniqueness(data, unique_columns, issues)
        if unique_together is not None:
            self._check_composite_uniqueness(data, unique_together, issues)

        progress.advance("label balance")
        label_distribution: dict[str, int] = {}
        if label_column is not None and label_column in data.columns:
            label_distribution = self._check_label_balance(data, label_column, issues)

        progress.advance("drift")
        drift_scores: dict[str, float] = {}
        correlation_drift_scores: dict[str, float] = {}
        if reference is not None:
            drift_scores = self._check_drift(data, reference, issues, label_column=label_column)
            correlation_drift_scores = self._correlation_drift(
                data, reference, issues, drift_threshold=self.drift_threshold
            )
            self._schema_diff(data, reference, issues)

        progress.advance("rules")
        # Per-column validation rules
        self._apply_rules(data, issues)
        # Relational constraints between column pairs
        self._check_cross_rules(data, issues)

        progress.advance("whitespace")
        if self.whitespace_check:
            self._check_whitespace_values(data, issues)

        progress.advance("null_patterns")
        if self.null_pattern_check:
            self._check_null_patterns(data, issues, threshold=self.null_pattern_threshold)

        progress.advance("sensitive")
        if self.sensitive_check:
            self._check_sensitive_values(data, issues)

        progress.advance("profiles")
        column_profiles = self._profile_columns(data)

        progress.advance("category_share")
        self._check_category_share(data, issues)

        progress.advance("redundancy")
        # Redundancy / collinearity check
        self._check_redundancy(
            data, issues, correlation_threshold=self.redundancy_threshold
        )

        # Identical-content columns that correlation cannot see
        self._check_duplicate_columns(data, issues)

        progress.close()

        all_drift_scores = {**drift_scores, **correlation_drift_scores}

        report = AuditReport(
            rows=int(len(data)),
            columns=column_count,
            duplicate_rows=duplicate_rows,
            missing_cells=missing_cells,
            missingness=missingness,
            column_profiles=column_profiles,
            label_distribution=label_distribution,
            drift_scores=all_drift_scores,
            issues=issues,
        )
        report.risk_score = self._risk_score(issues)
        return report

    def _risk_score(self, issues: Sequence[AuditIssue]) -> float:
        """Fold every blocking finding into one bounded 0-100 score.

        An error contributes its check's weight in full and a warning half of
        it; ``info`` findings describe the dataset rather than fault it, so
        they add nothing. Checks without an explicit weight count at
        :data:`DEFAULT_RISK_WEIGHT`, and the total is capped so a pathological
        dataset cannot leave the fixed range.
        """

        total = 0.0
        for issue in issues:
            if issue.severity == "info":
                continue
            weight = self.severity_weights.get(issue.check, DEFAULT_RISK_WEIGHT)
            multiplier = 1.0 if issue.severity == "error" else RISK_WARNING_FACTOR
            total += weight * multiplier
        return round(min(total, MAX_RISK_SCORE), 1)

    def audit_csv(
        self,
        data_path: str,
        *,
        reference_path: str | None = None,
        label_column: str | None = None,
        expected_columns: Sequence[str] | None = None,
        require_column_order: bool = False,
        unique_together: Sequence[Sequence[str]] | None = None,
        encoding: str | None = None,
        delimiter: str | None = None,
    ) -> AuditReport:
        return self.audit_file(
            data_path,
            reference_path=reference_path,
            label_column=label_column,
            expected_columns=expected_columns,
            require_column_order=require_column_order,
            unique_together=unique_together,
            encoding=encoding,
            delimiter=delimiter,
        )

    def audit_file(
        self,
        data_path: str,
        *,
        reference_path: str | None = None,
        label_column: str | None = None,
        expected_columns: Sequence[str] | None = None,
        require_column_order: bool = False,
        unique_columns: Sequence[str] | None = None,
        unique_together: Sequence[Sequence[str]] | None = None,
        encoding: str | None = None,
        delimiter: str | None = None,
        sample_rows: int | None = None,
        sample_seed: int | None = None,
    ) -> AuditReport:
        """Audit a dataset on disk.

        ``encoding`` and ``delimiter`` are handed to the reader for both the
        dataset and the reference, so a latin-1 or semicolon-separated pair
        loads the same way through the file API as through ``load_dataframe``.
        """

        data = self.load_dataframe(data_path, encoding=encoding, delimiter=delimiter)
        reference = (
            self.load_dataframe(reference_path, encoding=encoding, delimiter=delimiter)
            if reference_path
            else None
        )
        return self.audit_dataframe(
            data,
            reference=reference,
            label_column=label_column,
            expected_columns=expected_columns,
            require_column_order=require_column_order,
            unique_columns=unique_columns,
            unique_together=unique_together,
            sample_rows=sample_rows,
            sample_seed=sample_seed,
        )

    def audit_many(
        self,
        data_paths: Sequence[str | Path],
        *,
        reference_path: str | Path | None = None,
        label_column: str | None = None,
        expected_columns: Sequence[str] | None = None,
        require_column_order: bool = False,
        unique_columns: Sequence[str] | None = None,
        unique_together: Sequence[Sequence[str]] | None = None,
        encoding: str | None = None,
        delimiter: str | None = None,
    ) -> BatchAuditReport:
        """Audit multiple datasets while preserving input order.

        The same optional reference and validation settings are applied to
        every path.  Results are keyed by the caller-provided path spelling so
        a batch can be joined back to its manifest without path rewriting.
        """

        reports = {
            str(path): self.audit_file(
                path,
                reference_path=str(reference_path) if reference_path is not None else None,
                label_column=label_column,
                expected_columns=expected_columns,
                require_column_order=require_column_order,
                unique_columns=unique_columns,
                unique_together=unique_together,
                encoding=encoding,
                delimiter=delimiter,
            )
            for path in data_paths
        }
        return BatchAuditReport(reports=reports)

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

    def _check_missing_cooccurrence(
        self,
        data: pd.DataFrame,
        issues: list[AuditIssue],
    ) -> None:
        """Flag column pairs that are missing in the same rows.

        Two columns that are empty together are rarely an accident: a derived
        column computed from a sparsely populated source, a shared export bug,
        or an optional field gated behind the same condition. The per-column
        missingness check cannot see the pattern, so an opt-in scan looks at
        every pair of columns with any missing values and reports those whose
        missingness indicators are positively correlated. The score is the
        Pearson correlation of the two boolean masks (the phi coefficient for
        binary data), which reaches 1.0 when both columns are always empty
        together and is undefined — reported as zero — when either column is
        constant.
        """

        if not self.missing_cooccurrence_check:
            return
        missing_columns = [
            column for column in data.columns if bool(data[column].isna().any())
        ]
        if len(missing_columns) < 2:
            return

        missing = data[missing_columns].isna()
        pairs: list[tuple[int, float, str, str]] = []
        for i in range(len(missing_columns)):
            left = missing_columns[i]
            for right in missing_columns[i + 1:]:
                co_count = int((missing[left] & missing[right]).sum())
                if co_count < self.missing_cooccurrence_min_count:
                    continue
                score = float(missing[left].corr(missing[right]))
                if score != score:  # NaN: one mask is constant
                    score = 0.0
                pairs.append((co_count, score, str(left), str(right)))

        # Most joined first, then most strongly correlated, then by name so
        # the ordering stays deterministic across runs.
        pairs.sort(key=lambda pair: (-pair[0], -pair[1], pair[2], pair[3]))
        if self.missing_cooccurrence_top is not None:
            pairs = pairs[: self.missing_cooccurrence_top]

        for co_count, score, left, right in pairs:
            issues.append(
                AuditIssue(
                    check="missing_cooccurrence",
                    severity="warning",
                    message=(
                        f"Columns '{left}' and '{right}' are both missing in "
                        f"{co_count} row(s) (missingness correlation {score:.2f})."
                    ),
                    column=f"{left},{right}",
                    observed=co_count,
                    threshold=self.missing_cooccurrence_min_count,
                )
            )

    def _check_column_names(self, data: pd.DataFrame, issues: list[AuditIssue]) -> None:
        """Flag column names that tend to break downstream tooling.

        These are warnings, not errors: pandas is happy with any of them, but
        they routinely cause trouble once the frame reaches SQL, Parquet
        round-trips, ``df.query`` or attribute access.
        """

        seen: dict[str, str] = {}
        exact_seen: set[str] = set()
        for column in data.columns:
            name = str(column)

            if name in exact_seen:
                issues.append(
                    AuditIssue(
                        check="column_names",
                        severity="error",
                        message=(
                            "Duplicate column name: selecting it returns a frame "
                            "rather than a series, which breaks most downstream code."
                        ),
                        column=name,
                        observed=repr(name),
                    )
                )
                continue
            exact_seen.add(name)

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

    def _check_whitespace_values(
        self, data: pd.DataFrame, issues: list[AuditIssue]
    ) -> None:
        """Flag text values damaged by padding or invisible characters.

        These characters survive copy-paste from spreadsheets and web forms;
        only textual columns are scanned, since numeric parsing rejects the
        padding on its own.
        """

        for position, column in enumerate(data.columns):
            col = data.iloc[:, position]
            if not (
                isinstance(col.dtype, pd.CategoricalDtype)
                or col.dtype == object
                or pd.api.types.is_string_dtype(col)
            ):
                continue
            values = col.dropna().astype(str)
            if values.empty:
                continue
            padded = int((values != values.str.strip()).sum())
            if padded:
                issues.append(
                    AuditIssue(
                        check="whitespace",
                        severity="warning",
                        message=f"{padded} value(s) have leading or trailing whitespace.",
                        column=str(column),
                        observed=padded,
                    )
                )
            mask = values.str.contains(INVISIBLE_CHAR_PATTERN, regex=True)
            invisible = int(mask.sum())
            if invisible:
                issues.append(
                    AuditIssue(
                        check="whitespace",
                        severity="warning",
                        message=(
                            f"{invisible} value(s) contain invisible characters "
                            "such as zero-width spaces."
                        ),
                        column=str(column),
                        observed=invisible,
                    )
                )

    def _check_null_patterns(
        self,
        data: pd.DataFrame,
        issues: list[AuditIssue],
        *,
        threshold: float = DEFAULT_NULL_PATTERN_THRESHOLD,
    ) -> None:
        """Flag text columns containing literal null-pattern strings.

        Many datasets encode missingness as literal strings like "NA", "NULL",
        "N/A", "", "None", "nan", etc. instead of using proper NaN/None.
        This check detects such patterns in text/categorical columns when they
        exceed a configured fraction of non-null values.
        """
        if threshold <= 0:
            return
        for position, column in enumerate(data.columns):
            col = data.iloc[:, position]
            if not (
                isinstance(col.dtype, pd.CategoricalDtype)
                or col.dtype == object
                or pd.api.types.is_string_dtype(col)
            ):
                continue
            values = col.dropna().astype(str)
            if values.empty:
                continue
            total = len(values)
            # Count values that match known null patterns (case-insensitive)
            lowered = values.str.lower()
            null_like = lowered.isin(NULL_PATTERNS_LOWER)
            count = int(null_like.sum())
            if count == 0:
                continue
            ratio = count / total
            if ratio >= threshold:
                examples = (
                    values[null_like]
                    .unique()[:5]
                    .tolist()
                )
                issues.append(
                    AuditIssue(
                        check="null_pattern",
                        severity="warning",
                        message=(
                            f"{count} value(s) ({ratio:.1%}) match common null "
                            f"patterns (e.g., {', '.join(repr(e) for e in examples)}). "
                            f"Consider converting to proper missing values."
                        ),
                        column=str(column),
                        observed=ratio,
                        threshold=threshold,
                    )
                )

    def _check_sensitive_values(
        self,
        data: pd.DataFrame,
        issues: list[AuditIssue],
    ) -> None:
        """Flag text values shaped like email, phone, or SSN records.

        Opt-in because false positives are possible: a code column could
        legitimately contain ``123-456-7890``. The scan is meant to surface
        payloads that need a data-governance decision, so each flagged column
        names the pattern kind and the number of matching values.
        """

        for position, column in enumerate(data.columns):
            col = data.iloc[:, position]
            if not (
                isinstance(col.dtype, pd.CategoricalDtype)
                or col.dtype == object
                or pd.api.types.is_string_dtype(col)
            ):
                continue
            values = col.dropna().astype(str)
            if values.empty:
                continue
            for kind, pattern in SENSITIVE_PATTERNS:
                mask = values.str.contains(pattern)
                count = int(mask.sum())
                if count:
                    issues.append(
                        AuditIssue(
                            check="sensitive",
                            severity="warning",
                            message=(
                                f"{count} value(s) match {kind} pattern "
                                f"(e.g., {values[mask].iloc[0]!r})."
                            ),
                            column=str(column),
                            observed=count,
                        )
                    )

    def _check_category_share(
        self,
        data: pd.DataFrame,
        issues: list[AuditIssue],
    ) -> None:
        """Flag categorical columns dominated by or starved of values.

        A column whose most frequent value accounts for more than
        ``max_category_share`` of the non-missing values carries almost no
        information, and one containing categories rarer than
        ``rare_category_share`` usually means the class is too sparse to learn
        from. Both thresholds are off unless configured.
        """

        if self.max_category_share is None and self.rare_category_share is None:
            return
        for position, column in enumerate(data.columns):
            col = data.iloc[:, position]
            if not (
                isinstance(col.dtype, pd.CategoricalDtype)
                or col.dtype == object
                or pd.api.types.is_string_dtype(col)
            ):
                continue
            values = col.dropna().astype(str)
            if len(values) == 0:
                continue
            shares = values.value_counts(normalize=True)
            if self.max_category_share is not None:
                top = str(shares.index[0])
                top_share = float(shares.iloc[0])
                if top_share > self.max_category_share:
                    issues.append(
                        AuditIssue(
                            check="category_share",
                            severity="warning",
                            message=(
                                f"Category '{top}' dominates column with "
                                f"{top_share:.1%} of non-missing values "
                                f"(allowed {self.max_category_share:.1%})."
                            ),
                            column=str(column),
                            observed=round(top_share, 4),
                            threshold=self.max_category_share,
                        )
                    )
            if self.rare_category_share is not None:
                rare = shares[shares < self.rare_category_share]
                if len(rare):
                    examples = ", ".join(repr(str(v)) for v in list(rare.index)[:5])
                    issues.append(
                        AuditIssue(
                            check="category_share",
                            severity="warning",
                            message=(
                                f"{len(rare)} rare category(ies) below the "
                                f"{self.rare_category_share:.1%} share "
                                f"(e.g., {examples})."
                            ),
                            column=str(column),
                            observed=len(rare),
                            threshold=self.rare_category_share,
                        )
                    )

    def _check_schema(
        self,
        data: pd.DataFrame,
        expected_columns: Sequence[str],
        issues: list[AuditIssue],
        *,
        require_column_order: bool = False,
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
        if require_column_order and not missing and not extra and observed != expected:
            issues.append(
                AuditIssue(
                    check="schema",
                    severity="error",
                    message="Column order does not match the expected schema contract.",
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
            # Count the redundant rows: every occurrence after the first. The
            # old `duplicated(keep=False) // 2` undercounted whenever a value
            # repeated more than twice (three copies reported as one).
            duplicates = int(data[col].duplicated().sum())
            if duplicates > 0:
                repeated = int(data[col].duplicated(keep=False).sum()) - duplicates
                issues.append(
                    AuditIssue(
                        check="uniqueness",
                        severity="warning",
                        message=(
                            f"{duplicates} duplicate row(s) across {repeated} repeated "
                            f"value(s) in unique column '{col}'."
                        ),
                        column=col,
                        observed=duplicates,
                    )
                )

    @staticmethod
    def _check_composite_uniqueness(
        data: pd.DataFrame,
        groups: Sequence[Sequence[str]],
        issues: list[AuditIssue],
    ) -> None:
        """Validate uniqueness constraints spanning two or more columns."""

        for raw_group in groups:
            group = [str(column) for column in raw_group]
            if len(group) < 2:
                raise ValueError("unique_together groups must contain at least two columns")
            missing = [column for column in group if column not in data.columns]
            label = ", ".join(group)
            if missing:
                issues.append(
                    AuditIssue(
                        check="composite_uniqueness",
                        severity="error",
                        message=(
                            f"Composite key [{label}] references missing column(s): "
                            f"{', '.join(missing)}."
                        ),
                    )
                )
                continue
            duplicate_rows = int(data.duplicated(subset=group, keep="first").sum())
            if duplicate_rows:
                issues.append(
                    AuditIssue(
                        check="composite_uniqueness",
                        severity="error",
                        message=(
                            f"Composite key [{label}] has {duplicate_rows} duplicate row(s)."
                        ),
                        observed=duplicate_rows,
                        threshold=0,
                    )
                )

    def _check_label_balance(
        self,
        data: pd.DataFrame,
        label_column: str,
        issues: list[AuditIssue],
    ) -> dict[str, int]:
        labels = data[label_column]
        # Going through object first: filling a Categorical with a value that is
        # not one of its categories raises, so a categorical label column used
        # to crash the whole audit here.
        filled = labels.astype(object).where(labels.notna(), "<missing>").astype(str)
        counts = filled.value_counts()
        missing_count = int(labels.isna().sum())
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

        # Balance is a property of the labels that exist; the `<missing>` bucket
        # is reported separately above and must not be mistaken for a class.
        real_counts = counts.drop(labels="<missing>", errors="ignore")
        real_total = int(real_counts.sum()) or 1

        if real_counts.empty:
            issues.append(
                AuditIssue(
                    check="labels",
                    severity="warning",
                    message=f"No label values found in `{label_column}`.",
                    column=label_column,
                    observed=0,
                )
            )
        elif real_counts.size < 2:
            issues.append(
                AuditIssue(
                    check="labels",
                    severity="warning",
                    message="Only one label value is present.",
                    column=label_column,
                    observed=int(real_counts.iloc[0]),
                )
            )
        else:
            minority_share = real_counts.min() / real_total
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

    @staticmethod
    def _effective_date_formats(rule: ColumnRule) -> tuple[str, ...]:
        """Return the date formats a rule accepts, newest contract first.

        ``date_formats`` supersedes the single ``date_format`` key when both
        are present, mirroring how ``ValidationRules.from_dict`` resolves them.
        """

        if rule.date_formats is not None:
            return rule.date_formats
        if rule.date_format is not None:
            return (rule.date_format,)
        return ()

    @staticmethod
    def _validate_date_format(fmt: str) -> None:
        """Raise ValueError unless ``fmt`` round-trips a reference datetime.

        A format is usable when strftime can render a reference date and
        strptime can parse the rendering back. Directives neither understands
        (such as ``%Q``) raise, which is what guards the rules contract. The
        round-trip avoids the old sample-value probe, which rejected perfectly
        valid formats like ``%d/%m/%Y`` simply because they do not parse the
        literal probe ``2000-01-01``.
        """

        from datetime import datetime

        rendered = datetime(2000, 1, 1).strftime(fmt)
        datetime.strptime(rendered, fmt)

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

            # --- cardinality bounds ---
            if rule.min_unique is not None or rule.max_unique is not None:
                unique_count = int(col_data.nunique(dropna=True))
                if rule.min_unique is not None and unique_count < rule.min_unique:
                    issues.append(
                        AuditIssue(
                            check="rule",
                            severity="warning",
                            message=(
                                f"Only {unique_count} unique value(s); expected at least "
                                f"{rule.min_unique}."
                            ),
                            column=column_name,
                            observed=unique_count,
                            threshold=rule.min_unique,
                        )
                    )
                if rule.max_unique is not None and unique_count > rule.max_unique:
                    issues.append(
                        AuditIssue(
                            check="rule",
                            severity="warning",
                            message=(
                                f"{unique_count} unique value(s); expected at most "
                                f"{rule.max_unique}."
                            ),
                            column=column_name,
                            observed=unique_count,
                            threshold=rule.max_unique,
                        )
                    )

            # --- text length bounds ---
            if rule.min_length is not None or rule.max_length is not None:
                lengths = col_data.dropna().astype(str).str.len()
                if rule.min_length is not None:
                    violations = int((lengths < rule.min_length).sum())
                    if violations:
                        issues.append(
                            AuditIssue(
                                check="rule",
                                severity="warning",
                                message=(
                                    f"{violations} value(s) shorter than minimum length "
                                    f"{rule.min_length}."
                                ),
                                column=column_name,
                                observed=violations,
                                threshold=rule.min_length,
                            )
                        )
                if rule.max_length is not None:
                    violations = int((lengths > rule.max_length).sum())
                    if violations:
                        issues.append(
                            AuditIssue(
                                check="rule",
                                severity="warning",
                                message=(
                                    f"{violations} value(s) longer than maximum length "
                                    f"{rule.max_length}."
                                ),
                                column=column_name,
                                observed=violations,
                                threshold=rule.max_length,
                            )
                        )

            # --- numeric bounds ---
            if rule.min_value is not None or rule.max_value is not None:
                numeric = pd.to_numeric(col_data.dropna(), errors="coerce")
                if rule.min_value is not None:
                    effective_min = rule.min_value - rule.value_tolerance
                    if rule.min_inclusive:
                        violations = int((numeric < effective_min).sum())
                    else:
                        violations = int((numeric <= effective_min).sum())
                    if violations:
                        modifier = " (exclusive bound)" if not rule.min_inclusive else ""
                        if rule.value_tolerance:
                            modifier += f" beyond {rule.value_tolerance:g} tolerance"
                        issues.append(
                            AuditIssue(
                                check="rule",
                                severity="warning",
                                message=(
                                    f"{violations} value(s) below minimum "
                                    f"{rule.min_value}{modifier}."
                                ),
                                column=column_name,
                                observed=float(violations),
                                threshold=rule.min_value,
                            )
                        )
                if rule.max_value is not None:
                    effective_max = rule.max_value + rule.value_tolerance
                    if rule.max_inclusive:
                        violations = int((numeric > effective_max).sum())
                    else:
                        violations = int((numeric >= effective_max).sum())
                    if violations:
                        modifier = " (exclusive bound)" if not rule.max_inclusive else ""
                        if rule.value_tolerance:
                            modifier += f" beyond {rule.value_tolerance:g} tolerance"
                        issues.append(
                            AuditIssue(
                                check="rule",
                                severity="warning",
                                message=(
                                    f"{violations} value(s) above maximum "
                                    f"{rule.max_value}{modifier}."
                                ),
                                column=column_name,
                                observed=float(violations),
                                threshold=rule.max_value,
                            )
                        )

            # --- outlier allowance (IQR or percentile fences) ---
            if (
                rule.max_outlier_ratio is not None
                or rule.min_value is not None
                or rule.max_value is not None
                or rule.percentile_fences is not None
            ):
                numeric = pd.to_numeric(col_data.dropna(), errors="coerce")
                if len(numeric) >= 4:
                    if rule.percentile_fences is not None:
                        lower_q, upper_q = rule.percentile_fences
                        lower_fence = float(numeric.quantile(lower_q))
                        upper_fence = float(numeric.quantile(upper_q))
                        method_label = "percentile"
                    else:
                        q1 = float(numeric.quantile(0.25))
                        q3 = float(numeric.quantile(0.75))
                        iqr = q3 - q1
                        if iqr <= 0:
                            lower_fence = upper_fence = None
                        else:
                            lower_fence = q1 - 1.5 * iqr
                            upper_fence = q3 + 1.5 * iqr
                        method_label = "IQR"
                    if lower_fence is not None and upper_fence is not None:
                        # Bounds act as a hard fence for outlier counting, so
                        # the tolerance slack applies to them just as it does
                        # to the bound check itself.
                        bound_min = (
                            rule.min_value - rule.value_tolerance
                            if rule.min_value is not None
                            else lower_fence
                        )
                        bound_max = (
                            rule.max_value + rule.value_tolerance
                            if rule.max_value is not None
                            else upper_fence
                        )
                        low_outliers = int((numeric < max(lower_fence, bound_min)).sum())
                        high_outliers = int((numeric > min(upper_fence, bound_max)).sum())
                        total_outliers = low_outliers + high_outliers
                        total = len(numeric)
                        outlier_ratio = total_outliers / max(total, 1)
                        allowed_ratio = (
                            rule.max_outlier_ratio
                            if rule.max_outlier_ratio is not None
                            else 0.01
                        )
                        if total_outliers > 0 and outlier_ratio > allowed_ratio:
                            if rule.max_outlier_ratio is None:
                                message = (
                                    f"{total_outliers} {method_label} outlier(s) detected "
                                    f"({outlier_ratio * 100:.1f}% of values)."
                                )
                            else:
                                message = (
                                    f"{method_label} outlier ratio {outlier_ratio:.1%} exceeds allowed "
                                    f"{allowed_ratio:.1%} ({total_outliers} value(s))."
                                )
                            issues.append(
                                AuditIssue(
                                    check="rule",
                                    severity="info",
                                    message=message,
                                    column=column_name,
                                    observed=total_outliers,
                                    threshold=allowed_ratio,
                                )
                            )

            # --- allowed values ---
            if rule.allowed_values is not None:
                allowed_set = set(str(v) for v in rule.allowed_values)
                actual_values = col_data.dropna().astype(str).unique()
                if rule.ignore_case:
                    folded = {value.casefold() for value in allowed_set}
                    unexpected = [
                        str(value)
                        for value in actual_values
                        if str(value).casefold() not in folded
                    ]
                else:
                    unexpected = [
                        str(value) for value in actual_values if str(value) not in allowed_set
                    ]
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

            # --- text pattern contract ---
            if rule.pattern is not None:
                try:
                    matcher = re.compile(rule.pattern)
                except re.error as exc:
                    raise ValueError(
                        f"Invalid pattern for column '{column_name}': {exc}"
                    ) from exc
                values = col_data.dropna().astype(str)
                violations = int((~values.map(lambda value: bool(matcher.fullmatch(value)))).sum())
                if violations:
                    issues.append(
                        AuditIssue(
                            check="rule",
                            severity="warning",
                            message=(
                                f"{violations} value(s) do not match pattern "
                                f"'{rule.pattern}'."
                            ),
                            column=column_name,
                            observed=violations,
                        )
                    )

            # --- datetime format contract ---
            formats = self._effective_date_formats(rule)
            if formats:
                from datetime import datetime

                for fmt in formats:
                    try:
                        self._validate_date_format(fmt)
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid date_format for column '{column_name}': {exc}"
                        ) from exc
                values = col_data.dropna().astype(str)
                violations = 0
                for value in values:
                    matched = False
                    for fmt in formats:
                        try:
                            datetime.strptime(value, fmt)
                        except (ValueError, TypeError):
                            continue
                        matched = True
                        break
                    if not matched:
                        violations += 1
                if violations:
                    if len(formats) == 1:
                        message = (
                            f"{violations} value(s) do not parse with date "
                            f"format '{formats[0]}'."
                        )
                    else:
                        message = (
                            f"{violations} value(s) do not parse with any of the "
                            f"configured date formats ({', '.join(formats)})."
                        )
                    issues.append(
                        AuditIssue(
                            check="rule",
                            severity="warning",
                            message=message,
                            column=column_name,
                            observed=violations,
                        )
                    )

            # --- date-range bounds ---
            if (
                rule.min_date is not None
                or rule.max_date is not None
                or rule.no_future_dates
            ):
                from datetime import datetime

                lower_bound = None
                if rule.min_date is not None:
                    try:
                        lower_bound = pd.Timestamp(rule.min_date)
                    except (ValueError, TypeError) as exc:
                        raise ValueError(
                            f"Invalid min_date for column '{column_name}': {exc}"
                        ) from exc
                upper_bound = None
                if rule.max_date is not None:
                    try:
                        upper_bound = pd.Timestamp(rule.max_date)
                    except (ValueError, TypeError) as exc:
                        raise ValueError(
                            f"Invalid max_date for column '{column_name}': {exc}"
                        ) from exc
                if (
                    lower_bound is not None
                    and upper_bound is not None
                    and lower_bound > upper_bound
                ):
                    raise ValueError(
                        f"min_date cannot exceed max_date for column '{column_name}'"
                    )

                raw = col_data.dropna()
                if raw.empty:
                    continue
                formats = self._effective_date_formats(rule)
                if formats:
                    parsed_values: list[pd.Timestamp | None] = []
                    for value in raw.astype(str):
                        parsed_value = None
                        for fmt in formats:
                            try:
                                parsed_value = pd.Timestamp(
                                    datetime.strptime(value, fmt)
                                )
                            except (ValueError, TypeError):
                                continue
                            break
                        parsed_values.append(parsed_value)
                    parsed = pd.Series(parsed_values, index=raw.index)
                else:
                    parsed = pd.to_datetime(raw, errors="coerce")
                if not parsed.notna().any():
                    continue
                if lower_bound is not None:
                    violations = int((parsed < lower_bound).sum())
                    if violations:
                        issues.append(
                            AuditIssue(
                                check="rule",
                                severity="warning",
                                message=(
                                    f"{violations} value(s) before minimum date "
                                    f"'{rule.min_date}'."
                                ),
                                column=column_name,
                                observed=violations,
                            )
                        )
                if upper_bound is not None:
                    violations = int((parsed > upper_bound).sum())
                    if violations:
                        issues.append(
                            AuditIssue(
                                check="rule",
                                severity="warning",
                                message=(
                                    f"{violations} value(s) after maximum date "
                                    f"'{rule.max_date}'."
                                ),
                                column=column_name,
                                observed=violations,
                            )
                        )
                if rule.no_future_dates:
                    now = pd.Timestamp.now()
                    violations = int((parsed > now).sum())
                    if violations:
                        issues.append(
                            AuditIssue(
                                check="rule",
                                severity="warning",
                                message=(
                                    f"{violations} value(s) lie in the future "
                                    f"(after {now.date()})."
                                ),
                                column=column_name,
                                observed=violations,
                            )
                        )

    def _check_cross_rules(
        self,
        data: pd.DataFrame,
        issues: list[AuditIssue],
    ) -> None:
        """Evaluate relational constraints between column pairs."""
        if self.rules is None or not self.rules.cross:
            return

        op_names = {
            "le": ("less than or equal to", lambda a, b: a <= b),
            "lt": ("less than", lambda a, b: a < b),
            "ge": ("greater than or equal to", lambda a, b: a >= b),
            "gt": ("greater than", lambda a, b: a > b),
            "eq": ("equal to", lambda a, b: a == b),
            "ne": ("not equal to", lambda a, b: a != b),
        }

        for rule in self.rules.cross:
            if rule.left not in data.columns:
                issues.append(
                    AuditIssue(
                        check="cross-rule",
                        severity="error",
                        message=(
                            f"Cross-column rule references missing column "
                            f"'{rule.left}'."
                        ),
                        column=rule.left,
                    )
                )
                continue
            if rule.right not in data.columns:
                issues.append(
                    AuditIssue(
                        check="cross-rule",
                        severity="error",
                        message=(
                            f"Cross-column rule references missing column "
                            f"'{rule.right}'."
                        ),
                        column=rule.right,
                    )
                )
                continue

            left = pd.to_numeric(data[rule.left], errors="coerce")
            right = pd.to_numeric(data[rule.right], errors="coerce")
            if rule.missing_ok:
                present = left.notna() & right.notna()
                violations = int((~op_names[rule.op][1](left, right) & present).sum())
            else:
                violations = int((~op_names[rule.op][1](left, right)).sum())
            if violations:
                issues.append(
                    AuditIssue(
                        check="cross-rule",
                        severity="warning",
                        message=(
                            f"{violations} row(s) violate '{rule.left} {rule.op} "
                            f"{rule.right}'."
                        ),
                        column=f"{rule.left},{rule.right}",
                        observed=violations,
                    )
                )

    @staticmethod
    def _infer_dtype(series: pd.Series) -> str:
        """Infer a human-readable type for a series."""
        """Infer a human-readable type for a series."""
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if (
            isinstance(series.dtype, pd.CategoricalDtype)
            or series.dtype == object
            or pd.api.types.is_string_dtype(series)
        ):
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
                psi_score = self.population_stability_index(baseline, current)
                drift_scores[f"{column}__psi"] = psi_score
            else:
                score = self._categorical_drift(current.astype(str), baseline.astype(str))
                psi_score = None

            drift_scores[column] = score
            rule = self.rules.columns.get(column) if self.rules is not None else None
            threshold = (
                rule.max_drift
                if rule is not None and rule.max_drift is not None
                else self.drift_threshold
            )
            if score >= threshold:
                issues.append(
                    AuditIssue(
                        check="drift",
                        severity="warning",
                        message=f"Drift score {score:.3f} exceeds the {threshold:.3f} threshold.",
                        column=column,
                        observed=score,
                        threshold=threshold,
                    )
                )
            if psi_score is not None and psi_score >= threshold:
                issues.append(
                    AuditIssue(
                        check="psi",
                        severity="warning",
                        message=f"PSI {psi_score:.3f} exceeds the {threshold:.3f} threshold.",
                        column=column,
                        observed=psi_score,
                        threshold=threshold,
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

        # Positional access: with a duplicated column name `data[name]` hands
        # back a frame instead of a series, and every statistic below would
        # blow up on it.
        for position, column in enumerate(data.columns):
            col = data.iloc[:, position]
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
            elif isinstance(col.dtype, pd.CategoricalDtype) or col.dtype == object or pd.api.types.is_string_dtype(col):
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
    def population_stability_index(
        baseline: pd.Series,
        current: pd.Series,
        bins: int = 10,
    ) -> float:
        """Population Stability Index (PSI) between two numeric samples.

        The *baseline* distribution is split into ``bins`` equal-frequency
        buckets using its own quantiles; the *current* sample is measured
        against those same buckets. PSI sums
        ``(p_current - p_base) * ln(p_current / p_base)`` over the buckets.

        Conventional interpretation: PSI < 0.1 signals no material shift,
        0.1-0.25 a moderate one, and above 0.25 a large one. A value of 0.0
        means the two samples are distributionally identical within the
        chosen binning. Values outside the baseline range are folded into the
        nearest edge bucket, matching the standard PSI convention.
        """
        baseline_vals = pd.to_numeric(baseline, errors="coerce").dropna().to_numpy()
        current_vals = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
        if baseline_vals.size < 2 or current_vals.size < 1:
            return 0.0
        n_bins = max(2, min(int(bins), baseline_vals.size))
        edges = np.quantile(baseline_vals, [i / n_bins for i in range(n_bins + 1)])
        edges = np.unique(edges)
        if edges.size < 2:
            return 0.0
        base_counts, _ = np.histogram(baseline_vals, bins=edges)
        current_counts, _ = np.histogram(current_vals, bins=edges)
        base_pct = np.clip(base_counts / baseline_vals.size, 1e-6, None)
        current_pct = np.clip(current_counts / current_vals.size, 1e-6, None)
        psi = float(np.sum((current_pct - base_pct) * np.log(current_pct / base_pct)))
        return max(psi, 0.0)

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
    def _check_duplicate_columns(
        data: pd.DataFrame,
        issues: list[AuditIssue],
    ) -> None:
        """Flag column pairs that hold identical values cell for cell.

        Correlation redundancy only sees numeric columns, so a copied text or
        categorical column slips past it; an exact content match is caught
        here regardless of dtype. Columns that are entirely missing are
        skipped, since two empty columns say nothing about each other.
        """

        if len(data) == 0:
            return
        for i in range(len(data.columns)):
            left = data.iloc[:, i]
            if left.notna().sum() == 0:
                continue
            for j in range(i + 1, len(data.columns)):
                right = data.iloc[:, j]
                if right.notna().sum() == 0:
                    continue
                same = (left == right) | (left.isna() & right.isna())
                if bool(same.all()):
                    name_i, name_j = str(data.columns[i]), str(data.columns[j])
                    issues.append(
                        AuditIssue(
                            check="duplicate_columns",
                            severity="warning",
                            message=(
                                f"Columns '{name_i}' and '{name_j}' contain "
                                "identical values."
                            ),
                            column=f"{name_i},{name_j}",
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


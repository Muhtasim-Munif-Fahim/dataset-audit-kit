"""Apply all changes to core.py."""

import re

SRC = r"C:\Users\Admin\AppData\Local\Temp\kilo\dataset-audit-kit\src\dataset_audit_kit\core.py"

with open(SRC, "r", encoding="utf-8") as f:
    content = f.read()


# ============================================================
# Change 1: Remove duplicate fix_suggestions / to_file
# ============================================================
lines = content.split("\n")

fix_suggestions_starts = []
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped == "@property" and i + 1 < len(lines):
        if "fix_suggestions" in lines[i + 1]:
            fix_suggestions_starts.append(i)

to_file_starts = []
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith("def to_file(") and "self" in stripped:
        to_file_starts.append(i)

if len(fix_suggestions_starts) >= 2 and len(to_file_starts) >= 2:
    dup_start = fix_suggestions_starts[1]
    dup_to_file = to_file_starts[1]
    end_idx = len(lines)
    for i in range(dup_to_file, len(lines)):
        stripped = lines[i].strip()
        if stripped == "return path" and i > dup_to_file:
            end_idx = i
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == "":
                    continue
                elif lines[j].strip().startswith("class DatasetAuditor"):
                    end_idx = j
                    break
                else:
                    end_idx = j
                    break
            break
    del lines[dup_start:end_idx]
    content = "\n".join(lines)
    print("  [OK] Removed duplicate fix_suggestions / to_file")
else:
    print(f"  [!!] Could not find duplicates. fs={fix_suggestions_starts} tf={to_file_starts}")


# ============================================================
# Change 2: Replace _profile_columns
# ============================================================
old_profile = """    @staticmethod
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

        return profiles"""

new_profile = """    @staticmethod
    def _profile_columns(data: pd.DataFrame) -> dict[str, dict[str, object]]:
        \"\"\"Build statistical profiles for all columns in the dataset.\"\"\"
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

        return profiles"""

cnt = content.count(old_profile)
if cnt == 0:
    print("  [!!] Could not find old _profile_columns")
elif cnt > 1:
    print(f"  [!!] Found {cnt} occurrences, replacing all")
    content = content.replace(old_profile, new_profile)
else:
    content = content.replace(old_profile, new_profile)
    print("  [OK] Replaced _profile_columns")


# ============================================================
# Change 3: IQR outlier block
# ============================================================
iqr_block = """            # --- IQR outlier detection ---
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

"""

anchor_allowed = "            # --- allowed values ---"
cnt = content.count(anchor_allowed)
if cnt == 0:
    print("  [!!] Could not find allowed values anchor")
elif cnt > 1:
    print(f"  [!!] Found {cnt} occurrences")
    content = content.replace(anchor_allowed, iqr_block + anchor_allowed)
else:
    content = content.replace(anchor_allowed, iqr_block + anchor_allowed)
    print("  [OK] Added IQR outlier block")


# ============================================================
# Change 4: Update to_markdown
# ============================================================
old_issues = """        if self.issues:
            lines.extend(["", "## Issues"])
            for issue in self.issues:
                parts = [f"- **{issue.severity.upper()}**", f"`{issue.check}`", issue.message]
                if issue.column:
                    parts.insert(2, f"column `{issue.column}`")
                lines.append(" ".join(parts))
        else:
            lines.extend(["", "_No issues found._"])"""

new_issues = """        if self.issues:
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
            lines.extend(["", "_No issues found._"])"""

cnt = content.count(old_issues)
if cnt == 0:
    print("  [!!] Could not find old issues block")
elif cnt > 1:
    print(f"  [!!] Found {cnt} occurrences, replacing all")
    content = content.replace(old_issues, new_issues)
else:
    content = content.replace(old_issues, new_issues)
    print("  [OK] Updated to_markdown")


# ============================================================
# Change 5a: schema_diff call in audit_dataframe
# ============================================================
old_drift = """        if reference is not None:
            drift_scores = self._check_drift(data, reference, issues, label_column=label_column)
            correlation_drift_scores = self._correlation_drift(
                data, reference, issues, drift_threshold=self.drift_threshold
            )

        # Per-column validation rules"""

new_drift = """        if reference is not None:
            drift_scores = self._check_drift(data, reference, issues, label_column=label_column)
            correlation_drift_scores = self._correlation_drift(
                data, reference, issues, drift_threshold=self.drift_threshold
            )

        # Schema diff between reference and current
        schema_diff_summary: dict[str, dict[str, object]] = {}
        if reference is not None:
            schema_diff_summary = self._schema_diff(data, reference, issues)

        # Per-column validation rules"""

cnt = content.count(old_drift)
if cnt == 0:
    print("  [!!] Could not find drift end")
elif cnt > 1:
    print(f"  [!!] Found {cnt} occurrences")
    content = content.replace(old_drift, new_drift)
else:
    content = content.replace(old_drift, new_drift)
    print("  [OK] Added schema_diff call")


# ============================================================
# Change 5b: redundancy call in audit_dataframe
# ============================================================
old_prof = """        # Per-column validation rules
        self._apply_rules(data, issues)

        column_profiles = self._profile_columns(data)

        all_drift_scores"""

new_prof = """        # Per-column validation rules
        self._apply_rules(data, issues)

        column_profiles = self._profile_columns(data)

        # Redundancy / collinearity check
        self._check_redundancy(data, issues, correlation_threshold=0.95)

        all_drift_scores"""

cnt = content.count(old_prof)
if cnt == 0:
    print("  [!!] Could not find profiles end")
elif cnt > 1:
    print(f"  [!!] Found {cnt} occurrences")
    content = content.replace(old_prof, new_prof)
else:
    content = content.replace(old_prof, new_prof)
    print("  [OK] Added redundancy call")


# ============================================================
# Change 6: Add _check_redundancy and _schema_diff methods
# ============================================================
new_methods = """
    @staticmethod
    def _check_redundancy(
        data: pd.DataFrame,
        issues: list[AuditIssue],
        *,
        correlation_threshold: float = 0.95,
    ) -> None:
        \"\"\"Detect highly correlated numeric column pairs (redundancy).\"\"\"
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
        \"\"\"Compare column schemas between current and reference datasets.

        Returns a dict mapping column names to a diff description:
        ``{\"status\": \"added\"|\"removed\"|\"dtype_changed\"|\"same\", \"details\": ...}``
        \"\"\"
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

        return diff"""

old_cat = """        divergence = 0.0
        for category in categories:
            divergence += abs(float(current_dist.get(category, 0.0)) - float(baseline_dist.get(category, 0.0)))
        return divergence / 2.0"""

cnt = content.count(old_cat)
if cnt == 0:
    print("  [!!] Could not find categorical drift end")
elif cnt > 1:
    print(f"  [!!] Found {cnt} occurrences")
    content = content.replace(old_cat, old_cat + new_methods)
else:
    content = content.replace(old_cat, old_cat + new_methods)
    print("  [OK] Added _check_redundancy and _schema_diff")


# Write back
with open(SRC, "w", encoding="utf-8") as f:
    f.write(content)

print("DONE - All changes applied!")

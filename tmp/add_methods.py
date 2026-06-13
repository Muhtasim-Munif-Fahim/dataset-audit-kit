import re

path = r'C:\Users\Admin\AppData\Local\Temp\kilo\dataset-audit-kit\src\dataset_audit_kit\core.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

diff_method = """
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
"""

content = content.replace('class DatasetAuditor:', diff_method + '\nclass DatasetAuditor:')

new_methods = """
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
"""

content = content.rstrip() + '\n' + new_methods + '\n'

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')
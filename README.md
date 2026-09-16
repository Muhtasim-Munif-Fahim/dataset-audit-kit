# Dataset Audit Kit

[![Tests](https://github.com/Muhtasim-Munif-Fahim/dataset-audit-kit/actions/workflows/tests.yml/badge.svg)](https://github.com/Muhtasim-Munif-Fahim/dataset-audit-kit/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/dataset-audit-kit.svg)](https://pypi.org/project/dataset-audit-kit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`dataset-audit-kit` is a small Python library and CLI for dataset validation. It checks schema drift, missing values, duplicates, label consistency, and basic distribution shifts before a dataset reaches training or production.

The goal is to make data quality checks boring, repeatable, and easy to run in a maintainer-friendly OSS workflow.

## Problem

Many ML failures start with the data, not the model:

- a column disappears after a source change
- missing values silently spike
- duplicate rows leak into training
- labels become imbalanced
- a new dataset shifts far away from the reference baseline

This toolkit gives you a lightweight audit layer before you launch a training job or publish a dataset update.

## Features

- Schema checks against expected columns.
- Missingness summary by column.
- Duplicate-row detection.
- Label balance and label completeness checks.
- Numeric and categorical drift checks against a reference dataset, including Population Stability Index (PSI).
- Kolmogorov-Smirnov significance test for numeric distribution drift.
- Opt-in numeric outlier / extreme-value detection (IQR or z-score).
- Configurable per-column validation rules with JSON-based rule files.
- CI-friendly `check` command that exits with code 1 on issues.
- CSV, JSONL/NDJSON, and Parquet dataset loading.
- JSON, Markdown, and HTML report output.
- CLI and notebook demo paths for documentation and review.

## Installation

```bash
pip install dataset-audit-kit

# Development install
pip install -r requirements.txt
```

## Quickstart

```python
import pandas as pd
from dataset_audit_kit import DatasetAuditor

auditor = DatasetAuditor(missing_threshold=0.05, drift_threshold=0.20)
report = auditor.audit_file(
    "train.parquet",
    reference_path="reference.jsonl",
    label_column="target",
    expected_columns=["feature_1", "feature_2", "target"],
)

print(report.to_markdown())
```

## CLI

```bash
dataset-audit-kit audit data.parquet \
  --reference reference.jsonl \
  --label-column target \
  --expected-columns feature_1,feature_2,target \
  --select-columns feature_1,feature_2,target
```

Use `--json` if you want machine-readable output for automation. In `--json`
and `--minimal` mode, notices such as "report saved to ..." go to stderr, so
stdout stays parseable: `dataset-audit-kit audit data.csv --json | jq .`.

Use `--html-out report.html` to export a shareable standalone HTML report.

### Outlier / extreme-value checks

Numeric columns can be scanned for extreme values during `audit` and `check`. The check is off by default so a long-tailed column does not fail a run unless you ask for it.

```bash
dataset-audit-kit audit data.csv --check-outliers
dataset-audit-kit audit data.csv --check-outliers --outlier-method zscore --outlier-threshold 3
dataset-audit-kit audit data.csv --check-outliers --max-outlier-ratio 0.05
```

IQR (Tukey fences at `Q1 - k×IQR` / `Q3 + k×IQR`, default `k=1.5`) is the default method. Switch to `--outlier-method zscore` for a mean/standard-deviation rule. `--max-outlier-ratio` is the share of values a column may have outside the fences before a warning is raised (default `0`, so any outlier flags).

Findings are one issue per column, with the count, ratio, fences or `|z|` cutoff, and the extreme min/max. JSON reports also include `outlier_summary`; Markdown and HTML reports add an **Outliers** table with per-column IQR detail even when the check is off.

In Python:

```python
from dataset_audit_kit import DatasetAuditor

auditor = DatasetAuditor(
    outlier_check=True,
    outlier_method="iqr",      # or "zscore"
    outlier_threshold=1.5,     # IQR multiplier, or z-score cutoff
    outlier_max_ratio=0.0,
)
report = auditor.audit_file("data.csv")
```

Per-column rules still support `max_outlier_ratio`, `percentile_fences`, and `max_zscore` when you want a contract on specific fields rather than a dataset-wide scan.

### Population Stability Index (PSI)

When `--reference` is supplied, the audit computes PSI for every shared column except the label. Numeric columns are split into equal-frequency quantile bins from the reference (values outside that range fold into the nearest edge bucket). Categorical columns compare category proportions over the union of observed levels.

Findings show up in every report format:

- JSON: `{column}__psi` inside `drift_scores`, plus `issues` entries with `check: "psi"`
- Markdown / HTML: the drift-score table and the issue list

```bash
dataset-audit-kit audit data.csv --reference baseline.csv
dataset-audit-kit audit data.csv --reference baseline.csv --psi-threshold 0.25
dataset-audit-kit audit data.csv --reference baseline.csv --psi-bins 20
```

Conventionally, PSI < 0.10 is a stable shift, 0.10–0.25 is moderate, and above 0.25 is large. The warning threshold defaults to `--drift-threshold` (0.20) unless `--psi-threshold` or a per-column `max_drift` rule is set.

In Python:

```python
from dataset_audit_kit import DatasetAuditor

auditor = DatasetAuditor(psi_threshold=0.25, psi_bins=10)
report = auditor.audit_file("data.csv", reference_path="baseline.csv")
print(report.drift_scores["revenue__psi"])
```

`--psi-threshold` and `--psi-bins` are part of the report `config_hash`, so two saved reports with different PSI settings will not compare as the same contract.

Every `audit` run is stamped with provenance metadata — an `audit_id`, the UTC generation time, and a `config_hash` covering every setting that changes findings (thresholds, sampling, schema expectations, rules file contents). The stamps appear in the JSON report under `meta`, in SARIF run properties (`auditId`, `createdUtc`, `configHash`), and as a footer line in HTML reports, so two saved reports with equal config hashes were produced under the same contract.

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | No errors or warnings. Informational findings (a new column, an outlier note) do not fail the run. |
| 1 | At least one warning or error was reported. |
| 2 | The command could not run: bad arguments, unreadable input, or an unwritable output path. |

Supported formats are `.csv`, `.jsonl`, `.ndjson`, and `.parquet`.

### Shape

```bash
# Default output
dataset-audit-kit shape data.csv
# 1000 rows x 10 columns

# CSV output for scripting
dataset-audit-kit shape data.csv --csv
# 1000,10
```

### CI check

```bash
dataset-audit-kit check data.csv --rules rules.json
```

Exits with code `0` if all checks pass, `1` if any issues are found. Use it in CI:

```yaml
- name: Validate dataset
  run: dataset-audit-kit check data.csv --rules rules.json
```

## Per-column Validation Rules

Define stronger expectations than global thresholds with a JSON rule file:

```json
{
  "age": {
    "dtype": "numeric",
    "min_value": 0,
    "max_value": 120,
    "max_missing_ratio": 0.05
  },
  "income": {
    "dtype": "numeric",
    "min_value": 0
  },
  "category": {
    "dtype": "categorical",
    "allowed_values": ["A", "B", "C"]
  }
}
```

Use it via the CLI:

```bash
dataset-audit-kit audit data.csv --rules rules.json
```

Or in Python:

```python
from dataset_audit_kit import DatasetAuditor, ValidationRules

rules = ValidationRules.from_json("rules.json")
auditor = DatasetAuditor(rules=rules)
report = auditor.audit_file("data.csv")
```

Rules are checked per-column for:
- **Data type** — `numeric`, `categorical`, or `string`
- **Numeric bounds** — `min_value` / `max_value`
- **Allowed values** — `allowed_values` for categorical columns
- **Missing ratio** — `max_missing_ratio` (overrides the global threshold per column)

### Lint a Rules File

Check a rules contract before pointing an audit at it — bad JSON, malformed rules, uncompilable patterns, invalid date formats, and unknown dtypes each get one actionable line:

```bash
dataset-audit-kit validate-config rules.json
# OK: 3 column rule(s), 0 cross-column rule(s).
```

`validate-config` exits `0` when the file is sound, `1` when it has findings, and `2` when the file cannot be read. Pass `--profile` to lint one named profile (see below).

### Named Profiles

One rules file can hold several reusable rule sets under a top-level `profiles` object. Pick one at run time with `--profile`:

```json
{
  "profiles": {
    "strict": {
      "age": {"dtype": "numeric", "min_value": 0, "max_value": 120}
    },
    "loose": {
      "age": {"dtype": "numeric"}
    }
  }
}
```

```bash
dataset-audit-kit audit data.csv --rules rules.json --profile strict
```

`audit`, `check`, and `audit-glob` all accept `--profile`. Running against a profiles file without `--profile` fails with a list of the available names, so a CI job never audits against the wrong contract by accident.

## Demo

The repository includes a fully self-contained demo based on the public Iris dataset.

- Script: [`examples/demo.py`](examples/demo.py)
- Notebook: [`examples/demo.ipynb`](examples/demo.ipynb)

![Dataset audit demo](assets/demo-screenshot.svg)

## How it compares

`dataset-audit-kit` is intentionally **lightweight** — a fast pre-flight check, not a full data platform.

| Capability | dataset-audit-kit | Pandera | Great Expectations |
| --- | --- | --- | --- |
| Install size / setup | Small, single CLI | Medium | Large, suite-oriented |
| Schema + dtype checks | Yes | Yes | Yes |
| Missingness / duplicates | Yes | Partial | Yes |
| Reference drift signals | Yes (basic) | No | Yes (richer) |
| CI `check` exit codes | Yes | Yes | Yes |
| Best for | Quick audits before training | Typed DataFrame pipelines | Enterprise data contracts |

Use this when you want a **maintainer-friendly OSS audit layer** before a training job or dataset release — not when you need a full observability platform.

## What It Reports

- Total rows and columns.
- Missing values per column.
- Duplicate rows.
- Label distribution.
- Drift score summaries for reference comparisons, including PSI (`{column}__psi`) and KS statistic/p-value.
- Numeric outlier summaries (IQR fences, counts, and ratios) plus opt-in outlier issues.
- A short issue list with severity, column, and explanation.

## Roadmap

- ~~Add HTML report export~~ ✅ v0.1.1
- ~~Add Parquet and JSONL loaders~~ ✅ v0.1.2
- ~~Add per-column validation rules~~ ✅ v0.2.0
- ~~Add CI check for auditable sample datasets~~ ✅ v0.2.0
- ~~Add columns subcommand~~ ✅ v0.3.0
- ~~Add head subcommand~~ ✅ v0.3.0
- ~~Add tail subcommand~~ ✅ v0.3.3
- ~~Add unique subcommand~~ ✅ v0.3.3
- ~~Add dtype subcommand~~ ✅ v0.3.3
- ~~Add correlate subcommand~~ ✅ v0.3.3
- ~~Add --csv flag to shape subcommand~~ ✅ v0.3.4
- ~~Add --select-columns flag to audit subcommand~~ ✅ v0.3.4

## Tests

```bash
python -m pytest -q
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for local setup and pull request guidance.

## License

MIT - see [LICENSE](LICENSE).

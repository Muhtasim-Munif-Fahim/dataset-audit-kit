# Contributing to dataset-audit-kit

Thanks for helping improve a lightweight dataset audit tool.

## Local setup

```bash
git clone https://github.com/Muhtasim-Munif-Fahim/dataset-audit-kit.git
cd dataset-audit-kit
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Pull requests

1. Open an issue for large changes when possible.
2. Add or update tests for behavior changes.
3. Keep the CLI backward compatible unless the major version bumps.
4. Run `python -m pytest -q` before pushing.

## Good first issues

Look for issues labeled `good first issue` — schema checks, report formatting, and loader edge cases are great entry points.

"""Shared fixtures for the test suite."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def clean_frame() -> pd.DataFrame:
    """A small frame that raises no warnings or errors."""

    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["ann", "bo", "cy", "di"],
            "label": ["yes", "no", "yes", "no"],
        }
    )


@pytest.fixture
def clean_csv(tmp_path, clean_frame) -> str:
    path = tmp_path / "clean.csv"
    clean_frame.to_csv(path, index=False)
    return str(path)

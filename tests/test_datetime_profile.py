"""Tests for datetime column profiling in _profile_columns."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from dataset_audit_kit.core import DatasetAuditor


def _datetime_column() -> pd.Series:
    return pd.Series(
        pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"])
    )


class TestDatetimeProfile:
    def test_datetime_column_is_profiled_with_datetime_stats(self) -> None:
        data = pd.DataFrame({"ts": _datetime_column()})
        profile = DatasetAuditor._profile_columns(data)["ts"]

        assert profile["dtype"] == "datetime"
        assert profile["missing"] == 0
        assert profile["min"] == "2020-01-01T00:00:00"
        assert profile["max"] == "2020-04-01T00:00:00"
        assert profile["mean"] == "2020-02-15T12:00:00"
        assert profile["std_seconds"] == pytest.approx(2917519.6, abs=1.0)
        assert profile["range_seconds"] == pytest.approx(7862400.0, abs=1.0)

    def test_empty_datetime_column_has_dtype_only(self) -> None:
        data = pd.DataFrame({"ts": pd.Series(pd.to_datetime(["NaT", "NaT"]))})
        profile = DatasetAuditor._profile_columns(data)["ts"]
        assert profile["dtype"] == "datetime"
        assert profile["count"] == 2
        assert profile["missing"] == 2
        assert "min" not in profile
        assert "mean" not in profile

    def test_tz_aware_datetime_column_is_profiled(self) -> None:
        data = pd.DataFrame(
            {
                "ts": pd.to_datetime(
                    ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"],
                    utc=True,
                )
            }
        )
        profile = DatasetAuditor._profile_columns(data)["ts"]
        assert profile["dtype"] == "datetime"
        assert profile["min"] == "2020-01-01T00:00:00+00:00"
        assert profile["mean"] == "2020-02-15T12:00:00+00:00"

    def test_datetime_profile_serializes_to_json(self) -> None:
        data = pd.DataFrame({"ts": _datetime_column(), "n": [1.0, 2.0, 3.0, 4.0]})
        report = DatasetAuditor().audit_dataframe(data)
        payload = json.loads(report.to_json())
        ts = payload["column_profiles"]["ts"]
        assert ts["dtype"] == "datetime"
        assert ts["min"] == "2020-01-01T00:00:00"
        assert isinstance(ts["range_seconds"], float)


class TestDatetimeProfileRendering:
    def test_markdown_renders_datetime_stats(self) -> None:
        data = pd.DataFrame({"ts": _datetime_column()})
        report = DatasetAuditor().audit_dataframe(data)
        text = report.to_markdown()
        assert "### ts (datetime)" in text
        assert "Mean: 2020-02-15T12:00:00" in text
        assert "Std:" in text and "seconds" in text
        assert "Span:" in text and "seconds" in text

    def test_html_renders_datetime_stats(self) -> None:
        data = pd.DataFrame({"ts": _datetime_column()})
        report = DatasetAuditor().audit_dataframe(data)
        html = report.to_html()
        assert "Std (seconds)" in html
        assert "Span (seconds)" in html
        assert "2020-02-15T12:00:00" in html

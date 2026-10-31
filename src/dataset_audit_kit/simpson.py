"""Simpson's paradox detection.

For a target ``Y`` (numeric or binary), a two-level *factor* ``X`` and a
categorical *stratifier* ``Z``, compare

* the aggregate effect ``mean(Y | X=b) - mean(Y | X=a)`` over all rows, with
* the stratum-adjusted effect: the weighted average of within-stratum
  differences, using Mantel-Haenszel-style weights
  ``w_z = n_za * n_zb / (n_za + n_zb)`` over strata where both factor levels
  have at least ``min_cell`` rows.

A *Simpson reversal* is flagged when the two effects have opposite signs
(each larger in magnitude than ``min_effect``). Such pairs mean the
headline comparison is confounded by ``Z`` and any analysis or model that
ignores ``Z`` will draw the wrong conclusion.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_SIMPSON_MIN_CELL = 5
DEFAULT_SIMPSON_MAX_LEVELS = 20
DEFAULT_SIMPSON_MIN_EFFECT = 0.0


def _numeric_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    levels = sorted(series.dropna().unique(), key=str)
    if len(levels) != 2:
        raise ValueError(
            f"target '{series.name}' must be numeric, boolean or have exactly two levels"
        )
    return (series == levels[1]).astype(float).where(series.notna())


def _two_level(series: pd.Series) -> list | None:
    levels = series.dropna().unique()
    if len(levels) != 2:
        return None
    try:
        return sorted(levels)
    except TypeError:
        return sorted(levels, key=str)


def simpsons_paradox_report(
    data: pd.DataFrame,
    target: str,
    *,
    factors: Sequence[str] | None = None,
    strata: Sequence[str] | None = None,
    min_cell: int = DEFAULT_SIMPSON_MIN_CELL,
    max_levels: int = DEFAULT_SIMPSON_MAX_LEVELS,
    min_effect: float = DEFAULT_SIMPSON_MIN_EFFECT,
) -> dict[str, object]:
    """Scan (factor, stratifier) pairs for Simpson's paradox reversals.

    ``factors`` defaults to every non-target column with exactly two
    non-missing levels; ``strata`` defaults to every non-target column with
    2..``max_levels`` levels. Returns a dict with ``target``, ``n_rows``,
    ``n_pairs`` scanned, ``n_reversals`` and ``pairs`` (reversals first,
    then by ``|aggregate - adjusted|`` descending). Each pair row holds
    ``factor``, ``level_a``, ``level_b``, ``stratum``, ``aggregate_effect``,
    ``adjusted_effect``, ``n_strata`` used, ``reversed_strata`` (strata whose
    own effect has the opposite sign of the aggregate), and ``reversal``.
    """
    if target not in data.columns:
        raise ValueError(f"target column '{target}' not found")
    if min_cell < 1:
        raise ValueError("min_cell must be >= 1")
    if max_levels < 2:
        raise ValueError("max_levels must be >= 2")
    if min_effect < 0:
        raise ValueError("min_effect must be >= 0")
    y = _numeric_target(data[target])
    others = [c for c in data.columns if c != target]
    if factors is None:
        factors = [c for c in others if _two_level(data[c]) is not None]
    if strata is None:
        strata = [c for c in others if 2 <= data[c].nunique(dropna=True) <= max_levels]
    for col in list(factors) + list(strata):
        if col not in data.columns:
            raise ValueError(f"column '{col}' not found")

    rows: list[dict[str, object]] = []
    for factor in factors:
        levels = _two_level(data[factor])
        if levels is None:
            raise ValueError(f"factor '{factor}' must have exactly two non-missing levels")
        a, b = levels
        for stratum in strata:
            if stratum == factor:
                continue
            frame = pd.DataFrame({"y": y, "x": data[factor], "z": data[stratum]}).dropna()
            ya, yb = frame.loc[frame.x == a, "y"], frame.loc[frame.x == b, "y"]
            if len(ya) < min_cell or len(yb) < min_cell:
                continue
            aggregate = float(yb.mean() - ya.mean())
            num = den = 0.0
            used = reversed_count = 0
            for _, g in frame.groupby("z", sort=False):
                ga, gb = g.loc[g.x == a, "y"], g.loc[g.x == b, "y"]
                if len(ga) < min_cell or len(gb) < min_cell:
                    continue
                diff = float(gb.mean() - ga.mean())
                w = len(ga) * len(gb) / (len(ga) + len(gb))
                num += w * diff
                den += w
                used += 1
                if diff * aggregate < 0:
                    reversed_count += 1
            if used == 0:
                continue
            adjusted = num / den
            reversal = (
                aggregate * adjusted < 0
                and abs(aggregate) > min_effect
                and abs(adjusted) > min_effect
            )
            rows.append(
                {
                    "factor": factor,
                    "level_a": a,
                    "level_b": b,
                    "stratum": stratum,
                    "aggregate_effect": aggregate,
                    "adjusted_effect": adjusted,
                    "n_strata": used,
                    "reversed_strata": reversed_count,
                    "reversal": bool(reversal),
                }
            )
    rows.sort(key=lambda r: (not r["reversal"], -abs(r["aggregate_effect"] - r["adjusted_effect"])))
    return {
        "target": target,
        "n_rows": int(len(data)),
        "n_pairs": len(rows),
        "n_reversals": sum(r["reversal"] for r in rows),
        "pairs": rows,
    }

"""Day-level feature engineering from wearable + hormone panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURES = (
    "resting_hr",
    "nightly_temperature",
)


def add_rolling_features(
    df: pd.DataFrame,
    cols: tuple[str, ...] = BASE_FEATURES,
    windows: tuple[int, ...] = (3, 7),
) -> pd.DataFrame:
    """Per-person rolling means / deltas for wearable channels."""
    out = df.sort_values(["id", "day_in_study"]).copy()
    for col in cols:
        if col not in out.columns:
            continue
        g = out.groupby("id", group_keys=False)[col]
        for w in windows:
            out[f"{col}_roll{w}"] = g.transform(
                lambda s, window=w: s.rolling(window, min_periods=1).mean()
            )
        out[f"{col}_delta1"] = g.diff()
        # Deviation from personal expanding median (robust personal baseline).
        out[f"{col}_vs_baseline"] = g.transform(
            lambda s: s - s.expanding(min_periods=7).median()
        )
    return out


def add_cycle_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that use within-cycle context without leaking future labels."""
    out = df.copy()
    if "day_in_cycle" in out.columns:
        out["day_in_cycle_sin"] = np.sin(2 * np.pi * out["day_in_cycle"] / 28.0)
        out["day_in_cycle_cos"] = np.cos(2 * np.pi * out["day_in_cycle"] / 28.0)
    return out


def feature_matrix(
    df: pd.DataFrame,
    *,
    include_hormones_as_features: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build model-ready feature frame.

    Hormone columns (lh, estrogen, pdg) are labels / targets by default and
    are excluded from features unless explicitly requested (for ablation only).
    """
    out = add_rolling_features(df)
    out = add_cycle_relative_features(out)

    feature_cols: list[str] = []
    for col in out.columns:
        if col in {
            "id",
            "day_in_study",
            "phase",
            "cycle_id",
            "day_in_cycle",
            "cycle_length",
            "cycle_pct",
            "study_interval",
            "lh_surge",
            "phase_label",
        }:
            continue
        if not include_hormones_as_features and col in {"lh", "estrogen", "pdg", "e3g"}:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            feature_cols.append(col)

    X = out[feature_cols].copy()
    return X, feature_cols

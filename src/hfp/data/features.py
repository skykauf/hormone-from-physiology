"""Day-level feature engineering from wearable + hormone panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURES = (
    "resting_hr",
    "nightly_temperature",
)

TEMP_FAMILY_PREFIXES = (
    "nightly_temperature",
    "wrist_temperature",
    "wrist_temp",
    "baseline_relative",
    "temp_",
)

RHR_FAMILY_PREFIXES = (
    "resting_hr",
    "sleep_score_rhr",
)

SLEEP_FAMILY_PREFIXES = (
    "sleep_",
    "minutes_asleep",
    "minutes_awake",
    "time_in_bed",
    "efficiency",
)

HRV_FAMILY_PREFIXES = ("hrv_",)

RR_FAMILY_PREFIXES = ("rr_",)

STRESS_FAMILY_PREFIXES = ("stress_",)

LEAKY_COLS = {
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
    "biphasic_label",
    "pdg_rise",
    "pdg_baseline",
    "day_in_cycle_sin",
    "day_in_cycle_cos",
    # Hormone assays / Mira metabolites.
    "lh",
    "estrogen",
    "pdg",
    "e3g",
    # Self-report symptoms that are not wearable physiology (keep out of default).
    "flow_volume",
    "flow_color",
    "appetite",
    "exerciselevel",
    "headaches",
    "cramps",
    "sorebreasts",
    "fatigue",
    "sleepissue",
    "moodswing",
    "stress",
    "foodcravings",
    "indigestion",
    "bloating",
}


def add_rolling_features(
    df: pd.DataFrame,
    cols: tuple[str, ...] | list[str] | None = None,
    windows: tuple[int, ...] = (3, 7),
) -> pd.DataFrame:
    """Per-person rolling means / deltas / z-scores / temp-nadir for channels."""
    out = df.sort_values(["id", "day_in_study"]).copy()
    if cols is None:
        cols = [c for c in BASE_FEATURES if c in out.columns]
        # Also roll key multimodal channels when present.
        for c in (
            "hrv_rmssd",
            "rr_full_sleep",
            "stress_score",
            "sleep_duration_min",
            "sleep_overall_score",
            "wrist_temperature",
        ):
            if c in out.columns:
                cols.append(c)

    for col in cols:
        if col not in out.columns:
            continue
        g = out.groupby("id", group_keys=False)[col]
        for w in windows:
            out[f"{col}_roll{w}"] = g.transform(
                lambda s, window=w: s.rolling(window, min_periods=1).mean()
            )
        out[f"{col}_delta1"] = g.diff()
        out[f"{col}_vs_baseline"] = g.transform(
            lambda s: s - s.expanding(min_periods=7).median()
        )
        # Expanding z-score (personal).
        def _z(s: pd.Series) -> pd.Series:
            mu = s.expanding(min_periods=7).mean()
            sd = s.expanding(min_periods=7).std().replace(0, np.nan)
            return (s - mu) / sd

        out[f"{col}_z"] = g.transform(_z)

    # BBT-style: temperature vs recent 6-day nadir.
    if "nightly_temperature" in out.columns:
        gtemp = out.groupby("id", group_keys=False)["nightly_temperature"]
        nadir = gtemp.transform(
            lambda s: s.rolling(6, min_periods=3).min().shift(1)
        )
        out["temp_vs_nadir"] = out["nightly_temperature"] - nadir
        out["temp_rise_from_nadir"] = out["temp_vs_nadir"].clip(lower=0)
    return out


def add_cycle_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that use within-cycle context (excluded from leak-safe matrix)."""
    out = df.copy()
    if "day_in_cycle" in out.columns:
        out["day_in_cycle_sin"] = np.sin(2 * np.pi * out["day_in_cycle"] / 28.0)
        out["day_in_cycle_cos"] = np.cos(2 * np.pi * out["day_in_cycle"] / 28.0)
    return out


def _matches_family(col: str, prefixes: tuple[str, ...]) -> bool:
    return any(col == p or col.startswith(p) for p in prefixes)


def select_feature_cols(
    columns: list[str],
    *,
    channel_set: str = "multimodal",
) -> list[str]:
    """
    Filter feature columns by ablation channel set.

    channel_set: temp_only | rhr_only | rhr_temp | multimodal
    """
    keep: list[str] = []
    for col in columns:
        if col in LEAKY_COLS or col in {"is_weekend"} and channel_set in {
            "temp_only",
            "rhr_only",
        }:
            # weekend allowed for rhr_temp/multimodal only below
            if col == "is_weekend":
                if channel_set in {"rhr_temp", "multimodal"}:
                    keep.append(col)
                continue
            continue

        if channel_set == "temp_only":
            if _matches_family(col, TEMP_FAMILY_PREFIXES) or col.startswith("temp_"):
                keep.append(col)
        elif channel_set == "rhr_only":
            if _matches_family(col, RHR_FAMILY_PREFIXES):
                keep.append(col)
        elif channel_set == "rhr_temp":
            if (
                _matches_family(col, TEMP_FAMILY_PREFIXES)
                or _matches_family(col, RHR_FAMILY_PREFIXES)
                or col.startswith("temp_")
                or col == "is_weekend"
            ):
                keep.append(col)
        else:  # multimodal — all non-leaky numeric candidates already filtered
            keep.append(col)
    # Deduplicate preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for c in keep:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def feature_matrix(
    df: pd.DataFrame,
    *,
    include_hormones_as_features: bool = False,
    channel_set: str = "multimodal",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build model-ready feature frame.

    Hormone columns are labels / targets by default. ``channel_set`` selects an
    ablation family (temp_only, rhr_only, rhr_temp, multimodal).
    """
    out = add_rolling_features(df)
    out = add_cycle_relative_features(out)

    candidates: list[str] = []
    for col in out.columns:
        if col in LEAKY_COLS:
            continue
        if not include_hormones_as_features and col in {
            "lh",
            "estrogen",
            "pdg",
            "e3g",
            "pdg_rise",
            "pdg_baseline",
        }:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            candidates.append(col)

    feature_cols = select_feature_cols(candidates, channel_set=channel_set)
    # For multimodal, prefer wearable families if any extras exist; else rhr_temp.
    if channel_set == "multimodal":
        extra = [
            c
            for c in feature_cols
            if _matches_family(c, SLEEP_FAMILY_PREFIXES)
            or _matches_family(c, HRV_FAMILY_PREFIXES)
            or _matches_family(c, RR_FAMILY_PREFIXES)
            or _matches_family(c, STRESS_FAMILY_PREFIXES)
            or _matches_family(c, ("wrist_",))
        ]
        if not extra:
            feature_cols = select_feature_cols(candidates, channel_set="rhr_temp")

    X = out[feature_cols].copy()
    return X, feature_cols

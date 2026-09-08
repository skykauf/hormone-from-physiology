"""Dataset loading helpers for mcPHASES (PhysioNet)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

# Core tables used by the baseline pipeline.
CORE_TABLES = (
    "hormones_and_selfreport.csv",
    "resting_heart_rate.csv",
    "computed_temperature.csv",
)

# Optional Fitbit tables that improve multimodal models (mid-size; skip raw HR/calories).
OPTIONAL_TABLES = (
    "sleep.csv",
    "sleep_score.csv",
    "heart_rate_variability_details.csv",
    "respiratory_rate_summary.csv",
    "stress_score.csv",
    "wrist_temperature.csv",
)

_META_COLS = {
    "id",
    "study_interval",
    "is_weekend",
    "day_in_study",
    "sleep_start_day_in_study",
    "sleep_end_day_in_study",
    "sleep_start_timestamp",
    "sleep_end_timestamp",
    "timestamp",
    "type",
    "date",
    "time",
}


def default_data_dir() -> Path:
    """Return repo-local ``data/raw`` if present, else ``MCPHASES_DATA_DIR`` env."""
    import os

    env = os.environ.get("MCPHASES_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "data" / "raw"


def require_tables(data_dir: Path, tables: Iterable[str] = CORE_TABLES) -> None:
    missing = [t for t in tables if not (data_dir / t).exists()]
    if missing:
        raise FileNotFoundError(
            "mcPHASES tables not found under "
            f"{data_dir}:\n  - "
            + "\n  - ".join(missing)
            + "\n\nDownload from https://physionet.org/content/mcphases/1.0.0/ "
            "(PhysioNet Restricted Health Data Use Agreement required) and place "
            "CSVs in data/raw/ or set MCPHASES_DATA_DIR."
        )


def available_optional_tables(data_dir: Path | None = None) -> list[str]:
    data_dir = data_dir or default_data_dir()
    return [t for t in OPTIONAL_TABLES if (data_dir / t).exists()]


def _resolve_day_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "day_in_study" not in out.columns:
        for cand in ("sleep_end_day_in_study", "sleep_start_day_in_study"):
            if cand in out.columns:
                out = out.rename(columns={cand: "day_in_study"})
                break
    out["day_in_study"] = pd.to_numeric(out["day_in_study"], errors="coerce")
    return out


def _day_agg(
    df: pd.DataFrame,
    value_cols: list[str],
    *,
    how: str = "median",
) -> pd.DataFrame:
    """Aggregate to one row per id/day for selected numeric columns."""
    if "id" not in df.columns or "day_in_study" not in df.columns:
        return pd.DataFrame(columns=["id", "day_in_study"])
    present = [c for c in value_cols if c in df.columns]
    if not present:
        # Fall back: all numeric non-meta columns.
        present = [
            c
            for c in df.columns
            if c not in _META_COLS and pd.api.types.is_numeric_dtype(df[c])
        ]
    if not present:
        return pd.DataFrame(columns=["id", "day_in_study"])
    g = df.groupby(["id", "day_in_study"], as_index=False)[present]
    return g.median() if how == "median" else g.mean()


def load_hormones(data_dir: Path | None = None) -> pd.DataFrame:
    """Load daily hormone + self-report table."""
    data_dir = data_dir or default_data_dir()
    require_tables(data_dir, ["hormones_and_selfreport.csv"])
    df = pd.read_csv(data_dir / "hormones_and_selfreport.csv")
    df["day_in_study"] = pd.to_numeric(df["day_in_study"], errors="coerce")
    return df


def load_resting_heart_rate(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or default_data_dir()
    require_tables(data_dir, ["resting_heart_rate.csv"])
    df = pd.read_csv(data_dir / "resting_heart_rate.csv")
    df = df[df["value"] > 0].copy()
    return (
        df.groupby(["id", "day_in_study"], as_index=False)["value"]
        .median()
        .rename(columns={"value": "resting_hr"})
    )


def load_nightly_temperature(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or default_data_dir()
    require_tables(data_dir, ["computed_temperature.csv"])
    df = pd.read_csv(data_dir / "computed_temperature.csv")
    df = _resolve_day_col(df)
    cols = [
        "nightly_temperature",
        "baseline_relative_nightly_standard_deviation",
        "baseline_relative_sample_standard_deviation",
    ]
    return _day_agg(df, cols)


def load_sleep(data_dir: Path | None = None) -> pd.DataFrame | None:
    data_dir = data_dir or default_data_dir()
    path = data_dir / "sleep.csv"
    if not path.exists():
        return None
    df = _resolve_day_col(pd.read_csv(path))
    # Prefer main sleep bouts when flagged.
    if "mainsleep" in df.columns:
        main = df[df["mainsleep"].astype(str).str.lower().isin(["true", "1", "yes"])]
        if len(main):
            df = main
    elif "type" in df.columns:
        main = df[df["type"].astype(str).str.lower().isin(["stages", "main", "sleep"])]
        if len(main):
            df = main
    # Normalize Fitbit-ish lowercase column names from mcPHASES.
    rename_in = {
        "minutesasleep": "minutes_asleep",
        "minutesawake": "minutes_awake",
        "timeinbed": "time_in_bed",
        "minutestofallasleep": "minutes_to_fall_asleep",
        "minutesafterwakeup": "minutes_after_wakeup",
    }
    df = df.rename(columns={k: v for k, v in rename_in.items() if k in df.columns})
    preferred = [
        "duration",
        "minutes_asleep",
        "minutes_awake",
        "time_in_bed",
        "efficiency",
        "minutes_to_fall_asleep",
        "minutes_after_wakeup",
    ]
    out = _day_agg(df, preferred)
    if "duration" in out.columns and out["duration"].median(skipna=True) > 10_000:
        out["sleep_duration_min"] = out["duration"] / 60000.0
        out = out.drop(columns=["duration"])
    elif "duration" in out.columns:
        out = out.rename(columns={"duration": "sleep_duration_min"})
    if "minutes_asleep" in out.columns and "sleep_duration_min" not in out.columns:
        out["sleep_duration_min"] = out["minutes_asleep"]
    if "efficiency" in out.columns:
        out = out.rename(columns={"efficiency": "sleep_efficiency"})
    return out


def load_sleep_score(data_dir: Path | None = None) -> pd.DataFrame | None:
    data_dir = data_dir or default_data_dir()
    path = data_dir / "sleep_score.csv"
    if not path.exists():
        return None
    df = _resolve_day_col(pd.read_csv(path))
    preferred = [
        "overall_score",
        "composition_score",
        "revitalization_score",
        "duration_score",
        "deep_sleep_in_minutes",
        "resting_heart_rate",
        "restlessness",
    ]
    out = _day_agg(df, preferred)
    return out.rename(
        columns={
            "overall_score": "sleep_overall_score",
            "composition_score": "sleep_composition_score",
            "revitalization_score": "sleep_revitalization_score",
            "duration_score": "sleep_duration_score",
            "deep_sleep_in_minutes": "sleep_deep_min_score",
            "resting_heart_rate": "sleep_score_rhr",
            "restlessness": "sleep_restlessness",
        }
    )


def load_hrv(data_dir: Path | None = None) -> pd.DataFrame | None:
    data_dir = data_dir or default_data_dir()
    path = data_dir / "heart_rate_variability_details.csv"
    if not path.exists():
        return None
    df = _resolve_day_col(pd.read_csv(path))
    preferred = [
        "rmssd",
        "coverage",
        "hf",
        "high_frequency",
        "low_frequency",
        "nremhr",
        "entropy",
    ]
    out = _day_agg(df, preferred)
    if "high_frequency" in out.columns and "hf" not in out.columns:
        out = out.rename(columns={"high_frequency": "hf"})
    return out.rename(
        columns={
            "rmssd": "hrv_rmssd",
            "coverage": "hrv_coverage",
            "hf": "hrv_hf",
            "low_frequency": "hrv_lf",
            "nremhr": "hrv_nremhr",
            "entropy": "hrv_entropy",
        }
    )


def load_respiratory_rate(data_dir: Path | None = None) -> pd.DataFrame | None:
    data_dir = data_dir or default_data_dir()
    path = data_dir / "respiratory_rate_summary.csv"
    if not path.exists():
        return None
    df = _resolve_day_col(pd.read_csv(path))
    preferred = [
        "full_sleep_breathing_rate",
        "deep_sleep_breathing_rate",
        "light_sleep_breathing_rate",
        "rem_sleep_breathing_rate",
        "value",
    ]
    out = _day_agg(df, preferred)
    if "value" in out.columns and "full_sleep_breathing_rate" not in out.columns:
        out = out.rename(columns={"value": "full_sleep_breathing_rate"})
    return out.rename(
        columns={
            "full_sleep_breathing_rate": "rr_full_sleep",
            "deep_sleep_breathing_rate": "rr_deep",
            "light_sleep_breathing_rate": "rr_light",
            "rem_sleep_breathing_rate": "rr_rem",
        }
    )


def load_stress_score(data_dir: Path | None = None) -> pd.DataFrame | None:
    data_dir = data_dir or default_data_dir()
    path = data_dir / "stress_score.csv"
    if not path.exists():
        return None
    df = _resolve_day_col(pd.read_csv(path))
    preferred = [
        "stress_score",
        "sleep_points",
        "responsiveness_points",
        "exertion_points",
    ]
    out = _day_agg(df, preferred)
    return out.rename(
        columns={
            "sleep_points": "stress_sleep_points",
            "responsiveness_points": "stress_responsiveness_points",
            "exertion_points": "stress_exertion_points",
        }
    )


def load_wrist_temperature(data_dir: Path | None = None) -> pd.DataFrame | None:
    data_dir = data_dir or default_data_dir()
    path = data_dir / "wrist_temperature.csv"
    if not path.exists():
        return None
    df = _resolve_day_col(pd.read_csv(path))
    preferred = [
        "temperature",
        "nightly_temperature",
        "relative_temperature",
        "value",
    ]
    out = _day_agg(df, preferred)
    if "nightly_temperature" in out.columns:
        out = out.rename(columns={"nightly_temperature": "wrist_temperature"})
    elif "temperature" in out.columns:
        out = out.rename(columns={"temperature": "wrist_temperature"})
    elif "relative_temperature" in out.columns:
        out = out.rename(columns={"relative_temperature": "wrist_temp_relative"})
    elif "value" in out.columns:
        out = out.rename(columns={"value": "wrist_temperature"})
    return out


def load_daily_panel(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Merge hormones with day-level wearable summaries.

    Always includes RHR + computed nightly temperature. Optionally merges sleep,
    sleep score, HRV, respiratory rate, stress score, and wrist temperature when
    those CSVs are present under ``data_dir``.
    """
    data_dir = data_dir or default_data_dir()
    hormones = load_hormones(data_dir)
    panel = hormones.merge(load_resting_heart_rate(data_dir), on=["id", "day_in_study"], how="left")
    panel = panel.merge(load_nightly_temperature(data_dir), on=["id", "day_in_study"], how="left")

    for loader in (
        load_sleep,
        load_sleep_score,
        load_hrv,
        load_respiratory_rate,
        load_stress_score,
        load_wrist_temperature,
    ):
        extra = loader(data_dir)
        if extra is not None and len(extra):
            panel = panel.merge(extra, on=["id", "day_in_study"], how="left")

    return panel.sort_values(["id", "day_in_study"]).reset_index(drop=True)

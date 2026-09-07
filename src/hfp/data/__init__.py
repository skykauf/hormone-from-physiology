"""Dataset loading helpers for mcPHASES (PhysioNet)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

# Core tables used by the baseline pipeline.
CORE_TABLES = (
    "hormones_and_selfreport.csv",
    "resting_heart_rate.csv",
    "computed_temperature.csv",
)

# Optional Fitbit / CGM tables that improve multimodal models.
OPTIONAL_TABLES = (
    "heart_rate.csv",
    "hrv.csv",
    "sleep.csv",
    "respiratory_rate.csv",
    "spo2.csv",
    "stress_score.csv",
    "glucose.csv",
)


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
    if "sleep_end_day_in_study" in df.columns and "day_in_study" not in df.columns:
        df = df.rename(columns={"sleep_end_day_in_study": "day_in_study"})
    return (
        df.groupby(["id", "day_in_study"], as_index=False)["nightly_temperature"]
        .median()
    )


def load_daily_panel(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Merge hormones with day-level wearable summaries (RHR + nightly temperature).

    This is the primary table for baseline phase / surge models.
    """
    data_dir = data_dir or default_data_dir()
    hormones = load_hormones(data_dir)
    rhr = load_resting_heart_rate(data_dir)
    temp = load_nightly_temperature(data_dir)
    panel = hormones.merge(rhr, on=["id", "day_in_study"], how="left")
    panel = panel.merge(temp, on=["id", "day_in_study"], how="left")
    return panel.sort_values(["id", "day_in_study"]).reset_index(drop=True)

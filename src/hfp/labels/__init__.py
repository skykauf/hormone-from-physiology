"""Hormone-derived labels (ground truth from Mira urinalysis)."""

from __future__ import annotations

import numpy as np
import pandas as pd

PHASE_ORDER = ("Menstrual", "Follicular", "Fertility", "Luteal")
PHASE_TO_INT = {p: i for i, p in enumerate(PHASE_ORDER)}

# Biphasic: pre-luteal (0) vs luteal (1). Fertility excluded (ambiguous transition).
BIPHASIC_MAP = {
    "Menstrual": 0,
    "Follicular": 0,
    "Luteal": 1,
}


def encode_phase(series: pd.Series) -> pd.Series:
    """Map mcPHASES phase strings to integers; unknown → NA."""
    return series.map(PHASE_TO_INT).astype("Int64")


def encode_biphasic(series: pd.Series) -> pd.Series:
    """Map to pre-luteal (0) vs luteal (1); Fertility / unknown → NA."""
    return series.map(BIPHASIC_MAP).astype("Int64")


def label_lh_surge(
    df: pd.DataFrame,
    *,
    lh_col: str = "lh",
    threshold_quantile: float = 0.90,
    min_absolute: float = 15.0,
) -> pd.DataFrame:
    """
    Binary LH-surge label per day.

    Uses a per-person high quantile of LH, floored at ``min_absolute`` mIU/mL.
    """
    out = df.copy()
    thresholds = (
        out.groupby("id")[lh_col]
        .transform(
            lambda s: max(float(np.nanquantile(s.dropna(), threshold_quantile)), min_absolute)
        )
        if out[lh_col].notna().any()
        else pd.Series(np.nan, index=out.index)
    )
    out["lh_surge"] = ((out[lh_col] >= thresholds) & out[lh_col].notna()).astype(int)
    return out


def label_pdg_rise(
    df: pd.DataFrame,
    *,
    pdg_col: str = "pdg",
    rise_factor: float = 2.0,
    window: int = 5,
) -> pd.DataFrame:
    """Mark luteal PdG confirmation days vs early-cycle baseline."""
    out = df.sort_values(["id", "day_in_study"]).copy()
    if "cycle_id" in out.columns and "day_in_cycle" in out.columns:
        early = out[out["day_in_cycle"] <= window]
        baseline = early.groupby(["id", "cycle_id"])[pdg_col].median()
        out = out.merge(
            baseline.rename("pdg_baseline").reset_index(),
            on=["id", "cycle_id"],
            how="left",
        )
    else:
        out["pdg_baseline"] = out.groupby("id")[pdg_col].transform(
            lambda s: s.expanding(min_periods=3).median()
        )
    out["pdg_rise"] = (
        (out[pdg_col] >= rise_factor * out["pdg_baseline"]) & out[pdg_col].notna()
    ).astype(int)
    return out

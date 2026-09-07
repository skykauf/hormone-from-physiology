"""Cycle segmentation helpers."""

from __future__ import annotations

import pandas as pd


def assign_menstrual_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Number cycles per participant using menstrual-phase onset.

    Adds columns:
      - ``cycle_id``: integer cycle index within participant (starts at 1)
      - ``day_in_cycle``: 1-indexed day within cycle
      - ``cycle_length``: length of that cycle in days
      - ``cycle_pct``: percent through cycle (0–100)
    """
    out = df.sort_values(["id", "day_in_study"]).copy()
    phase = out["phase"]
    new_cycle = (phase == "Menstrual") & (phase.shift(1) != "Menstrual")
    # First row of each person: if already Menstrual, start a cycle.
    first = ~out["id"].eq(out["id"].shift(1))
    new_cycle = new_cycle | (first & phase.eq("Menstrual"))
    out["cycle_id"] = new_cycle.groupby(out["id"]).cumsum().astype("Int64")
    out.loc[out["cycle_id"] == 0, "cycle_id"] = pd.NA

    grouped = out.groupby(["id", "cycle_id"], dropna=True)
    out["day_in_cycle"] = grouped.cumcount() + 1
    out["cycle_length"] = grouped["day_in_cycle"].transform("max")
    out["cycle_pct"] = (out["day_in_cycle"] / out["cycle_length"]) * 100.0
    return out


def complete_cycles(
    df: pd.DataFrame,
    *,
    phases: tuple[str, ...] = ("Menstrual", "Follicular", "Fertility", "Luteal"),
    min_length: int = 21,
    max_length: int = 45,
) -> pd.DataFrame:
    """Keep only cycles that contain all expected phases and plausible length."""
    if "cycle_id" not in df.columns:
        df = assign_menstrual_cycles(df)

    def _ok(g: pd.DataFrame) -> bool:
        if g["cycle_length"].iloc[0] < min_length or g["cycle_length"].iloc[0] > max_length:
            return False
        present = set(g["phase"].dropna().unique())
        return set(phases).issubset(present)

    keep = (
        df.dropna(subset=["cycle_id"])
        .groupby(["id", "cycle_id"], group_keys=False)
        .filter(_ok)
    )
    return keep.reset_index(drop=True)

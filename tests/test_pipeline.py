"""Synthetic panel fixtures for unit tests (no PhysioNet data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hfp.data.cycles import assign_menstrual_cycles
from hfp.data.features import feature_matrix
from hfp.labels import encode_phase, label_lh_surge, label_pdg_rise
from hfp.models import train_phase_classifier


def _fake_person(pid: str, start: int = 1, n_cycles: int = 2) -> pd.DataFrame:
    rows = []
    day = start
    phases = (
        ["Menstrual"] * 5
        + ["Follicular"] * 7
        + ["Fertility"] * 4
        + ["Luteal"] * 12
    )
    rng = np.random.default_rng(abs(hash(pid)) % (2**32))
    for _ in range(n_cycles):
        fertility_day = 0
        for phase in phases:
            # Luteal: higher temp, higher HR, higher PdG; early fertility: LH spike
            temp = 34.2 + (0.35 if phase == "Luteal" else 0.0) + rng.normal(0, 0.05)
            rhr = 62 + (3.5 if phase == "Luteal" else 0.0) + rng.normal(0, 0.8)
            if phase == "Fertility":
                lh = 45.0 if fertility_day == 0 else 12.0
                fertility_day += 1
            else:
                lh = 8.0 + rng.normal(0, 0.5)
                fertility_day = 0
            e3g = 20 + (30 if phase in {"Follicular", "Fertility"} else 5) + rng.normal(0, 2)
            pdg = 2 + (8 if phase == "Luteal" else 0) + rng.normal(0, 0.3)
            rows.append(
                {
                    "id": pid,
                    "day_in_study": day,
                    "phase": phase,
                    "lh": lh,
                    "estrogen": e3g,
                    "pdg": pdg,
                    "resting_hr": rhr,
                    "nightly_temperature": temp,
                    "study_interval": 2024,
                }
            )
            day += 1
    return pd.DataFrame(rows)


def test_cycle_assignment_and_labels():
    df = pd.concat([_fake_person("a"), _fake_person("b")], ignore_index=True)
    df = assign_menstrual_cycles(df)
    assert df["cycle_id"].notna().all()
    assert df["day_in_cycle"].min() == 1

    df = label_lh_surge(df)
    df = label_pdg_rise(df)
    assert df["lh_surge"].sum() > 0
    assert df["pdg_rise"].sum() > 0
    assert set(encode_phase(df["phase"]).dropna().unique()).issubset({0, 1, 2, 3})


def test_biphasic_and_ablation_channel_sets():
    from hfp.labels import encode_biphasic
    from hfp.models import train_biphasic_classifier

    df = pd.concat(
        [_fake_person("p1"), _fake_person("p2"), _fake_person("p3")],
        ignore_index=True,
    )
    df = assign_menstrual_cycles(df)
    assert encode_biphasic(df["phase"]).dropna().isin([0, 1]).all()
    X, cols = feature_matrix(df, channel_set="rhr_temp")
    panel = df.copy()
    for c in cols:
        panel[c] = X[c]
    result = train_biphasic_classifier(panel, cols, fit_final=False, max_iter=50)
    assert "accuracy" in result.metrics
    assert result.metrics["accuracy"] > 0.5

    for cs in ("temp_only", "rhr_only", "rhr_temp", "multimodal"):
        _, c = feature_matrix(df, channel_set=cs)
        assert len(c) > 0


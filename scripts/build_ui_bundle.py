"""Build a UI-safe aggregate JSON bundle (no raw participant rows)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from hfp.data import load_daily_panel
from hfp.data.cycles import assign_menstrual_cycles, complete_cycles
from hfp.data.features import feature_matrix
from hfp.eval import per_subject_accuracy, phase_metrics
from hfp.labels import PHASE_ORDER, label_lh_surge, label_pdg_rise
from hfp.models import train_lh_surge_detector, train_phase_classifier


def _anon_map(ids: pd.Series) -> dict[str, str]:
    unique = sorted(ids.dropna().astype(str).unique())
    return {uid: f"P{i:02d}" for i, uid in enumerate(unique, start=1)}


def _phase_physiology(panel: pd.DataFrame) -> dict:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for signal in ("resting_hr", "nightly_temperature"):
        if signal not in panel.columns:
            continue
        out[signal] = {}
        for phase in PHASE_ORDER:
            vals = panel.loc[panel["phase"] == phase, signal].dropna()
            if len(vals) == 0:
                continue
            out[signal][phase] = {
                "n": int(len(vals)),
                "median": float(vals.median()),
                "q25": float(vals.quantile(0.25)),
                "q75": float(vals.quantile(0.75)),
                "mean": float(vals.mean()),
            }
    return out


def _hormone_curves(panel: pd.DataFrame, grid: np.ndarray | None = None) -> dict:
    """Population mean hormone trajectories on a normalized cycle % grid."""
    grid = grid if grid is not None else np.linspace(0, 100, 50)
    df = panel.dropna(subset=["cycle_pct", "cycle_id"]).copy()
    curves: dict[str, list[dict[str, float]]] = {}
    for col, key in (("lh", "lh"), ("estrogen", "e3g"), ("pdg", "pdg")):
        points = []
        for x in grid:
            # nearest-bin mean within ±1%
            mask = (df["cycle_pct"] - x).abs() <= 1.5
            vals = df.loc[mask, col].dropna()
            if len(vals) == 0:
                continue
            points.append(
                {
                    "cycle_pct": float(x),
                    "mean": float(vals.mean()),
                    "sem": float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                    "n": int(len(vals)),
                }
            )
        curves[key] = points
    return curves


def build_bundle(data_dir: Path | None = None) -> dict:
    panel = load_daily_panel(data_dir)
    panel = assign_menstrual_cycles(panel)
    panel = complete_cycles(panel)
    panel = label_lh_surge(panel)
    panel = label_pdg_rise(panel)

    X, feature_cols = feature_matrix(panel)
    panel = panel.loc[X.index].copy()
    for c in feature_cols:
        panel[c] = X[c]

    anon = _anon_map(panel["id"])
    phase_result = train_phase_classifier(panel, feature_cols)
    lh_result = train_lh_surge_detector(panel, feature_cols)

    phase_m = phase_metrics(phase_result.y_true, phase_result.y_pred)
    subject_df = per_subject_accuracy(phase_result.y_true, phase_result.y_pred, phase_result.groups)
    subject_df["id"] = subject_df["id"].astype(str).map(anon)

    n_subjects = panel["id"].nunique()
    n_cycles = panel.dropna(subset=["cycle_id"]).groupby(["id", "cycle_id"]).ngroups
    n_days = len(panel)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "mcPHASES",
            "version": "1.0.0",
            "citation_doi": "10.13026/zx6a-2c81",
            "n_subjects": int(n_subjects),
            "n_complete_cycles": int(n_cycles),
            "n_days": int(n_days),
            "features": feature_cols,
            "note": "Aggregates only — raw PhysioNet rows are not included.",
        },
        "phase_model": {
            "evaluation": "leave-one-subject-out",
            "accuracy": phase_m["accuracy"],
            "macro_f1": phase_m["macro_f1"],
            "phase_names": phase_m["phase_names"],
            "confusion_matrix": phase_m["confusion_matrix"],
            "per_subject": subject_df.to_dict(orient="records"),
        },
        "lh_surge_model": {
            "evaluation": "leave-one-subject-out",
            "accuracy": float(lh_result.metrics["accuracy"]),
            "macro_f1": float(lh_result.metrics["macro_f1"]),
            "train_auroc_final_model": lh_result.metrics.get("train_auroc_final_model"),
        },
        "physiology_by_phase": _phase_physiology(panel),
        "hormone_curves": _hormone_curves(panel),
        "phase_counts": {
            str(k): int(v) for k, v in panel["phase"].value_counts().to_dict().items()
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("artifacts/ui_bundle.json"))
    args = parser.parse_args(argv)

    bundle = build_bundle(args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=2))
    print(f"Wrote {args.out}")
    print(
        f"Phase LOSO accuracy: {bundle['phase_model']['accuracy']:.3f} "
        f"({bundle['dataset']['n_subjects']} subjects, "
        f"{bundle['dataset']['n_complete_cycles']} cycles)"
    )


if __name__ == "__main__":
    main()

"""Build a UI-safe aggregate JSON bundle (no raw participant rows)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hfp.data import available_optional_tables, load_daily_panel
from hfp.data.cycles import assign_menstrual_cycles, complete_cycles
from hfp.labels import PHASE_ORDER, label_lh_surge, label_pdg_rise
from run_ablations import run_ablations


def _anon_map(ids: list | pd.Series) -> dict[str, str]:
    unique = sorted({str(x) for x in ids if pd.notna(x)})
    return {uid: f"P{i:02d}" for i, uid in enumerate(unique, start=1)}


def _phase_physiology(panel: pd.DataFrame) -> dict:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for signal in ("resting_hr", "nightly_temperature", "hrv_rmssd", "stress_score"):
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
    grid = grid if grid is not None else np.linspace(0, 100, 50)
    df = panel.dropna(subset=["cycle_pct", "cycle_id"]).copy()
    curves: dict[str, list[dict[str, float]]] = {}
    for col, key in (("lh", "lh"), ("estrogen", "e3g"), ("pdg", "pdg")):
        points = []
        for x in grid:
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

    ablations = run_ablations(data_dir)
    primary_key = "multimodal" if "multimodal" in ablations["configs"] else "rhr_temp"
    primary = ablations["configs"][primary_key]
    feature_cols = primary["features"]

    anon = _anon_map(
        [row["id"] for row in (primary["phase"].get("per_subject") or [])]
    )
    per_subject = [
        {**row, "id": anon.get(str(row["id"]), str(row["id"]))}
        for row in (primary["phase"].get("per_subject") or [])
    ]

    # Keep ablation JSON lean for the public UI.
    lean_ablations = json.loads(json.dumps(ablations))
    for cfg in lean_ablations.get("configs", {}).values():
        if isinstance(cfg.get("phase"), dict):
            cfg["phase"].pop("per_subject", None)

    n_subjects = panel["id"].nunique()
    n_cycles = panel.dropna(subset=["cycle_id"]).groupby(["id", "cycle_id"]).ngroups

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "mcPHASES",
            "version": "1.0.0",
            "citation_doi": "10.13026/zx6a-2c81",
            "n_subjects": int(n_subjects),
            "n_complete_cycles": int(n_cycles),
            "n_days": int(len(panel)),
            "features": feature_cols,
            "channel_set": primary_key,
            "optional_tables_present": available_optional_tables(data_dir),
            "note": "Aggregates only — raw PhysioNet rows are not included.",
        },
        "phase_model": {
            "evaluation": "leave-one-subject-out + majority smooth (w=3)",
            "accuracy": primary["phase"]["accuracy"],
            "balanced_accuracy": primary["phase"].get("balanced_accuracy"),
            "macro_f1": primary["phase"]["macro_f1"],
            "per_class_recall": primary["phase"].get("per_class_recall"),
            "phase_names": primary["phase"].get("phase_names") or list(PHASE_ORDER),
            "confusion_matrix": primary["phase"].get("confusion_matrix"),
            "per_subject": per_subject,
        },
        "biphasic_model": {
            "evaluation": "leave-one-subject-out + majority smooth (w=3)",
            "definition": "pre-luteal (Menstrual+Follicular) vs Luteal; Fertility excluded",
            "accuracy": primary["biphasic"].get("accuracy"),
            "balanced_accuracy": primary["biphasic"].get("balanced_accuracy"),
            "macro_f1": primary["biphasic"].get("macro_f1"),
            "pr_auc": primary["biphasic"].get("pr_auc"),
        },
        "lh_surge_model": {
            "evaluation": "leave-one-subject-out",
            "accuracy": primary.get("lh_surge", {}).get("accuracy"),
            "balanced_accuracy": primary.get("lh_surge", {}).get("balanced_accuracy"),
            "macro_f1": primary.get("lh_surge", {}).get("macro_f1"),
            "pr_auc": primary.get("lh_surge", {}).get("pr_auc"),
            "positive_rate": primary.get("lh_surge", {}).get("positive_rate"),
            "sensitivity_at_fpr": primary.get("lh_surge", {}).get("sensitivity_at_fpr"),
        },
        "ablations": lean_ablations,
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
    bal = bundle["phase_model"].get("balanced_accuracy") or 0.0
    print(
        f"Phase LOSO bal_acc: {bal:.3f} "
        f"macro_f1: {bundle['phase_model']['macro_f1']:.3f} | "
        f"biphasic acc: {bundle['biphasic_model']['accuracy']:.3f} "
        f"({bundle['dataset']['n_subjects']} subjects, "
        f"{bundle['dataset']['n_complete_cycles']} cycles)"
    )


if __name__ == "__main__":
    main()

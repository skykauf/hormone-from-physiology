"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from hfp.data import load_daily_panel
from hfp.data.cycles import assign_menstrual_cycles, complete_cycles
from hfp.data.features import feature_matrix
from hfp.eval import per_subject_accuracy, phase_metrics
from hfp.labels import label_lh_surge, label_pdg_rise
from hfp.models import train_lh_surge_detector, train_phase_classifier


def _prepare_panel(data_dir: Path | None):
    panel = load_daily_panel(data_dir)
    panel = assign_menstrual_cycles(panel)
    panel = complete_cycles(panel)
    panel = label_lh_surge(panel)
    panel = label_pdg_rise(panel)
    X, feature_cols = feature_matrix(panel)
    panel = panel.loc[X.index].copy()
    for c in feature_cols:
        panel[c] = X[c]
    return panel, feature_cols


def train_phase_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train LOSO phase classifier on mcPHASES")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("artifacts/phase_model.joblib"))
    args = parser.parse_args(argv)

    panel, feature_cols = _prepare_panel(args.data_dir)
    result = train_phase_classifier(panel, feature_cols)
    metrics = phase_metrics(result.y_true, result.y_pred)
    metrics["loso_accuracy"] = result.metrics["accuracy"]
    metrics["loso_macro_f1"] = result.metrics["macro_f1"]
    subject_df = per_subject_accuracy(result.y_true, result.y_pred, result.groups)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": result.model, "feature_cols": feature_cols, "metrics": metrics},
        args.out,
    )
    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    subject_df.to_csv(args.out.with_suffix(".per_subject.csv"), index=False)

    print(result.metrics["report"])
    print(f"LOSO accuracy: {metrics['loso_accuracy']:.3f}")
    print(f"Wrote {args.out} and {metrics_path}")


def eval_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train LOSO LH-surge detector on mcPHASES")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("artifacts/lh_surge_model.joblib"))
    args = parser.parse_args(argv)

    panel, feature_cols = _prepare_panel(args.data_dir)
    result = train_lh_surge_detector(panel, feature_cols)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": result.model, "feature_cols": feature_cols, "metrics": result.metrics}
    joblib.dump(payload, args.out)
    print(result.metrics.get("report", result.metrics))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    train_phase_main()

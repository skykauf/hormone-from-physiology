"""Run channel ablations (temp / RHR / RHR+temp / multimodal) with LOSO metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hfp.data import available_optional_tables, load_daily_panel
from hfp.data.cycles import assign_menstrual_cycles, complete_cycles
from hfp.data.features import feature_matrix
from hfp.eval import per_subject_accuracy
from hfp.labels import label_lh_surge, label_pdg_rise
from hfp.models import (
    train_biphasic_classifier,
    train_lh_surge_detector,
    train_phase_classifier,
)

CHANNEL_SETS = ("temp_only", "rhr_only", "rhr_temp", "multimodal")


def _prepare(data_dir: Path | None):
    panel = load_daily_panel(data_dir)
    panel = assign_menstrual_cycles(panel)
    panel = complete_cycles(panel)
    panel = label_lh_surge(panel)
    panel = label_pdg_rise(panel)
    return panel


def run_ablations(data_dir: Path | None = None) -> dict:
    panel = _prepare(data_dir)
    optional = available_optional_tables(data_dir)
    results: dict = {
        "optional_tables_present": optional,
        "configs": {},
    }

    for channel_set in CHANNEL_SETS:
        X, feature_cols = feature_matrix(panel, channel_set=channel_set)
        if not feature_cols:
            results["configs"][channel_set] = {"error": "no features"}
            continue
        p = panel.loc[X.index].copy()
        for c in feature_cols:
            p[c] = X[c]

        phase = train_phase_classifier(
            p, feature_cols, smooth_window=3, fit_final=False, max_iter=100
        )
        biphasic = train_biphasic_classifier(
            p, feature_cols, smooth_window=3, fit_final=False, max_iter=100
        )
        # LH surge only on the primary multimodal/rhr_temp configs (expensive + imbalanced).
        lh_payload: dict
        if channel_set in {"rhr_temp", "multimodal"}:
            lh = train_lh_surge_detector(p, feature_cols, fit_final=False, max_iter=100)
            lh_payload = {
                "accuracy": lh.metrics.get("accuracy"),
                "balanced_accuracy": lh.metrics.get("balanced_accuracy"),
                "macro_f1": lh.metrics.get("macro_f1"),
                "pr_auc": lh.metrics.get("pr_auc"),
                "positive_rate": lh.metrics.get("positive_rate"),
                "sensitivity_at_fpr": lh.metrics.get("sensitivity_at_fpr"),
            }
        else:
            lh_payload = {}

        subject_df = per_subject_accuracy(phase.y_true, phase.y_pred, phase.groups)

        results["configs"][channel_set] = {
            "n_features": len(feature_cols),
            "features": feature_cols,
            "phase": {
                "accuracy": phase.metrics.get("accuracy"),
                "balanced_accuracy": phase.metrics.get("balanced_accuracy"),
                "macro_f1": phase.metrics.get("macro_f1"),
                "per_class_recall": phase.metrics.get("per_class_recall"),
                "confusion_matrix": phase.metrics.get("confusion_matrix"),
                "phase_names": phase.metrics.get("phase_names"),
                "per_subject_median_accuracy": float(subject_df["accuracy"].median()),
                # Full per-subject list kept only for the primary config in ui_bundle.
                "per_subject": subject_df.to_dict(orient="records")
                if channel_set in {"rhr_temp", "multimodal"}
                else None,
            },
            "biphasic": {
                "accuracy": biphasic.metrics.get("accuracy"),
                "balanced_accuracy": biphasic.metrics.get("balanced_accuracy"),
                "macro_f1": biphasic.metrics.get("macro_f1"),
                "pr_auc": biphasic.metrics.get("pr_auc"),
            },
            "lh_surge": lh_payload,
        }
        print(
            f"{channel_set:12s} phase bal_acc="
            f"{results['configs'][channel_set]['phase']['balanced_accuracy']:.3f} "
            f"macro_f1={results['configs'][channel_set]['phase']['macro_f1']:.3f} "
            f"biphasic_acc={results['configs'][channel_set]['biphasic']['accuracy']:.3f} "
            f"n_feat={len(feature_cols)}"
        )

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("artifacts/ablations.json"))
    args = parser.parse_args(argv)

    payload = run_ablations(args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

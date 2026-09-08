"""Evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
)

from hfp.labels import PHASE_ORDER


def phase_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = list(range(len(PHASE_ORDER)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_recall = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_recall": {
            PHASE_ORDER[i]: float(per_class_recall[i]) for i in range(len(PHASE_ORDER))
        },
        "confusion_matrix": cm.tolist(),
        "phase_names": list(PHASE_ORDER),
    }


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    *,
    fpr_target: float = 0.1,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "positive_rate": float(np.mean(y_true)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        out["pr_auc"] = float(average_precision_score(y_true, y_proba))
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        # Sensitivity at approximate FPR target via score threshold sweep on preds.
        # Approximate FPR from ranking: pick threshold with FPR closest to target.
        order = np.argsort(-y_proba)
        y_sorted = y_true[order]
        n_neg = max(int((y_true == 0).sum()), 1)
        fp = 0
        tp = 0
        n_pos = max(int((y_true == 1).sum()), 1)
        best = {"sensitivity": 0.0, "fpr": 1.0}
        for i, label in enumerate(y_sorted):
            if label == 1:
                tp += 1
            else:
                fp += 1
            fpr = fp / n_neg
            sens = tp / n_pos
            if abs(fpr - fpr_target) < abs(best["fpr"] - fpr_target):
                best = {"sensitivity": float(sens), "fpr": float(fpr)}
        out["sensitivity_at_fpr"] = {
            "fpr_target": fpr_target,
            "fpr": best["fpr"],
            "sensitivity": best["sensitivity"],
        }
        _ = (precision, recall, thresholds)  # retained for future plotting
    return out


def per_subject_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> pd.DataFrame:
    rows = []
    for sid in np.unique(groups):
        mask = groups == sid
        rows.append(
            {
                "id": sid,
                "n_days": int(mask.sum()),
                "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            }
        )
    return pd.DataFrame(rows).sort_values("accuracy")

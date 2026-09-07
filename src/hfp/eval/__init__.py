"""Evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from hfp.labels import PHASE_ORDER


def phase_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = list(range(len(PHASE_ORDER)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "phase_names": list(PHASE_ORDER),
    }


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

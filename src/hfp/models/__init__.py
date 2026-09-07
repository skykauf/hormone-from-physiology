"""Baseline models: phase classification and LH-surge detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


@dataclass
class TrainResult:
    model: Pipeline
    feature_cols: list[str]
    metrics: dict[str, Any]
    y_true: np.ndarray
    y_pred: np.ndarray
    groups: np.ndarray


def _clf_pipeline(**kwargs: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_depth=4,
                    learning_rate=0.08,
                    max_iter=200,
                    random_state=42,
                    **kwargs,
                ),
            ),
        ]
    )


def leave_one_subject_out(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    task: str = "classification",
) -> TrainResult:
    """
    LOSO cross-validation — the conservative scheme Clair reports.

    Trains a fresh model per held-out subject and concatenates predictions.
    Also fits a final model on all data for export.
    """
    subjects = groups.unique()
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    groups_all: list[np.ndarray] = []

    for sid in subjects:
        test_mask = groups == sid
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        if task == "classification":
            pipe = _clf_pipeline()
            pipe.fit(X.loc[train_mask], y.loc[train_mask])
            pred = pipe.predict(X.loc[test_mask])
        else:
            pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_depth=4, learning_rate=0.08, max_iter=200, random_state=42
                        ),
                    ),
                ]
            )
            pipe.fit(X.loc[train_mask], y.loc[train_mask])
            pred = pipe.predict(X.loc[test_mask])
        y_true_all.append(y.loc[test_mask].to_numpy())
        y_pred_all.append(np.asarray(pred))
        groups_all.append(groups.loc[test_mask].to_numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    g = np.concatenate(groups_all)

    if task == "classification":
        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "report": classification_report(y_true, y_pred, zero_division=0),
        }
        final = _clf_pipeline()
    else:
        metrics = {"mae": float(mean_absolute_error(y_true, y_pred))}
        final = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_depth=4, learning_rate=0.08, max_iter=200, random_state=42
                    ),
                ),
            ]
        )

    final.fit(X, y)
    return TrainResult(
        model=final,
        feature_cols=list(X.columns),
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        groups=g,
    )


def train_phase_classifier(panel: pd.DataFrame, feature_cols: list[str]) -> TrainResult:
    from hfp.labels import encode_phase

    df = panel.dropna(subset=["phase"]).copy()
    df["phase_label"] = encode_phase(df["phase"])
    df = df.dropna(subset=["phase_label"])
    X = df[feature_cols]
    y = df["phase_label"].astype(int)
    groups = df["id"]
    return leave_one_subject_out(X, y, groups, task="classification")


def train_lh_surge_detector(panel: pd.DataFrame, feature_cols: list[str]) -> TrainResult:
    df = panel.dropna(subset=["lh_surge"]).copy()
    X = df[feature_cols]
    y = df["lh_surge"].astype(int)
    groups = df["id"]
    result = leave_one_subject_out(X, y, groups, task="classification")
    if len(np.unique(result.y_true)) > 1 and hasattr(result.model, "predict_proba"):
        # Final-model AUROC on all data (optimistic); primary metric remains LOSO accuracy/F1.
        proba = result.model.predict_proba(X)[:, 1]
        result.metrics["train_auroc_final_model"] = float(roc_auc_score(y, proba))
    return result

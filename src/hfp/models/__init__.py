"""Baseline models: phase classification, biphasic, and LH-surge detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from hfp.eval import binary_metrics, phase_metrics


@dataclass
class TrainResult:
    model: Pipeline
    feature_cols: list[str]
    metrics: dict[str, Any]
    y_true: np.ndarray
    y_pred: np.ndarray
    groups: np.ndarray
    y_proba: np.ndarray | None = None


def _clf_pipeline(max_iter: int = 200, **kwargs: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_depth=4,
                    learning_rate=0.08,
                    max_iter=max_iter,
                    random_state=42,
                    **kwargs,
                ),
            ),
        ]
    )


def smooth_phase_predictions(
    y_pred: np.ndarray,
    groups: np.ndarray,
    *,
    window: int = 3,
) -> np.ndarray:
    """
    Majority-smooth predicted phase labels within each subject (inference only).

    Encourages temporally coherent phase stretches without using labels at train time.
    """
    out = y_pred.copy()
    half = max(window // 2, 1)
    for sid in np.unique(groups):
        idx = np.where(groups == sid)[0]
        seq = out[idx].astype(int)
        smoothed = seq.copy()
        for i in range(len(seq)):
            lo = max(0, i - half)
            hi = min(len(seq), i + half + 1)
            vals, counts = np.unique(seq[lo:hi], return_counts=True)
            smoothed[i] = int(vals[np.argmax(counts)])
        out[idx] = smoothed
    return out


def leave_one_subject_out(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    task: str = "classification",
    class_weight: str | None = "balanced",
    smooth_window: int = 0,
    fit_final: bool = True,
    max_iter: int = 200,
) -> TrainResult:
    """
    LOSO cross-validation — the conservative scheme Clair reports.

    Optionally applies balanced sample weights and post-hoc phase smoothing.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    subjects = list(groups.unique())

    def _one_fold(sid: object) -> tuple | None:
        test_mask = groups == sid
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            return None
        if task == "classification":
            pipe = _clf_pipeline(max_iter=max_iter)
            sw = None
            if class_weight == "balanced":
                sw = compute_sample_weight("balanced", y.loc[train_mask])
            pipe.fit(X.loc[train_mask], y.loc[train_mask], model__sample_weight=sw)
            pred = pipe.predict(X.loc[test_mask])
            proba = pipe.predict_proba(X.loc[test_mask]) if hasattr(pipe, "predict_proba") else None
        else:
            pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_depth=4,
                            learning_rate=0.08,
                            max_iter=max_iter,
                            random_state=42,
                        ),
                    ),
                ]
            )
            pipe.fit(X.loc[train_mask], y.loc[train_mask])
            pred = pipe.predict(X.loc[test_mask])
            proba = None
        return (
            y.loc[test_mask].to_numpy(),
            np.asarray(pred),
            groups.loc[test_mask].to_numpy(),
            proba,
        )

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    y_proba_all: list[np.ndarray] = []
    groups_all: list[np.ndarray] = []
    collect_proba = False

    # Thread pool: sklearn releases the GIL in native HGB loops.
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(subjects)))) as pool:
        futures = [pool.submit(_one_fold, sid) for sid in subjects]
        for fut in as_completed(futures):
            got = fut.result()
            if got is None:
                continue
            yt, yp, gg, proba = got
            y_true_all.append(yt)
            y_pred_all.append(yp)
            groups_all.append(gg)
            if proba is not None:
                y_proba_all.append(proba)
                collect_proba = True

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    g = np.concatenate(groups_all)
    y_proba = None
    if collect_proba and y_proba_all:
        y_proba = np.concatenate(y_proba_all, axis=0)

    if task == "classification" and smooth_window and smooth_window > 1:
        y_pred = smooth_phase_predictions(y_pred, g, window=smooth_window)

    if task == "classification":
        n_classes = int(pd.Series(y_true).nunique())
        if n_classes > 2:
            metrics: dict[str, Any] = phase_metrics(y_true, y_pred)
        else:
            pos_proba = y_proba[:, 1] if y_proba is not None and y_proba.shape[1] > 1 else None
            metrics = binary_metrics(y_true, y_pred, pos_proba)
        metrics["report"] = classification_report(y_true, y_pred, zero_division=0)
        final = _clf_pipeline(max_iter=max_iter)
        if fit_final:
            sw_final = (
                compute_sample_weight("balanced", y) if class_weight == "balanced" else None
            )
            final.fit(X, y, model__sample_weight=sw_final)
    else:
        metrics = {"mae": float(mean_absolute_error(y_true, y_pred))}
        final = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_depth=4,
                        learning_rate=0.08,
                        max_iter=max_iter,
                        random_state=42,
                    ),
                ),
            ]
        )
        if fit_final:
            final.fit(X, y)

    return TrainResult(
        model=final,
        feature_cols=list(X.columns),
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        groups=g,
        y_proba=y_proba,
    )


def train_phase_classifier(
    panel: pd.DataFrame,
    feature_cols: list[str],
    *,
    smooth_window: int = 3,
    fit_final: bool = True,
    max_iter: int = 200,
) -> TrainResult:
    from hfp.labels import encode_phase

    df = panel.dropna(subset=["phase"]).copy()
    df["phase_label"] = encode_phase(df["phase"])
    df = df.dropna(subset=["phase_label"])
    X = df[feature_cols]
    y = df["phase_label"].astype(int)
    groups = df["id"]
    return leave_one_subject_out(
        X,
        y,
        groups,
        task="classification",
        smooth_window=smooth_window,
        fit_final=fit_final,
        max_iter=max_iter,
    )


def train_biphasic_classifier(
    panel: pd.DataFrame,
    feature_cols: list[str],
    *,
    smooth_window: int = 3,
    fit_final: bool = True,
    max_iter: int = 200,
) -> TrainResult:
    from hfp.labels import encode_biphasic

    df = panel.dropna(subset=["phase"]).copy()
    df["biphasic_label"] = encode_biphasic(df["phase"])
    df = df.dropna(subset=["biphasic_label"])
    X = df[feature_cols]
    y = df["biphasic_label"].astype(int)
    groups = df["id"]
    return leave_one_subject_out(
        X,
        y,
        groups,
        task="classification",
        smooth_window=smooth_window,
        fit_final=fit_final,
        max_iter=max_iter,
    )


def train_lh_surge_detector(
    panel: pd.DataFrame,
    feature_cols: list[str],
    *,
    fit_final: bool = True,
    max_iter: int = 200,
) -> TrainResult:
    df = panel.dropna(subset=["lh_surge"]).copy()
    X = df[feature_cols]
    y = df["lh_surge"].astype(int)
    groups = df["id"]
    return leave_one_subject_out(
        X,
        y,
        groups,
        task="classification",
        smooth_window=0,
        fit_final=fit_final,
        max_iter=max_iter,
    )

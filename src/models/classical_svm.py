from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def train_svm_with_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    c_grid: list[float],
    max_iter: int,
    seed: int,
) -> tuple[Pipeline, dict]:
    best_model = None
    best_score = -1.0
    best_c = None

    for c in c_grid:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LinearSVC(
                        C=float(c),
                        max_iter=max_iter,
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        score = f1_score(y_val, y_val_pred, average="macro")
        if score > best_score:
            best_score = score
            best_model = model
            best_c = c

    info = {"best_c": float(best_c), "val_macro_f1": float(best_score)}
    return best_model, info

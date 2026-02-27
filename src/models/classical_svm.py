from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm


def train_svm_with_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    c_grid: list[float],
    max_iter: int,
    seed: int,
    verbose: bool = True,
) -> tuple[Pipeline, dict]:
    best_model = None
    best_score = -1.0
    best_c = None

    progress = tqdm(c_grid, desc="LinearSVC tuning", leave=True) if verbose else c_grid
    for idx, c in enumerate(progress, start=1):
        fit_start = time.perf_counter()
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
        fit_time = time.perf_counter() - fit_start
        if verbose:
            tqdm.write(
                f"[LinearSVC {idx}/{len(c_grid)}] C={float(c):g} "
                f"val_macro_f1={score:.4f} fit_time={fit_time:.1f}s"
            )
        if score > best_score:
            best_score = score
            best_model = model
            best_c = c

    info = {"best_c": float(best_c), "val_macro_f1": float(best_score)}
    return best_model, info

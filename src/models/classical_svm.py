from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm


def _to_numpy(arr: Any) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


class _CumlLinearSVMWrapper:
    def __init__(self, c: float, max_iter: int, seed: int):
        from cuml.svm import LinearSVC as CuLinearSVC  # type: ignore

        self.scaler = StandardScaler()
        self.clf = CuLinearSVC(C=float(c), max_iter=int(max_iter), random_state=int(seed))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_CumlLinearSVMWrapper":
        Xs = self.scaler.fit_transform(X).astype(np.float32)
        ys = y.astype(np.int32)
        self.clf.fit(Xs, ys)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X).astype(np.float32)
        out = self.clf.predict(Xs)
        return _to_numpy(out).astype(np.int64, copy=False)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X).astype(np.float32)
        out = self.clf.decision_function(Xs)
        return _to_numpy(out)


def train_svm_with_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    c_grid: list[float],
    max_iter: int,
    seed: int,
    use_gpu: bool = True,
    verbose: bool = True,
) -> tuple[Any, dict]:
    best_model = None
    best_score = -1.0
    best_c = None
    backend = "sklearn_cpu"

    gpu_available = False
    if use_gpu:
        try:
            import cuml  # noqa: F401

            gpu_available = True
            backend = "cuml_gpu"
        except Exception:
            gpu_available = False

    progress = tqdm(c_grid, desc="LinearSVC tuning", leave=True) if verbose else c_grid
    for idx, c in enumerate(progress, start=1):
        fit_start = time.perf_counter()
        if gpu_available:
            model = _CumlLinearSVMWrapper(c=float(c), max_iter=max_iter, seed=seed)
        else:
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
                f"[LinearSVC {idx}/{len(c_grid)}][{backend}] C={float(c):g} "
                f"val_macro_f1={score:.4f} fit_time={fit_time:.1f}s"
            )
        if score > best_score:
            best_score = score
            best_model = model
            best_c = c

    info = {
        "best_c": float(best_c),
        "val_macro_f1": float(best_score),
        "backend": backend,
        "requested_gpu": bool(use_gpu),
        "used_gpu": bool(gpu_available),
    }
    return best_model, info

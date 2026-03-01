from __future__ import annotations

from itertools import product

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score


def train_lgbm_with_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    param_grid: dict,
    subsample: float,
    colsample_bytree: float,
    seed: int,
    use_gpu: bool = False,
):
    best_model = None
    best_score = -1.0
    best_params = None
    gpu_requested = bool(use_gpu)
    used_gpu = False
    gpu_unavailable = False

    leaves_grid = param_grid.get("num_leaves", [31])
    lr_grid = param_grid.get("learning_rate", [0.1])
    est_grid = param_grid.get("n_estimators", [200])

    for num_leaves, learning_rate, n_estimators in product(leaves_grid, lr_grid, est_grid):
        model_kwargs = dict(
            objective="multiclass",
            num_class=num_classes,
            num_leaves=int(num_leaves),
            learning_rate=float(learning_rate),
            n_estimators=int(n_estimators),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            random_state=seed,
            n_jobs=-1,
        )
        if gpu_requested and not gpu_unavailable:
            model_kwargs["device_type"] = "gpu"
        model = lgb.LGBMClassifier(**model_kwargs)
        try:
            model.fit(X_train, y_train)
            if gpu_requested and not gpu_unavailable:
                used_gpu = True
        except Exception:
            # GPU build/runtime is not always available; retry on CPU.
            if gpu_requested and not gpu_unavailable:
                gpu_unavailable = True
                model_kwargs.pop("device_type", None)
                model = lgb.LGBMClassifier(**model_kwargs)
                model.fit(X_train, y_train)
            else:
                raise
        y_val_pred = model.predict(X_val)
        score = f1_score(y_val, y_val_pred, average="macro")
        if score > best_score:
            best_score = score
            best_model = model
            best_params = {
                "num_leaves": int(num_leaves),
                "learning_rate": float(learning_rate),
                "n_estimators": int(n_estimators),
            }
    return best_model, {
        "best_params": best_params,
        "val_macro_f1": float(best_score),
        "requested_gpu": gpu_requested,
        "used_gpu": used_gpu,
    }

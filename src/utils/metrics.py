from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)


def compute_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: list[int],
    class_names: list[str],
) -> pd.DataFrame:
    rows = []
    for cid, cname in zip(class_ids, class_names):
        idx = y_true == cid
        count = int(np.sum(idx))
        acc = float(np.mean(y_pred[idx] == y_true[idx])) if count > 0 else 0.0
        rows.append({"class_id": cid, "class_name": cname, "count": count, "accuracy": acc})
    df = pd.DataFrame(rows)
    return df


def compute_topk(y_true: np.ndarray, scores: np.ndarray | None, k: int, num_classes: int) -> float | None:
    if scores is None:
        return None
    if scores.ndim != 2:
        return None
    if scores.shape[1] != num_classes:
        return None
    k = min(k, num_classes)
    labels = np.arange(num_classes)
    return float(top_k_accuracy_score(y_true, scores, k=k, labels=labels))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    scores: np.ndarray | None = None,
    topk: int = 5,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, pd.DataFrame]:
    num_classes = len(class_names)
    class_ids = list(range(num_classes))

    acc = float(accuracy_score(y_true, y_pred))
    topk_acc = compute_topk(y_true, scores, k=topk, num_classes=num_classes)

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm_norm / row_sums

    per_class_df = compute_per_class_accuracy(y_true, y_pred, class_ids=class_ids, class_names=class_names)

    metrics = {
        "accuracy": acc,
        "top5_accuracy": topk_acc,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f_weighted),
    }
    return metrics, cm, cm_norm, per_class_df

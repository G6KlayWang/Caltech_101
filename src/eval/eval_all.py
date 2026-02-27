from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.utils.io import load_json, save_json
from src.utils.metrics import compute_classification_metrics
from src.utils.plotting import plot_confusion_matrix, plot_per_class_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved predictions and regenerate metrics/plots.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--y_true", type=str, default=None)
    parser.add_argument("--y_pred", type=str, default=None)
    parser.add_argument("--scores", type=str, default=None)
    parser.add_argument("--class_names_json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)

    y_true_path = Path(args.y_true) if args.y_true else run_dir / "preds" / "y_true.npy"
    y_pred_path = Path(args.y_pred) if args.y_pred else run_dir / "preds" / "y_pred.npy"
    scores_path = Path(args.scores) if args.scores else run_dir / "preds" / "scores.npy"
    class_names_path = (
        Path(args.class_names_json) if args.class_names_json else run_dir / "preds" / "class_names.json"
    )

    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)
    scores = np.load(scores_path) if scores_path.exists() else None
    class_names = load_json(class_names_path)["class_names"]

    metrics, cm, cm_norm, per_class_df = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        scores=scores,
        topk=5,
    )

    save_json(metrics, run_dir / "metrics.json")
    per_class_df.to_csv(run_dir / "per_class_accuracy.csv", index=False)
    plot_confusion_matrix(cm, class_names, str(run_dir / "confusion_matrix.png"), normalized=False)
    plot_confusion_matrix(
        cm_norm, class_names, str(run_dir / "confusion_matrix_normalized.png"), normalized=True
    )
    plot_per_class_accuracy(per_class_df, str(run_dir / "per_class_accuracy.png"))

    print(f"Evaluation artifacts updated in: {run_dir.resolve()}")


if __name__ == "__main__":
    main()

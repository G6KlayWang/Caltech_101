from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: str,
    normalized: bool = False,
) -> None:
    plt.figure(figsize=(16, 14))
    cmap = "Blues"
    sns.heatmap(cm, cmap=cmap, cbar=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    title = "Confusion Matrix (Normalized)" if normalized else "Confusion Matrix"
    plt.title(title)

    # Only set sparse ticks for readability with 102 classes.
    if len(class_names) <= 30:
        plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=90)
        plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_per_class_accuracy(per_class_df: pd.DataFrame, output_path: str) -> None:
    df = per_class_df.sort_values("accuracy", ascending=True).reset_index(drop=True)
    plt.figure(figsize=(16, 10))
    plt.bar(np.arange(len(df)), df["accuracy"].values)
    plt.xlabel("Classes (sorted by accuracy)")
    plt.ylabel("Accuracy")
    plt.title("Per-class Accuracy (Ascending)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_training_curves(history: dict[str, list[float]], output_path: str) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

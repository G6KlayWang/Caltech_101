from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.datasets import get_class_names_from_df
from src.data.feature_engineered import build_engineered_features
from src.data.feature_hog import maybe_load_or_build_hog
from src.utils.io import create_run_dir, deep_merge, ensure_dir, load_yaml, save_json, save_yaml
from src.utils.logging import get_logger
from src.utils.metrics import compute_classification_metrics
from src.utils.plotting import plot_confusion_matrix, plot_per_class_accuracy
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate classical Caltech-101 models.")
    parser.add_argument("--model", type=str, choices=["classical_svm", "classical_lgbm"], required=True)
    parser.add_argument("--exp_name", type=str, default="classical_exp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svm_use_gpu", type=int, choices=[0, 1], default=None)
    parser.add_argument("--lgbm_use_gpu", type=int, choices=[0, 1], default=None)
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    parser.add_argument("--model_config", type=str, default=None)
    return parser.parse_args()


def _maybe_load_or_build_engineered(df: pd.DataFrame, cache_path: Path) -> np.ndarray:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return np.load(cache_path, allow_pickle=False)["X"]
    X = build_engineered_features(df)
    np.savez_compressed(cache_path, X=X)
    return X


def _load_configs(base_config: str, model: str, model_config_override: str | None) -> dict:
    base = load_yaml(base_config)
    if model_config_override is not None:
        model_cfg = load_yaml(model_config_override)
    else:
        default_map = {
            "classical_svm": "configs/classical_svm.yaml",
            "classical_lgbm": "configs/classical_lgbm.yaml",
        }
        model_cfg = load_yaml(default_map[model])
    return deep_merge(base, model_cfg)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = _load_configs(args.base_config, args.model, model_config_override=args.model_config)
    cfg["seed"] = args.seed
    cfg["model"]["name"] = args.model

    runs_root = cfg["project"]["runs_root"]
    processed_dir = Path(cfg["project"]["processed_dir"])
    train_csv = cfg["data"]["train_csv"]
    val_csv = cfg["data"]["val_csv"]
    test_csv = cfg["data"]["test_csv"]

    run_dir = create_run_dir(runs_root=runs_root, exp_name=args.exp_name)
    logger = get_logger("train_classical", run_dir / "logs.txt")
    save_yaml(cfg, run_dir / "config.yaml")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    class_names = get_class_names_from_df(all_df)
    num_classes = len(class_names)

    logger.info("Model: %s | num_classes=%d", args.model, num_classes)
    logger.info("Run dir: %s", run_dir)

    hog_size = int(cfg["features"]["hog_size"])
    orientations = int(cfg["features"]["orientations"])
    pixels_per_cell = int(cfg["features"]["pixels_per_cell"])
    cells_per_block = int(cfg["features"]["cells_per_block"])

    hog_cache_dir = ensure_dir(processed_dir / f"hog_{hog_size}")
    X_train_hog, y_train = maybe_load_or_build_hog(
        train_df,
        cache_path=hog_cache_dir / "train.npz",
        hog_size=hog_size,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    X_val_hog, y_val = maybe_load_or_build_hog(
        val_df,
        cache_path=hog_cache_dir / "val.npz",
        hog_size=hog_size,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    X_test_hog, y_test = maybe_load_or_build_hog(
        test_df,
        cache_path=hog_cache_dir / "test.npz",
        hog_size=hog_size,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )

    if args.model == "classical_svm":
        from src.models.classical_svm import train_svm_with_val

        training_cfg = cfg["training"]
        use_gpu = bool(training_cfg.get("use_gpu", True))
        if args.svm_use_gpu is not None:
            use_gpu = bool(args.svm_use_gpu)
        model, tune_info = train_svm_with_val(
            X_train=X_train_hog,
            y_train=y_train,
            X_val=X_val_hog,
            y_val=y_val,
            c_grid=list(training_cfg["c_grid"]),
            max_iter=int(training_cfg["max_iter"]),
            seed=args.seed,
            use_gpu=use_gpu,
        )
        X_test = X_test_hog
    else:
        from src.models.classical_lgbm import train_lgbm_with_val

        eng_cache_dir = ensure_dir(processed_dir / "engineered9")
        X_train_eng = _maybe_load_or_build_engineered(train_df, eng_cache_dir / "train.npz")
        X_val_eng = _maybe_load_or_build_engineered(val_df, eng_cache_dir / "val.npz")
        X_test_eng = _maybe_load_or_build_engineered(test_df, eng_cache_dir / "test.npz")

        if X_train_eng.shape[1] != 9 or X_val_eng.shape[1] != 9 or X_test_eng.shape[1] != 9:
            raise RuntimeError("Engineered feature dimensions are not exactly 9.")

        X_train = np.concatenate([X_train_hog, X_train_eng], axis=1)
        X_val = np.concatenate([X_val_hog, X_val_eng], axis=1)
        X_test = np.concatenate([X_test_hog, X_test_eng], axis=1)

        training_cfg = cfg["training"]
        use_gpu = bool(training_cfg.get("use_gpu", False))
        if args.lgbm_use_gpu is not None:
            use_gpu = bool(args.lgbm_use_gpu)
        model, tune_info = train_lgbm_with_val(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            num_classes=num_classes,
            param_grid=training_cfg["param_grid"],
            subsample=float(training_cfg["subsample"]),
            colsample_bytree=float(training_cfg["colsample_bytree"]),
            seed=args.seed,
            use_gpu=use_gpu,
        )

    y_pred = model.predict(X_test)
    if args.model == "classical_svm":
        scores = model.decision_function(X_test)
    else:
        scores = model.predict_proba(X_test)

    metrics, cm, cm_norm, per_class_df = compute_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        scores=scores,
        topk=5,
    )
    metrics["model"] = args.model
    metrics["seed"] = args.seed
    metrics["num_classes"] = num_classes
    metrics["tuning"] = tune_info

    save_json(metrics, run_dir / "metrics.json")
    per_class_df.to_csv(run_dir / "per_class_accuracy.csv", index=False)
    plot_confusion_matrix(cm, class_names, str(run_dir / "confusion_matrix.png"), normalized=False)
    plot_confusion_matrix(
        cm_norm, class_names, str(run_dir / "confusion_matrix_normalized.png"), normalized=True
    )
    plot_per_class_accuracy(per_class_df, str(run_dir / "per_class_accuracy.png"))

    np.save(run_dir / "preds" / "y_true.npy", y_test)
    np.save(run_dir / "preds" / "y_pred.npy", y_pred)
    np.save(run_dir / "preds" / "scores.npy", scores)
    save_json({"class_names": class_names}, run_dir / "preds" / "class_names.json")

    joblib.dump(model, run_dir / "model" / "model.joblib")

    logger.info("Test accuracy: %.4f | Macro-F1: %.4f", metrics["accuracy"], metrics["f1_macro"])
    print(f"RUN_DIR={run_dir.resolve()}")


if __name__ == "__main__":
    main()

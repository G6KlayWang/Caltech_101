from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import CaltechCSVDataset, get_class_names_from_df
from src.utils.io import create_run_dir, deep_merge, load_yaml, save_json, save_yaml
from src.utils.logging import get_logger
from src.utils.metrics import compute_classification_metrics
from src.utils.plotting import plot_confusion_matrix, plot_per_class_accuracy, plot_training_curves
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate transfer learning models on Caltech-101.")
    parser.add_argument("--model", type=str, choices=["resnet50", "efficientnet", "vit"], required=True)
    parser.add_argument("--img_size", type=int, default=128, choices=[64, 128, 224])
    parser.add_argument("--aug", type=int, default=1, choices=[0, 1])
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--exp_name", type=str, default="dl_exp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    parser.add_argument("--model_config", type=str, default=None)
    return parser.parse_args()


def _load_configs(base_config: str, model: str, model_config_override: str | None) -> dict:
    base = load_yaml(base_config)
    if model_config_override is not None:
        model_cfg = load_yaml(model_config_override)
    else:
        default_map = {
            "resnet50": "configs/dl_resnet50.yaml",
            "efficientnet": "configs/dl_efficientnet.yaml",
            "vit": "configs/dl_vit.yaml",
        }
        model_cfg = load_yaml(default_map[model])
    return deep_merge(base, model_cfg)


def _build_model(model_name: str, num_classes: int, pretrained: bool):
    from src.models import dl_efficientnet, dl_resnet50, dl_vit

    if model_name == "resnet50":
        return dl_resnet50.build_model(num_classes=num_classes, pretrained=pretrained), dl_resnet50
    if model_name == "efficientnet":
        return dl_efficientnet.build_model(num_classes=num_classes, pretrained=pretrained), dl_efficientnet
    if model_name == "vit":
        return dl_vit.build_model(num_classes=num_classes, pretrained=pretrained), dl_vit
    raise ValueError(f"Unsupported model: {model_name}")


def _make_optimizer(model: nn.Module, optimizer_name: str, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)


def _run_train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, labels, _ in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total += batch_size

    return total_loss / total, total_correct / total


@torch.no_grad()
def _run_eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    y_true, y_pred, prob_list = [], [], []

    for images, labels, _ in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total += batch_size

        y_true.append(labels.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
        prob_list.append(probs.cpu().numpy())

    return (
        total_loss / total,
        total_correct / total,
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(prob_list),
    )


def main() -> None:
    args = parse_args()
    from src.data.transforms import build_eval_transform, build_train_transform

    set_seed(args.seed)

    # torchvision vit_b_16 expects 224x224 inputs.
    vit_img_size_forced = False
    if args.model == "vit" and int(args.img_size) != 224:
        args.img_size = 224
        vit_img_size_forced = True

    cfg = _load_configs(args.base_config, args.model, model_config_override=args.model_config)
    cfg["seed"] = args.seed
    cfg["model"]["name"] = args.model
    cfg["training"]["optimizer"] = args.optimizer
    cfg["data"]["img_size"] = int(args.img_size)
    cfg["data"]["aug"] = int(args.aug)

    runs_root = cfg["project"]["runs_root"]
    train_csv = cfg["data"]["train_csv"]
    val_csv = cfg["data"]["val_csv"]
    test_csv = cfg["data"]["test_csv"]
    num_workers = int(cfg["data"].get("num_workers", 4))

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    class_names = get_class_names_from_df(pd.concat([train_df, val_df, test_df], ignore_index=True))
    num_classes = len(class_names)

    run_dir = create_run_dir(runs_root=runs_root, exp_name=args.exp_name)
    logger = get_logger("train_dl", run_dir / "logs.txt")
    save_yaml(cfg, run_dir / "config.yaml")

    if vit_img_size_forced:
        logger.warning("ViT requires img_size=224. Overriding requested size to 224.")
    logger.info("Model=%s | img_size=%d | aug=%d | optimizer=%s", args.model, args.img_size, args.aug, args.optimizer)
    logger.info("Classes=%d | Run dir=%s", num_classes, run_dir)

    batch_size = int(cfg["training"]["batch_size"])
    img_size = int(cfg["data"]["img_size"])
    aug = bool(int(cfg["data"]["aug"]))
    train_tfms = build_train_transform(img_size=img_size, aug=aug)
    eval_tfms = build_eval_transform(img_size=img_size)

    train_ds = CaltechCSVDataset(train_csv, transform=train_tfms)
    val_ds = CaltechCSVDataset(val_csv, transform=eval_tfms)
    test_ds = CaltechCSVDataset(test_csv, transform=eval_tfms)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_module = _build_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=bool(cfg["model"].get("pretrained", True)),
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    epochs = int(cfg["training"]["epochs"])
    freeze_epochs = int(cfg["training"]["freeze_epochs"])
    lr = float(cfg["training"]["lr"])
    unfreeze_lr = float(cfg["training"]["unfreeze_lr"])
    weight_decay = float(cfg["training"]["weight_decay"])

    model_module.set_backbone_trainable(model, trainable=False)
    optimizer = _make_optimizer(model, optimizer_name=args.optimizer, lr=lr, weight_decay=weight_decay)
    logger.info("Starting training: freeze_epochs=%d, total_epochs=%d", freeze_epochs, epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_macro_f1": []}
    best_val_macro_f1 = -1.0
    best_ckpt_path = run_dir / "model" / "best.pt"

    for epoch in range(1, epochs + 1):
        if epoch == freeze_epochs + 1:
            model_module.set_backbone_trainable(model, trainable=True)
            optimizer = _make_optimizer(
                model, optimizer_name=args.optimizer, lr=unfreeze_lr, weight_decay=weight_decay
            )
            logger.info("Unfroze backbone at epoch %d.", epoch)

        train_loss, train_acc = _run_train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, y_val, y_val_pred, val_probs = _run_eval_epoch(model, val_loader, criterion, device)
        val_metrics, _, _, _ = compute_classification_metrics(
            y_true=y_val,
            y_pred=y_val_pred,
            class_names=class_names,
            scores=val_probs,
            topk=5,
        )
        val_macro_f1 = float(val_metrics["f1_macro"])

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["val_macro_f1"].append(val_macro_f1)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_macro_f1,
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_macro_f1": best_val_macro_f1,
                },
                best_ckpt_path,
            )

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded best checkpoint from epoch %s with val_macro_f1=%.4f", ckpt["epoch"], ckpt["best_val_macro_f1"])

    test_loss, test_acc, y_test, y_pred, test_probs = _run_eval_epoch(model, test_loader, criterion, device)
    metrics, cm, cm_norm, per_class_df = compute_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        scores=test_probs,
        topk=5,
    )
    metrics["model"] = args.model
    metrics["seed"] = args.seed
    metrics["img_size"] = img_size
    metrics["aug"] = int(aug)
    metrics["optimizer"] = args.optimizer
    metrics["num_classes"] = num_classes
    metrics["test_loss"] = float(test_loss)
    metrics["best_val_macro_f1"] = float(best_val_macro_f1)

    save_json(metrics, run_dir / "metrics.json")
    save_json(history, run_dir / "training_history.json")
    per_class_df.to_csv(run_dir / "per_class_accuracy.csv", index=False)

    plot_confusion_matrix(cm, class_names, str(run_dir / "confusion_matrix.png"), normalized=False)
    plot_confusion_matrix(
        cm_norm, class_names, str(run_dir / "confusion_matrix_normalized.png"), normalized=True
    )
    plot_per_class_accuracy(per_class_df, str(run_dir / "per_class_accuracy.png"))
    plot_training_curves(history, str(run_dir / "training_curves.png"))

    np.save(run_dir / "preds" / "y_true.npy", y_test)
    np.save(run_dir / "preds" / "y_pred.npy", y_pred)
    np.save(run_dir / "preds" / "scores.npy", test_probs)
    save_json({"class_names": class_names}, run_dir / "preds" / "class_names.json")

    logger.info("Test accuracy=%.4f | top5=%.4f | macro_f1=%.4f", metrics["accuracy"], metrics["top5_accuracy"], metrics["f1_macro"])
    print(f"RUN_DIR={run_dir.resolve()}")


if __name__ == "__main__":
    main()

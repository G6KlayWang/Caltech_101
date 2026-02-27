from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def maybe_download_from_kaggle(raw_dir: Path, kaggle_dataset: str) -> None:
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin is None:
        raise FileNotFoundError("Kaggle CLI not found. Install kaggle or place dataset manually in data/raw.")
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [kaggle_bin, "datasets", "download", "-d", kaggle_dataset, "-p", str(raw_dir), "--unzip"]
    subprocess.run(cmd, check=True)


def detect_class_root(raw_dir: Path) -> Path:
    candidates = [
        raw_dir,
        raw_dir / "Caltech101",
        raw_dir / "101_ObjectCategories",
        raw_dir / "Caltech101" / "101_ObjectCategories",
        raw_dir / "caltech101",
        raw_dir / "caltech101" / "101_ObjectCategories",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            subdirs = [d for d in c.iterdir() if d.is_dir()]
            if len(subdirs) >= 50:
                return c

    # Fallback: search two levels down for 101_ObjectCategories.
    for p in raw_dir.rglob("101_ObjectCategories"):
        if p.is_dir():
            return p
    raise FileNotFoundError(
        f"Could not detect class root under {raw_dir}. "
        "Expected a directory containing category folders including BACKGROUND_Google."
    )


def build_index(class_root: Path) -> pd.DataFrame:
    class_dirs = sorted([d for d in class_root.iterdir() if d.is_dir()], key=lambda p: p.name)
    class_names = [d.name for d in class_dirs]

    if "BACKGROUND_Google" not in class_names:
        raise ValueError("BACKGROUND_Google was not found. This project requires keeping it as a class.")

    class_to_id = {name: i for i, name in enumerate(class_names)}
    records = []
    for cdir in tqdm(class_dirs, desc="Scanning classes"):
        cname = cdir.name
        cid = class_to_id[cname]
        for fp in cdir.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in IMAGE_EXTS:
                records.append(
                    {
                        "filepath": str(fp.resolve()),
                        "class_name": cname,
                        "class_id": cid,
                    }
                )
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No image files found while building index.")
    return df


def split_index(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df["class_id"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["class_id"],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def validate_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    s_train = set(train_df["filepath"].tolist())
    s_val = set(val_df["filepath"].tolist())
    s_test = set(test_df["filepath"].tolist())
    if s_train & s_val or s_train & s_test or s_val & s_test:
        raise RuntimeError("Split overlap detected.")


def print_split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print(f"Train size: {len(train_df)}")
    print(f"Val size:   {len(val_df)}")
    print(f"Test size:  {len(test_df)}")

    merged = []
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        c = split_df.groupby(["class_id", "class_name"]).size().reset_index(name=split_name)
        merged.append(c)
    out = merged[0]
    out = out.merge(merged[1], on=["class_id", "class_name"], how="outer")
    out = out.merge(merged[2], on=["class_id", "class_name"], how="outer")
    out = out.fillna(0).astype({"train": int, "val": int, "test": int})
    print("Per-class counts (head):")
    print(out.sort_values("class_id").head(12).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified train/val/test splits for Caltech-101.")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--splits_dir", type=str, default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download_if_missing", type=int, default=0)
    parser.add_argument("--kaggle_dataset", type=str, default="jessicali9530/caltech101")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    try:
        class_root = detect_class_root(raw_dir)
    except FileNotFoundError:
        if int(args.download_if_missing) == 1:
            print("Dataset not found. Attempting Kaggle download...")
            maybe_download_from_kaggle(raw_dir, args.kaggle_dataset)
            class_root = detect_class_root(raw_dir)
        else:
            raise

    print(f"Detected class root: {class_root}")
    index_df = build_index(class_root)
    train_df, val_df, test_df = split_index(index_df, seed=args.seed)
    validate_splits(train_df, val_df, test_df)

    index_df.to_csv(splits_dir / "index.csv", index=False)
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    print_split_summary(train_df, val_df, test_df)
    print(f"Saved split CSVs to: {splits_dir.resolve()}")


if __name__ == "__main__":
    main()

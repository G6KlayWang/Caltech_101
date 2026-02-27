from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from tqdm import tqdm


def _extract_single_hog(
    filepath: str,
    hog_size: int,
    orientations: int,
    pixels_per_cell: int,
    cells_per_block: int,
) -> np.ndarray:
    img = Image.open(filepath).convert("RGB").resize((hog_size, hog_size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    gray = rgb2gray(arr)
    feat = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feat.astype(np.float32)


def build_hog_features(
    df: pd.DataFrame,
    hog_size: int = 128,
    orientations: int = 9,
    pixels_per_cell: int = 8,
    cells_per_block: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    feats = []
    labels = df["class_id"].to_numpy(dtype=np.int64)
    for fp in tqdm(df["filepath"].tolist(), desc=f"HOG {hog_size}"):
        feats.append(
            _extract_single_hog(
                fp,
                hog_size=hog_size,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
            )
        )
    return np.vstack(feats), labels


def maybe_load_or_build_hog(
    df: pd.DataFrame,
    cache_path: Path,
    hog_size: int = 128,
    orientations: int = 9,
    pixels_per_cell: int = 8,
    cells_per_block: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        obj = np.load(cache_path, allow_pickle=False)
        return obj["X"], obj["y"]
    X, y = build_hog_features(
        df,
        hog_size=hog_size,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    np.savez_compressed(cache_path, X=X, y=y)
    return X, y

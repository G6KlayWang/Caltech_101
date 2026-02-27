from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def _extract_engineered_9(filepath: str) -> np.ndarray:
    img = Image.open(filepath).convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[0], arr.shape[1]
    ratio = float(w) / float(h) if h > 0 else 0.0

    mean_rgb = arr.mean(axis=(0, 1))
    std_rgb = arr.std(axis=(0, 1))

    feats = np.array(
        [
            float(w),
            float(h),
            float(ratio),
            float(mean_rgb[0]),
            float(mean_rgb[1]),
            float(mean_rgb[2]),
            float(std_rgb[0]),
            float(std_rgb[1]),
            float(std_rgb[2]),
        ],
        dtype=np.float32,
    )
    assert feats.shape[0] == 9, "Engineered features must be exactly 9 dimensions."
    return feats


def build_engineered_features(df: pd.DataFrame) -> np.ndarray:
    feats = []
    for fp in tqdm(df["filepath"].tolist(), desc="Engineered(9)"):
        feats.append(_extract_engineered_9(fp))
    X = np.vstack(feats)
    if X.shape[1] != 9:
        raise RuntimeError(f"Engineered feature dimension is {X.shape[1]} but expected 9.")
    return X

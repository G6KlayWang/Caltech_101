from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def load_split_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def get_class_names_from_df(df: pd.DataFrame) -> list[str]:
    pairs = (
        df[["class_id", "class_name"]]
        .drop_duplicates()
        .sort_values("class_id")
        .reset_index(drop=True)
    )
    return pairs["class_name"].tolist()


class CaltechCSVDataset(Dataset):
    def __init__(self, csv_path: str | Path, transform=None) -> None:
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = int(row["class_id"])
        return img, label, row["filepath"]

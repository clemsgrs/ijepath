from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import torch
from torchvision.datasets.folder import default_loader


class EvalDataset(torch.utils.data.Dataset):
    """Evaluation dataset that returns (index, image, label)."""

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Callable | None = None,
        image_col: str = "image_path",
        label_col: str = "label",
        label_to_int: dict[str, int] | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col
        self.loader = default_loader

        labels = [str(v) for v in self.df[label_col].tolist()]
        if label_to_int is None:
            unique = sorted(set(labels))
            label_to_int = {name: idx for idx, name in enumerate(unique)}
        self.label_to_int = dict(label_to_int)
        self.labels = [int(self.label_to_int[str(v)]) for v in labels]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = self.loader(str(row[self.image_col]))
        if self.transform is not None:
            image = self.transform(image)
        return idx, image, int(self.labels[idx])

    def __len__(self) -> int:
        return len(self.df)

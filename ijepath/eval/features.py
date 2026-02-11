from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def extract_teacher_features_single_process(
    teacher,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Extract L2-normalized mean-pooled token features from the teacher encoder."""
    if hasattr(teacher, "module"):
        teacher = teacher.module

    teacher.eval()
    features: np.ndarray | None = None
    total_rows = len(loader.dataset) if hasattr(loader, "dataset") else None
    write_offset = 0

    for idx, image, _label in loader:
        image = image.to(device, non_blocking=True)
        tokens = teacher(image, masks=None)
        pooled = tokens.mean(dim=1)
        arr = pooled.detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        normalized = arr / norms

        if features is None:
            if total_rows is None or int(total_rows) <= 0:
                raise RuntimeError("Feature extraction requires a non-empty dataset-backed loader")
            features = np.empty((int(total_rows), int(normalized.shape[1])), dtype=np.float32)

        if isinstance(idx, torch.Tensor):
            batch_indices = idx.detach().cpu().numpy().astype(np.int64)
            features[batch_indices, :] = normalized
        else:
            batch_size = int(normalized.shape[0])
            features[write_offset : write_offset + batch_size, :] = normalized
            write_offset += batch_size

    if features is None:
        raise RuntimeError("Feature extraction received an empty loader")
    return features

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
    features: list[np.ndarray] = []

    for _idx, image, _label in loader:
        image = image.to(device, non_blocking=True)
        tokens = teacher(image, masks=None)
        pooled = tokens.mean(dim=1)
        arr = pooled.detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        features.append(arr / norms)

    if not features:
        raise RuntimeError("Feature extraction received an empty loader")
    return np.concatenate(features, axis=0)

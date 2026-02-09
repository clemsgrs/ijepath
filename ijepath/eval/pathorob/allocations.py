"""Paper-specified PathoROB allocation matrices (Camelyon phase-1)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

CAMELYON_CLASSES = ["normal", "tumor"]
CAMELYON_ID_CENTERS = ["RUMC", "UMCU"]
CAMELYON_OOD_CENTERS = ["CWZ", "RST", "LPON"]

CAMELYON_ALLOCATIONS: Dict[float, np.ndarray] = {
    0.00: np.array([[2100, 2100], [2100, 2100]]),
    0.14: np.array([[1800, 2400], [2400, 1800]]),
    0.29: np.array([[1500, 2700], [2700, 1500]]),
    0.43: np.array([[1200, 3000], [3000, 1200]]),
    0.57: np.array([[900, 3300], [3300, 900]]),
    0.71: np.array([[600, 3600], [3600, 600]]),
    0.86: np.array([[300, 3900], [3900, 300]]),
    1.00: np.array([[0, 4200], [4200, 0]]),
}

CAMELYON_V_LEVELS = [0.00, 0.14, 0.29, 0.43, 0.57, 0.71, 0.86, 1.00]

_CAMELYON_DATASET_ALIASES = {"camelyon", "camelyon16", "camelyon17"}


def _is_camelyon_dataset(dataset: str) -> bool:
    return str(dataset).lower() in _CAMELYON_DATASET_ALIASES


def get_paper_allocations(dataset: str) -> Dict[float, np.ndarray]:
    if _is_camelyon_dataset(dataset):
        return CAMELYON_ALLOCATIONS
    dataset = str(dataset).lower()
    raise ValueError(f"Unknown dataset: {dataset}. Available: camelyon")


def get_paper_v_levels(dataset: str) -> List[float]:
    if _is_camelyon_dataset(dataset):
        return CAMELYON_V_LEVELS.copy()
    dataset = str(dataset).lower()
    raise ValueError(f"Unknown dataset: {dataset}. Available: camelyon")


def get_paper_metadata(dataset: str) -> Tuple[List[str], List[str], List[str]]:
    if _is_camelyon_dataset(dataset):
        return CAMELYON_CLASSES.copy(), CAMELYON_ID_CENTERS.copy(), CAMELYON_OOD_CENTERS.copy()
    dataset = str(dataset).lower()
    raise ValueError(f"Unknown dataset: {dataset}. Available: camelyon")


def scale_allocation(alloc: np.ndarray, target_total: int) -> np.ndarray:
    alloc = np.asarray(alloc, dtype=float)
    current_total = float(alloc.sum())
    if current_total <= 0:
        return np.zeros_like(alloc, dtype=int)

    scaled = alloc * (float(target_total) / current_total)
    result = np.floor(scaled).astype(int)
    remainder = int(target_total) - int(result.sum())
    if remainder > 0:
        frac = scaled - result
        frac[alloc == 0] = -1.0
        flat = np.argsort(frac.ravel())[::-1]
        for i in range(min(remainder, len(flat))):
            idx = np.unravel_index(flat[i], result.shape)
            if frac[idx] < 0:
                break
            result[idx] += 1
    elif remainder < 0:
        for _ in range(-remainder):
            idx = np.unravel_index(np.argmax(result), result.shape)
            if result[idx] <= 0:
                break
            result[idx] -= 1
    return result

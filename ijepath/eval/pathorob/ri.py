from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .datasets import infer_2x2_pairs, subset_by_pair


@dataclass
class RIResult:
    dataset: str
    k: int
    value: float
    std: float
    n_pairs: int


def _require_sklearn():
    try:
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.neighbors import NearestNeighbors
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "PathoROB RI requires scikit-learn. Install `scikit-learn` to run robustness tuning."
        ) from exc
    return balanced_accuracy_score, NearestNeighbors


def _normalized_ri_from_neighbors(
    labels: np.ndarray,
    centers: np.ndarray,
    neigh_idx: np.ndarray,
    k: int,
) -> float:
    neigh = neigh_idx[:, :k]
    sample_labels = labels[:, None]
    sample_centers = centers[:, None]

    neigh_labels = labels[neigh]
    neigh_centers = centers[neigh]

    so = np.logical_and(neigh_labels == sample_labels, neigh_centers != sample_centers).sum()
    os = np.logical_and(neigh_labels != sample_labels, neigh_centers == sample_centers).sum()
    denom = float(so + os)
    if denom <= 0:
        return 0.5
    return float(so / denom)


def _prepare_neighbors(features: np.ndarray, slide_ids: np.ndarray, kmax: int) -> np.ndarray:
    _balanced_accuracy_score, NearestNeighbors = _require_sklearn()

    nn = NearestNeighbors(n_neighbors=min(kmax + 64, len(features) - 1), metric="cosine")
    nn.fit(features)
    _distances, neigh = nn.kneighbors(features)

    out = np.full((len(features), kmax), -1, dtype=int)
    for i in range(len(features)):
        vals = []
        for j in neigh[i].tolist():
            if j == i:
                continue
            if slide_ids[j] == slide_ids[i]:
                continue
            vals.append(j)
            if len(vals) == kmax:
                break
        if len(vals) < kmax:
            for j in neigh[i].tolist():
                if j == i or j in vals:
                    continue
                vals.append(j)
                if len(vals) == kmax:
                    break
        if len(vals) < kmax:
            raise RuntimeError("Not enough neighbors to compute RI")
        out[i, :] = np.asarray(vals, dtype=int)

    return out


def _optimal_k_by_knn_balanced_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    slide_ids: np.ndarray,
    k_values: Sequence[int],
) -> int:
    balanced_accuracy_score, _NearestNeighbors = _require_sklearn()

    kmax = int(max(k_values))
    neigh = _prepare_neighbors(features, slide_ids, kmax)

    best_k = int(k_values[0])
    best_score = -1.0

    for k in k_values:
        topk = neigh[:, : int(k)]
        neigh_labels = labels[topk]
        pred = []
        for row in neigh_labels:
            vals, cnt = np.unique(row, return_counts=True)
            pred.append(int(vals[np.argmax(cnt)]))
        score = float(balanced_accuracy_score(labels, np.asarray(pred, dtype=int)))
        if score > best_score:
            best_score = score
            best_k = int(k)

    return best_k


def compute_ri(
    dataset_name: str,
    features: np.ndarray,
    manifest_df: pd.DataFrame,
    k_candidates: Sequence[int],
    max_pairs: Optional[int],
    random_state: int,
) -> RIResult:
    df = manifest_df.reset_index(drop=True).copy()
    pairs = infer_2x2_pairs(df, dataset_name=dataset_name, max_pairs=max_pairs, random_state=random_state)
    if not pairs:
        raise RuntimeError(f"{dataset_name}: no valid 2x2 pairs for RI")

    k = _optimal_k_by_knn_balanced_accuracy(
        features=features,
        labels=pd.factorize(df["label"])[0].astype(int),
        slide_ids=df["slide_id"].astype(str).to_numpy(),
        k_values=k_candidates,
    )

    values: list[float] = []
    for pair in pairs:
        sub = subset_by_pair(df, pair)
        if len(sub) <= k + 1:
            continue

        idx = sub.index.to_numpy()
        f = features[idx]
        norms = np.linalg.norm(f, axis=1, keepdims=True) + 1e-12
        f = f / norms

        lbl = pd.factorize(sub["label"])[0].astype(int)
        ctr = pd.factorize(sub["medical_center"])[0].astype(int)
        sid = sub["slide_id"].astype(str).to_numpy()

        neigh = _prepare_neighbors(f, sid, k)
        values.append(float(_normalized_ri_from_neighbors(lbl, ctr, neigh, k)))

    if not values:
        raise RuntimeError(f"{dataset_name}: RI failed on all inferred 2x2 pairs")

    arr = np.asarray(values, dtype=float)
    return RIResult(
        dataset=str(dataset_name),
        k=int(k),
        value=float(arr.mean()),
        std=float(arr.std(ddof=0)),
        n_pairs=int(len(arr)),
    )

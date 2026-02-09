from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .datasets import infer_2x2_pairs, subset_by_pair


@dataclass
class ClusteringResult:
    dataset: str
    score: float
    std: float
    n_pairs: int


def _comb2(n: np.ndarray) -> np.ndarray:
    return n * (n - 1.0) * 0.5


def _adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError("labels_true and labels_pred must have the same length")

    n = labels_true.shape[0]
    if n < 2:
        return 0.0

    true_classes, true_inv = np.unique(labels_true, return_inverse=True)
    pred_classes, pred_inv = np.unique(labels_pred, return_inverse=True)

    contingency = np.zeros((len(true_classes), len(pred_classes)), dtype=np.float64)
    np.add.at(contingency, (true_inv, pred_inv), 1.0)

    nij = _comb2(contingency).sum()
    ai = _comb2(contingency.sum(axis=1)).sum()
    bj = _comb2(contingency.sum(axis=0)).sum()
    total = float(_comb2(np.array([n], dtype=np.float64))[0])

    if total <= 0:
        return 0.0

    expected = (ai * bj) / total
    max_index = 0.5 * (ai + bj)
    denom = max_index - expected
    if abs(denom) < 1e-12:
        return 0.0
    return float((nij - expected) / denom)


def clustering_score(cluster_assignments: np.ndarray, bio_labels: np.ndarray, center_labels: np.ndarray) -> float:
    ari_bio = _adjusted_rand_index(bio_labels, cluster_assignments)
    ari_center = _adjusted_rand_index(center_labels, cluster_assignments)
    return float(ari_bio - ari_center)


def _require_sklearn():
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PathoROB clustering requires scikit-learn. Install `scikit-learn` to run robustness tuning."
        ) from exc
    return KMeans, silhouette_score


def _best_k_by_silhouette(features: np.ndarray, k_min: int, k_max: int, random_state: int) -> int:
    KMeans, silhouette_score = _require_sklearn()

    best_k = int(k_min)
    best_s = -1.0
    for k in range(int(k_min), min(int(k_max), len(features) - 1) + 1):
        if k < 2:
            continue
        km = KMeans(n_clusters=int(k), n_init=10, random_state=int(random_state))
        pred = km.fit_predict(features)
        if len(np.unique(pred)) < 2:
            continue
        s = float(silhouette_score(features, pred, metric="euclidean"))
        if s > best_s:
            best_s = s
            best_k = int(k)
    return best_k


def compute_clustering_score(
    dataset_name: str,
    features: np.ndarray,
    manifest_df: pd.DataFrame,
    repeats: int,
    k_min: int,
    k_max: int,
    max_pairs: Optional[int],
    random_state: int,
) -> ClusteringResult:
    KMeans, _silhouette_score = _require_sklearn()

    pairs = infer_2x2_pairs(
        manifest_df,
        dataset_name=dataset_name,
        max_pairs=max_pairs,
        random_state=random_state,
    )
    if not pairs:
        raise RuntimeError(f"{dataset_name}: no valid 2x2 pairs for clustering")

    pair_scores: list[float] = []
    for pidx, pair in enumerate(pairs):
        sub = subset_by_pair(manifest_df, pair)
        if len(sub) < 8:
            continue
        idx = sub.index.to_numpy()
        f = features[idx]
        norms = np.linalg.norm(f, axis=1, keepdims=True) + 1e-12
        f = f / norms

        y_bio = pd.factorize(sub["label"])[0].astype(int)
        y_ctr = pd.factorize(sub["medical_center"])[0].astype(int)

        best_k = _best_k_by_silhouette(
            f,
            k_min=max(2, int(k_min)),
            k_max=int(k_max),
            random_state=int(random_state) + pidx,
        )

        rep_scores: list[float] = []
        for rep in range(int(repeats)):
            km = KMeans(n_clusters=int(best_k), n_init=5, random_state=int(random_state) + rep + pidx * 1000)
            pred = km.fit_predict(f)
            rep_scores.append(clustering_score(pred, y_bio, y_ctr))

        pair_scores.append(float(np.mean(rep_scores)))

    if not pair_scores:
        raise RuntimeError(f"{dataset_name}: clustering failed on all 2x2 pairs")

    arr = np.asarray(pair_scores, dtype=float)
    return ClusteringResult(
        dataset=str(dataset_name),
        score=float(arr.mean()),
        std=float(arr.std(ddof=0)),
        n_pairs=int(len(arr)),
    )

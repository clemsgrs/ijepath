from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd


@dataclass
class APDResult:
    dataset: str
    apd_id: float
    apd_ood: float
    apd_avg: float
    apd_id_std: float
    apd_ood_std: float
    apd_avg_std: float
    acc_id_by_rho: Dict[float, tuple[float, float]]
    acc_ood_by_rho: Dict[float, tuple[float, float]]


def _require_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PathoROB APD requires scikit-learn. Install `scikit-learn` to run robustness tuning."
        ) from exc
    return LogisticRegression, accuracy_score


def _train_linear_probe(train_x: np.ndarray, train_y: np.ndarray, seed: int):
    LogisticRegression, _accuracy_score = _require_sklearn()
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=int(seed))
    clf.fit(train_x, train_y)
    return clf


def _apd_from_split_acc(acc_by_split: Dict[int, float]) -> float:
    if 1 not in acc_by_split:
        raise RuntimeError("Split 1 (balanced) missing for APD")
    base = float(acc_by_split[1])
    keys = sorted([k for k in acc_by_split if k != 1])
    if not keys:
        return 0.0
    drops = [((float(acc_by_split[k]) - base) / max(base, 1e-12)) for k in keys]
    return float(np.mean(drops))


def compute_apd(
    dataset_name: str,
    features: np.ndarray,
    all_splits: Sequence[pd.DataFrame],
    seed: int,
) -> APDResult:
    _LogisticRegression, accuracy_score = _require_sklearn()

    if len(all_splits) == 0:
        raise RuntimeError(f"{dataset_name}: no APD splits provided")

    id_by_rep: Dict[int, Dict[int, float]] = {}
    ood_by_rep: Dict[int, Dict[int, float]] = {}
    id_by_rho: Dict[float, list[float]] = {}
    ood_by_rho: Dict[float, list[float]] = {}

    for split_df in all_splits:
        rep = int(split_df["rep"].iloc[0])
        split_id = int(split_df["split_id"].iloc[0])
        rho = float(split_df["correlation_level"].iloc[0])

        if "feature_index" not in split_df.columns:
            raise RuntimeError("Split dataframe missing feature_index")

        y = pd.factorize(split_df["label"])[0].astype(int)
        feat_idx = split_df["feature_index"].to_numpy(dtype=int)

        train_mask = (split_df["partition"] == "train").to_numpy()
        id_mask = (split_df["partition"] == "id_test").to_numpy()
        ood_mask = (split_df["partition"] == "ood_test").to_numpy()

        if train_mask.sum() == 0 or id_mask.sum() == 0 or ood_mask.sum() == 0:
            raise RuntimeError(f"{dataset_name}: split {split_id} rep {rep} missing partitions")

        x_train = features[feat_idx[train_mask]]
        y_train = y[train_mask]
        x_id = features[feat_idx[id_mask]]
        y_id = y[id_mask]
        x_ood = features[feat_idx[ood_mask]]
        y_ood = y[ood_mask]

        clf = _train_linear_probe(x_train, y_train, int(seed) + rep * 100 + split_id)

        acc_id = float(accuracy_score(y_id, clf.predict(x_id)))
        acc_ood = float(accuracy_score(y_ood, clf.predict(x_ood)))

        id_by_rep.setdefault(rep, {})[split_id] = acc_id
        ood_by_rep.setdefault(rep, {})[split_id] = acc_ood
        id_by_rho.setdefault(rho, []).append(acc_id)
        ood_by_rho.setdefault(rho, []).append(acc_ood)

    apd_id: list[float] = []
    apd_ood: list[float] = []
    apd_avg: list[float] = []
    reps = sorted(set(id_by_rep).intersection(ood_by_rep))
    for rep in reps:
        rep_apd_id = _apd_from_split_acc(id_by_rep[rep])
        rep_apd_ood = _apd_from_split_acc(ood_by_rep[rep])
        apd_id.append(rep_apd_id)
        apd_ood.append(rep_apd_ood)
        apd_avg.append((rep_apd_id + rep_apd_ood) / 2.0)

    apd_id_arr = np.asarray(apd_id, dtype=float)
    apd_ood_arr = np.asarray(apd_ood, dtype=float)
    apd_avg_arr = np.asarray(apd_avg, dtype=float)

    acc_id_by_rho = {
        float(rho): (float(np.mean(vals)), float(np.std(vals, ddof=0)))
        for rho, vals in sorted(id_by_rho.items(), key=lambda kv: kv[0])
    }
    acc_ood_by_rho = {
        float(rho): (float(np.mean(vals)), float(np.std(vals, ddof=0)))
        for rho, vals in sorted(ood_by_rho.items(), key=lambda kv: kv[0])
    }

    return APDResult(
        dataset=str(dataset_name),
        apd_id=float(apd_id_arr.mean() if len(apd_id_arr) else 0.0),
        apd_ood=float(apd_ood_arr.mean() if len(apd_ood_arr) else 0.0),
        apd_avg=float(apd_avg_arr.mean() if len(apd_avg_arr) else 0.0),
        apd_id_std=float(apd_id_arr.std(ddof=0) if len(apd_id_arr) else 0.0),
        apd_ood_std=float(apd_ood_arr.std(ddof=0) if len(apd_ood_arr) else 0.0),
        apd_avg_std=float(apd_avg_arr.std(ddof=0) if len(apd_avg_arr) else 0.0),
        acc_id_by_rho=acc_id_by_rho,
        acc_ood_by_rho=acc_ood_by_rho,
    )

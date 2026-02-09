from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .allocations import get_paper_allocations, get_paper_v_levels, scale_allocation

logger = logging.getLogger("ijepath")
EPS = 1e-12


def cramers_v_from_counts(counts: np.ndarray) -> float:
    arr = np.asarray(counts, dtype=float)
    if arr.ndim != 2:
        raise ValueError("counts must be 2D")
    n = float(arr.sum())
    if n <= 0:
        return 0.0

    row = arr.sum(axis=1, keepdims=True)
    col = arr.sum(axis=0, keepdims=True)
    expected = row @ col / n
    mask = expected > 0
    chi2 = (((arr - expected) ** 2)[mask] / expected[mask]).sum()

    r, k = arr.shape
    denom = n * max(min(r - 1, k - 1), 1)
    return float(np.sqrt(max(float(chi2) / max(denom, EPS), 0.0)))


def _matrix_from_df(df: pd.DataFrame, labels: list[str], centers: list[str]) -> np.ndarray:
    mat = np.zeros((len(labels), len(centers)), dtype=int)
    for i, lbl in enumerate(labels):
        for j, ctr in enumerate(centers):
            mat[i, j] = int(((df["label"] == lbl) & (df["medical_center"] == ctr)).sum())
    return mat


def _choose_train_id_slides(
    id_df: pd.DataFrame,
    labels: list[str],
    centers: list[str],
    id_test_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(random_state))

    train_parts = []
    id_test_parts = []
    for lbl in labels:
        for ctr in centers:
            cell = id_df[(id_df["label"] == lbl) & (id_df["medical_center"] == ctr)]
            slides = sorted(cell["slide_id"].unique().tolist())
            if not slides:
                continue

            n_test = max(1, int(round(len(slides) * float(id_test_fraction)))) if len(slides) > 1 else 0
            if n_test >= len(slides):
                n_test = len(slides) - 1
            test_slides = set(rng.choice(slides, size=n_test, replace=False).tolist()) if n_test > 0 else set()
            train_slides = set(slides) - test_slides

            train_parts.append(cell[cell["slide_id"].isin(train_slides)])
            id_test_parts.append(cell[cell["slide_id"].isin(test_slides)])

    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True) if train_parts else id_df.iloc[0:0].copy()
    id_test_df = pd.concat(id_test_parts, axis=0).reset_index(drop=True) if id_test_parts else id_df.iloc[0:0].copy()

    train_slides = set(train_df["slide_id"].unique().tolist())
    id_slides = set(id_test_df["slide_id"].unique().tolist())
    if train_slides.intersection(id_slides):
        raise RuntimeError("Slide leakage detected between train and id_test")

    return train_df, id_test_df


def _sample_cell_rows(cell_df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n <= 0:
        return cell_df.iloc[0:0].copy()
    if len(cell_df) < n:
        raise RuntimeError(f"Not enough samples in cell: need {n}, have {len(cell_df)}")
    idx = rng.choice(cell_df.index.to_numpy(), size=int(n), replace=False)
    return cell_df.loc[idx]


def _sample_from_matrix(
    train_pool: pd.DataFrame,
    labels: list[str],
    centers: list[str],
    matrix: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    parts = []
    for i, lbl in enumerate(labels):
        for j, ctr in enumerate(centers):
            n = int(matrix[i, j])
            cell = train_pool[(train_pool["label"] == lbl) & (train_pool["medical_center"] == ctr)]
            parts.append(_sample_cell_rows(cell, n, rng))
    return pd.concat(parts, axis=0).reset_index(drop=True) if parts else train_pool.iloc[0:0].copy()


def _build_custom_allocation(
    avail: np.ndarray,
    labels: list[str],
    centers: list[str],
    rho: float,
    target_train_total: int,
) -> np.ndarray:
    base = int(target_train_total // (len(labels) * len(centers)))
    row_totals = np.full(len(labels), base * len(centers), dtype=int)
    col_totals = np.full(len(centers), base * len(labels), dtype=int)
    uniform = np.full((len(labels), len(centers)), base, dtype=float)

    # high-association matrix in 2x2 style by assigning each class to one preferred center
    preferred_cols = np.asarray([i % len(centers) for i in range(len(labels))], dtype=int)
    max_assoc = np.zeros_like(avail, dtype=float)
    row_left = row_totals.astype(float).copy()
    col_left = col_totals.astype(float).copy()
    cap = avail.astype(float).copy()
    for i in range(len(labels)):
        order = [int(preferred_cols[i])] + [j for j in range(len(centers)) if j != int(preferred_cols[i])]
        for j in order:
            if row_left[i] <= 0:
                break
            take = min(row_left[i], col_left[j], cap[i, j])
            if take > 0:
                max_assoc[i, j] += take
                row_left[i] -= take
                col_left[j] -= take
                cap[i, j] -= take

    target = (1.0 - float(rho)) * uniform + float(rho) * max_assoc
    mat = np.floor(target).astype(int)
    mat = np.minimum(mat, avail)

    # reconcile totals using available slack
    deficit = int(target_train_total - mat.sum())
    if deficit > 0:
        slack = avail - mat
        for _ in range(deficit):
            idx = np.unravel_index(np.argmax(slack), slack.shape)
            if slack[idx] <= 0:
                break
            mat[idx] += 1
            slack[idx] -= 1
    elif deficit < 0:
        for _ in range(-deficit):
            idx = np.unravel_index(np.argmax(mat), mat.shape)
            if mat[idx] <= 0:
                break
            mat[idx] -= 1

    return mat


def generate_apd_splits(
    df: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    repetitions: int,
    correlation_levels: Sequence[float],
    id_centers: Sequence[str],
    ood_centers: Sequence[str],
    id_test_fraction: float,
    seed: int,
    mode: Literal["custom", "paper"] = "paper",
) -> list[pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(df["label"].unique().tolist())
    centers = sorted([c for c in id_centers if c in set(df["medical_center"].unique().tolist())])
    if not labels or not centers:
        raise ValueError(f"{dataset_name}: empty labels/centers for APD")

    id_df = df[df["medical_center"].isin(centers)].copy()
    ood_df = df[df["medical_center"].isin(set(ood_centers))].copy()
    if len(ood_df) == 0:
        raise ValueError(f"{dataset_name}: OOD set is empty")

    paper_allocations = None
    if mode == "paper":
        try:
            paper_allocations = get_paper_allocations(dataset_name)
            correlation_levels = get_paper_v_levels(dataset_name)
            logger.info("[APD] Using paper allocations for %s", dataset_name)
        except ValueError:
            mode = "custom"

    all_splits: list[pd.DataFrame] = []

    for rep in range(int(repetitions)):
        rep_rng = np.random.default_rng(int(seed) + rep)

        train_pool, id_test_df = _choose_train_id_slides(
            id_df=id_df,
            labels=labels,
            centers=centers,
            id_test_fraction=float(id_test_fraction),
            random_state=int(seed) + rep,
        )

        rep_ood_df = ood_df.copy().reset_index(drop=True)
        avail = _matrix_from_df(train_pool, labels, centers)
        if (avail <= 0).any():
            raise RuntimeError(
                f"{dataset_name}: some train pool class-center cells are empty; cannot build APD splits"
            )

        if mode == "paper" and paper_allocations is not None:
            paper_base = paper_allocations[0.0]
            paper_total = int(paper_base.sum())
            scale_factors = avail / np.maximum(paper_base, 1)
            scale_factors[paper_base == 0] = np.inf
            max_scale = float(scale_factors.min())
            target_train_total = min(int(round(paper_total * max_scale)), int(avail.sum()))
            target_train_total = max(target_train_total, len(labels) * len(centers))
        else:
            base = max(1, int(avail.min()) // 2)
            target_train_total = int(base * len(labels) * len(centers))

        for split_idx, rho in enumerate(correlation_levels):
            rho = float(rho)
            if mode == "paper" and paper_allocations is not None:
                paper_alloc = paper_allocations.get(rho)
                if paper_alloc is None:
                    # nearest available for non-paper level request
                    nearest = min(paper_allocations.keys(), key=lambda x: abs(float(x) - rho))
                    paper_alloc = paper_allocations[nearest]
                split_matrix = scale_allocation(paper_alloc, target_train_total)
                split_matrix = np.minimum(split_matrix, avail)
                # distribute missing counts due clipping
                deficit = int(target_train_total - split_matrix.sum())
                if deficit > 0:
                    slack = avail - split_matrix
                    for _ in range(deficit):
                        idx = np.unravel_index(np.argmax(slack), slack.shape)
                        if slack[idx] <= 0:
                            break
                        split_matrix[idx] += 1
                        slack[idx] -= 1
            else:
                split_matrix = _build_custom_allocation(
                    avail=avail,
                    labels=labels,
                    centers=centers,
                    rho=rho,
                    target_train_total=target_train_total,
                )

            train_df = _sample_from_matrix(train_pool, labels, centers, split_matrix, rep_rng)

            train_part = train_df.copy()
            train_part["partition"] = "train"
            id_part = id_test_df.copy()
            id_part["partition"] = "id_test"
            ood_part = rep_ood_df.copy()
            ood_part["partition"] = "ood_test"

            merged = pd.concat([train_part, id_part, ood_part], axis=0).reset_index(drop=True)
            merged["rep"] = int(rep)
            merged["split_id"] = int(split_idx + 1)
            merged["correlation_level"] = float(rho)
            merged["cramers_v_target"] = float(rho)
            merged["dataset"] = str(dataset_name)

            train_ct = pd.crosstab(
                merged[merged["partition"] == "train"]["label"],
                merged[merged["partition"] == "train"]["medical_center"],
            )
            merged["cramers_v_realized"] = float(cramers_v_from_counts(train_ct.to_numpy()))

            rep_dir = output_dir / str(dataset_name) / f"rep_{rep:02d}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            merged.to_csv(rep_dir / f"split_{split_idx+1:02d}.csv", index=False)

            all_splits.append(merged)

    return all_splits

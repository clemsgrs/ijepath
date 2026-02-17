"""Canonical I-JEPA mask visualization utilities.

This module keeps plotting logic out of the notebook so the notebook stays readable
while preserving a versioned source of truth for figure generation.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from stagec_plot_utils import token_ids_to_map  # reuse from cross-res utils

ENC_COLOR = "#1dace4"  # green for encoder-kept tokens
PRED_COLORS = ["#ef4444", "#f97316", "#eab308", "#8b5cf6"]  # red/orange/yellow/purple


def draw_mask_grid_on_image(
    ax,
    image_rgb: np.ndarray,
    token_map: np.ndarray,
    grid_h: int,
    grid_w: int,
    color: str,
    alpha: float = 0.5,
) -> None:
    """Overlay colored semi-transparent rectangles on each active token cell."""
    h_img, w_img = image_rgb.shape[:2]
    cell_h = h_img / grid_h
    cell_w = w_img / grid_w

    for row in range(grid_h):
        for col in range(grid_w):
            if token_map[row, col] > 0.5:
                ax.add_patch(
                    plt.Rectangle(
                        (col * cell_w, row * cell_h),
                        cell_w,
                        cell_h,
                        fill=True,
                        facecolor=color,
                        edgecolor="none",
                        alpha=alpha,
                    )
                )


def plot_canonical_masks(
    image_rgb: np.ndarray,
    masks_enc: list,
    masks_pred: list,
    grid_h: int,
    grid_w: int,
    sample_idx: int = 0,
):
    """Return a figure with 1 + num_pred_masks subplots.

    Panel 0: image + encoder-keep overlay (green)
    Panels 1..k: image + predictor-mask overlay per pred mask
    """
    num_pred = len(masks_pred)
    n_panels = 1 + num_pred

    enc_map = token_ids_to_map(masks_enc[0][sample_idx], grid_h=grid_h, grid_w=grid_w)
    pred_maps = [
        token_ids_to_map(pm[sample_idx], grid_h=grid_h, grid_w=grid_w)
        for pm in masks_pred
    ]

    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    enc_count = int(enc_map.sum())
    axes[0].imshow(image_rgb)
    draw_mask_grid_on_image(axes[0], image_rgb, enc_map, grid_h, grid_w, color=ENC_COLOR, alpha=0.4)
    axes[0].set_title(f"Encoder kept ({enc_count}/{grid_h * grid_w} tokens)", fontsize=10)
    axes[0].axis("off")

    for i, pred_map in enumerate(pred_maps):
        ax = axes[i + 1]
        color = PRED_COLORS[i % len(PRED_COLORS)]
        pred_count = int(pred_map.sum())
        ax.imshow(image_rgb)
        draw_mask_grid_on_image(ax, image_rgb, pred_map, grid_h, grid_w, color=color, alpha=0.5)
        ax.set_title(f"Pred mask {i + 1} ({pred_count} tokens)", fontsize=10)
        ax.axis("off")

    fig.tight_layout()
    return fig

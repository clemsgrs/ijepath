"""Canonical I-JEPA mask visualization utilities.

This module keeps plotting logic out of the notebook so the notebook stays readable
while preserving a versioned source of truth for figure generation.
"""

from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from ijepath.datasets.cross_resolution_wsi_dataset import IMAGENET_MEAN, IMAGENET_STD
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tensor_to_rgb_uint8(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().numpy().transpose(1, 2, 0)
    arr = (arr * IMAGENET_STD[None, None, :]) + IMAGENET_MEAN[None, None, :]
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


# ── High-level figure builders ────────────────────────────────────────────────

def visualize_sample(
    sample: dict,
    anchor_data_by_id: dict[str, dict],
    wsi_l0_mpp_by_slide: dict[str, float],
    source_tile_size_px: int,
    input_mpp: float,
    downsample: int,
    spacing_tolerance: float,
    backend: str,
    load_fn: Callable,
    record_check_fn: Callable | None = None,
) -> plt.Figure:
    """Build a 3-panel figure: source tile + mask, model input, augmentation summary.

    If *record_check_fn* is provided, QC checks for tissue fraction and tile size
    are recorded (results land in the notebook's QC_RESULTS list via the callback).
    Returns the figure; caller is responsible for saving it.
    """
    img_rgb = _tensor_to_rgb_uint8(sample['image'])
    meta = sample['sample_metadata']
    anchor_row = anchor_data_by_id[meta['anchor_id']]

    source_tile, mask_tile, tissue_pct = load_fn(
        wsi_path=str(anchor_row['wsi_path']),
        mask_path=str(anchor_row['mask_path']),
        center_x_l0=int(anchor_row['center_x_level0']),
        center_y_l0=int(anchor_row['center_y_level0']),
        source_tile_size_px=source_tile_size_px,
        input_mpp=input_mpp,
        wsi_l0_mpp=wsi_l0_mpp_by_slide[anchor_row['slide_id']],
        downsample=downsample,
        backend=backend,
        spacing_tolerance=spacing_tolerance,
    )

    if record_check_fn is not None:
        record_check_fn(
            'Anchor tissue fraction matches readback',
            tissue_pct == float(anchor_row['tissue_fraction']),
            f'{tissue_pct:.4f} vs expected {anchor_row["tissue_fraction"]:.4f}',
        )
        record_check_fn(
            'Source tile size correct',
            source_tile.size == (source_tile_size_px, source_tile_size_px),
            f'{source_tile.size} vs expected {(source_tile_size_px, source_tile_size_px)}',
        )

    mask_arr = np.array(mask_tile)
    green_rgba = np.zeros((*source_tile.size, 4), dtype=np.uint8)
    green_rgba[mask_arr > 0] = [0, 200, 50, 100]

    req_sz = int(meta['source_tile_size_px_requested'])
    req_mpp = float(meta['requested_input_mpp'])
    src_sz = int(meta['source_tile_size_px_at_effective_spacing'])
    src_mpp = float(meta['source_input_mpp'])
    model_sz = int(meta['model_crop_size_px'])
    crop_min = float(meta['crop_scale_min'])
    crop_max = float(meta['crop_scale_max'])
    anchor_tf_pct = float(anchor_row['tissue_fraction']) * 100.0
    anchor_id_short = meta['anchor_id'].rsplit('_', 1)[-1]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(source_tile)
    axes[0].imshow(green_rgba)
    axes[0].set_title(f'source tile + tissue mask\n{req_sz}px', fontsize=9)
    axes[0].axis('off')

    axes[1].imshow(img_rgb)
    axes[1].set_title(f'model input\n{model_sz}px', fontsize=9)
    axes[1].axis('off')

    axes[2].axis('off')
    summary = '\n'.join([
        '─── info ───────────────────',
        f"  slide:    {meta['slide_id']}",
        f"  anchor:   {anchor_id_short}",
        f"  request:  {req_sz}px @ {req_mpp:.2f} mpp",
        f"  read:     {src_sz}px @ {src_mpp:.3f} mpp → {req_sz}px",
        f"  tissue:   {anchor_tf_pct:.1f}%",
        '',
        '─── augmentations ─────────────',
        f"  crop:          RandomResizedCrop(",
        f"                   size={model_sz}px,",
        f"                   scale=({crop_min:.1f}, {crop_max:.1f}),",
        f"                 )",
        f"  hflip:         {'on' if meta['use_horizontal_flip'] else 'off'}"
        f"  (p={float(meta['horizontal_flip_prob']):.2f})",
        f"  color jitter:  {'on' if meta['use_color_distortion'] else 'off'}"
        + (f"  (s={float(meta['color_jitter_strength']):.2f})" if meta['use_color_distortion'] else ''),
        f"  gaussian blur: {'on' if meta['use_gaussian_blur'] else 'off'}",
    ])
    axes[2].text(0.02, 0.98, summary, va='top', ha='left', fontsize=9,
                 family='monospace', transform=axes[2].transAxes)

    fig.tight_layout()
    return fig


def visualize_batch(
    batch_data: dict,
    masks_enc: list,
    masks_pred: list,
    anchor_data_by_id: dict[str, dict],
    wsi_l0_mpp_by_slide: dict[str, float],
    source_tile_size_px: int,
    input_mpp: float,
    patch_size: int,
    downsample: int,
    spacing_tolerance: float,
    backend: str,
    load_fn: Callable,
    record_check_fn: Callable | None = None,
) -> plt.Figure:
    """Build a grid figure: rows=batch samples, cols=anchor + raw + enc_keep + pred_masks.

    If *record_check_fn* is provided, QC checks for tissue fraction and tile size
    are recorded for each sample in the batch.
    Returns the figure; caller is responsible for saving it.
    """
    images = batch_data['image']
    batch_size = images.shape[0]
    num_pred_masks = len(masks_pred)
    crop_size_px = images.shape[-1]
    grid_h = crop_size_px // patch_size
    grid_w = crop_size_px // patch_size

    n_cols = 1 + 1 + len(masks_enc) + num_pred_masks  # anchor + raw + enc_keep + pred masks
    fig, axes = plt.subplots(batch_size, n_cols, figsize=(3.0 * n_cols, 3.0 * batch_size), squeeze=False)

    for b in range(batch_size):
        img_rgb = _tensor_to_rgb_uint8(images[b])
        meta = batch_data['sample_metadata'][b]
        slide_id = meta['slide_id']
        enc_map = token_ids_to_map(masks_enc[0][b], grid_h=grid_h, grid_w=grid_w)
        pred_maps = [token_ids_to_map(pm[b], grid_h=grid_h, grid_w=grid_w) for pm in masks_pred]

        anchor_row = anchor_data_by_id[meta['anchor_id']]
        anchor_tf = float(anchor_row['tissue_fraction'])
        source_tile, mask_tile, tissue_pct = load_fn(
            wsi_path=str(anchor_row['wsi_path']),
            mask_path=str(anchor_row['mask_path']),
            center_x_l0=int(anchor_row['center_x_level0']),
            center_y_l0=int(anchor_row['center_y_level0']),
            source_tile_size_px=source_tile_size_px,
            input_mpp=input_mpp,
            wsi_l0_mpp=wsi_l0_mpp_by_slide[slide_id],
            downsample=downsample,
            backend=backend,
            spacing_tolerance=spacing_tolerance,
        )

        if record_check_fn is not None:
            record_check_fn(
                'Anchor tissue fraction matches readback',
                tissue_pct == float(anchor_row['tissue_fraction']),
                f'{tissue_pct:.4f} vs expected {anchor_row["tissue_fraction"]:.4f}',
            )
            record_check_fn(
                'Source tile size correct',
                source_tile.size == (source_tile_size_px, source_tile_size_px),
                f'{source_tile.size} vs expected {(source_tile_size_px, source_tile_size_px)}',
            )

        mask_arr = np.array(mask_tile)
        green_rgba = np.zeros((*source_tile.size, 4), dtype=np.uint8)
        green_rgba[mask_arr > 0] = [0, 200, 50, 100]

        col = 0

        # Anchor: source tile + tissue mask
        axes[b, col].imshow(source_tile)
        axes[b, col].imshow(green_rgba)
        axes[b, col].set_title(f"anchor\ntissue: {anchor_tf * 100:.1f}%", fontsize=7)
        axes[b, col].axis('off')
        col += 1

        # Raw model input
        axes[b, col].imshow(img_rgb)
        axes[b, col].set_title(f"sample {b}\n{slide_id}", fontsize=7)
        axes[b, col].axis('off')
        col += 1

        # Encoder keep
        axes[b, col].imshow(img_rgb)
        draw_mask_grid_on_image(axes[b, col], img_rgb, enc_map, grid_h, grid_w, color=ENC_COLOR, alpha=0.4)
        axes[b, col].set_title(f"enc kept\n({int(enc_map.sum())}/{grid_h * grid_w})", fontsize=7)
        axes[b, col].axis('off')
        col += 1

        # Predictor masks
        for i, pred_map in enumerate(pred_maps):
            color = PRED_COLORS[i % len(PRED_COLORS)]
            axes[b, col].imshow(img_rgb)
            draw_mask_grid_on_image(axes[b, col], img_rgb, pred_map, grid_h, grid_w, color=color, alpha=0.5)
            axes[b, col].set_title(f"pred {i + 1}\n({int(pred_map.sum())} tokens)", fontsize=7)
            axes[b, col].axis('off')
            col += 1

    fig.suptitle(
        f'Batch overview  ({batch_size} samples × anchor + raw + enc_keep + {num_pred_masks} pred masks)',
        fontsize=10,
    )
    # Reserve a fixed ~0.3-inch strip at the top for the suptitle regardless of batch size.
    # tight_layout alone doesn't account for suptitle, so it overlaps at large batch sizes.
    suptitle_fraction = 0.3 / (3.0 * batch_size)
    fig.tight_layout(rect=[0, 0, 1, 1 - suptitle_fraction])
    return fig

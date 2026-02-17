"""WSI I/O and lookup helpers for the canonical I-JEPA QC notebook.

Extracted from the big helper cell so the notebook stays readable
while the implementation lives in a versioned Python file.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import hs2p
from PIL import Image
import pyarrow.parquet as pq

from ijepath.datasets.cross_resolution_wsi_dataset import IMAGENET_MEAN, IMAGENET_STD


class SegmentationParameters(NamedTuple):
    """Parameters for filtering contours."""

    downsample: int   # downsample factor for loading segmentation mask
    sthresh: int      # segmentation threshold (higher → less foreground)
    sthresh_up: int   # upper threshold for scaling the binary mask
    mthresh: int      # median filter size (positive, odd integer)
    close: int        # additional morphological closing after thresholding
    use_otsu: bool    # whether to use Otsu's method for thresholding
    use_hsv: bool     # whether to use HSV thresholding


class SamplingParameters(NamedTuple):
    """Parameters for sampling."""

    pixel_mapping: dict[str, int]           # annotation name → pixel value
    color_mapping: dict[str, list[int] | None]  # annotation name → RGB color
    tissue_percentage: dict[str, float | None]  # minimum tile coverage per category


_SEG_PARAMS = SegmentationParameters(
    downsample=16,
    sthresh=15,
    sthresh_up=255,
    mthresh=5,
    close=5,
    use_otsu=False,
    use_hsv=False,
)

_SAMPLING_PARAMS = SamplingParameters(
    pixel_mapping={'background': 0, 'tissue': 1},
    color_mapping={'background': [255, 255, 255], 'tissue': [255, 0, 0]},
    tissue_percentage={'background': None, 'tissue': 0.5},
)


def load_source_tile_and_mask(
    wsi_path: str,
    mask_path: str,
    center_x_l0: int,
    center_y_l0: int,
    source_tile_size_px: int,
    input_mpp: float,
    wsi_l0_mpp: float,
    downsample: int,
    backend: str = 'openslide',
    spacing_tolerance: float = 0.05,
) -> tuple[Image.Image, Image.Image, float]:
    """Read source tile (RGB) and binary tissue mask at the anchor center.

    Returns (tile_rgb, binary_mask, tissue_pct):
    - tile_rgb: PIL Image (RGB), shape (source_tile_size_px, source_tile_size_px)
    - binary_mask: PIL Image (L), 1 where tissue, 0 elsewhere
    - tissue_pct: fraction of mask pixels that are tissue

    Tissue fraction
    ---------------
    Computed on the same downsampled mask level used by the anchor catalog
    (selected via the profile's `downsample` factor), so readback values
    agree with the stored tissue_fraction within rounding of the coarse patch.

    Coordinate handling
    -------------------
    The mask center is derived from center_x/y_l0 (not from the top-left corner),
    matching the anchor catalog's center-anchored patch extraction. Computing
    top-left from x0_wsi instead would introduce a ~1-pixel rounding error.
    """
    # ── WSI RGB tile ──────────────────────────────────────────────────────────
    tile_l0_px = int(round(source_tile_size_px * input_mpp / wsi_l0_mpp))
    x0_wsi = center_x_l0 - tile_l0_px // 2
    y0_wsi = center_y_l0 - tile_l0_px // 2

    wsi = hs2p.wsi.WholeSlideImage(
        path=Path(wsi_path),
        mask_path=Path(mask_path),
        backend=backend,
        segment_params=_SEG_PARAMS,
        sampling_params=_SAMPLING_PARAMS,
    )
    tile_level, is_within_tolerance = wsi.get_best_level_for_spacing(
        target_spacing=input_mpp,
        tolerance=spacing_tolerance,
    )
    effective_spacing = wsi.get_level_spacing(tile_level)
    tile_arr = wsi.get_tile(
        x0_wsi,
        y0_wsi,
        tile_l0_px,
        tile_l0_px,
        spacing=effective_spacing,
    )
    tile = Image.fromarray(tile_arr).convert("RGB")
    if source_tile_size_px != tile.size[0]:
        tile = tile.resize((source_tile_size_px, source_tile_size_px))

    # ── Tissue mask ───────────────────────────────────────────────────────────
    mask = hs2p.wsi.WholeSlideImage(
        path=Path(mask_path),
        backend=backend,
        segment=False,
    )
    mask_spacing0 = mask.get_level_spacing(0)

    # Compute mask-level-0 center from WSI-level-0 center (not from top-left x0_wsi).
    # This matches the anchor catalog, which iterates in mask coords and extracts
    # center-anchored patches. Computing from x0_wsi introduces a rounding error
    # (round(x0_wsi * r) ≠ round(center * r) - half) that shifts the region by ~1 px.
    cx_mask_l0 = int(round(center_x_l0 * wsi_l0_mpp / mask_spacing0))
    cy_mask_l0 = int(round(center_y_l0 * wsi_l0_mpp / mask_spacing0))

    # Pick the same downsampled mask level as the anchor catalog (profile downsample param).
    downsample_level = wsi.get_best_level_for_downsample_custom(downsample)
    downsample_spacing = wsi.get_level_spacing(downsample_level)
    mask_level, _ = mask.get_best_level_for_spacing(downsample_spacing, tolerance=spacing_tolerance)
    mask_spacing = mask.spacings[mask_level]

    # Patch size at the chosen mask level (mirrors catalog: round(context_fov_um / mask_spacing)).
    context_fov_um = source_tile_size_px * input_mpp
    context_px_at_spacing = max(1, int(round(context_fov_um / mask_spacing)))
    # Top-left in mask-level-0, derived from center (not from x0_wsi) to match catalog.
    half_l0 = int(round(context_px_at_spacing * mask_spacing / mask_spacing0)) // 2
    mask_tile_arr = mask.get_tile(
        cx_mask_l0 - half_l0,
        cy_mask_l0 - half_l0,
        context_px_at_spacing,
        context_px_at_spacing,
        spacing=mask_spacing,
    )
    if mask_tile_arr.ndim == 3:
        mask_tile_arr = mask_tile_arr[..., 0]
    tissue_pct = (mask_tile_arr > 0).sum() / mask_tile_arr.size

    # Visualization mask: full source-tile footprint at mask level-0.
    mask_tile_px_l0 = int(round(tile_l0_px * wsi_l0_mpp / mask_spacing0))
    mask_tile_arr_viz = mask.get_tile(
        cx_mask_l0 - mask_tile_px_l0 // 2,
        cy_mask_l0 - mask_tile_px_l0 // 2,
        mask_tile_px_l0,
        mask_tile_px_l0,
        spacing=mask_spacing0,
    )
    if mask_tile_arr_viz.ndim == 3:
        mask_tile_arr_viz = mask_tile_arr_viz[..., 0]
    mask_tile = Image.fromarray(mask_tile_arr_viz).convert("L")
    if source_tile_size_px != mask_tile.size[0]:
        mask_tile = mask_tile.resize((source_tile_size_px, source_tile_size_px), resample=Image.NEAREST)

    return tile, mask_tile, tissue_pct


def compute_tissue_token_map(
    img_tensor: torch.Tensor, grid_h: int, grid_w: int, patch_size: int
) -> np.ndarray:
    """Per-token tissue proxy (inverted luminance): white background → 0, stained tissue → 1."""
    img_f = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img_f = img_f * IMAGENET_STD + IMAGENET_MEAN
    img_f = np.clip(img_f, 0.0, 1.0)
    luminance = 0.299 * img_f[..., 0] + 0.587 * img_f[..., 1] + 0.114 * img_f[..., 2]
    tissue_proxy = (1.0 - luminance).astype(np.float32)
    token_tissue = np.zeros((grid_h, grid_w), dtype=np.float32)
    for gy in range(grid_h):
        for gx in range(grid_w):
            py0, px0 = gy * patch_size, gx * patch_size
            token_tissue[gy, gx] = tissue_proxy[py0:py0 + patch_size, px0:px0 + patch_size].mean()
    return token_tissue


def build_slide_lookups(slide_meta_rows: list[dict]) -> tuple[dict, dict, dict, dict]:
    """Build per-slide lookup dicts from slide metadata rows.

    Returns (wsi_l0_mpp_by_slide, mask_spacings_by_slide, mask_scale_x_by_slide, mask_scale_y_by_slide).
    """
    wsi_l0_mpp_by_slide = {str(r['slide_id']): float(r['wsi_level0_spacing_mpp']) for r in slide_meta_rows}
    mask_spacings_by_slide = {
        str(r['slide_id']): [float(s) for s in r['mask_spacings_mpp']]
        for r in slide_meta_rows
    }
    mask_scale_x_by_slide = {str(r['slide_id']): float(r['mask_to_wsi_scale_x']) for r in slide_meta_rows}
    mask_scale_y_by_slide = {str(r['slide_id']): float(r['mask_to_wsi_scale_y']) for r in slide_meta_rows}
    return wsi_l0_mpp_by_slide, mask_spacings_by_slide, mask_scale_x_by_slide, mask_scale_y_by_slide


def load_anchor_index(anchor_shards: list[dict]) -> dict[str, dict]:
    """Load anchor rows from all shards and index them by anchor_id."""
    anchor_data_by_id: dict[str, dict] = {}
    for shard_info in anchor_shards:
        tbl = pq.read_table(
            str(Path(shard_info['path'])),
            columns=[
                'anchor_id', 'slide_id', 'wsi_path', 'mask_path',
                'center_x_level0', 'center_y_level0', 'tissue_fraction',
            ],
        )
        for row in tbl.to_pylist():
            anchor_data_by_id[str(row['anchor_id'])] = row
    return anchor_data_by_id

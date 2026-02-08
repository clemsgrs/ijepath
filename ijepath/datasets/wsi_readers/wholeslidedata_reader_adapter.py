import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import wholeslidedata as wsd


def spacing_pixels_to_level0_pixels(
    size_pixels_at_spacing: int,
    spacing: float,
    spacing_at_level0: float,
) -> int:
    """Convert pixel size measured at `spacing` to level-0 pixel size."""
    if spacing_at_level0 <= 0:
        raise ValueError("spacing_at_level0 must be positive")
    return int(round(float(size_pixels_at_spacing) * float(spacing) / float(spacing_at_level0)))


def level0_pixels_to_spacing_pixels(
    size_pixels_at_level0: int,
    spacing: float,
    spacing_at_level0: float,
) -> int:
    """Convert level-0 pixel size to pixel size measured at `spacing`."""
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    return int(round(float(size_pixels_at_level0) * float(spacing_at_level0) / float(spacing)))


def context_box_from_center_level0(
    center_x_level0: int,
    center_y_level0: int,
    size_pixels_at_spacing: int,
    spacing: float,
    spacing_at_level0: float,
) -> Tuple[int, int, int, int]:
    """Return an axis-aligned level-0 box [x0, y0, x1, y1] centered at (x,y)."""
    size_level0 = spacing_pixels_to_level0_pixels(
        size_pixels_at_spacing=size_pixels_at_spacing,
        spacing=spacing,
        spacing_at_level0=spacing_at_level0,
    )
    half = size_level0 // 2
    x0 = int(round(center_x_level0)) - half
    y0 = int(round(center_y_level0)) - half
    x1 = x0 + size_level0
    y1 = y0 + size_level0
    return x0, y0, x1, y1


@dataclass(frozen=True)
class SlideGeometry:
    spacing_at_level0_mpp: float
    level0_width: int
    level0_height: int


class WholeSlideDataReaderAdapter:
    """Thin adapter around `wholeslidedata` with explicit geometry helpers."""

    def __init__(
        self,
        wsi_path: str,
        mask_path: Optional[str] = None,
        backend: str = "openslide",
    ) -> None:
        self.wsi_path = str(wsi_path)
        self.mask_path = None if mask_path is None else str(mask_path)
        self.backend = backend

        self.wsi = wsd.WholeSlideImage(Path(self.wsi_path), backend=self.backend)
        self.mask = None
        if self.mask_path:
            self.mask = wsd.WholeSlideImage(Path(self.mask_path), backend=self.backend)

    @property
    def wsi_spacings(self):
        return list(self.wsi.spacings)

    @property
    def wsi_shapes(self):
        return list(self.wsi.shapes)

    @property
    def mask_spacings(self):
        return None if self.mask is None else list(self.mask.spacings)

    @property
    def mask_shapes(self):
        return None if self.mask is None else list(self.mask.shapes)

    @property
    def geometry(self) -> SlideGeometry:
        w0, h0 = self.wsi.shapes[0]
        return SlideGeometry(
            spacing_at_level0_mpp=float(self.wsi.spacings[0]),
            level0_width=int(w0),
            level0_height=int(h0),
        )

    def get_best_level_for_spacing(
        self,
        target_spacing_mpp: float,
        tolerance: float,
        use_mask: bool = False,
    ) -> Tuple[int, bool]:
        spacings = self.mask_spacings if use_mask else self.wsi_spacings
        if spacings is None:
            raise ValueError("Mask is not available")
        if target_spacing_mpp <= 0:
            raise ValueError("target_spacing_mpp must be > 0")

        # Prefer the coarsest level that does not require upsampling.
        candidate = 0
        for idx, spacing in enumerate(spacings):
            if spacing <= target_spacing_mpp:
                candidate = idx
            else:
                break

        best_spacing = float(spacings[candidate])
        within_tolerance = abs(best_spacing - target_spacing_mpp) / target_spacing_mpp <= tolerance
        if within_tolerance:
            return candidate, True

        # Fall back to nearest spacing if no level matches the tolerance.
        nearest = min(range(len(spacings)), key=lambda i: abs(float(spacings[i]) - target_spacing_mpp))
        nearest_spacing = float(spacings[nearest])
        within_tolerance = abs(nearest_spacing - target_spacing_mpp) / target_spacing_mpp <= tolerance
        return int(nearest), bool(within_tolerance)

    def _ensure_patch_shape(self, patch: np.ndarray, width: int, height: int) -> np.ndarray:
        if patch.ndim == 2:
            patch = patch[..., None]
        if patch.shape[0] == height and patch.shape[1] == width:
            return patch

        out = np.zeros((height, width, patch.shape[2]), dtype=patch.dtype)
        h = min(height, patch.shape[0])
        w = min(width, patch.shape[1])
        out[:h, :w] = patch[:h, :w]
        return out

    def get_patch_by_center_level0(
        self,
        center_x_level0: int,
        center_y_level0: int,
        width_pixels_at_spacing: int,
        height_pixels_at_spacing: int,
        spacing_mpp: float,
        use_mask: bool = False,
        center_is_wsi_level0: bool = False,
    ) -> np.ndarray:
        spacing_at_level0 = self.mask_spacings[0] if use_mask else self.wsi_spacings[0]
        center_x_source_level0 = int(center_x_level0)
        center_y_source_level0 = int(center_y_level0)
        if use_mask and center_is_wsi_level0:
            if self.mask is None:
                raise ValueError("Mask is not available")
            scale_wsi_to_mask = float(self.wsi_spacings[0]) / float(self.mask_spacings[0])
            center_x_source_level0 = int(round(center_x_level0 * scale_wsi_to_mask))
            center_y_source_level0 = int(round(center_y_level0 * scale_wsi_to_mask))

        x0, y0, _, _ = context_box_from_center_level0(
            center_x_level0=center_x_source_level0,
            center_y_level0=center_y_source_level0,
            size_pixels_at_spacing=width_pixels_at_spacing,
            spacing=spacing_mpp,
            spacing_at_level0=spacing_at_level0,
        )
        source = self.mask if use_mask else self.wsi
        if source is None:
            raise ValueError("Mask is not available")
        patch = source.get_patch(
            int(x0),
            int(y0),
            int(width_pixels_at_spacing),
            int(height_pixels_at_spacing),
            spacing=float(spacing_mpp),
            center=False,
        )
        return self._ensure_patch_shape(np.asarray(patch), width_pixels_at_spacing, height_pixels_at_spacing)

    def level0_center_in_bounds(
        self,
        center_x_level0: int,
        center_y_level0: int,
        size_pixels_at_spacing: int,
        spacing_mpp: float,
    ) -> bool:
        x0, y0, x1, y1 = context_box_from_center_level0(
            center_x_level0=center_x_level0,
            center_y_level0=center_y_level0,
            size_pixels_at_spacing=size_pixels_at_spacing,
            spacing=spacing_mpp,
            spacing_at_level0=self.geometry.spacing_at_level0_mpp,
        )
        return x0 >= 0 and y0 >= 0 and x1 <= self.geometry.level0_width and y1 <= self.geometry.level0_height

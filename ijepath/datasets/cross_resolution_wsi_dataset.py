import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ijepath.datasets.wsi_readers.wholeslidedata_reader_adapter import (
    WholeSlideDataReaderAdapter,
    spacing_pixels_to_level0_pixels,
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def snap_size_to_patch_multiple(size_px: int, patch_size: int) -> int:
    """Snap size to nearest patch-size multiple (ties round down)."""
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    size = max(1, int(size_px))
    lower = (size // patch_size) * patch_size
    upper = ((size + patch_size - 1) // patch_size) * patch_size
    lower = max(int(patch_size), int(lower))
    upper = max(int(patch_size), int(upper))
    if (size - lower) <= (upper - size):
        return int(lower)
    return int(upper)


class CrossResolutionWSIDataset(Dataset):
    """Online context/target extraction dataset from a profile-specific anchor catalog."""

    def __init__(
        self,
        anchor_catalog_csv: str,
        context_mpp: float,
        target_mpp: float,
        context_fov_um: float,
        target_fov_um: float,
        patch_size: int,
        targets_per_context: int,
        seed: int,
        spacing_tolerance: float = 0.05,
        min_target_tissue_fraction: float = 0.25,
        insufficient_target_policy: str = "skip_anchor",
        min_target_tissue_fraction_floor: float | None = None,
        min_target_tissue_fraction_step: float = 0.05,
        backend: str = "openslide",
        align_targets_to_patch_grid: bool = False,
    ) -> None:
        self.anchor_catalog_csv = str(anchor_catalog_csv)
        self.context_mpp = float(context_mpp)
        self.target_mpp = float(target_mpp)
        self.context_fov_um = float(context_fov_um)
        self.target_fov_um = float(target_fov_um)
        self.patch_size = int(patch_size)
        self.targets_per_context = int(targets_per_context)
        self.seed = int(seed)
        self.spacing_tolerance = float(spacing_tolerance)
        self.min_target_tissue_fraction = float(min_target_tissue_fraction)
        self.insufficient_target_policy = str(insufficient_target_policy)
        if min_target_tissue_fraction_floor is None:
            default_floor = 0.0 if self.insufficient_target_policy == "lower_threshold" else self.min_target_tissue_fraction
            self.min_target_tissue_fraction_floor = float(default_floor)
        else:
            self.min_target_tissue_fraction_floor = float(min_target_tissue_fraction_floor)
        self.min_target_tissue_fraction_step = float(min_target_tissue_fraction_step)
        self.backend = backend
        self.align_targets_to_patch_grid = align_targets_to_patch_grid

        valid_policies = {"skip_anchor", "skip_slide", "lower_threshold"}
        if self.insufficient_target_policy not in valid_policies:
            raise ValueError(
                "insufficient_target_policy must be one of "
                f"{sorted(valid_policies)}, got {self.insufficient_target_policy!r}"
            )
        if self.min_target_tissue_fraction_floor < 0.0 or self.min_target_tissue_fraction_floor > self.min_target_tissue_fraction:
            raise ValueError(
                "min_target_tissue_fraction_floor must be in [0, min_target_tissue_fraction]"
            )
        if self.min_target_tissue_fraction_step <= 0.0:
            raise ValueError("min_target_tissue_fraction_step must be > 0")

        context_size_raw_px = max(1, int(round(self.context_fov_um / self.context_mpp)))
        target_size_raw_px = max(1, int(round(self.target_fov_um / self.target_mpp)))
        self.context_size_requested_px = snap_size_to_patch_multiple(
            size_px=context_size_raw_px,
            patch_size=self.patch_size,
        )
        self.target_size_requested_px = snap_size_to_patch_multiple(
            size_px=target_size_raw_px,
            patch_size=self.patch_size,
        )
        self.context_effective_mpp = float(self.context_fov_um / float(self.context_size_requested_px))
        self.target_effective_mpp = float(self.target_fov_um / float(self.target_size_requested_px))
        self.target_size_context_requested_px = max(
            1,
            int(round(self.target_fov_um / self.context_effective_mpp)),
        )

        self.anchors = self._load_anchor_rows(Path(self.anchor_catalog_csv))
        if not self.anchors:
            raise ValueError(f"No anchors found in catalog: {self.anchor_catalog_csv}")
        self.current_pass_index = 0
        self._reader_cache: Dict[str, WholeSlideDataReaderAdapter] = {}
        self._resolution_plan_cache: Dict[str, dict] = {}

    def _load_anchor_rows(self, path: Path) -> List[dict]:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]

    def __len__(self) -> int:
        return len(self.anchors)

    def set_pass_index(self, pass_index: int) -> None:
        self.current_pass_index = int(pass_index)

    def _rng_seed_for(self, index: int, anchor_attempt: int) -> int:
        # Pass-aware seeding ensures repeated indices can yield different
        # target placements across anchor passes while remaining deterministic.
        return (
            int(self.seed)
            + int(index)
            + int(anchor_attempt) * 1_000_003
            + int(self.current_pass_index) * 10_000_019
        )

    def _get_reader(self, row: dict) -> WholeSlideDataReaderAdapter:
        slide_id = row["slide_id"]
        if slide_id not in self._reader_cache:
            self._reader_cache[slide_id] = WholeSlideDataReaderAdapter(
                wsi_path=row["wsi_path"],
                mask_path=row.get("mask_path"),
                backend=self.backend,
            )
        return self._reader_cache[slide_id]

    def _to_rgb(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        if image.shape[2] > 3:
            image = image[..., :3]
        return image

    def _normalize_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
        return torch.from_numpy(image).permute(2, 0, 1).contiguous()

    def _choose_source_spacing(self, spacings: list[float], requested_mpp: float) -> tuple[float, str]:
        if not spacings:
            raise ValueError("Slide does not expose any pyramid spacing")
        sorted_spacings = sorted(float(s) for s in spacings)

        nearest = min(sorted_spacings, key=lambda s: abs(s - requested_mpp))
        rel_err = abs(nearest - requested_mpp) / requested_mpp
        if rel_err <= self.spacing_tolerance:
            return nearest, "native_or_close"

        finer = [s for s in sorted_spacings if s < requested_mpp]
        if finer:
            return max(finer), "fallback_from_finer"

        coarser = [s for s in sorted_spacings if s > requested_mpp]
        if coarser:
            return min(coarser), "fallback_from_coarser"

        return nearest, "fallback_nearest"

    def _get_resolution_plan(self, slide_id: str, reader: WholeSlideDataReaderAdapter) -> dict:
        if slide_id in self._resolution_plan_cache:
            return self._resolution_plan_cache[slide_id]

        context_source_mpp, context_mode = self._choose_source_spacing(
            spacings=reader.wsi_spacings,
            requested_mpp=self.context_effective_mpp,
        )
        target_source_mpp, target_mode = self._choose_source_spacing(
            spacings=reader.wsi_spacings,
            requested_mpp=self.target_effective_mpp,
        )

        context_source_size_px = max(
            1,
            int(round(self.context_size_requested_px * self.context_effective_mpp / context_source_mpp)),
        )
        target_source_size_px = max(
            1,
            int(round(self.target_size_requested_px * self.target_effective_mpp / target_source_mpp)),
        )

        plan = {
            "context_source_mpp": float(context_source_mpp),
            "target_source_mpp": float(target_source_mpp),
            "context_resolution_mode": context_mode,
            "target_resolution_mode": target_mode,
            "context_source_size_px": int(context_source_size_px),
            "target_source_size_px": int(target_source_size_px),
            "context_size_requested_px": int(self.context_size_requested_px),
            "target_size_requested_px": int(self.target_size_requested_px),
            "target_size_context_requested_px": int(self.target_size_context_requested_px),
        }
        self._resolution_plan_cache[slide_id] = plan
        return plan

    def _extract_context_tissue_mask(
        self,
        reader: WholeSlideDataReaderAdapter,
        center_x_level0: int,
        center_y_level0: int,
        context_size_requested_px: int,
    ) -> Optional[np.ndarray]:
        if reader.mask is None:
            return None
        if not reader.mask_spacings:
            return None

        mask_source_mpp, _ = self._choose_source_spacing(
            spacings=reader.mask_spacings,
            requested_mpp=self.context_effective_mpp,
        )
        mask_source_size_px = max(
            1,
            int(round(context_size_requested_px * self.context_effective_mpp / mask_source_mpp)),
        )
        mask_patch = reader.get_patch_by_center_level0(
            center_x_level0=center_x_level0,
            center_y_level0=center_y_level0,
            width_pixels_at_spacing=mask_source_size_px,
            height_pixels_at_spacing=mask_source_size_px,
            spacing_mpp=mask_source_mpp,
            use_mask=True,
            center_is_wsi_level0=True,
        )
        if mask_patch.ndim == 3:
            mask_patch = mask_patch[..., 0]
        if mask_source_size_px != context_size_requested_px:
            mask_patch = cv2.resize(
                mask_patch,
                (context_size_requested_px, context_size_requested_px),
                interpolation=cv2.INTER_NEAREST,
            )
        return (mask_patch > 0).astype(np.float32)

    def _box_tissue_fraction(self, tissue_mask: Optional[np.ndarray], box_xyxy: np.ndarray) -> Optional[float]:
        if tissue_mask is None:
            return None
        h, w = tissue_mask.shape[:2]
        x0, y0, x1, y1 = [int(v) for v in box_xyxy.tolist()]
        x0 = max(0, min(w, x0))
        y0 = max(0, min(h, y0))
        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return float(tissue_mask[y0:y1, x0:x1].mean())

    def _make_box_from_center(
        self,
        center_x: int,
        center_y: int,
        target_size_context_px: int,
    ) -> np.ndarray:
        x0 = int(center_x - target_size_context_px // 2)
        y0 = int(center_y - target_size_context_px // 2)
        return np.array(
            [x0, y0, x0 + int(target_size_context_px), y0 + int(target_size_context_px)],
            dtype=np.float32,
        )

    def _is_in_bounds(self, box_xyxy: np.ndarray, context_size_px: int) -> bool:
        x0, y0, x1, y1 = [float(v) for v in box_xyxy.tolist()]
        return x0 >= 0.0 and y0 >= 0.0 and x1 <= float(context_size_px) and y1 <= float(context_size_px)

    def _overlaps_too_much(
        self,
        candidate: np.ndarray,
        selected_boxes: list[np.ndarray],
        max_overlap_ratio: float,
        target_size_context_px: int,
    ) -> bool:
        for other in selected_boxes:
            ix0 = max(candidate[0], other[0])
            iy0 = max(candidate[1], other[1])
            ix1 = min(candidate[2], other[2])
            iy1 = min(candidate[3], other[3])
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            if inter > max_overlap_ratio * float(target_size_context_px * target_size_context_px):
                return True
        return False

    def _sample_target_boxes_in_context(
        self,
        rng: np.random.Generator,
        target_margin_context_px: int,
        context_size_px: int,
        target_size_context_px: int,
        context_tissue_mask: Optional[np.ndarray],
        min_tissue_fraction: Optional[float] = None,
    ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        threshold = self.min_target_tissue_fraction if min_tissue_fraction is None else float(min_tissue_fraction)
        min_center = target_margin_context_px
        max_center = context_size_px - target_margin_context_px - 1
        if max_center < min_center:
            return None

        patch = max(1, int(self.patch_size))

        def _aligned_x0_bounds() -> tuple[int, int] | None:
            half = 0.5 * float(target_size_context_px)
            min_x0 = max(0, int(math.ceil(float(min_center) - half)))
            max_x0 = min(
                int(context_size_px - target_size_context_px),
                int(math.floor(float(max_center) - half)),
            )
            if max_x0 < min_x0:
                return None
            return min_x0, max_x0

        def _sample_candidate_box() -> np.ndarray | None:
            if not self.align_targets_to_patch_grid:
                cx = int(rng.integers(min_center, max_center + 1))
                cy = int(rng.integers(min_center, max_center + 1))
                return self._make_box_from_center(
                    center_x=cx,
                    center_y=cy,
                    target_size_context_px=target_size_context_px,
                )

            bounds = _aligned_x0_bounds()
            if bounds is None:
                return None
            min_x0, max_x0 = bounds
            min_y0, max_y0 = bounds

            start_x0 = ((min_x0 + patch - 1) // patch) * patch
            start_y0 = ((min_y0 + patch - 1) // patch) * patch
            if start_x0 > max_x0 or start_y0 > max_y0:
                return None

            nx = ((max_x0 - start_x0) // patch) + 1
            ny = ((max_y0 - start_y0) // patch) + 1
            x0 = int(start_x0 + patch * int(rng.integers(0, nx)))
            y0 = int(start_y0 + patch * int(rng.integers(0, ny)))
            return np.array(
                [x0, y0, x0 + int(target_size_context_px), y0 + int(target_size_context_px)],
                dtype=np.float32,
            )

        boxes: list[np.ndarray] = []
        tissue_fractions: list[float] = []
        max_attempts = 4000
        attempts = 0
        while len(boxes) < self.targets_per_context and attempts < max_attempts:
            attempts += 1
            box = _sample_candidate_box()
            if box is None:
                return None
            if not self._is_in_bounds(box, context_size_px):
                continue
            if self._overlaps_too_much(
                candidate=box,
                selected_boxes=boxes,
                max_overlap_ratio=0.25,
                target_size_context_px=target_size_context_px,
            ):
                continue
            tissue_fraction = self._box_tissue_fraction(context_tissue_mask, box)
            if tissue_fraction is not None and tissue_fraction < threshold:
                continue

            boxes.append(box)
            if tissue_fraction is not None:
                tissue_fractions.append(float(tissue_fraction))

        if len(boxes) < self.targets_per_context:
            # Deterministic fallback: evaluate a dense grid and greedily keep highest-coverage boxes.
            candidates: list[tuple[float, np.ndarray, Optional[float]]] = []

            if self.align_targets_to_patch_grid:
                bounds = _aligned_x0_bounds()
                if bounds is not None:
                    min_x0, max_x0 = bounds
                    min_y0, max_y0 = bounds
                    start_x0 = ((min_x0 + patch - 1) // patch) * patch
                    start_y0 = ((min_y0 + patch - 1) // patch) * patch
                    x0_values = np.arange(start_x0, max_x0 + 1, patch, dtype=np.int32)
                    y0_values = np.arange(start_y0, max_y0 + 1, patch, dtype=np.int32)
                    for y0 in y0_values:
                        for x0 in x0_values:
                            box = np.array(
                                [
                                    int(x0),
                                    int(y0),
                                    int(x0) + int(target_size_context_px),
                                    int(y0) + int(target_size_context_px),
                                ],
                                dtype=np.float32,
                            )
                            if not self._is_in_bounds(box, context_size_px):
                                continue
                            tissue_fraction = self._box_tissue_fraction(context_tissue_mask, box)
                            if tissue_fraction is not None and tissue_fraction < threshold:
                                continue
                            score = 1.0 if tissue_fraction is None else float(tissue_fraction)
                            candidates.append((score, box, tissue_fraction))
            else:
                grid_side = max(3, int(np.ceil(np.sqrt(self.targets_per_context * 8))))
                centers = np.linspace(min_center, max_center, num=grid_side, dtype=np.int32)
                for cy in centers:
                    for cx in centers:
                        box = self._make_box_from_center(
                            center_x=int(cx),
                            center_y=int(cy),
                            target_size_context_px=target_size_context_px,
                        )
                        if not self._is_in_bounds(box, context_size_px):
                            continue
                        tissue_fraction = self._box_tissue_fraction(context_tissue_mask, box)
                        if tissue_fraction is not None and tissue_fraction < threshold:
                            continue
                        score = 1.0 if tissue_fraction is None else float(tissue_fraction)
                        candidates.append((score, box, tissue_fraction))

            candidates.sort(key=lambda x: x[0], reverse=True)
            for _, box, tissue_fraction in candidates:
                if len(boxes) >= self.targets_per_context:
                    break
                if self._overlaps_too_much(
                    candidate=box,
                    selected_boxes=boxes,
                    max_overlap_ratio=0.25,
                    target_size_context_px=target_size_context_px,
                ):
                    continue
                boxes.append(box)
                if tissue_fraction is not None:
                    tissue_fractions.append(float(tissue_fraction))

        if len(boxes) < self.targets_per_context:
            return None

        return (
            np.stack(boxes[: self.targets_per_context], axis=0),
            None if context_tissue_mask is None else np.asarray(tissue_fractions[: self.targets_per_context], dtype=np.float32),
        )

    def _build_sample_from_anchor(
        self,
        anchor: dict,
        index: int,
        anchor_attempt: int,
        min_target_tissue_fraction: Optional[float] = None,
    ) -> Optional[dict]:
        active_threshold = (
            self.min_target_tissue_fraction
            if min_target_tissue_fraction is None
            else float(min_target_tissue_fraction)
        )
        reader = self._get_reader(anchor)
        slide_id = anchor["slide_id"]
        resolution_plan = self._get_resolution_plan(
            slide_id=slide_id,
            reader=reader,
        )
        context_source_mpp = float(resolution_plan["context_source_mpp"])
        target_source_mpp = float(resolution_plan["target_source_mpp"])
        context_source_size_px = int(resolution_plan["context_source_size_px"])
        target_source_size_px = int(resolution_plan["target_source_size_px"])
        context_size_requested_px = int(resolution_plan["context_size_requested_px"])
        target_size_requested_px = int(resolution_plan["target_size_requested_px"])
        target_size_context_requested_px = int(resolution_plan["target_size_context_requested_px"])

        center_x_level0 = int(float(anchor["center_x_level0"]))
        center_y_level0 = int(float(anchor["center_y_level0"]))
        target_margin_um = float(anchor.get("target_margin_um", 16.0))
        target_margin_context_px = max(
            int(round(target_margin_um / self.context_effective_mpp)),
            target_size_context_requested_px // 2,
        )
        target_margin_context_px = min(target_margin_context_px, max(1, context_size_requested_px // 2 - 1))

        context_patch = reader.get_patch_by_center_level0(
            center_x_level0=center_x_level0,
            center_y_level0=center_y_level0,
            width_pixels_at_spacing=context_source_size_px,
            height_pixels_at_spacing=context_source_size_px,
            spacing_mpp=context_source_mpp,
            use_mask=False,
        )
        context_patch = self._to_rgb(context_patch)
        if context_source_size_px != context_size_requested_px:
            context_patch = cv2.resize(
                context_patch,
                (context_size_requested_px, context_size_requested_px),
                interpolation=cv2.INTER_AREA if context_source_size_px > context_size_requested_px else cv2.INTER_LINEAR,
            )

        context_tissue_mask = self._extract_context_tissue_mask(
            reader=reader,
            center_x_level0=center_x_level0,
            center_y_level0=center_y_level0,
            context_size_requested_px=context_size_requested_px,
        )

        rng_seed = self._rng_seed_for(index=index, anchor_attempt=anchor_attempt)
        rng = np.random.default_rng(rng_seed)
        sampled = self._sample_target_boxes_in_context(
            rng=rng,
            target_margin_context_px=target_margin_context_px,
            context_size_px=context_size_requested_px,
            target_size_context_px=target_size_context_requested_px,
            context_tissue_mask=context_tissue_mask,
            min_tissue_fraction=active_threshold,
        )
        if sampled is None:
            return None
        boxes_context_px, target_tissue_fractions = sampled

        context_size_level0 = spacing_pixels_to_level0_pixels(
            size_pixels_at_spacing=context_size_requested_px,
            spacing=self.context_effective_mpp,
            spacing_at_level0=float(anchor["wsi_level0_spacing_mpp"]),
        )
        context_x0_level0 = center_x_level0 - context_size_level0 // 2
        context_y0_level0 = center_y_level0 - context_size_level0 // 2

        scale_context_px_to_level0 = self.context_effective_mpp / float(anchor["wsi_level0_spacing_mpp"])

        target_patches = []
        for box in boxes_context_px:
            cx_context = 0.5 * (box[0] + box[2])
            cy_context = 0.5 * (box[1] + box[3])
            target_center_x_level0 = int(round(context_x0_level0 + cx_context * scale_context_px_to_level0))
            target_center_y_level0 = int(round(context_y0_level0 + cy_context * scale_context_px_to_level0))

            target_patch = reader.get_patch_by_center_level0(
                center_x_level0=target_center_x_level0,
                center_y_level0=target_center_y_level0,
                width_pixels_at_spacing=target_source_size_px,
                height_pixels_at_spacing=target_source_size_px,
                spacing_mpp=target_source_mpp,
                use_mask=False,
            )
            target_patch = self._to_rgb(target_patch)
            if target_source_size_px != target_size_requested_px:
                target_patch = cv2.resize(
                    target_patch,
                    (target_size_requested_px, target_size_requested_px),
                    interpolation=cv2.INTER_AREA if target_source_size_px > target_size_requested_px else cv2.INTER_LINEAR,
                )
            target_patches.append(target_patch)

        context_tensor = self._normalize_to_tensor(context_patch)
        target_tensors = torch.stack([self._normalize_to_tensor(tile) for tile in target_patches], dim=0)

        return {
            "context_image": context_tensor,
            "target_images": target_tensors,
            "target_boxes_in_context_pixels": torch.as_tensor(boxes_context_px, dtype=torch.float32),
            "sample_metadata": {
                "slide_id": anchor["slide_id"],
                "anchor_id": anchor["anchor_id"],
                "anchor_retry_offset": int(anchor_attempt),
                "dataset_pass_index": int(self.current_pass_index),
                "requested_context_mpp": self.context_mpp,
                "requested_target_mpp": self.target_mpp,
                "effective_context_mpp": self.context_effective_mpp,
                "effective_target_mpp": self.target_effective_mpp,
                "source_context_mpp": context_source_mpp,
                "source_target_mpp": target_source_mpp,
                "context_resolution_mode": resolution_plan["context_resolution_mode"],
                "target_resolution_mode": resolution_plan["target_resolution_mode"],
                "context_size_px_at_effective_spacing": context_source_size_px,
                "target_size_context_px_at_effective_spacing": target_size_context_requested_px,
                "target_size_target_px_at_effective_spacing": target_source_size_px,
                "context_size_px_requested_spacing": context_size_requested_px,
                "target_size_px_requested_spacing": target_size_requested_px,
                "context_input_size_px": context_size_requested_px,
                "target_input_size_px": target_size_requested_px,
                "target_tissue_fractions": None
                if target_tissue_fractions is None
                else [float(x) for x in target_tissue_fractions.tolist()],
                "min_target_tissue_fraction": active_threshold,
                "configured_min_target_tissue_fraction": self.min_target_tissue_fraction,
                "align_targets_to_patch_grid": self.align_targets_to_patch_grid,
            },
        }

    def _iter_target_thresholds(self) -> list[float]:
        if self.insufficient_target_policy != "lower_threshold":
            return [self.min_target_tissue_fraction]

        thresholds: list[float] = []
        current = self.min_target_tissue_fraction
        floor = self.min_target_tissue_fraction_floor
        step = self.min_target_tissue_fraction_step
        eps = 1e-9
        while current > floor + eps:
            thresholds.append(float(round(current, 6)))
            current -= step
        thresholds.append(float(round(floor, 6)))

        deduped: list[float] = []
        seen = set()
        for value in thresholds:
            key = round(value, 6)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(float(key))
        return deduped

    def _ordered_anchor_indices(self, index: int) -> list[int]:
        num_anchors = len(self.anchors)
        return [(index + offset) % num_anchors for offset in range(num_anchors)]

    def _grouped_anchor_indices_by_slide(self, index: int) -> list[list[int]]:
        ordered_indices = self._ordered_anchor_indices(index)
        slide_order: list[str] = []
        grouped: dict[str, list[int]] = {}
        for anchor_idx in ordered_indices:
            slide_id = str(self.anchors[anchor_idx]["slide_id"])
            if slide_id not in grouped:
                grouped[slide_id] = []
                slide_order.append(slide_id)
            grouped[slide_id].append(anchor_idx)
        return [grouped[slide_id] for slide_id in slide_order]

    def __getitem__(self, index: int):
        thresholds = self._iter_target_thresholds()

        for threshold in thresholds:
            if self.insufficient_target_policy == "skip_slide":
                grouped_anchor_indices = self._grouped_anchor_indices_by_slide(index)
                attempted_anchors = 0
                exhausted_slides = 0
                for group in grouped_anchor_indices:
                    for anchor_idx in group:
                        anchor = self.anchors[anchor_idx]
                        sample = self._build_sample_from_anchor(
                            anchor=anchor,
                            index=index,
                            anchor_attempt=attempted_anchors,
                            min_target_tissue_fraction=threshold,
                        )
                        attempted_anchors += 1
                        if sample is None:
                            continue
                        sample["sample_metadata"]["effective_min_target_tissue_fraction"] = float(threshold)
                        sample["sample_metadata"]["insufficient_target_policy"] = self.insufficient_target_policy
                        sample["sample_metadata"]["threshold_schedule"] = thresholds
                        sample["sample_metadata"]["anchors_attempted_at_threshold"] = attempted_anchors
                        sample["sample_metadata"]["slides_exhausted_before_success"] = exhausted_slides
                        return sample
                    exhausted_slides += 1
                continue

            ordered_indices = self._ordered_anchor_indices(index)
            for attempted_anchors, anchor_idx in enumerate(ordered_indices, start=1):
                anchor = self.anchors[anchor_idx]
                sample = self._build_sample_from_anchor(
                    anchor=anchor,
                    index=index,
                    anchor_attempt=attempted_anchors - 1,
                    min_target_tissue_fraction=threshold,
                )
                if sample is not None:
                    sample["sample_metadata"]["effective_min_target_tissue_fraction"] = float(threshold)
                    sample["sample_metadata"]["insufficient_target_policy"] = self.insufficient_target_policy
                    sample["sample_metadata"]["threshold_schedule"] = thresholds
                    sample["sample_metadata"]["anchors_attempted_at_threshold"] = attempted_anchors
                    return sample

        raise RuntimeError(
            "Could not sample enough target boxes for index="
            f"{index} after trying policy={self.insufficient_target_policy}, "
            f"thresholds={thresholds}, anchors={len(self.anchors)}."
        )

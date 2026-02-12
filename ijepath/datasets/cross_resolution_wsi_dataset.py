from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ijepath.datasets.wsi_readers.wholeslidedata_reader_adapter import (
    WholeSlideDataReaderAdapter,
    spacing_pixels_to_level0_pixels,
)
from ijepath.utils.parquet import require_pyarrow


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


class CrossResolutionWSIDataset(IterableDataset):
    """Manifest-driven iterable dataset for online context/target extraction."""

    def __init__(
        self,
        anchor_catalog_manifest: str,
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
        backend: str = "asap",
        align_targets_to_patch_grid: bool = False,
        world_size: int = 1,
        rank: int = 0,
        sampling_strategy: str = "stratified_weighted",
        sampling_stratum_key: str = "organ",
        sampling_stratum_weights: str | dict = "inverse_frequency",
        max_open_slides_per_worker: int = 16,
        anchor_stream_batch_size: int = 2048,
    ) -> None:
        super().__init__()
        self.anchor_catalog_manifest = str(anchor_catalog_manifest)
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
        self.world_size = max(1, int(world_size))
        self.rank = int(rank)
        self.sampling_strategy = str(sampling_strategy)
        self.sampling_stratum_key = str(sampling_stratum_key)
        self.sampling_stratum_weights = sampling_stratum_weights
        self.max_open_slides_per_worker = max(1, int(max_open_slides_per_worker))
        self.anchor_stream_batch_size = max(1, int(anchor_stream_batch_size))

        valid_policies = {"skip_anchor", "lower_threshold"}
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

        manifest = json.loads(Path(self.anchor_catalog_manifest).read_text(encoding="utf-8"))
        self.total_anchors = int(manifest.get("total_anchors", 0))
        if self.total_anchors <= 0:
            raise ValueError(f"No anchors found in manifest: {self.anchor_catalog_manifest}")
        self.anchor_shards = [dict(x) for x in manifest.get("anchor_shards", [])]
        if not self.anchor_shards:
            raise ValueError(f"Manifest has no shard entries: {self.anchor_catalog_manifest}")
        self.stratum_counts = {
            str(k): int(v)
            for k, v in dict(manifest.get("stratum_counts", {}) or {}).items()
        }

        self.current_pass_index = 0
        self._reader_cache: OrderedDict[str, WholeSlideDataReaderAdapter] = OrderedDict()
        self._resolution_plan_cache: Dict[str, dict] = {}
        self._reader_cache_hits = 0
        self._reader_cache_misses = 0
        self._reader_cache_evictions = 0
        self._last_reader_cache_event: dict[str, int | str] = {
            "event": "none",
            "open_slides": 0,
            "evicted": 0,
            "hits_total": 0,
            "misses_total": 0,
            "evictions_total": 0,
        }

    def __len__(self) -> int:
        return max(1, int(math.ceil(float(self.total_anchors) / float(max(1, self.world_size)))))

    def set_pass_index(self, pass_index: int) -> None:
        self.current_pass_index = int(pass_index)

    def _rng_seed_for(self, index: int, anchor_attempt: int) -> int:
        return (
            int(self.seed)
            + int(index)
            + int(anchor_attempt) * 1_000_003
            + int(self.current_pass_index) * 10_000_019
        )

    def _partition_for_anchor(self, anchor_id: str, total_partitions: int) -> int:
        digest = hashlib.blake2b(anchor_id.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False) % int(total_partitions)

    def _get_reader(self, row: dict) -> WholeSlideDataReaderAdapter:
        slide_id = str(row["slide_id"])
        cached = self._reader_cache.pop(slide_id, None)
        if cached is not None:
            self._reader_cache_hits += 1
            self._reader_cache[slide_id] = cached
            self._last_reader_cache_event = {
                "event": "hit",
                "open_slides": int(len(self._reader_cache)),
                "evicted": 0,
                "hits_total": int(self._reader_cache_hits),
                "misses_total": int(self._reader_cache_misses),
                "evictions_total": int(self._reader_cache_evictions),
            }
            return cached

        self._reader_cache_misses += 1
        adapter = WholeSlideDataReaderAdapter(
            wsi_path=str(row["wsi_path"]),
            mask_path=row.get("mask_path"),
            backend=self.backend,
        )
        self._reader_cache[slide_id] = adapter
        evicted_count = 0
        while len(self._reader_cache) > int(self.max_open_slides_per_worker):
            _, evicted = self._reader_cache.popitem(last=False)
            close_fn = getattr(evicted, "close", None)
            if callable(close_fn):
                close_fn()
            evicted_count += 1
        self._reader_cache_evictions += int(evicted_count)
        self._last_reader_cache_event = {
            "event": "miss",
            "open_slides": int(len(self._reader_cache)),
            "evicted": int(evicted_count),
            "hits_total": int(self._reader_cache_hits),
            "misses_total": int(self._reader_cache_misses),
            "evictions_total": int(self._reader_cache_evictions),
        }
        return adapter

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
        reader_cache_event = dict(self._last_reader_cache_event)
        slide_id = str(anchor["slide_id"])
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
                "slide_id": slide_id,
                "anchor_id": str(anchor["anchor_id"]),
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
                "stratum_id": str(anchor.get("stratum_id", "unknown")),
                "anchor_stream_batch_id": int(anchor.get("_anchor_stream_batch_id", -1)),
                "anchor_stream_row_in_batch": int(anchor.get("_anchor_stream_row_in_batch", -1)),
                "anchor_stream_batch_size": int(anchor.get("_anchor_stream_batch_size", -1)),
                "anchor_stream_shard_path": str(anchor.get("_anchor_stream_shard_path", "")),
                "reader_cache_event": str(reader_cache_event.get("event", "none")),
                "reader_cache_hit": 1 if str(reader_cache_event.get("event", "none")) == "hit" else 0,
                "reader_cache_miss": 1 if str(reader_cache_event.get("event", "none")) == "miss" else 0,
                "reader_cache_evictions_on_event": int(reader_cache_event.get("evicted", 0)),
                "reader_cache_open_slides": int(reader_cache_event.get("open_slides", 0)),
                "reader_cache_hits_total": int(reader_cache_event.get("hits_total", 0)),
                "reader_cache_misses_total": int(reader_cache_event.get("misses_total", 0)),
                "reader_cache_evictions_total": int(reader_cache_event.get("evictions_total", 0)),
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

    def _parse_sampling_weights(self) -> dict[str, float] | str:
        if isinstance(self.sampling_stratum_weights, dict):
            return {str(k): float(v) for k, v in self.sampling_stratum_weights.items()}
        if isinstance(self.sampling_stratum_weights, str):
            raw = self.sampling_stratum_weights.strip()
            if raw.lower() in {"inverse_frequency", "uniform"}:
                return raw.lower()
            if raw.startswith("{"):
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("sampling_stratum_weights JSON must decode to an object")
                return {str(k): float(v) for k, v in parsed.items()}
            return raw.lower()
        raise ValueError("Unsupported sampling_stratum_weights type")

    def _resolve_strata_and_weights(self) -> tuple[list[str], np.ndarray]:
        if self.sampling_strategy == "global_uniform":
            return ["__all__"], np.asarray([1.0], dtype=np.float64)

        if not self.stratum_counts:
            return ["unknown"], np.asarray([1.0], dtype=np.float64)

        strata = sorted(str(k) for k in self.stratum_counts.keys())
        mode_or_map = self._parse_sampling_weights()

        weights: list[float] = []
        if mode_or_map == "uniform":
            weights = [1.0 for _ in strata]
        elif mode_or_map == "inverse_frequency":
            for stratum in strata:
                count = max(1, int(self.stratum_counts.get(stratum, 0)))
                weights.append(1.0 / float(count))
        elif isinstance(mode_or_map, dict):
            for stratum in strata:
                weights.append(float(mode_or_map.get(stratum, 0.0)))
        else:
            raise ValueError(
                "sampling_stratum_weights must be 'inverse_frequency', 'uniform', or a map"
            )

        weights_arr = np.asarray(weights, dtype=np.float64)
        if np.any(weights_arr < 0):
            raise ValueError("sampling_stratum_weights cannot include negative values")
        if float(weights_arr.sum()) <= 0:
            weights_arr = np.asarray([1.0 for _ in strata], dtype=np.float64)
        weights_arr = weights_arr / weights_arr.sum()
        return strata, weights_arr

    def _iter_anchor_rows_for_stratum(
        self,
        requested_stratum: str | None,
        partition_id: int,
        total_partitions: int,
        rng: np.random.Generator,
        repeat: bool = True,
    ) -> Iterator[dict]:
        _, pq, _ = require_pyarrow()

        eligible_shards = []
        for shard in self.anchor_shards:
            stratum_counts = dict(shard.get("stratum_counts", {}) or {})
            if requested_stratum is not None and int(stratum_counts.get(requested_stratum, 0)) <= 0:
                continue
            eligible_shards.append(dict(shard))

        if not eligible_shards:
            return

        while True:
            order = list(range(len(eligible_shards)))
            rng.shuffle(order)
            produced = 0

            for idx in order:
                shard = eligible_shards[idx]
                parquet_file = pq.ParquetFile(str(shard["path"]))
                batch_id = 0
                for batch in parquet_file.iter_batches(batch_size=int(self.anchor_stream_batch_size)):
                    columns = batch.to_pydict()
                    if not columns:
                        continue
                    first_col = next(iter(columns.values()))
                    row_count = len(first_col)
                    for row_idx in range(row_count):
                        row = {k: columns[k][row_idx] for k in columns.keys()}
                        row["_anchor_stream_batch_id"] = int(batch_id)
                        row["_anchor_stream_row_in_batch"] = int(row_idx)
                        row["_anchor_stream_batch_size"] = int(row_count)
                        row["_anchor_stream_shard_path"] = str(shard["path"])
                        anchor_id = str(row.get("anchor_id", ""))
                        if not anchor_id:
                            continue
                        if requested_stratum is not None and str(row.get("stratum_id", "unknown")) != requested_stratum:
                            continue
                        if self._partition_for_anchor(anchor_id, total_partitions) != partition_id:
                            continue
                        produced += 1
                        yield row
                    batch_id += 1

            if produced == 0:
                return
            if not repeat:
                return

    def _build_sample_with_policy(self, anchor: dict, index: int) -> Optional[dict]:
        thresholds = self._iter_target_thresholds()
        for threshold in thresholds:
            sample = self._build_sample_from_anchor(
                anchor=anchor,
                index=index,
                anchor_attempt=0,
                min_target_tissue_fraction=threshold,
            )
            if sample is None:
                continue
            sample["sample_metadata"]["effective_min_target_tissue_fraction"] = float(threshold)
            sample["sample_metadata"]["insufficient_target_policy"] = self.insufficient_target_policy
            sample["sample_metadata"]["threshold_schedule"] = thresholds
            sample["sample_metadata"]["anchors_attempted_at_threshold"] = 1
            return sample
        return None

    def __iter__(self):
        worker = get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        num_workers = int(worker.num_workers) if worker is not None else 1

        partition_id = int(self.rank) * int(num_workers) + int(worker_id)
        total_partitions = int(self.world_size) * int(num_workers)

        base_seed = (
            int(self.seed)
            + int(self.current_pass_index) * 1_000_003
            + int(partition_id) * 7_919
        )
        rng = np.random.default_rng(base_seed)

        strata, weights = self._resolve_strata_and_weights()
        per_stratum_iters: dict[str, Iterator[dict]] = {}
        for stratum in strata:
            requested_stratum = None if stratum == "__all__" else stratum
            per_stratum_iters[stratum] = self._iter_anchor_rows_for_stratum(
                requested_stratum=requested_stratum,
                partition_id=partition_id,
                total_partitions=total_partitions,
                rng=rng,
            )

        sample_index = 0
        while True:
            chosen_idx = int(rng.choice(len(strata), p=weights))
            chosen_stratum = strata[chosen_idx]
            it = per_stratum_iters[chosen_stratum]

            try:
                anchor = next(it)
            except StopIteration:
                requested_stratum = None if chosen_stratum == "__all__" else chosen_stratum
                it = self._iter_anchor_rows_for_stratum(
                    requested_stratum=requested_stratum,
                    partition_id=partition_id,
                    total_partitions=total_partitions,
                    rng=rng,
                )
                per_stratum_iters[chosen_stratum] = it
                try:
                    anchor = next(it)
                except StopIteration:
                    continue

            sample = self._build_sample_with_policy(anchor=anchor, index=sample_index)
            sample_index += 1
            if sample is None:
                continue
            yield sample

    def __getitem__(self, index: int):
        if index < 0:
            raise IndexError("index must be >= 0")

        worker = get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        num_workers = int(worker.num_workers) if worker is not None else 1
        partition_id = int(self.rank) * int(num_workers) + int(worker_id)
        total_partitions = int(self.world_size) * int(num_workers)

        rng = np.random.default_rng(int(self.seed) + int(self.current_pass_index) * 1_000_003 + int(partition_id) * 7_919)
        anchors: list[dict] = []
        anchors.extend(
            self._iter_anchor_rows_for_stratum(
                requested_stratum=None,
                partition_id=partition_id,
                total_partitions=total_partitions,
                rng=rng,
                repeat=False,
            )
        )
        if not anchors:
            raise RuntimeError("No anchors available for current partition.")

        start = int(index) % len(anchors)
        for offset in range(len(anchors)):
            anchor = anchors[(start + offset) % len(anchors)]
            sample = self._build_sample_with_policy(anchor=anchor, index=index + offset)
            if sample is not None:
                return sample

        raise RuntimeError(
            "Could not sample enough target boxes for index="
            f"{index} after trying policy={self.insufficient_target_policy}, "
            f"anchors={len(anchors)}."
        )

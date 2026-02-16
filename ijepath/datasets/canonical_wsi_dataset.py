from __future__ import annotations

import cv2
import torch
from torchvision import transforms

from ijepath.datasets.cross_resolution_wsi_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    CrossResolutionWSIDataset,
)


class CanonicalWSIDataset(CrossResolutionWSIDataset):
    """Single-scale canonical I-JEPA dataset backed by anchor parquet streaming."""

    def __init__(
        self,
        anchor_catalog_manifest: str,
        input_mpp: float,
        source_tile_size_px: int,
        crop_size_px: int,
        patch_size: int,
        seed: int,
        spacing_tolerance: float = 0.05,
        backend: str = "asap",
        crop_scale: tuple[float, float] = (0.3, 1.0),
        use_horizontal_flip: bool = True,
        horizontal_flip_prob: float = 0.5,
        use_color_distortion: bool = False,
        color_jitter_strength: float = 0.0,
        use_gaussian_blur: bool = False,
        world_size: int = 1,
        rank: int = 0,
        sampling_strategy: str = "stratified_weighted",
        sampling_stratum_key: str = "organ",
        sampling_stratum_weights: str | dict = "inverse_frequency",
        max_open_slides_per_worker: int = 16,
        anchor_stream_batch_size: int = 2048,
    ) -> None:
        self.input_mpp = float(input_mpp)
        self.source_tile_size_px = int(source_tile_size_px)
        self.crop_size_px = int(crop_size_px)
        self.crop_scale = tuple(float(x) for x in crop_scale)
        self.use_horizontal_flip = bool(use_horizontal_flip)
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.use_color_distortion = bool(use_color_distortion)
        self.color_jitter_strength = float(color_jitter_strength)
        self.use_gaussian_blur = bool(use_gaussian_blur)
        if self.input_mpp <= 0:
            raise ValueError("input_mpp must be > 0")
        if self.source_tile_size_px <= 0:
            raise ValueError("source_tile_size_px must be > 0")
        if self.crop_size_px <= 0:
            raise ValueError("crop_size_px must be > 0")
        if self.source_tile_size_px < self.crop_size_px:
            raise ValueError("source_tile_size_px must be >= crop_size_px")

        super().__init__(
            anchor_catalog_manifest=anchor_catalog_manifest,
            context_mpp=float(self.input_mpp),
            target_mpp=float(self.input_mpp),
            context_fov_um=float(self.source_tile_size_px * self.input_mpp),
            target_fov_um=float(self.source_tile_size_px * self.input_mpp),
            patch_size=patch_size,
            targets_per_context=1,
            seed=seed,
            spacing_tolerance=spacing_tolerance,
            min_target_tissue_fraction=0.0,
            insufficient_target_policy="skip_anchor",
            min_target_tissue_fraction_floor=0.0,
            min_target_tissue_fraction_step=0.05,
            backend=backend,
            align_targets_to_patch_grid=False,
            world_size=world_size,
            rank=rank,
            sampling_strategy=sampling_strategy,
            sampling_stratum_key=sampling_stratum_key,
            sampling_stratum_weights=sampling_stratum_weights,
            max_open_slides_per_worker=max_open_slides_per_worker,
            anchor_stream_batch_size=anchor_stream_batch_size,
        )

        self.image_transform = self._build_transform(
            crop_size_px=self.crop_size_px,
            crop_scale=self.crop_scale,
            use_horizontal_flip=self.use_horizontal_flip,
            horizontal_flip_prob=self.horizontal_flip_prob,
            use_color_distortion=self.use_color_distortion,
            color_jitter_strength=self.color_jitter_strength,
            use_gaussian_blur=self.use_gaussian_blur,
        )

    @staticmethod
    def _build_transform(
        *,
        crop_size_px: int,
        crop_scale: tuple[float, float],
        use_horizontal_flip: bool,
        horizontal_flip_prob: float,
        use_color_distortion: bool,
        color_jitter_strength: float,
        use_gaussian_blur: bool,
    ):
        steps = [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(
                size=int(crop_size_px),
                scale=tuple(float(x) for x in crop_scale),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        ]
        if bool(use_horizontal_flip):
            steps.append(transforms.RandomHorizontalFlip(p=float(horizontal_flip_prob)))
        if bool(use_color_distortion):
            jitter_strength = float(color_jitter_strength)
            steps.append(
                transforms.ColorJitter(
                    brightness=0.8 * jitter_strength,
                    contrast=0.8 * jitter_strength,
                    saturation=0.8 * jitter_strength,
                    hue=0.2 * jitter_strength,
                )
            )
        if bool(use_gaussian_blur):
            kernel_size = max(3, int(round(float(crop_size_px) * 0.1)))
            if kernel_size % 2 == 0:
                kernel_size += 1
            steps.append(transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)))

        steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=tuple(float(x) for x in IMAGENET_MEAN.tolist()),
                    std=tuple(float(x) for x in IMAGENET_STD.tolist()),
                ),
            ]
        )
        return transforms.Compose(steps)

    def _build_sample_with_policy(self, anchor: dict, index: int):
        del index
        reader = self._get_reader(anchor)
        reader_cache_event = dict(self._last_reader_cache_event)

        slide_id = str(anchor["slide_id"])
        center_x_level0 = int(float(anchor["center_x_level0"]))
        center_y_level0 = int(float(anchor["center_y_level0"]))
        source_mpp, source_mode = self._choose_source_spacing(
            spacings=reader.wsi_spacings,
            requested_mpp=float(self.input_mpp),
        )
        source_size_px_at_spacing = max(
            1,
            int(round(float(self.source_tile_size_px) * float(self.input_mpp) / float(source_mpp))),
        )

        tile = reader.get_patch_by_center_level0(
            center_x_level0=center_x_level0,
            center_y_level0=center_y_level0,
            width_pixels_at_spacing=source_size_px_at_spacing,
            height_pixels_at_spacing=source_size_px_at_spacing,
            spacing_mpp=float(source_mpp),
            use_mask=False,
        )
        tile = self._to_rgb(tile)
        if int(source_size_px_at_spacing) != int(self.source_tile_size_px):
            tile = cv2.resize(
                tile,
                (int(self.source_tile_size_px), int(self.source_tile_size_px)),
                interpolation=(
                    cv2.INTER_AREA
                    if int(source_size_px_at_spacing) > int(self.source_tile_size_px)
                    else cv2.INTER_LINEAR
                ),
            )

        image_tensor = self.image_transform(tile)
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError("canonical transform must produce torch.Tensor")
        if tuple(image_tensor.shape[-2:]) != (int(self.crop_size_px), int(self.crop_size_px)):
            raise ValueError("canonical transform produced unexpected crop size")

        sample_metadata = {
            "slide_id": slide_id,
            "anchor_id": str(anchor["anchor_id"]),
            "dataset_pass_index": int(self.current_pass_index),
            "requested_input_mpp": float(self.input_mpp),
            "source_input_mpp": float(source_mpp),
            "source_resolution_mode": str(source_mode),
            "source_tile_size_px_at_effective_spacing": int(source_size_px_at_spacing),
            "source_tile_size_px_requested": int(self.source_tile_size_px),
            "model_crop_size_px": int(self.crop_size_px),
            "crop_scale_min": float(self.crop_scale[0]),
            "crop_scale_max": float(self.crop_scale[1]),
            "use_horizontal_flip": int(self.use_horizontal_flip),
            "horizontal_flip_prob": float(self.horizontal_flip_prob),
            "use_color_distortion": int(self.use_color_distortion),
            "color_jitter_strength": float(self.color_jitter_strength),
            "use_gaussian_blur": int(self.use_gaussian_blur),
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
        }
        return {
            "image": image_tensor,
            "sample_metadata": sample_metadata,
        }

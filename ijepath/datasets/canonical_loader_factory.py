from __future__ import annotations

from torch.utils.data import DataLoader

from ijepath.datasets.canonical_wsi_dataset import CanonicalWSIDataset
from ijepath.masks.multiblock import MaskCollator


class CanonicalCollateWithMetadata:
    """Pickle-safe collate wrapper for canonical DataLoader workers."""

    def __init__(
        self,
        *,
        crop_size_px: int,
        patch_size: int,
        enc_mask_scale: tuple[float, float],
        pred_mask_scale: tuple[float, float],
        aspect_ratio: tuple[float, float],
        num_enc_masks: int,
        num_pred_masks: int,
        min_keep: int,
        allow_overlap: bool,
    ) -> None:
        self.crop_size_px = int(crop_size_px)
        self.patch_size = int(patch_size)
        self.enc_mask_scale = tuple(float(x) for x in enc_mask_scale)
        self.pred_mask_scale = tuple(float(x) for x in pred_mask_scale)
        self.aspect_ratio = tuple(float(x) for x in aspect_ratio)
        self.num_enc_masks = int(num_enc_masks)
        self.num_pred_masks = int(num_pred_masks)
        self.min_keep = int(min_keep)
        self.allow_overlap = bool(allow_overlap)
        self._collator: MaskCollator | None = None

    def _get_collator(self) -> MaskCollator:
        if self._collator is None:
            # Build collator lazily in each worker process (spawn-safe).
            self._collator = MaskCollator(
                input_size=int(self.crop_size_px),
                patch_size=int(self.patch_size),
                enc_mask_scale=self.enc_mask_scale,
                pred_mask_scale=self.pred_mask_scale,
                aspect_ratio=self.aspect_ratio,
                nenc=int(self.num_enc_masks),
                npred=int(self.num_pred_masks),
                min_keep=int(self.min_keep),
                allow_overlap=bool(self.allow_overlap),
            )
        return self._collator

    def __call__(self, batch: list[dict]):
        collator = self._get_collator()
        images = [sample["image"] for sample in batch]
        metadata = [dict(sample.get("sample_metadata", {})) for sample in batch]
        collated_images, masks_enc, masks_pred = collator(images)
        collated_batch = {
            "image": collated_images,
            "sample_metadata": metadata,
        }
        return collated_batch, masks_enc, masks_pred


def validate_canonical_size_alignment(
    *,
    dataset: CanonicalWSIDataset,
    crop_size_px: int,
    patch_size: int,
) -> None:
    if int(crop_size_px) != int(dataset.crop_size_px):
        raise ValueError(
            "Canonical crop size drift: "
            f"dataset={int(dataset.crop_size_px)} loader={int(crop_size_px)}"
        )
    if int(crop_size_px) % int(patch_size) != 0:
        raise ValueError(
            "Canonical crop size must be divisible by patch size: "
            f"crop={int(crop_size_px)} patch={int(patch_size)}"
        )


def make_canonical_loader(
    *,
    batch_size: int,
    pin_mem: bool,
    num_workers: int,
    world_size: int,
    rank: int,
    drop_last: bool,
    anchor_catalog_manifest: str,
    patch_size: int,
    input_mpp: float,
    source_tile_size_px: int,
    crop_size_px: int,
    crop_scale: tuple[float, float],
    use_horizontal_flip: bool,
    horizontal_flip_prob: float,
    use_color_distortion: bool,
    color_jitter_strength: float,
    use_gaussian_blur: bool,
    enc_mask_scale: tuple[float, float],
    pred_mask_scale: tuple[float, float],
    aspect_ratio: tuple[float, float],
    num_enc_masks: int,
    num_pred_masks: int,
    min_keep: int,
    allow_overlap: bool,
    seed: int,
    spacing_tolerance: float,
    backend: str = "asap",
    sampling_strategy: str = "stratified_weighted",
    sampling_stratum_key: str = "organ",
    sampling_stratum_weights: str | dict = "inverse_frequency",
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    max_open_slides_per_worker: int = 16,
    anchor_stream_batch_size: int = 2048,
):
    dataset = CanonicalWSIDataset(
        anchor_catalog_manifest=anchor_catalog_manifest,
        input_mpp=input_mpp,
        source_tile_size_px=source_tile_size_px,
        crop_size_px=crop_size_px,
        patch_size=patch_size,
        seed=seed,
        spacing_tolerance=spacing_tolerance,
        backend=backend,
        crop_scale=crop_scale,
        use_horizontal_flip=use_horizontal_flip,
        horizontal_flip_prob=horizontal_flip_prob,
        use_color_distortion=use_color_distortion,
        color_jitter_strength=color_jitter_strength,
        use_gaussian_blur=use_gaussian_blur,
        world_size=world_size,
        rank=rank,
        sampling_strategy=sampling_strategy,
        sampling_stratum_key=sampling_stratum_key,
        sampling_stratum_weights=sampling_stratum_weights,
        max_open_slides_per_worker=max_open_slides_per_worker,
        anchor_stream_batch_size=anchor_stream_batch_size,
    )

    validate_canonical_size_alignment(
        dataset=dataset,
        crop_size_px=int(crop_size_px),
        patch_size=int(patch_size),
    )

    loader_kwargs = {
        "dataset": dataset,
        "collate_fn": CanonicalCollateWithMetadata(
            crop_size_px=int(crop_size_px),
            patch_size=int(patch_size),
            enc_mask_scale=tuple(float(x) for x in enc_mask_scale),
            pred_mask_scale=tuple(float(x) for x in pred_mask_scale),
            aspect_ratio=tuple(float(x) for x in aspect_ratio),
            num_enc_masks=int(num_enc_masks),
            num_pred_masks=int(num_pred_masks),
            min_keep=int(min_keep),
            allow_overlap=bool(allow_overlap),
        ),
        "batch_size": int(batch_size),
        "drop_last": bool(drop_last),
        "pin_memory": bool(pin_mem),
        "num_workers": int(num_workers),
        "persistent_workers": bool(persistent_workers and int(num_workers) > 0),
    }
    if int(num_workers) > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))

    loader = DataLoader(**loader_kwargs)
    return dataset, loader, None

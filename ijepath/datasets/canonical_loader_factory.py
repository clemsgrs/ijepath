from __future__ import annotations

from torch.utils.data import DataLoader

from ijepath.datasets.canonical_wsi_dataset import CanonicalWSIDataset
from ijepath.masks.multiblock import MaskCollator


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
    transform_preset: str,
    num_enc_masks: int,
    num_pred_masks: int,
    min_keep: int,
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
        transform_preset=transform_preset,
        world_size=world_size,
        rank=rank,
        sampling_strategy=sampling_strategy,
        sampling_stratum_key=sampling_stratum_key,
        sampling_stratum_weights=sampling_stratum_weights,
        max_open_slides_per_worker=max_open_slides_per_worker,
        anchor_stream_batch_size=anchor_stream_batch_size,
    )

    collator = MaskCollator(
        input_size=int(crop_size_px),
        patch_size=int(patch_size),
        nenc=int(num_enc_masks),
        npred=int(num_pred_masks),
        min_keep=int(min_keep),
    )

    def _collate_with_metadata(batch: list[dict]):
        images = [sample["image"] for sample in batch]
        metadata = [dict(sample.get("sample_metadata", {})) for sample in batch]
        collated_images, masks_enc, masks_pred = collator(images)
        collated_batch = {
            "image": collated_images,
            "sample_metadata": metadata,
        }
        return collated_batch, masks_enc, masks_pred

    validate_canonical_size_alignment(
        dataset=dataset,
        crop_size_px=int(crop_size_px),
        patch_size=int(patch_size),
    )

    loader_kwargs = {
        "dataset": dataset,
        "collate_fn": _collate_with_metadata,
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

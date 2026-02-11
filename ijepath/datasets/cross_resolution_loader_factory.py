from torch.utils.data import DataLoader

from ijepath.datasets.cross_resolution_wsi_dataset import (
    CrossResolutionWSIDataset,
    snap_size_to_patch_multiple,
)
from ijepath.masks.context_target_footprint_mask_collator import ContextTargetFootprintMaskCollator


def validate_dataset_collator_size_alignment(
    dataset: CrossResolutionWSIDataset,
    context_input_size_px: int,
    target_input_size_px: int,
    patch_size: int,
) -> None:
    dataset_context_px = int(dataset.context_size_requested_px)
    dataset_target_px = int(dataset.target_size_requested_px)
    context_input_px = int(context_input_size_px)
    target_input_px = int(target_input_size_px)
    patch = int(patch_size)

    if dataset_context_px != context_input_px:
        raise ValueError(
            "Context input size drift: "
            f"dataset={dataset_context_px} collator={context_input_px}"
        )
    if dataset_target_px != target_input_px:
        raise ValueError(
            "Target input size drift: "
            f"dataset={dataset_target_px} encoder={target_input_px}"
        )
    if context_input_px % patch != 0:
        raise ValueError(
            "Context input size must be divisible by patch size: "
            f"context={context_input_px} patch={patch}"
        )
    if target_input_px % patch != 0:
        raise ValueError(
            "Target input size must be divisible by patch size: "
            f"target={target_input_px} patch={patch}"
        )


def make_cross_resolution_loader(
    batch_size: int,
    pin_mem: bool,
    num_workers: int,
    world_size: int,
    rank: int,
    drop_last: bool,
    anchor_catalog_manifest: str,
    patch_size: int,
    context_mpp: float,
    target_mpp: float,
    context_fov_um: float,
    target_fov_um: float,
    targets_per_context: int,
    seed: int,
    spacing_tolerance: float,
    min_target_tissue_fraction: float,
    insufficient_target_policy: str,
    min_target_tissue_fraction_floor: float | None,
    min_target_tissue_fraction_step: float,
    min_keep: int,
    num_enc_masks: int,
    backend: str = "asap",
    align_targets_to_patch_grid: bool = False,
    sampling_strategy: str = "stratified_weighted",
    sampling_stratum_key: str = "organ",
    sampling_stratum_weights: str | dict = "inverse_frequency",
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    max_open_slides_per_worker: int = 16,
):
    context_size_raw_px = max(1, int(round(float(context_fov_um) / float(context_mpp))))
    target_size_raw_px = max(1, int(round(float(target_fov_um) / float(target_mpp))))
    context_input_size_px = snap_size_to_patch_multiple(
        size_px=context_size_raw_px,
        patch_size=int(patch_size),
    )
    target_input_size_px = snap_size_to_patch_multiple(
        size_px=target_size_raw_px,
        patch_size=int(patch_size),
    )
    dataset = CrossResolutionWSIDataset(
        anchor_catalog_manifest=anchor_catalog_manifest,
        context_mpp=context_mpp,
        target_mpp=target_mpp,
        context_fov_um=context_fov_um,
        target_fov_um=target_fov_um,
        patch_size=patch_size,
        targets_per_context=targets_per_context,
        seed=seed,
        spacing_tolerance=spacing_tolerance,
        min_target_tissue_fraction=min_target_tissue_fraction,
        insufficient_target_policy=insufficient_target_policy,
        min_target_tissue_fraction_floor=min_target_tissue_fraction_floor,
        min_target_tissue_fraction_step=min_target_tissue_fraction_step,
        backend=backend,
        align_targets_to_patch_grid=align_targets_to_patch_grid,
        world_size=world_size,
        rank=rank,
        sampling_strategy=sampling_strategy,
        sampling_stratum_key=sampling_stratum_key,
        sampling_stratum_weights=sampling_stratum_weights,
        max_open_slides_per_worker=max_open_slides_per_worker,
    )

    collator = ContextTargetFootprintMaskCollator(
        input_size=context_input_size_px,
        patch_size=patch_size,
        nenc=num_enc_masks,
        min_keep=min_keep,
    )
    validate_dataset_collator_size_alignment(
        dataset=dataset,
        context_input_size_px=context_input_size_px,
        target_input_size_px=target_input_size_px,
        patch_size=patch_size,
    )

    loader_kwargs = {
        "dataset": dataset,
        "collate_fn": collator,
        "batch_size": batch_size,
        "drop_last": drop_last,
        "pin_memory": pin_mem,
        "num_workers": num_workers,
        "persistent_workers": bool(persistent_workers and num_workers > 0),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))

    loader = DataLoader(**loader_kwargs)

    return dataset, loader, None

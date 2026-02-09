from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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
    anchor_catalog_csv: str,
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
    backend: str = "openslide",
    align_targets_to_patch_grid: bool = False,
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
        anchor_catalog_csv=anchor_catalog_csv,
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
    )

    dist_sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
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

    loader = DataLoader(
        dataset,
        sampler=dist_sampler,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
    )

    return dataset, loader, dist_sampler

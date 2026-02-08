import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from ijepath.datasets.cross_resolution_loader_factory import make_cross_resolution_loader
from ijepath.datasets.cross_resolution_wsi_dataset import CrossResolutionWSIDataset
from ijepath.datasets.wsi_readers.wholeslidedata_reader_adapter import (
    WholeSlideDataReaderAdapter,
    spacing_pixels_to_level0_pixels,
)


def _write_min_anchor_csv(path: Path, row: dict):
    fieldnames = [
        "anchor_id",
        "slide_id",
        "wsi_path",
        "mask_path",
        "center_x_level0",
        "center_y_level0",
        "wsi_level0_spacing_mpp",
        "target_margin_um",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def test_reader_level0_border_in_bounds_threshold():
    repo_root = Path(__file__).resolve().parents[1]
    wsi_path = repo_root / "data/tcga-prad/wsi/TCGA-HC-8257.tif"
    if not wsi_path.exists():
        pytest.skip("WSI test data is not available")

    reader = WholeSlideDataReaderAdapter(wsi_path=str(wsi_path), backend="openslide")
    size_spacing_px = 512
    spacing_mpp = 1.0

    half_level0 = spacing_pixels_to_level0_pixels(
        size_pixels_at_spacing=size_spacing_px,
        spacing=spacing_mpp,
        spacing_at_level0=reader.geometry.spacing_at_level0_mpp,
    ) // 2

    assert reader.level0_center_in_bounds(
        center_x_level0=half_level0,
        center_y_level0=half_level0,
        size_pixels_at_spacing=size_spacing_px,
        spacing_mpp=spacing_mpp,
    )
    assert not reader.level0_center_in_bounds(
        center_x_level0=half_level0 - 1,
        center_y_level0=half_level0,
        size_pixels_at_spacing=size_spacing_px,
        spacing_mpp=spacing_mpp,
    )


def test_reader_patch_shape_is_stable_at_border():
    repo_root = Path(__file__).resolve().parents[1]
    wsi_path = repo_root / "data/tcga-prad/wsi/TCGA-HC-8257.tif"
    if not wsi_path.exists():
        pytest.skip("WSI test data is not available")

    reader = WholeSlideDataReaderAdapter(wsi_path=str(wsi_path), backend="openslide")
    patch = reader.get_patch_by_center_level0(
        center_x_level0=8,
        center_y_level0=8,
        width_pixels_at_spacing=512,
        height_pixels_at_spacing=512,
        spacing_mpp=1.0,
        use_mask=False,
    )

    assert patch.shape == (512, 512, 3)


def test_dataset_choose_source_spacing_prefers_finer_when_requested_missing(tmp_path):
    dummy_csv = tmp_path / "anchors.csv"
    _write_min_anchor_csv(
        dummy_csv,
        {
            "anchor_id": "dummy_0",
            "slide_id": "dummy",
            "wsi_path": "/tmp/nonexistent.tif",
            "mask_path": "",
            "center_x_level0": 0,
            "center_y_level0": 0,
            "wsi_level0_spacing_mpp": 0.25,
            "target_margin_um": 16.0,
        },
    )

    dataset = CrossResolutionWSIDataset(
        anchor_catalog_csv=str(dummy_csv),
        context_mpp=1.0,
        target_mpp=0.5,
        context_fov_um=512.0,
        target_fov_um=128.0,
        patch_size=16,
        targets_per_context=4,
        seed=0,
        spacing_tolerance=0.05,
    )

    spacing, mode = dataset._choose_source_spacing([0.252, 1.008, 4.032], requested_mpp=0.5)
    assert mode == "fallback_from_finer"
    assert spacing == 0.252

    spacing, mode = dataset._choose_source_spacing([0.252, 1.008, 4.032], requested_mpp=1.0)
    assert mode == "native_or_close"
    assert spacing == 1.008


def test_dataset_border_anchor_sample_shapes(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    wsi_path = repo_root / "data/tcga-prad/wsi/TCGA-HC-8257.tif"
    if not wsi_path.exists():
        pytest.skip("WSI test data is not available")

    anchor_csv = tmp_path / "edge_anchor.csv"
    _write_min_anchor_csv(
        anchor_csv,
        {
            "anchor_id": "edge_0",
            "slide_id": "TCGA-HC-8257",
            "wsi_path": str(wsi_path),
            "mask_path": "",
            "center_x_level0": 8,
            "center_y_level0": 8,
            "wsi_level0_spacing_mpp": 0.25200000393750005,
            "target_margin_um": 16.0,
        },
    )

    dataset = CrossResolutionWSIDataset(
        anchor_catalog_csv=str(anchor_csv),
        context_mpp=1.0,
        target_mpp=0.5,
        context_fov_um=512.0,
        target_fov_um=128.0,
        patch_size=16,
        targets_per_context=4,
        seed=0,
        spacing_tolerance=0.05,
        backend="openslide",
    )

    sample = dataset[0]
    assert sample["context_image"].shape == (3, 512, 512)
    assert sample["target_images"].shape == (4, 3, 256, 256)

    boxes = sample["target_boxes_in_context_pixels"].detach().cpu().numpy()
    assert np.isfinite(boxes).all()
    assert boxes.min() >= 0.0
    assert boxes.max() <= 512.0


def test_target_sampling_respects_min_tissue_fraction(tmp_path):
    dummy_csv = tmp_path / "anchors.csv"
    _write_min_anchor_csv(
        dummy_csv,
        {
            "anchor_id": "dummy_0",
            "slide_id": "dummy",
            "wsi_path": "/tmp/nonexistent.tif",
            "mask_path": "",
            "center_x_level0": 0,
            "center_y_level0": 0,
            "wsi_level0_spacing_mpp": 0.25,
            "target_margin_um": 8.0,
        },
    )
    dataset = CrossResolutionWSIDataset(
        anchor_catalog_csv=str(dummy_csv),
        context_mpp=1.0,
        target_mpp=0.5,
        context_fov_um=64.0,
        target_fov_um=16.0,
        patch_size=16,
        targets_per_context=4,
        seed=0,
        min_target_tissue_fraction=0.5,
    )

    tissue_mask = np.zeros((64, 64), dtype=np.float32)
    tissue_mask[16:48, 16:48] = 1.0

    sampled = dataset._sample_target_boxes_in_context(
        rng=np.random.default_rng(0),
        target_margin_context_px=8,
        context_size_px=64,
        target_size_context_px=16,
        context_tissue_mask=tissue_mask,
    )
    assert sampled is not None
    boxes, fractions = sampled

    assert boxes.shape == (4, 4)
    assert fractions is not None
    assert np.all(fractions >= 0.5)
    assert np.all(boxes[:, 0] >= 0)
    assert np.all(boxes[:, 1] >= 0)
    assert np.all(boxes[:, 2] <= 64)
    assert np.all(boxes[:, 3] <= 64)


def test_target_sampling_alignment_toggle(tmp_path):
    dummy_csv = tmp_path / "anchors.csv"
    _write_min_anchor_csv(
        dummy_csv,
        {
            "anchor_id": "dummy_0",
            "slide_id": "dummy",
            "wsi_path": "/tmp/nonexistent.tif",
            "mask_path": "",
            "center_x_level0": 0,
            "center_y_level0": 0,
            "wsi_level0_spacing_mpp": 0.25,
            "target_margin_um": 8.0,
        },
    )

    aligned_dataset = CrossResolutionWSIDataset(
        anchor_catalog_csv=str(dummy_csv),
        context_mpp=1.0,
        target_mpp=0.5,
        context_fov_um=64.0,
        target_fov_um=16.0,
        patch_size=8,
        targets_per_context=4,
        seed=0,
        align_targets_to_patch_grid=True,
    )
    sampled_aligned = aligned_dataset._sample_target_boxes_in_context(
        rng=np.random.default_rng(0),
        target_margin_context_px=8,
        context_size_px=64,
        target_size_context_px=16,
        context_tissue_mask=None,
    )
    assert sampled_aligned is not None
    aligned_boxes, _ = sampled_aligned
    assert np.all(np.mod(aligned_boxes, 8) == 0), "Aligned mode should snap box edges to patch grid"

    free_dataset = CrossResolutionWSIDataset(
        anchor_catalog_csv=str(dummy_csv),
        context_mpp=1.0,
        target_mpp=0.5,
        context_fov_um=64.0,
        target_fov_um=16.0,
        patch_size=8,
        targets_per_context=4,
        seed=0,
        align_targets_to_patch_grid=False,
    )
    sampled_free = free_dataset._sample_target_boxes_in_context(
        rng=np.random.default_rng(0),
        target_margin_context_px=8,
        context_size_px=64,
        target_size_context_px=16,
        context_tissue_mask=None,
    )
    assert sampled_free is not None
    free_boxes, _ = sampled_free
    assert not np.all(np.mod(free_boxes, 8) == 0), "Non-aligned mode should retain sub-patch offsets"


def test_target_sampling_returns_none_when_threshold_is_impossible(tmp_path):
    dummy_csv = tmp_path / "anchors.csv"
    _write_min_anchor_csv(
        dummy_csv,
        {
            "anchor_id": "dummy_0",
            "slide_id": "dummy",
            "wsi_path": "/tmp/nonexistent.tif",
            "mask_path": "",
            "center_x_level0": 0,
            "center_y_level0": 0,
            "wsi_level0_spacing_mpp": 0.25,
            "target_margin_um": 8.0,
        },
    )
    dataset = CrossResolutionWSIDataset(
        anchor_catalog_csv=str(dummy_csv),
        context_mpp=1.0,
        target_mpp=0.5,
        context_fov_um=64.0,
        target_fov_um=16.0,
        patch_size=16,
        targets_per_context=4,
        seed=0,
        min_target_tissue_fraction=0.95,
    )

    tissue_mask = np.zeros((64, 64), dtype=np.float32)
    tissue_mask[24:40, 24:40] = 1.0

    sampled = dataset._sample_target_boxes_in_context(
        rng=np.random.default_rng(0),
        target_margin_context_px=8,
        context_size_px=64,
        target_size_context_px=16,
        context_tissue_mask=tissue_mask,
    )
    assert sampled is None


def test_loader_factory_propagates_patch_alignment_toggle(tmp_path):
    anchor_csv = tmp_path / "anchors.csv"
    _write_min_anchor_csv(
        anchor_csv,
        {
            "anchor_id": "dummy_0",
            "slide_id": "dummy",
            "wsi_path": "/tmp/nonexistent.tif",
            "mask_path": "",
            "center_x_level0": 0,
            "center_y_level0": 0,
            "wsi_level0_spacing_mpp": 0.25,
            "target_margin_um": 8.0,
        },
    )

    dataset, _, _ = make_cross_resolution_loader(
        batch_size=1,
        pin_mem=False,
        num_workers=0,
        world_size=1,
        rank=0,
        drop_last=False,
        anchor_catalog_csv=str(anchor_csv),
        patch_size=8,
        context_mpp=1.0,
        target_mpp=0.5,
        context_fov_um=64.0,
        target_fov_um=16.0,
        targets_per_context=4,
        seed=0,
        spacing_tolerance=0.05,
        min_target_tissue_fraction=0.25,
        insufficient_target_policy="skip_anchor",
        min_target_tissue_fraction_floor=None,
        min_target_tissue_fraction_step=0.05,
        min_keep=4,
        num_enc_masks=1,
        backend="openslide",
        samples_per_epoch=1,
        align_targets_to_patch_grid=True,
    )

    assert dataset.align_targets_to_patch_grid is True


def test_rng_seed_varies_across_epochs_for_same_index(tmp_path):
    dummy_csv = tmp_path / "anchors.csv"
    _write_min_anchor_csv(
        dummy_csv,
        {
            "anchor_id": "dummy_0",
            "slide_id": "dummy",
            "wsi_path": "/tmp/nonexistent.tif",
            "mask_path": "",
            "center_x_level0": 0,
            "center_y_level0": 0,
            "wsi_level0_spacing_mpp": 0.25,
            "target_margin_um": 8.0,
        },
    )

    dataset = CrossResolutionWSIDataset(
        anchor_catalog_csv=str(dummy_csv),
        context_mpp=1.0,
        target_mpp=0.5,
        context_fov_um=64.0,
        target_fov_um=16.0,
        patch_size=16,
        targets_per_context=4,
        seed=123,
    )

    seed_e0_a = dataset._rng_seed_for(index=7, anchor_attempt=2)
    seed_e0_b = dataset._rng_seed_for(index=7, anchor_attempt=2)
    assert seed_e0_a == seed_e0_b

    dataset.set_epoch(1)
    seed_e1 = dataset._rng_seed_for(index=7, anchor_attempt=2)
    assert seed_e1 != seed_e0_a

    dataset.set_epoch(1)
    seed_e1_repeat = dataset._rng_seed_for(index=7, anchor_attempt=2)
    assert seed_e1 == seed_e1_repeat


def _make_dummy_sample(anchor_id: str, slide_id: str):
    return {
        "context_image": torch.zeros(3, 224, 224),
        "target_images": torch.zeros(4, 3, 224, 224),
        "target_boxes_in_context_pixels": torch.zeros(4, 4),
        "sample_metadata": {
            "anchor_id": anchor_id,
            "slide_id": slide_id,
        },
    }


def test_getitem_skip_slide_policy_exhausts_slide_before_switching_slides(tmp_path):
    anchor_csv = tmp_path / "anchors.csv"
    fieldnames = [
        "anchor_id",
        "slide_id",
        "wsi_path",
        "mask_path",
        "center_x_level0",
        "center_y_level0",
        "wsi_level0_spacing_mpp",
        "target_margin_um",
    ]
    with anchor_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "anchor_id": "a0",
                    "slide_id": "slideA",
                    "wsi_path": "/tmp/a.tif",
                    "mask_path": "",
                    "center_x_level0": 0,
                    "center_y_level0": 0,
                    "wsi_level0_spacing_mpp": 0.25,
                    "target_margin_um": 8.0,
                },
                {
                    "anchor_id": "a1",
                    "slide_id": "slideA",
                    "wsi_path": "/tmp/a.tif",
                    "mask_path": "",
                    "center_x_level0": 0,
                    "center_y_level0": 0,
                    "wsi_level0_spacing_mpp": 0.25,
                    "target_margin_um": 8.0,
                },
                {
                    "anchor_id": "b0",
                    "slide_id": "slideB",
                    "wsi_path": "/tmp/b.tif",
                    "mask_path": "",
                    "center_x_level0": 0,
                    "center_y_level0": 0,
                    "wsi_level0_spacing_mpp": 0.25,
                    "target_margin_um": 8.0,
                },
            ]
        )

    dataset = CrossResolutionWSIDataset(
        anchor_catalog_csv=str(anchor_csv),
        context_mpp=1.0,
        target_mpp=0.5,
        context_fov_um=64.0,
        target_fov_um=16.0,
        patch_size=16,
        targets_per_context=4,
        seed=0,
        insufficient_target_policy="skip_slide",
    )

    seen = []

    def fake_build(anchor, index, anchor_attempt, min_target_tissue_fraction=None):
        seen.append(anchor["anchor_id"])
        if anchor["anchor_id"] == "b0":
            return _make_dummy_sample(anchor_id="b0", slide_id="slideB")
        return None

    dataset._build_sample_from_anchor = fake_build  # type: ignore[method-assign]
    sample = dataset[0]
    assert sample["sample_metadata"]["anchor_id"] == "b0"
    assert seen == ["a0", "a1", "b0"]


def test_getitem_lower_threshold_policy_relaxes_threshold_until_success(tmp_path):
    anchor_csv = tmp_path / "anchors.csv"
    _write_min_anchor_csv(
        anchor_csv,
        {
            "anchor_id": "a0",
            "slide_id": "slideA",
            "wsi_path": "/tmp/a.tif",
            "mask_path": "",
            "center_x_level0": 0,
            "center_y_level0": 0,
            "wsi_level0_spacing_mpp": 0.25,
            "target_margin_um": 8.0,
        },
    )

    dataset = CrossResolutionWSIDataset(
        anchor_catalog_csv=str(anchor_csv),
        context_mpp=1.0,
        target_mpp=0.5,
        context_fov_um=64.0,
        target_fov_um=16.0,
        patch_size=16,
        targets_per_context=4,
        seed=0,
        min_target_tissue_fraction=0.5,
        insufficient_target_policy="lower_threshold",
        min_target_tissue_fraction_floor=0.2,
        min_target_tissue_fraction_step=0.1,
    )

    seen_thresholds = []

    def fake_build(anchor, index, anchor_attempt, min_target_tissue_fraction=None):
        seen_thresholds.append(min_target_tissue_fraction)
        if float(min_target_tissue_fraction) <= 0.3:
            return _make_dummy_sample(anchor_id="a0", slide_id="slideA")
        return None

    dataset._build_sample_from_anchor = fake_build  # type: ignore[method-assign]
    sample = dataset[0]

    assert sample["sample_metadata"]["anchor_id"] == "a0"
    assert seen_thresholds[:3] == [0.5, 0.4, 0.3]
    assert sample["sample_metadata"]["effective_min_target_tissue_fraction"] == 0.3


def test_getitem_lower_threshold_policy_raises_if_no_threshold_works(tmp_path):
    anchor_csv = tmp_path / "anchors.csv"
    _write_min_anchor_csv(
        anchor_csv,
        {
            "anchor_id": "a0",
            "slide_id": "slideA",
            "wsi_path": "/tmp/a.tif",
            "mask_path": "",
            "center_x_level0": 0,
            "center_y_level0": 0,
            "wsi_level0_spacing_mpp": 0.25,
            "target_margin_um": 8.0,
        },
    )

    dataset = CrossResolutionWSIDataset(
        anchor_catalog_csv=str(anchor_csv),
        context_mpp=1.0,
        target_mpp=0.5,
        context_fov_um=64.0,
        target_fov_um=16.0,
        patch_size=16,
        targets_per_context=4,
        seed=0,
        min_target_tissue_fraction=0.5,
        insufficient_target_policy="lower_threshold",
        min_target_tissue_fraction_floor=0.2,
        min_target_tissue_fraction_step=0.1,
    )

    dataset._build_sample_from_anchor = lambda *args, **kwargs: None  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="policy=lower_threshold"):
        _ = dataset[0]

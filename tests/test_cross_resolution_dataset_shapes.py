from pathlib import Path

import pytest
import torch

from src.datasets.cross_resolution_wsi_dataset import CrossResolutionWSIDataset


def test_cross_resolution_dataset_shapes_and_determinism():
    repo_root = Path(__file__).resolve().parents[1]
    anchor_csv = repo_root / "data/tcga-prad/indexes/anchors_profile_ctx1p0_tgt0p5_fov512um_k4.csv"

    if not anchor_csv.exists():
        pytest.skip("Anchor catalog is not available yet")

    dataset = CrossResolutionWSIDataset(
        anchor_catalog_csv=str(anchor_csv),
        crop_size=224,
        context_mpp=1.0,
        target_mpp=0.5,
        context_fov_um=512.0,
        target_fov_um=128.0,
        targets_per_context=4,
        seed=0,
        spacing_tolerance=0.05,
        min_target_tissue_fraction=0.25,
    )

    sample_a = dataset[0]
    sample_b = dataset[0]

    assert sample_a["context_image"].shape == (3, 224, 224)
    assert sample_a["target_images"].shape == (4, 3, 224, 224)
    assert sample_a["target_boxes_in_context_pixels"].shape == (4, 4)
    assert sample_a["sample_metadata"]["anchor_id"] == sample_b["sample_metadata"]["anchor_id"]

    assert torch.allclose(sample_a["context_image"], sample_b["context_image"])
    assert torch.allclose(sample_a["target_images"], sample_b["target_images"])
    assert torch.allclose(
        sample_a["target_boxes_in_context_pixels"],
        sample_b["target_boxes_in_context_pixels"],
    )

    metadata = sample_a["sample_metadata"]
    assert metadata["source_target_mpp"] <= metadata["output_target_mpp"]
    assert metadata["output_target_mpp"] == metadata["requested_target_mpp"]
    assert metadata["source_context_mpp"] > 0.0
    if metadata["target_resolution_mode"] == "fallback_from_finer":
        assert (
            metadata["target_size_target_px_at_effective_spacing"]
            > metadata["target_size_px_requested_spacing"]
        )

    if metadata.get("target_tissue_fractions") is not None:
        assert min(metadata["target_tissue_fractions"]) >= 0.25

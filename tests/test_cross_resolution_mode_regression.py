from pathlib import Path

import pytest
import yaml

from ijepath.config_loading import load_training_config
from ijepath.train import main as route_train


def _write_yaml(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def test_cross_resolution_mode_still_loads_without_canonical_overrides(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(
        cfg_path,
        {
            "pretraining": {"mode": "cross_resolution"},
            "data": {
                "slide_manifest_csv": "a.csv",
                "slide_metadata_parquet": "index.parquet",
                "anchor_catalog_manifest": "manifest.json",
                "batch_size_per_gpu": 2,
                "context_mpp": 1.0,
                "target_mpp": 0.5,
                "context_fov_um": 512.0,
                "target_fov_um": 128.0,
                "targets_per_context": 4,
            },
            "canonical": {
                "input_mpp": 0.5,
                "source_tile_size_px": 256,
                "crop_size_px": 224,
                "transform_preset": "official_ijepa",
                "mask_preset": "official_ijepa_multiblock",
                "enc_mask_scale": [0.85, 1.0],
                "pred_mask_scale": [0.15, 0.2],
                "aspect_ratio": [0.75, 1.5],
                "num_enc_masks": 1,
                "num_pred_masks": 4,
                "min_keep": 16,
            },
            "mask": {"num_pred_masks": 4, "num_enc_masks": 1, "min_keep": 16},
            "meta": {"architecture": "vit_small", "patch_size": 16},
            "optimization": {"total_images_budget": 1000},
        },
    )

    loaded = load_training_config(config_file=str(cfg_path))
    assert loaded["pretraining"]["mode"] == "cross_resolution"


def test_cross_resolution_mode_requires_geometry(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(
        cfg_path,
        {
            "pretraining": {"mode": "cross_resolution"},
            "data": {
                "slide_manifest_csv": "a.csv",
                "slide_metadata_parquet": "index.parquet",
                "anchor_catalog_manifest": "manifest.json",
                "batch_size_per_gpu": 2,
                "context_mpp": None,
                "target_mpp": 0.5,
                "context_fov_um": 512.0,
                "target_fov_um": 128.0,
                "targets_per_context": 4,
            },
            "canonical": {
                "input_mpp": 0.5,
                "source_tile_size_px": 256,
                "crop_size_px": 224,
                "transform_preset": "official_ijepa",
                "mask_preset": "official_ijepa_multiblock",
                "enc_mask_scale": [0.85, 1.0],
                "pred_mask_scale": [0.15, 0.2],
                "aspect_ratio": [0.75, 1.5],
                "num_enc_masks": 1,
                "num_pred_masks": 4,
                "min_keep": 16,
            },
            "mask": {"num_pred_masks": 4, "num_enc_masks": 1, "min_keep": 16},
            "meta": {"architecture": "vit_small", "patch_size": 16},
            "optimization": {"total_images_budget": 1000},
        },
    )

    with pytest.raises(ValueError, match="data.context_mpp"):
        load_training_config(config_file=str(cfg_path))

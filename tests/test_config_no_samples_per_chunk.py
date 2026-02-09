from pathlib import Path

import pytest
import yaml

from ijepath.config_loading import load_training_config


def _write_yaml(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _base_cfg() -> dict:
    return {
        "data": {
            "slide_manifest_csv": "a.csv",
            "slide_metadata_index_jsonl": "b.jsonl",
            "anchor_catalog_csv": "c.csv",
            "batch_size_per_gpu": 2,
            "context_mpp": 1.0,
            "target_mpp": 0.5,
            "context_fov_um": 512.0,
            "target_fov_um": 128.0,
            "targets_per_context": 4,
        },
        "meta": {"architecture": "vit_small", "patch_size": 16},
        "mask": {"num_enc_masks": 1, "num_pred_masks": 4, "min_keep": 8},
        "optimization": {"total_images_budget": 1000},
    }


def test_samples_per_chunk_is_rejected(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg = _base_cfg()
    cfg["data"]["samples_per_chunk"] = 10
    _write_yaml(cfg_path, cfg)

    with pytest.raises(ValueError, match="data.samples_per_chunk"):
        load_training_config(config_file=str(cfg_path))

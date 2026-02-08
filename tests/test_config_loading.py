from pathlib import Path

import pytest
import yaml

from src.config_loading import load_training_config


def _write_yaml(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def test_layered_config_merge_and_opts_override(tmp_path: Path):
    default_cfg = tmp_path / "default.yaml"
    profile_cfg = tmp_path / "profile.yaml"
    run_cfg = tmp_path / "run.yaml"

    _write_yaml(
        default_cfg,
        {
            "data": {
                "slide_manifest_csv": "default_manifest.csv",
                "slide_metadata_index_jsonl": "default_index.jsonl",
                "anchor_catalog_csv": "default_anchor.csv",
                "context_mpp": 1.0,
                "target_mpp": 0.5,
                "context_fov_um": 512.0,
                "target_fov_um": 128.0,
                "targets_per_context": 4,
                "batch_size_per_gpu": 2,
            },
            "mask": {"num_pred_masks": None, "patch_size": 16, "num_enc_masks": 1, "min_keep": 10},
        },
    )
    _write_yaml(
        profile_cfg,
        {
            "context_mpp": 0.8,
            "target_mpp": 0.4,
            "context_fov_um": 480.0,
            "target_fov_um": 120.0,
            "targets_per_context": 6,
        },
    )
    _write_yaml(
        run_cfg,
        {
            "data": {
                "slide_manifest_csv": "run_manifest.csv",
                "slide_metadata_index_jsonl": "run_index.jsonl",
                "anchor_catalog_csv": "run_anchor.csv",
            }
        },
    )

    cfg = load_training_config(
        default_config=str(default_cfg),
        profile_config=str(profile_cfg),
        run_config=str(run_cfg),
        opts=["data.batch_size_per_gpu=8", "data.targets_per_context=5"],
    )

    assert cfg["data"]["slide_manifest_csv"] == "run_manifest.csv"
    assert cfg["data"]["context_mpp"] == 0.8
    assert cfg["data"]["target_mpp"] == 0.4
    assert cfg["data"]["batch_size_per_gpu"] == 8
    assert cfg["data"]["targets_per_context"] == 5
    assert cfg["mask"]["num_pred_masks"] == 5
    assert cfg["_config_sources"]["mode"] == "layered"
    assert cfg["_config_sources"]["profile_config"] == str(profile_cfg)


def test_layered_config_requires_required_paths(tmp_path: Path):
    default_cfg = tmp_path / "default.yaml"
    profile_cfg = tmp_path / "profile.yaml"
    run_cfg = tmp_path / "run.yaml"

    _write_yaml(
        default_cfg,
        {
            "data": {
                "slide_manifest_csv": None,
                "slide_metadata_index_jsonl": None,
                "anchor_catalog_csv": None,
                "batch_size_per_gpu": 2,
                "context_mpp": 1.0,
                "target_mpp": 0.5,
                "context_fov_um": 512.0,
                "target_fov_um": 128.0,
                "targets_per_context": 4,
            },
            "mask": {"num_pred_masks": 4, "patch_size": 16, "num_enc_masks": 1, "min_keep": 10},
        },
    )
    _write_yaml(profile_cfg, {"data": {}})
    _write_yaml(run_cfg, {"data": {}})

    with pytest.raises(ValueError, match="data.slide_manifest_csv"):
        load_training_config(
            default_config=str(default_cfg),
            profile_config=str(profile_cfg),
            run_config=str(run_cfg),
        )


def test_layered_config_enforces_num_pred_masks_matches_targets(tmp_path: Path):
    default_cfg = tmp_path / "default.yaml"
    profile_cfg = tmp_path / "profile.yaml"
    run_cfg = tmp_path / "run.yaml"

    _write_yaml(
        default_cfg,
        {
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
            "mask": {"num_pred_masks": 3, "patch_size": 16, "num_enc_masks": 1, "min_keep": 10},
        },
    )
    _write_yaml(profile_cfg, {"data": {}})
    _write_yaml(run_cfg, {"data": {}})

    with pytest.raises(ValueError, match="mask.num_pred_masks"):
        load_training_config(
            default_config=str(default_cfg),
            profile_config=str(profile_cfg),
            run_config=str(run_cfg),
        )


def test_layered_config_rejects_conflicting_duplicate_values_within_profile(tmp_path: Path):
    default_cfg = tmp_path / "default.yaml"
    profile_cfg = tmp_path / "profile.yaml"
    run_cfg = tmp_path / "run.yaml"

    _write_yaml(
        default_cfg,
        {
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
            "mask": {"num_pred_masks": 4, "patch_size": 16, "num_enc_masks": 1, "min_keep": 10},
        },
    )
    _write_yaml(profile_cfg, {"context_mpp": 0.8, "data": {"context_mpp": 1.2}})
    _write_yaml(run_cfg, {"data": {}})

    with pytest.raises(ValueError, match="Conflicting values within profile config"):
        load_training_config(
            default_config=str(default_cfg),
            profile_config=str(profile_cfg),
            run_config=str(run_cfg),
        )


def test_legacy_batch_size_is_accepted_and_projected(tmp_path: Path):
    default_cfg = tmp_path / "default.yaml"
    profile_cfg = tmp_path / "profile.yaml"
    run_cfg = tmp_path / "run.yaml"

    _write_yaml(
        default_cfg,
        {
            "data": {
                "slide_manifest_csv": "a.csv",
                "slide_metadata_index_jsonl": "b.jsonl",
                "anchor_catalog_csv": "c.csv",
                "batch_size": 6,
                "context_mpp": 1.0,
                "target_mpp": 0.5,
                "context_fov_um": 512.0,
                "target_fov_um": 128.0,
                "targets_per_context": 4,
            },
            "mask": {"num_pred_masks": 4, "patch_size": 16, "num_enc_masks": 1, "min_keep": 10},
        },
    )
    _write_yaml(profile_cfg, {"data": {}})
    _write_yaml(run_cfg, {"data": {}})

    cfg = load_training_config(
        default_config=str(default_cfg),
        profile_config=str(profile_cfg),
        run_config=str(run_cfg),
    )
    assert cfg["data"]["batch_size_per_gpu"] == 6

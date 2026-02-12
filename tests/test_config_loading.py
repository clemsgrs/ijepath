from pathlib import Path

import pytest
import yaml

from ijepath.config_loading import load_training_config


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
                "slide_metadata_parquet": "default_index.jsonl",
                "anchor_catalog_manifest": "default_anchor.csv",
                "context_mpp": 1.0,
                "target_mpp": 0.5,
                "context_fov_um": 512.0,
                "target_fov_um": 128.0,
                "targets_per_context": 4,
                "batch_size_per_gpu": 2,
            },
            "mask": {"num_pred_masks": None, "num_enc_masks": 1, "min_keep": 10},
            "meta": {"architecture": "vit_small", "patch_size": 16},
            "optimization": {"total_images_budget": 1000},
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
                "slide_metadata_parquet": "run_index.jsonl",
                "anchor_catalog_manifest": "run_anchor.csv",
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
                "slide_metadata_parquet": None,
                "anchor_catalog_manifest": None,
                "batch_size_per_gpu": 2,
                "context_mpp": 1.0,
                "target_mpp": 0.5,
                "context_fov_um": 512.0,
                "target_fov_um": 128.0,
                "targets_per_context": 4,
            },
            "mask": {"num_pred_masks": 4, "num_enc_masks": 1, "min_keep": 10},
            "meta": {"architecture": "vit_small", "patch_size": 16},
            "optimization": {"total_images_budget": 1000},
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
                "slide_metadata_parquet": "b.jsonl",
                "anchor_catalog_manifest": "c.csv",
                "batch_size_per_gpu": 2,
                "context_mpp": 1.0,
                "target_mpp": 0.5,
                "context_fov_um": 512.0,
                "target_fov_um": 128.0,
                "targets_per_context": 4,
            },
            "mask": {"num_pred_masks": 3, "num_enc_masks": 1, "min_keep": 10},
            "meta": {"architecture": "vit_small", "patch_size": 16},
            "optimization": {"total_images_budget": 1000},
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
                "slide_metadata_parquet": "b.jsonl",
                "anchor_catalog_manifest": "c.csv",
                "batch_size_per_gpu": 2,
                "context_mpp": 1.0,
                "target_mpp": 0.5,
                "context_fov_um": 512.0,
                "target_fov_um": 128.0,
                "targets_per_context": 4,
            },
            "mask": {"num_pred_masks": 4, "num_enc_masks": 1, "min_keep": 10},
            "meta": {"architecture": "vit_small", "patch_size": 16},
            "optimization": {"total_images_budget": 1000},
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


def test_legacy_batch_size_is_rejected(tmp_path: Path):
    default_cfg = tmp_path / "default.yaml"
    profile_cfg = tmp_path / "profile.yaml"
    run_cfg = tmp_path / "run.yaml"

    _write_yaml(
        default_cfg,
        {
            "data": {
                "slide_manifest_csv": "a.csv",
                "slide_metadata_parquet": "b.jsonl",
                "anchor_catalog_manifest": "c.csv",
                "batch_size": 6,
                "context_mpp": 1.0,
                "target_mpp": 0.5,
                "context_fov_um": 512.0,
                "target_fov_um": 128.0,
                "targets_per_context": 4,
            },
            "mask": {"num_pred_masks": 4, "num_enc_masks": 1, "min_keep": 10},
            "meta": {"architecture": "vit_small", "patch_size": 16},
            "optimization": {"total_images_budget": 1000},
        },
    )
    _write_yaml(profile_cfg, {"data": {}})
    _write_yaml(run_cfg, {"data": {}})

    with pytest.raises(ValueError, match="data.batch_size"):
        load_training_config(
            default_config=str(default_cfg),
            profile_config=str(profile_cfg),
            run_config=str(run_cfg),
        )


def test_legacy_anchor_catalog_and_slide_index_keys_are_rejected(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(
        cfg_path,
        {
            "data": {
                "slide_manifest_csv": "a.csv",
                "slide_metadata_index_jsonl": "legacy_index.jsonl",
                "anchor_catalog_csv": "legacy_anchor.csv",
                "slide_metadata_parquet": "new_index.parquet",
                "anchor_catalog_manifest": "new_manifest.json",
                "batch_size_per_gpu": 2,
                "context_mpp": 1.0,
                "target_mpp": 0.5,
                "context_fov_um": 512.0,
                "target_fov_um": 128.0,
                "targets_per_context": 4,
            },
            "mask": {"num_pred_masks": 4, "num_enc_masks": 1, "min_keep": 10},
            "meta": {"architecture": "vit_small", "patch_size": 16},
            "optimization": {"total_images_budget": 1000},
        },
    )

    with pytest.raises(ValueError, match="data.slide_metadata_index_jsonl"):
        load_training_config(config_file=str(cfg_path))


def test_training_cadence_values_must_be_positive_when_provided(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(
        cfg_path,
        {
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
            "mask": {"num_pred_masks": 4, "num_enc_masks": 1, "min_keep": 10},
            "meta": {"architecture": "vit_small", "patch_size": 16},
            "optimization": {"total_images_budget": 1000},
            "training": {"log_every": 0, "save_every": 1000},
            "tuning": {"tune_every": 1000},
        },
    )

    with pytest.raises(ValueError, match="training.log_every must be > 0"):
        load_training_config(config_file=str(cfg_path))


def test_legacy_checkpoint_and_tuning_schedule_keys_are_rejected(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(
        cfg_path,
        {
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
            "mask": {"num_pred_masks": 4, "num_enc_masks": 1, "min_keep": 10},
            "meta": {"architecture": "vit_small", "patch_size": 16},
            "optimization": {"total_images_budget": 1000},
            "logging": {"checkpoint_every_images": 1000},
            "tuning": {"tune_every": 1000},
        },
    )

    with pytest.raises(ValueError, match="logging.checkpoint_every_images"):
        load_training_config(config_file=str(cfg_path))


def test_legacy_tuning_schedule_key_is_rejected(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(
        cfg_path,
        {
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
            "mask": {"num_pred_masks": 4, "num_enc_masks": 1, "min_keep": 10},
            "meta": {"architecture": "vit_small", "patch_size": 16},
            "optimization": {"total_images_budget": 1000},
            "tuning": {"schedule": {"interval_images": 1000, "run_baseline_at_zero": True}},
        },
    )

    with pytest.raises(ValueError, match="tuning.schedule"):
        load_training_config(config_file=str(cfg_path))


def test_legacy_wandb_log_every_images_key_is_rejected(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(
        cfg_path,
        {
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
            "mask": {"num_pred_masks": 4, "num_enc_masks": 1, "min_keep": 10},
            "meta": {"architecture": "vit_small", "patch_size": 16},
            "optimization": {"total_images_budget": 1000},
            "wandb": {"log_every_images": 1000},
        },
    )

    with pytest.raises(ValueError, match="wandb.log_every_images"):
        load_training_config(config_file=str(cfg_path))


def test_anchor_stream_batch_size_must_be_positive(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(
        cfg_path,
        {
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
                "anchor_stream_batch_size": 0,
            },
            "mask": {"num_pred_masks": 4, "num_enc_masks": 1, "min_keep": 10},
            "meta": {"architecture": "vit_small", "patch_size": 16},
            "optimization": {"total_images_budget": 1000},
        },
    )

    with pytest.raises(ValueError, match="data.anchor_stream_batch_size must be > 0"):
        load_training_config(config_file=str(cfg_path))


def test_performance_debug_logging_values_must_be_positive_when_enabled(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(
        cfg_path,
        {
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
            "mask": {"num_pred_masks": 4, "num_enc_masks": 1, "min_keep": 10},
            "meta": {"architecture": "vit_small", "patch_size": 16},
            "optimization": {"total_images_budget": 1000},
            "logging": {
                "performance_debug": {
                    "enable": True,
                    "log_every_images": 0,
                    "slow_step_ms": 250.0,
                    "slow_data_wait_ms": 50.0,
                }
            },
        },
    )

    with pytest.raises(ValueError, match="logging.performance_debug.log_every_images must be > 0"):
        load_training_config(config_file=str(cfg_path))


def test_legacy_step_log_every_iters_key_is_rejected(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(
        cfg_path,
        {
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
            "mask": {"num_pred_masks": 4, "num_enc_masks": 1, "min_keep": 10},
            "meta": {"architecture": "vit_small", "patch_size": 16},
            "optimization": {"total_images_budget": 1000},
            "logging": {"step_log_every_iters": 10},
        },
    )

    with pytest.raises(ValueError, match="logging.step_log_every_iters"):
        load_training_config(config_file=str(cfg_path))


def test_step_log_every_images_rejects_invalid_type_and_range(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    base_cfg = {
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
        "mask": {"num_pred_masks": 4, "num_enc_masks": 1, "min_keep": 10},
        "meta": {"architecture": "vit_small", "patch_size": 16},
        "optimization": {"total_images_budget": 1000},
    }

    bad_cfg = dict(base_cfg)
    bad_cfg["logging"] = {"step_log_every_images": "10%"}
    _write_yaml(cfg_path, bad_cfg)
    with pytest.raises(ValueError, match="logging.step_log_every_images"):
        load_training_config(config_file=str(cfg_path))

    bad_cfg = dict(base_cfg)
    bad_cfg["logging"] = {"step_log_every_images": True}
    _write_yaml(cfg_path, bad_cfg)
    with pytest.raises(ValueError, match="logging.step_log_every_images"):
        load_training_config(config_file=str(cfg_path))

    bad_cfg = dict(base_cfg)
    bad_cfg["logging"] = {"step_log_every_images": -5}
    _write_yaml(cfg_path, bad_cfg)
    with pytest.raises(ValueError, match="logging.step_log_every_images"):
        load_training_config(config_file=str(cfg_path))

    bad_cfg = dict(base_cfg)
    bad_cfg["logging"] = {"step_log_every_images": 1.1}
    _write_yaml(cfg_path, bad_cfg)
    with pytest.raises(ValueError, match="logging.step_log_every_images"):
        load_training_config(config_file=str(cfg_path))

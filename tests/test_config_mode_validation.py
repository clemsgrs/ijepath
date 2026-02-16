from pathlib import Path

import pytest
import yaml

from ijepath.config_loading import infer_pretraining_mode, load_training_config


def _write_yaml(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _base_cfg() -> dict:
    return {
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
            "enc_mask_scale": [0.85, 1.0],
            "pred_mask_scale": [0.15, 0.2],
            "aspect_ratio": [0.75, 1.5],
            "crop_scale": [0.3, 1.0],
            "use_horizontal_flip": True,
            "horizontal_flip_prob": 0.5,
            "use_color_distortion": False,
            "color_jitter_strength": 0.0,
            "use_gaussian_blur": False,
            "allow_overlap": False,
            "num_enc_masks": 1,
            "num_pred_masks": 4,
            "min_keep": 16,
        },
        "mask": {"num_pred_masks": 4, "num_enc_masks": 1, "min_keep": 16},
        "meta": {"architecture": "vit_small", "patch_size": 16},
        "optimization": {"total_images_budget": 1000},
    }


def test_pretraining_mode_is_required(tmp_path: Path):
    cfg = _base_cfg()
    cfg.pop("pretraining")
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(cfg_path, cfg)

    with pytest.raises(ValueError, match="pretraining.mode"):
        load_training_config(config_file=str(cfg_path))


def test_pretraining_mode_is_required_in_layered_configs(tmp_path: Path):
    default_cfg = {
        "data": {
            "batch_size_per_gpu": 2,
        },
        "meta": {
            "architecture": "vit_small",
            "patch_size": 16,
        },
        "optimization": {
            "total_images_budget": 1000,
        },
    }
    profile_cfg = {
        "context_mpp": 1.0,
        "target_mpp": 0.5,
        "context_fov_um": 512.0,
        "target_fov_um": 128.0,
        "targets_per_context": 4,
    }
    run_cfg = {
        "data": {
            "slide_manifest_csv": "a.csv",
            "slide_metadata_parquet": "index.parquet",
            "anchor_catalog_manifest": "manifest.json",
        },
    }

    default_path = tmp_path / "defaults.yaml"
    profile_path = tmp_path / "profile.yaml"
    run_path = tmp_path / "run.yaml"
    _write_yaml(default_path, default_cfg)
    _write_yaml(profile_path, profile_cfg)
    _write_yaml(run_path, run_cfg)

    with pytest.raises(ValueError, match="pretraining.mode"):
        load_training_config(
            default_config=str(default_path),
            profile_config=str(profile_path),
            run_config=str(run_path),
        )


def test_pretraining_mode_must_be_valid(tmp_path: Path):
    cfg = _base_cfg()
    cfg["pretraining"]["mode"] = "invalid"
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(cfg_path, cfg)

    with pytest.raises(ValueError, match="pretraining.mode"):
        load_training_config(config_file=str(cfg_path))


def test_canonical_mode_requires_canonical_fields(tmp_path: Path):
    cfg = _base_cfg()
    cfg["pretraining"]["mode"] = "canonical"
    cfg["canonical"]["input_mpp"] = None
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(cfg_path, cfg)

    with pytest.raises(ValueError, match="canonical.input_mpp"):
        load_training_config(config_file=str(cfg_path))


def test_canonical_mode_does_not_require_cross_resolution_geometry(tmp_path: Path):
    cfg = _base_cfg()
    cfg["pretraining"]["mode"] = "canonical"
    cfg["data"]["context_mpp"] = None
    cfg["data"]["target_mpp"] = None
    cfg["data"]["context_fov_um"] = None
    cfg["data"]["target_fov_um"] = None
    cfg["data"]["targets_per_context"] = None
    cfg["mask"]["num_pred_masks"] = None
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(cfg_path, cfg)

    loaded = load_training_config(config_file=str(cfg_path))
    assert loaded["pretraining"]["mode"] == "canonical"
    assert loaded["canonical"]["crop_size_px"] == 224


def test_cross_resolution_mode_does_not_require_canonical_fields(tmp_path: Path):
    cfg = _base_cfg()
    cfg["pretraining"]["mode"] = "cross_resolution"
    cfg["canonical"] = {}
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(cfg_path, cfg)

    loaded = load_training_config(config_file=str(cfg_path))
    assert loaded["pretraining"]["mode"] == "cross_resolution"


def test_canonical_mode_does_not_require_transform_preset(tmp_path: Path):
    cfg = _base_cfg()
    cfg["pretraining"]["mode"] = "canonical"
    cfg["canonical"].pop("transform_preset", None)
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(cfg_path, cfg)

    loaded = load_training_config(config_file=str(cfg_path))
    assert loaded["pretraining"]["mode"] == "canonical"


def test_canonical_allow_overlap_must_be_boolean(tmp_path: Path):
    cfg = _base_cfg()
    cfg["pretraining"]["mode"] = "canonical"
    cfg["canonical"]["allow_overlap"] = "yes"
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(cfg_path, cfg)

    with pytest.raises(ValueError, match="canonical.allow_overlap"):
        load_training_config(config_file=str(cfg_path))


def test_canonical_crop_scale_must_be_unit_interval_pair(tmp_path: Path):
    cfg = _base_cfg()
    cfg["pretraining"]["mode"] = "canonical"
    cfg["canonical"]["crop_scale"] = [0.0, 1.2]
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(cfg_path, cfg)

    with pytest.raises(ValueError, match="canonical.crop_scale"):
        load_training_config(config_file=str(cfg_path))


def test_canonical_horizontal_flip_prob_must_be_unit_interval(tmp_path: Path):
    cfg = _base_cfg()
    cfg["pretraining"]["mode"] = "canonical"
    cfg["canonical"]["horizontal_flip_prob"] = 1.5
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(cfg_path, cfg)

    with pytest.raises(ValueError, match="canonical.horizontal_flip_prob"):
        load_training_config(config_file=str(cfg_path))


def test_infer_pretraining_mode_prefers_opts_over_run_and_profile(tmp_path: Path):
    profile_path = tmp_path / "profile.yaml"
    run_path = tmp_path / "run.yaml"
    _write_yaml(profile_path, {"pretraining": {"mode": "cross_resolution"}})
    _write_yaml(run_path, {"pretraining": {"mode": "cross_resolution"}})

    mode = infer_pretraining_mode(
        profile_config=str(profile_path),
        run_config=str(run_path),
        opts=["pretraining.mode=canonical"],
    )
    assert mode == "canonical"


def test_infer_pretraining_mode_uses_run_before_profile(tmp_path: Path):
    profile_path = tmp_path / "profile.yaml"
    run_path = tmp_path / "run.yaml"
    _write_yaml(profile_path, {"pretraining": {"mode": "cross_resolution"}})
    _write_yaml(run_path, {"pretraining": {"mode": "canonical"}})

    mode = infer_pretraining_mode(
        profile_config=str(profile_path),
        run_config=str(run_path),
        opts=[],
    )
    assert mode == "canonical"


def test_infer_pretraining_mode_errors_when_missing_everywhere(tmp_path: Path):
    profile_path = tmp_path / "profile.yaml"
    run_path = tmp_path / "run.yaml"
    _write_yaml(profile_path, {"x": 1})
    _write_yaml(run_path, {"y": 2})

    with pytest.raises(ValueError, match="Unable to infer pretraining.mode"):
        infer_pretraining_mode(
            profile_config=str(profile_path),
            run_config=str(run_path),
            opts=[],
        )

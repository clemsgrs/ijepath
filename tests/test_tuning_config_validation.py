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
        "tuning": {"enable": False, "plugins": []},
    }


def test_total_images_budget_required(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg = _base_cfg()
    cfg["optimization"]["total_images_budget"] = None
    _write_yaml(cfg_path, cfg)

    with pytest.raises(ValueError, match="optimization.total_images_budget"):
        load_training_config(config_file=str(cfg_path))


def test_samples_per_epoch_is_rejected(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg = _base_cfg()
    cfg["data"]["samples_per_epoch"] = 12
    _write_yaml(cfg_path, cfg)

    with pytest.raises(ValueError, match="data.samples_per_epoch"):
        load_training_config(config_file=str(cfg_path))


def test_optimization_epochs_is_rejected(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg = _base_cfg()
    cfg["optimization"]["epochs"] = 100
    _write_yaml(cfg_path, cfg)

    with pytest.raises(ValueError, match="optimization.epochs"):
        load_training_config(config_file=str(cfg_path))


def test_checkpoint_every_epochs_is_rejected(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg = _base_cfg()
    cfg["logging"] = {"checkpoint_every_epochs": 10}
    _write_yaml(cfg_path, cfg)

    with pytest.raises(ValueError, match="logging.checkpoint_every_epochs"):
        load_training_config(config_file=str(cfg_path))


def test_tuning_early_stopping_requires_selected_plugin(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg = _base_cfg()
    cfg["tuning"] = {
        "enable": True,
        "plugins": [
            {
                "type": "pathorob",
                "enable": True,
                "use_for_early_stopping": False,
                "datasets": {"camelyon": {"enable": False}},
            }
        ],
        "early_stopping": {
            "enable": True,
            "patience_evals": 2,
            "min_evals": 1,
            "stop_training": False,
            "save_best_checkpoint": True,
            "best_checkpoint_name": "best-robustness.pth.tar",
        },
    }
    _write_yaml(cfg_path, cfg)

    with pytest.raises(ValueError, match="use_for_early_stopping"):
        load_training_config(config_file=str(cfg_path))


def test_tuning_rejects_multiple_selected_plugins(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg = _base_cfg()
    cfg["tuning"] = {
        "enable": True,
        "plugins": [
            {"type": "pathorob", "enable": True, "use_for_early_stopping": True, "datasets": {"camelyon": {"enable": False}}},
            {"type": "pathorob", "enable": True, "use_for_early_stopping": True, "datasets": {"camelyon": {"enable": False}}},
        ],
        "early_stopping": {"enable": True},
    }
    _write_yaml(cfg_path, cfg)

    with pytest.raises(ValueError, match="At most one enabled plugin"):
        load_training_config(config_file=str(cfg_path))


def test_selected_early_stopping_plugin_requires_metric_key(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg = _base_cfg()
    cfg["tuning"] = {
        "enable": True,
        "plugins": [
            {
                "type": "pathorob",
                "enable": True,
                "use_for_early_stopping": True,
                "datasets": {"camelyon": {"enable": False}},
            }
        ],
        "early_stopping": {"enable": True},
    }
    _write_yaml(cfg_path, cfg)

    with pytest.raises(ValueError, match="early_stopping_metric"):
        load_training_config(config_file=str(cfg_path))


def test_selected_early_stopping_plugin_rejects_invalid_mode(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg = _base_cfg()
    cfg["tuning"] = {
        "enable": True,
        "plugins": [
            {
                "type": "pathorob",
                "enable": True,
                "use_for_early_stopping": True,
                "early_stopping_metric": "camelyon/apd_avg",
                "early_stopping_mode": "upward",
                "datasets": {"camelyon": {"enable": False}},
            }
        ],
        "early_stopping": {"enable": True},
    }
    _write_yaml(cfg_path, cfg)

    with pytest.raises(ValueError, match="early_stopping_mode"):
        load_training_config(config_file=str(cfg_path))

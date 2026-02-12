from pathlib import Path

import pytest
import torch

from ijepath.eval.plugins.pathorob import PathoROBPlugin
from ijepath.eval.tuner import Tuner
from ijepath.eval.plugins.base import PluginResult


def test_unknown_plugin_type_raises(tmp_path: Path):
    cfg = {
        "enable": True,
        "plugins": [
            {"type": "unknown", "enable": True},
        ],
    }

    with pytest.raises(ValueError, match="Unknown tuning plugin type"):
        Tuner(cfg=cfg, device=torch.device("cpu"), output_dir=tmp_path)


def test_disabled_plugins_are_skipped(tmp_path: Path):
    cfg = {
        "enable": True,
        "plugins": [
            {"type": "pathorob", "enable": False},
        ],
    }
    tuner = Tuner(cfg=cfg, device=torch.device("cpu"), output_dir=tmp_path)
    assert tuner.plugins == []
    assert tuner.primary_plugin_name is None


def test_selected_plugin_name_is_resolved(tmp_path: Path):
    cfg = {
        "enable": True,
        "early_stopping": {
            "enable": True,
            "selection": {"plugin": "pathorob", "dataset": "camelyon", "metric": "ri"},
        },
        "plugins": [
            {
                "type": "pathorob",
                "enable": True,
                "datasets": {"camelyon": {"enable": False}},
            }
        ],
    }
    tuner = Tuner(cfg=cfg, device=torch.device("cpu"), output_dir=tmp_path)
    assert tuner.primary_plugin_name == "pathorob"


def test_selected_plugin_must_emit_selection_metric_value(tmp_path: Path):
    cfg = {
        "enable": True,
        "early_stopping": {
            "enable": True,
            "selection": {"plugin": "pathorob", "dataset": "camelyon", "metric": "ri"},
        },
        "plugins": [
            {
                "type": "pathorob",
                "enable": True,
                "datasets": {"camelyon": {"enable": False}},
            }
        ],
    }
    tuner = Tuner(cfg=cfg, device=torch.device("cpu"), output_dir=tmp_path)

    class _DummyPlugin:
        name = "pathorob"

        def should_run(self, images_seen: int, tune_index: int) -> bool:
            return True

        def run(self, teacher, tune_index: int, images_seen: int):
            return PluginResult(
                name="pathorob",
                payload={},
                log_metrics={},
                selection_metric_value=None,
                selection_mode="max",
            )

    tuner.plugins = [_DummyPlugin()]
    with pytest.raises(ValueError, match="Configured early-stopping target was not emitted"):
        tuner.tune(teacher=object(), tune_index=0, images_seen=0)


def test_pathorob_transform_can_disable_center_crop(tmp_path: Path):
    plugin = PathoROBPlugin(
        cfg={
            "type": "pathorob",
            "enable": True,
            "transforms": {"resize": 256, "crop_size": None, "normalize": "imagenet"},
            "datasets": {"camelyon": {"enable": False}},
        },
        device=torch.device("cpu"),
        output_dir=tmp_path,
    )
    op_names = [type(op).__name__ for op in plugin.transform.transforms]
    assert "CenterCrop" not in op_names

from pathlib import Path

import pandas as pd
import torch

from ijepath.eval.plugins.base import PluginResult
from ijepath.eval.tuner import Tuner


class _DummyPlugin:
    name = "dummy"

    def bind_runtime(self, _output_dir):
        return None

    def should_run(self, images_seen: int, eval_index: int) -> bool:
        return True

    def run(self, teacher, eval_index: int, images_seen: int):
        return PluginResult(
            name=self.name,
            payload={"rows": [{"metric": "m", "value": 0.3}]},
            log_metrics={"score": 0.3},
        )


def test_tuner_persists_eval_named_artifacts(tmp_path: Path):
    tuner = Tuner(cfg={"enable": True, "plugins": []}, device=torch.device("cpu"), output_dir=tmp_path)
    tuner.plugins = [_DummyPlugin()]

    out = tuner.tune(teacher=object(), eval_index=3, images_seen=1_000_000)

    assert out["log_metrics"]["dummy/score"] == 0.3
    event_csv = tmp_path / "metrics" / "eval_0003.csv"
    assert event_csv.exists()
    df = pd.read_csv(event_csv)
    assert int(df.iloc[0]["eval_index"]) == 3
    assert int(df.iloc[0]["images_seen"]) == 1_000_000

from pathlib import Path

import torch

from ijepath.eval.early_stopping import RobustnessEarlyStopper


def test_robustness_early_stopper_tracks_best_and_stops(tmp_path: Path):
    stopper = RobustnessEarlyStopper(
        mode="max",
        patience_evals=2,
        min_evals=1,
        checkpoint_path=tmp_path / "best-robustness.pth.tar",
    )

    snapshot = {"x": 1}
    stopper.on_eval(metric_value=0.1, snapshot=snapshot)
    stopper.on_eval(metric_value=0.09, snapshot=snapshot)
    assert stopper.should_stop is False
    stopper.on_eval(metric_value=0.08, snapshot=snapshot)
    assert stopper.should_stop is True
    assert (tmp_path / "best-robustness.pth.tar").exists()


def test_robustness_early_stopper_saves_latest_best_metric(tmp_path: Path):
    stopper = RobustnessEarlyStopper(
        mode="min",
        patience_evals=5,
        min_evals=1,
        checkpoint_path=tmp_path / "best-robustness.pth.tar",
    )
    snapshot_a = {"metric": 3.0}
    snapshot_b = {"metric": 1.0}

    stopper.on_eval(metric_value=3.0, snapshot=snapshot_a)
    stopper.on_eval(metric_value=1.0, snapshot=snapshot_b)

    loaded = torch.load(tmp_path / "best-robustness.pth.tar", map_location="cpu")
    assert loaded["metric"] == 1.0


def test_robustness_early_stopper_can_skip_checkpoint_saving(tmp_path: Path):
    stopper = RobustnessEarlyStopper(
        mode="max",
        patience_evals=2,
        min_evals=1,
        checkpoint_path=tmp_path / "best-robustness.pth.tar",
        save_best_checkpoint=False,
    )

    stopper.on_eval(metric_value=0.3, snapshot={"metric": 0.3})
    assert stopper.best_metric == 0.3
    assert not (tmp_path / "best-robustness.pth.tar").exists()

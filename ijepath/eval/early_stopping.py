from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class RobustnessEarlyStopper:
    mode: str
    patience_evals: int
    min_evals: int
    checkpoint_path: Path
    save_best_checkpoint: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if int(self.patience_evals) <= 0:
            raise ValueError("patience_evals must be > 0")
        if int(self.min_evals) < 0:
            raise ValueError("min_evals must be >= 0")
        self.best_metric: float | None = None
        self.num_tunes = 0
        self.bad_tune_streak = 0
        self.should_stop = False
        self.checkpoint_path = Path(self.checkpoint_path)

    def _is_improved(self, value: float) -> bool:
        if self.best_metric is None:
            return True
        if self.mode == "max":
            return float(value) > float(self.best_metric)
        return float(value) < float(self.best_metric)

    def on_tune(self, metric_value: float, snapshot: dict) -> None:
        self.num_tunes += 1

        if self._is_improved(metric_value):
            self.best_metric = float(metric_value)
            self.bad_tune_streak = 0
            if bool(self.save_best_checkpoint):
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(snapshot, self.checkpoint_path)
            return

        self.bad_tune_streak += 1
        if self.num_tunes >= int(self.min_evals) and self.bad_tune_streak >= int(self.patience_evals):
            self.should_stop = True

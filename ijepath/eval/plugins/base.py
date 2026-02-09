from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PluginResult:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    log_metrics: dict[str, float] = field(default_factory=dict)
    selection_metric_value: float | None = None
    selection_mode: str | None = None


class BenchmarkPlugin(ABC):
    name: str
    tuning_output_dir: Path | None = None

    def bind_runtime(self, tuning_output_dir: Path | None) -> None:
        self.tuning_output_dir = tuning_output_dir

    def should_run(self, images_seen: int, eval_index: int) -> bool:
        return True

    @abstractmethod
    def run(self, teacher: Any, eval_index: int, images_seen: int) -> PluginResult:
        raise NotImplementedError

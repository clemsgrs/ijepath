from __future__ import annotations

import copy
import logging
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .tuner import Tuner

logger = logging.getLogger("ijepath")


@dataclass
class _TuneJob:
    tune_index: int
    images_seen: int
    snapshot_path: str
    enqueued_at_s: float


@dataclass
class _TuneCompleted:
    tune_index: int
    images_seen: int
    enqueued_at_s: float
    started_at_s: float
    finished_at_s: float
    result: dict[str, Any] | None
    error: str | None
    traceback_text: str | None
    queue_depth: int


class AsyncTuningRuntime:
    """Rank-0 async tuning runtime that runs tuning work on a dedicated device."""

    def __init__(
        self,
        *,
        tuning_cfg: dict[str, Any],
        teacher_template: torch.nn.Module,
        output_dir: Path,
    ) -> None:
        self.tuning_cfg = dict(tuning_cfg or {})
        self.execution_cfg = dict(self.tuning_cfg.get("execution", {}) or {})
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.output_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self.device_str = self._resolve_execution_device(str(self.execution_cfg.get("device", "auto")))
        self.device = torch.device(self.device_str)
        if self.device.type != "cuda":
            raise ValueError("AsyncTuningRuntime requires CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable but async tuning requires dedicated CUDA device")

        self.max_pending_jobs = int(self.execution_cfg.get("max_pending_jobs", 2))
        self.coalesce_policy = str(self.execution_cfg.get("coalesce_policy", "newest")).strip().lower()
        self.fail_on_backlog = bool(self.execution_cfg.get("fail_on_backlog", False))
        self.keep_last_n_snapshots = int(self.execution_cfg.get("keep_last_n_snapshots", 2))

        if self.max_pending_jobs <= 0:
            raise ValueError("tuning.execution.max_pending_jobs must be > 0")
        if self.coalesce_policy != "newest":
            raise ValueError("tuning.execution.coalesce_policy must be 'newest'")
        if self.keep_last_n_snapshots < 0:
            raise ValueError("tuning.execution.keep_last_n_snapshots must be >= 0")

        self._teacher_template = copy.deepcopy(teacher_template).cpu().eval()
        self._jobs: deque[_TuneJob] = deque()
        self._completed: deque[_TuneCompleted] = deque()
        self._snapshot_history: deque[str] = deque()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._stop_requested = False
        self._inflight = 0
        self._dropped_tunes = 0
        self._startup_error: str | None = None
        self._worker = threading.Thread(target=self._worker_loop, name="ijepath-async-tuner", daemon=True)
        self._worker.start()

    def submit(self, *, tune_index: int, images_seen: int, snapshot_path: str) -> dict[str, int]:
        with self._cv:
            self._raise_if_startup_failed_locked()

            while len(self._jobs) >= self.max_pending_jobs:
                if self.fail_on_backlog:
                    raise RuntimeError(
                        "Async tuning backlog limit reached: "
                        f"max_pending_jobs={self.max_pending_jobs} queue_depth={self.queue_depth_locked()}"
                    )
                dropped = self._jobs.popleft()
                self._dropped_tunes += 1
                self._safe_unlink(dropped.snapshot_path)
                logger.warning(
                    "Dropping queued tuning job due backlog (policy=%s): tune_index=%d images_seen=%d",
                    self.coalesce_policy,
                    int(dropped.tune_index),
                    int(dropped.images_seen),
                )

            self._jobs.append(
                _TuneJob(
                    tune_index=int(tune_index),
                    images_seen=int(images_seen),
                    snapshot_path=str(snapshot_path),
                    enqueued_at_s=float(time.time()),
                )
            )
            self._cv.notify_all()
            return {
                "queue_depth": int(self.queue_depth_locked()),
                "dropped_tunes": int(self._dropped_tunes),
            }

    def poll_completed(self) -> list[dict[str, Any]]:
        with self._lock:
            self._raise_if_startup_failed_locked()
            out: list[dict[str, Any]] = []
            while self._completed:
                item = self._completed.popleft()
                out.append(
                    {
                        "tune_index": int(item.tune_index),
                        "images_seen": int(item.images_seen),
                        "enqueued_at_s": float(item.enqueued_at_s),
                        "started_at_s": float(item.started_at_s),
                        "finished_at_s": float(item.finished_at_s),
                        "latency_seconds": float(item.finished_at_s - item.enqueued_at_s),
                        "worker_seconds": float(item.finished_at_s - item.started_at_s),
                        "queue_depth": int(item.queue_depth),
                        "result": item.result,
                        "error": item.error,
                        "traceback": item.traceback_text,
                    }
                )
            return out

    def queue_depth(self) -> int:
        with self._lock:
            return int(self.queue_depth_locked())

    def dropped_tunes(self) -> int:
        with self._lock:
            return int(self._dropped_tunes)

    def shutdown(self, *, wait: bool, timeout_s: float = 5.0) -> None:
        with self._cv:
            self._stop_requested = True
            if not wait:
                while self._jobs:
                    dropped = self._jobs.popleft()
                    self._safe_unlink(dropped.snapshot_path)
            self._cv.notify_all()
        self._worker.join(timeout=max(0.0, float(timeout_s)))

    def queue_depth_locked(self) -> int:
        return int(len(self._jobs) + self._inflight)

    def _raise_if_startup_failed_locked(self) -> None:
        if self._startup_error is not None:
            raise RuntimeError(f"Async tuning worker failed to initialize: {self._startup_error}")

    def _register_snapshot(self, snapshot_path: str) -> None:
        if self.keep_last_n_snapshots <= 0:
            self._safe_unlink(snapshot_path)
            return

        self._snapshot_history.append(str(snapshot_path))
        while len(self._snapshot_history) > self.keep_last_n_snapshots:
            old = self._snapshot_history.popleft()
            self._safe_unlink(old)

    def _worker_loop(self) -> None:
        teacher_backbone: torch.nn.Module | None = None
        tuner: Tuner | None = None

        try:
            torch.cuda.set_device(self.device)
            teacher_backbone = self._teacher_template.to(self.device).eval()
            tuner = Tuner(
                cfg=self.tuning_cfg,
                device=self.device,
                output_dir=self.output_dir,
            )
        except Exception:
            with self._cv:
                self._startup_error = traceback.format_exc()
                self._cv.notify_all()
            return

        while True:
            with self._cv:
                while not self._jobs and not self._stop_requested:
                    self._cv.wait(timeout=0.5)
                if self._stop_requested and not self._jobs:
                    return
                job = self._jobs.popleft()
                self._inflight += 1

            started_at_s = float(time.time())
            result: dict[str, Any] | None = None
            error: str | None = None
            traceback_text: str | None = None
            try:
                state = torch.load(job.snapshot_path, map_location=self.device)
                if teacher_backbone is None or tuner is None:
                    raise RuntimeError("Async tuning worker was not initialized")
                teacher_backbone.load_state_dict(state, strict=True)
                teacher_backbone.eval()
                result = tuner.tune(
                    teacher=teacher_backbone,
                    tune_index=int(job.tune_index),
                    images_seen=int(job.images_seen),
                )
            except Exception as exc:
                error = str(exc)
                traceback_text = traceback.format_exc()
            finally:
                finished_at_s = float(time.time())
                with self._cv:
                    self._inflight -= 1
                    self._register_snapshot(job.snapshot_path)
                    self._completed.append(
                        _TuneCompleted(
                            tune_index=int(job.tune_index),
                            images_seen=int(job.images_seen),
                            enqueued_at_s=float(job.enqueued_at_s),
                            started_at_s=float(started_at_s),
                            finished_at_s=float(finished_at_s),
                            result=result,
                            error=error,
                            traceback_text=traceback_text,
                            queue_depth=int(self.queue_depth_locked()),
                        )
                    )
                    self._cv.notify_all()

    @staticmethod
    def _safe_unlink(path: str) -> None:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            logger.debug("Failed to delete snapshot path: %s", path, exc_info=True)

    @staticmethod
    def _resolve_execution_device(raw_device: str) -> str:
        device = str(raw_device or "auto").strip().lower()
        if device != "auto":
            return device
        count = int(torch.cuda.device_count())
        if count < 2:
            raise RuntimeError(
                "tuning.execution.device=auto requires at least 2 visible GPUs on rank 0 "
                "(one for training, one for tuning)"
            )
        return "cuda:1"

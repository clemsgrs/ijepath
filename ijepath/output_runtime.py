from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
import re
import uuid


def _utc_timestamp_run_id(now_utc: datetime | None = None) -> str:
    now = now_utc or datetime.now(timezone.utc)
    return now.strftime("%Y%m%d-%H%M%SZ")


def _default_wandb_id_factory() -> str:
    try:
        import wandb  # type: ignore

        util = getattr(wandb, "util", None)
        if util is not None and callable(getattr(util, "generate_id", None)):
            generated = str(util.generate_id()).strip()
            if generated:
                return generated
    except Exception:
        pass
    return uuid.uuid4().hex[:8]


def _normalize_run_id(raw_value: str) -> str:
    value = str(raw_value or "").strip()
    if not value:
        raise ValueError("Resolved run_id is empty")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value)


def choose_run_id(
    cfg: Mapping[str, Any],
    *,
    now_utc: datetime | None = None,
    wandb_id_factory: Callable[[], str] | None = None,
) -> str:
    wandb_cfg = dict(cfg.get("wandb", {}) or {})
    resume_id = str(wandb_cfg.get("resume_id") or "").strip()
    if resume_id:
        return _normalize_run_id(resume_id)

    if bool(wandb_cfg.get("enable", False)):
        preset_run_id = str(wandb_cfg.get("run_id") or "").strip()
        if preset_run_id:
            return _normalize_run_id(preset_run_id)
        factory = wandb_id_factory or _default_wandb_id_factory
        return _normalize_run_id(factory())

    return _normalize_run_id(_utc_timestamp_run_id(now_utc=now_utc))


def resolve_output_paths(cfg: Mapping[str, Any], *, run_id: str) -> dict[str, Path | str]:
    output_cfg = dict(cfg.get("output", {}) or {})

    output_root_raw = output_cfg.get("root", "outputs")
    output_root = Path(str(output_root_raw)).expanduser().resolve()
    run_root = (output_root / "runs").resolve()

    shared_cache_root_raw = output_cfg.get("shared_cache_root", output_root / "cache")
    shared_cache_root = Path(str(shared_cache_root_raw)).expanduser().resolve()

    normalized_run_id = _normalize_run_id(run_id)
    run_dir = (run_root / normalized_run_id).resolve()
    return {
        "run_id": normalized_run_id,
        "output_root": output_root,
        "run_dir": run_dir,
        "shared_cache_root": shared_cache_root,
    }

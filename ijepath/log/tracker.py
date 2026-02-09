import numbers
import os
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover - runtime optional dependency
    wandb = None


def _require_wandb():
    if wandb is None:
        raise RuntimeError(
            "W&B logging is enabled, but `wandb` is not installed. "
            "Install it with `pip install wandb`."
        )
    return wandb


def initialize_wandb(
    cfg: Mapping[str, Any],
    key: Optional[str] = None,
):
    wb = _require_wandb()
    wandb_cfg = dict(cfg.get("wandb", {}) or {})
    tags = [str(t) for t in (wandb_cfg.get("tags") or [])]

    if key:
        wb.login(key=key, relogin=True)
    elif os.environ.get("WANDB_API_KEY"):
        wb.login(key=os.environ["WANDB_API_KEY"], relogin=False)

    exp_name = wandb_cfg.get("exp_name") or cfg.get("logging", {}).get("write_tag", "ijepath")
    init_kwargs = {
        "project": wandb_cfg.get("project", "ijepath"),
        "entity": wandb_cfg.get("username"),
        "name": exp_name,
        "group": wandb_cfg.get("group"),
        "dir": wandb_cfg.get("dir"),
        "config": dict(cfg),
        "tags": tags,
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

    resume_id = wandb_cfg.get("resume_id")
    if resume_id:
        init_kwargs.update({"id": resume_id, "resume": "must"})

    run = wb.init(**init_kwargs)
    run.define_metric("images_seen", summary="max")
    return run


def save_run_config_to_wandb(run, cfg: Mapping[str, Any], name: str = "run_config.yaml") -> str:
    wb = _require_wandb()
    config_file_path = Path(run.dir, name)
    with config_file_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False)
    wb.save(str(config_file_path))
    return str(config_file_path)


def update_log_dict(
    prefix: str,
    results: Mapping[str, Any],
    log_dict: dict[str, Any],
    step: str = "step",
) -> None:
    wb = _require_wandb()
    for result_key, value in results.items():
        metric_name = f"{prefix}/{result_key}"
        if isinstance(value, numbers.Number):
            wb.define_metric(metric_name, step_metric=step)
        log_dict[metric_name] = value


def log_images_seen_dict(log_dict: Mapping[str, Any], images_seen: int) -> None:
    wb = _require_wandb()
    wb.log(dict(log_dict), step=int(images_seen))


def finish_wandb(exit_code: Optional[int] = None) -> None:
    if wandb is not None and getattr(wandb, "run", None) is not None:
        if exit_code is None:
            wandb.finish()
        else:
            wandb.finish(exit_code=int(exit_code))

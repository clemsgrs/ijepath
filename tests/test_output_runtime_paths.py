from datetime import datetime, timezone
from pathlib import Path

from ijepath.output_runtime import choose_run_id, resolve_output_paths


def test_choose_run_id_uses_timestamp_when_wandb_disabled():
    cfg = {"wandb": {"enable": False}}
    run_id = choose_run_id(
        cfg,
        now_utc=datetime(2026, 2, 12, 15, 30, 45, tzinfo=timezone.utc),
        wandb_id_factory=lambda: "unused",
    )
    assert run_id == "20260212-153045Z"


def test_choose_run_id_prefers_resume_id_when_present():
    cfg = {"wandb": {"enable": True, "resume_id": "resume-abc"}}
    run_id = choose_run_id(cfg, wandb_id_factory=lambda: "generated-id")
    assert run_id == "resume-abc"


def test_choose_run_id_uses_generated_wandb_id_when_enabled():
    cfg = {"wandb": {"enable": True}}
    run_id = choose_run_id(cfg, wandb_id_factory=lambda: "generated-id")
    assert run_id == "generated-id"


def test_choose_run_id_uses_preset_wandb_run_id_when_enabled():
    cfg = {"wandb": {"enable": True, "run_id": "preset-id"}}
    run_id = choose_run_id(cfg, wandb_id_factory=lambda: "generated-id")
    assert run_id == "preset-id"


def test_resolve_output_paths_defaults_to_output_root_runs():
    paths = resolve_output_paths(
        {"output": {"root": "outputs", "shared_cache_root": "outputs/cache"}},
        run_id="rid-123",
    )
    assert paths["run_dir"] == (Path("outputs").resolve() / "runs" / "rid-123")
    assert paths["shared_cache_root"] == (Path("outputs").resolve() / "cache")


def test_resolve_output_paths_is_always_under_output_root_runs():
    paths = resolve_output_paths(
        {
            "output": {
                "root": "outputs",
                "shared_cache_root": "my-cache",
            }
        },
        run_id="rid-456",
    )
    assert paths["run_dir"] == (Path("outputs").resolve() / "runs" / "rid-456")
    assert paths["shared_cache_root"] == Path("my-cache").resolve()

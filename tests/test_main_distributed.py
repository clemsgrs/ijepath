import pytest
import importlib

pytest.importorskip("submitit")
main_distributed_entry = importlib.import_module("main_distributed")


def test_validate_master_port_rejects_invalid_value():
    with pytest.raises(ValueError, match="MASTER_PORT"):
        main_distributed_entry._validate_master_port("not-a-port")


def test_trainer_requires_master_port_for_multi_task_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.delenv("MASTER_PORT", raising=False)

    trainer = main_distributed_entry.Trainer(
        profile_config="profile.yaml",
        run_config="run.yaml",
        opts=[],
    )

    with pytest.raises(SystemExit, match="MASTER_PORT is required"):
        trainer()


def test_trainer_uses_master_port_argument(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.delenv("MASTER_PORT", raising=False)
    monkeypatch.setattr(main_distributed_entry, "setup_logging", lambda **_: None)
    monkeypatch.setattr(main_distributed_entry, "load_training_config", lambda **_: {"ok": True})

    def _fake_app_main(*, args, **_kwargs):
        captured["args"] = args

    monkeypatch.setattr(main_distributed_entry, "app_main", _fake_app_main)

    trainer = main_distributed_entry.Trainer(
        profile_config="profile.yaml",
        run_config="run.yaml",
        opts=["x=1"],
        master_port=29671,
    )
    trainer()

    assert captured["args"] == {"ok": True}


def test_trainer_logs_traceback_and_reraises_app_main_failure(monkeypatch):
    exception_logs: list[str] = []

    class _FakeLogger:
        def info(self, *_args, **_kwargs):
            return None

        def exception(self, msg, *args):
            if args:
                msg = msg % args
            exception_logs.append(str(msg))

    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.delenv("MASTER_PORT", raising=False)
    monkeypatch.setattr(main_distributed_entry, "setup_logging", lambda **_: _FakeLogger())
    monkeypatch.setattr(main_distributed_entry, "load_training_config", lambda **_: {"ok": True})

    def _boom(*, args, **_kwargs):
        raise RuntimeError("trainer exploded")

    monkeypatch.setattr(main_distributed_entry, "app_main", _boom)

    trainer = main_distributed_entry.Trainer(
        profile_config="profile.yaml",
        run_config="run.yaml",
        opts=["x=1"],
    )

    with pytest.raises(RuntimeError, match="trainer exploded"):
        trainer()

    assert any("Training entrypoint crashed with unhandled exception" in line for line in exception_logs)

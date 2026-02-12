import sys
import types

if "submitit" not in sys.modules:
    submitit_stub = types.SimpleNamespace(
        AutoExecutor=object,
        helpers=types.SimpleNamespace(DelayedSubmission=object),
    )
    sys.modules["submitit"] = submitit_stub

import main_distributed as main_dist


def test_resolve_distributed_gpu_request_without_tuning(monkeypatch):
    monkeypatch.setattr(
        main_dist,
        "load_training_config",
        lambda **_kwargs: {"tuning": {"enable": False}},
    )
    out = main_dist._resolve_distributed_gpu_request(
        profile_config="profile.yaml",
        run_config="run.yaml",
        opts=[],
        tasks_per_node=4,
    )
    assert out == 4


def test_resolve_distributed_gpu_request_requires_spare_gpu_index(monkeypatch):
    monkeypatch.setattr(
        main_dist,
        "load_training_config",
        lambda **_kwargs: {"tuning": {"enable": True, "execution": {"mode": "async", "device": "cuda:1"}}},
    )
    try:
        main_dist._resolve_distributed_gpu_request(
            profile_config="profile.yaml",
            run_config="run.yaml",
            opts=[],
            tasks_per_node=2,
        )
        raise AssertionError("Expected SystemExit for conflicting dedicated tuning GPU")
    except SystemExit as exc:
        assert "conflicts with training ranks" in str(exc)


def test_resolve_distributed_gpu_request_adds_one_gpu_for_tuning(monkeypatch):
    monkeypatch.setattr(
        main_dist,
        "load_training_config",
        lambda **_kwargs: {"tuning": {"enable": True, "execution": {"mode": "async", "device": "cuda:4"}}},
    )
    out = main_dist._resolve_distributed_gpu_request(
        profile_config="profile.yaml",
        run_config="run.yaml",
        opts=[],
        tasks_per_node=4,
    )
    assert out == 5


def test_resolve_distributed_gpu_request_auto_uses_spare_index(monkeypatch):
    monkeypatch.setattr(
        main_dist,
        "load_training_config",
        lambda **_kwargs: {"tuning": {"enable": True, "execution": {"mode": "async", "device": "auto"}}},
    )
    out = main_dist._resolve_distributed_gpu_request(
        profile_config="profile.yaml",
        run_config="run.yaml",
        opts=[],
        tasks_per_node=3,
    )
    assert out == 4


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
    monkeypatch.setattr(main_dist, "setup_logging", lambda **_: None)
    monkeypatch.setattr(main_dist, "logger", _FakeLogger())
    monkeypatch.setattr(main_dist, "load_training_config", lambda **_: {"ok": True})

    def _boom(*, args, **_kwargs):
        raise RuntimeError("trainer exploded")

    monkeypatch.setattr(main_dist, "app_main", _boom)

    trainer = main_dist.Trainer(
        profile_config="profile.yaml",
        run_config="run.yaml",
        opts=["x=1"],
    )
    try:
        trainer()
        raise AssertionError("Expected RuntimeError to propagate from app_main")
    except RuntimeError as exc:
        assert "trainer exploded" in str(exc)

    assert any("Training entrypoint crashed with unhandled exception" in line for line in exception_logs)

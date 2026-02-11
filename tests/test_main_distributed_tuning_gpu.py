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

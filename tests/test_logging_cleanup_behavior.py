import main as main_entry
import torch
import pytest
from pathlib import Path


def test_process_main_defers_config_logging_to_trainer(monkeypatch):
    captured_logs: list[str] = []
    captured_app_args: dict[str, object] = {}

    class _FakeLogger:
        def info(self, msg, *args):
            if args:
                msg = msg % args
            captured_logs.append(str(msg))

    monkeypatch.setattr(main_entry, "setup_logging", lambda **_: _FakeLogger())
    monkeypatch.setattr(main_entry, "load_training_config", lambda **_: {"ok": True})
    monkeypatch.setattr(main_entry, "init_distributed", lambda **_kwargs: (1, 0))

    def _fake_app_main(*, args, **_kwargs):
        captured_app_args["args"] = args

    monkeypatch.setattr(main_entry, "app_main", _fake_app_main)

    main_entry.process_main(
        rank=0,
        profile_config="profile.yaml",
        run_config="run.yaml",
        opts=["x=1"],
        world_size=1,
        visible_devices=["cpu"],
        master_addr=None,
        master_port=None,
    )

    assert captured_app_args["args"] == {"ok": True}
    assert not hasattr(main_entry, "render_config_yaml")
    assert any("called-params" in line for line in captured_logs)
    assert any("loaded layered config" in line for line in captured_logs)


def test_should_log_iteration_respects_frequency_and_anomalies():
    from ijepath.train_cross_resolution_jepa import should_log_iteration

    assert should_log_iteration(itr=0, step_log_every_iters=0, loss=1.0) is False
    assert should_log_iteration(itr=10, step_log_every_iters=10, loss=1.0) is True
    assert should_log_iteration(itr=11, step_log_every_iters=10, loss=1.0) is False
    assert should_log_iteration(itr=3, step_log_every_iters=0, loss=float("nan")) is True
    assert should_log_iteration(itr=3, step_log_every_iters=0, loss=float("inf")) is True


def test_resolve_checkpoint_every_images_defaults_and_validates():
    from ijepath.train_cross_resolution_jepa import resolve_checkpoint_every_images

    assert resolve_checkpoint_every_images({}) == 1_000_000
    assert resolve_checkpoint_every_images({"checkpoint_every_images": 3}) == 3

    try:
        resolve_checkpoint_every_images({"checkpoint_every_images": 0})
        raise AssertionError("Expected ValueError for non-positive checkpoint frequency")
    except ValueError:
        pass


def test_build_pass_train_results_uses_standardized_throughput_and_explicit_mask_avg_keys():
    from ijepath.train_cross_resolution_jepa import build_pass_train_results

    payload = build_pass_train_results(
        loss_avg=0.3,
        loss_min=0.2,
        loss_max=0.5,
        context_keep_tokens_avg=16.0,
        target_predict_tokens_avg=64.0,
        iter_time_ms=12.0,
        pass_time_s=8.0,
        images_seen=256,
        anchor_passes_seen=1.2,
        images_per_sec=32.0,
        iterations_per_sec=2.0,
        lr=1e-4,
        wd=0.04,
        global_batch_size=32,
    )

    assert payload["images_per_sec"] == 32.0
    assert payload["iterations_per_sec"] == 2.0
    assert payload["context_keep_tokens_avg"] == 16.0
    assert payload["target_predict_tokens_avg"] == 64.0
    assert "images_per_s" not in payload
    assert "iters_per_s" not in payload
    assert "mask_a" not in payload
    assert "mask_b" not in payload
    assert "context_keep_tokens" not in payload
    assert "target_predict_tokens" not in payload


def test_train_step_csv_schema_is_standardized():
    from ijepath.train_cross_resolution_jepa import get_train_step_csv_columns

    columns = get_train_step_csv_columns()
    headers = [header for _, header in columns]

    assert headers == [
        "pass_index",
        "iteration",
        "loss",
        "context_keep_tokens",
        "target_predict_tokens",
        "iteration_time_ms",
        "learning_rate",
        "weight_decay",
    ]


def test_build_step_log_line_uses_standardized_labels():
    from ijepath.train_cross_resolution_jepa import build_step_log_line

    line = build_step_log_line(
        pass_index=2,
        iteration=12,
        loss_avg=0.3456,
        context_keep_tokens=18.0,
        target_predict_tokens=64.0,
        weight_decay=0.04,
        learning_rate=2e-4,
        max_memory_mb=1536.0,
        iteration_time_ms=11.2,
    )

    assert line.startswith("pass_index=2 iteration=12")
    assert "loss_avg=0.346" in line
    assert "context_keep_tokens=18.0" in line
    assert "target_predict_tokens=64.0" in line
    assert "weight_decay=4.00e-02" in line
    assert "learning_rate=2.00e-04" in line
    assert "max_memory_mb=1.54e+03" in line
    assert "iteration_time_ms=11.2" in line
    assert "masks:" not in line


def test_build_grad_stats_log_line_uses_standardized_labels():
    from ijepath.train_cross_resolution_jepa import build_grad_stats_log_line

    line = build_grad_stats_log_line(
        pass_index=3,
        iteration=7,
        first_layer_grad_norm=1.23e-2,
        last_layer_grad_norm=4.56e-2,
        grad_min=7.89e-4,
        grad_max=9.99e-1,
    )

    assert line.startswith("pass_index=3 iteration=7")
    assert "first_layer_grad_norm=1.23e-02" in line
    assert "last_layer_grad_norm=4.56e-02" in line
    assert "grad_norm_min=7.89e-04" in line
    assert "grad_norm_max=9.99e-01" in line
    assert "grad_stats:" not in line


def test_resolve_use_bfloat16_disables_on_cpu_only_runtime():
    from ijepath.train_cross_resolution_jepa import resolve_use_bfloat16

    assert resolve_use_bfloat16(requested_use_bfloat16=True, cuda_available=False) is False
    assert resolve_use_bfloat16(requested_use_bfloat16=False, cuda_available=True) is False
    assert resolve_use_bfloat16(requested_use_bfloat16=True, cuda_available=True) is True


def test_init_opt_does_not_construct_grad_scaler_without_cuda(monkeypatch):
    from ijepath import helper

    monkeypatch.setattr(helper.torch.cuda, "is_available", lambda: False)

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("GradScaler should not be constructed when CUDA is unavailable")

    monkeypatch.setattr(helper.torch.cuda.amp, "GradScaler", _raise_if_called)

    encoder = torch.nn.Linear(4, 4)
    predictor = torch.nn.Linear(4, 4)
    _optimizer, scaler, _scheduler, _wd_scheduler = helper.init_opt(
        encoder=encoder,
        predictor=predictor,
        total_steps=10,
        start_lr=1e-5,
        ref_lr=1e-4,
        warmup=0.1,
        use_bfloat16=True,
    )
    assert scaler is None


def test_init_opt_uses_total_steps_for_schedule_horizon():
    from ijepath import helper

    encoder = torch.nn.Linear(4, 4)
    predictor = torch.nn.Linear(4, 4)
    _optimizer, _scaler, scheduler, wd_scheduler = helper.init_opt(
        encoder=encoder,
        predictor=predictor,
        total_steps=100,
        start_lr=1e-5,
        ref_lr=1e-4,
        warmup=0.1,
        ipe_scale=1.25,
        use_bfloat16=False,
    )

    assert scheduler.warmup_steps == 10
    # WarmupCosineSchedule stores post-warmup span.
    assert scheduler.T_max == 115
    assert wd_scheduler.T_max == 125


def test_launch_worker_processes_waits_for_all_children(monkeypatch):
    starts: list[tuple] = []
    joins: list[int] = []

    class _FakeProcess:
        def __init__(self, target, args):
            self._target = target
            self._args = args
            self.exitcode = 0

        def start(self):
            starts.append(self._args)

        def join(self, timeout=None):
            joins.append(self._args[0])

    monkeypatch.setattr(main_entry.mp, "Process", _FakeProcess)
    monkeypatch.setattr(main_entry, "_resolve_master_addr", lambda world_size: "127.0.0.1")
    monkeypatch.setattr(main_entry, "_resolve_master_port", lambda world_size: 29456)

    main_entry.launch_worker_processes(
        profile_config="profile.yaml",
        run_config="run.yaml",
        opts=["x=1"],
        visible_devices=["cpu", "1"],
    )

    assert len(starts) == 2
    assert len(joins) >= 2
    assert set(joins) == {0, 1}
    assert all(args[6] == "127.0.0.1" for args in starts)
    assert all(args[7] == 29456 for args in starts)


def test_launch_worker_processes_raises_on_failed_child(monkeypatch):
    class _FakeProcess:
        def __init__(self, target, args):
            self._target = target
            self._args = args
            self.exitcode = 0 if args[0] == 0 else 2

        def start(self):
            return None

        def join(self, timeout=None):
            return None

    monkeypatch.setattr(main_entry.mp, "Process", _FakeProcess)
    monkeypatch.setattr(main_entry, "_resolve_master_addr", lambda world_size: "127.0.0.1")
    monkeypatch.setattr(main_entry, "_resolve_master_port", lambda world_size: 29501)

    try:
        main_entry.launch_worker_processes(
            profile_config="profile.yaml",
            run_config="run.yaml",
            opts=[],
            visible_devices=["cpu", "1"],
        )
        raise AssertionError("Expected SystemExit when a worker exits with nonzero status")
    except SystemExit as exc:
        assert "rank=1" in str(exc)


def test_resolve_master_port_honors_env_override(monkeypatch):
    monkeypatch.setenv("MASTER_PORT", "29651")
    assert main_entry._resolve_master_port(world_size=2) == 29651


def test_resolve_master_port_rejects_invalid_env(monkeypatch):
    monkeypatch.setenv("MASTER_PORT", "bad-port")
    with pytest.raises(ValueError, match="MASTER_PORT"):
        main_entry._resolve_master_port(world_size=2)


def test_launch_worker_processes_cleans_up_on_keyboard_interrupt(monkeypatch):
    lifecycle: dict[str, int] = {"terminated": 0, "killed": 0}

    class _FakeProcess:
        def __init__(self, target, args):
            self._target = target
            self._args = args
            self.exitcode = None

        def start(self):
            return None

        def join(self, timeout=None):
            raise KeyboardInterrupt()

        def terminate(self):
            lifecycle["terminated"] += 1
            self.exitcode = -15

        def kill(self):
            lifecycle["killed"] += 1
            self.exitcode = -9

        def is_alive(self):
            return self.exitcode is None

    monkeypatch.setattr(main_entry.mp, "Process", _FakeProcess)
    monkeypatch.setattr(main_entry, "_resolve_master_addr", lambda world_size: "127.0.0.1")
    monkeypatch.setattr(main_entry, "_resolve_master_port", lambda world_size: 29522)

    with pytest.raises(SystemExit, match="Interrupted while waiting for worker processes"):
        main_entry.launch_worker_processes(
            profile_config="profile.yaml",
            run_config="run.yaml",
            opts=[],
            visible_devices=["cpu", "1"],
        )

    assert lifecycle["terminated"] >= 1


def test_has_opt_detects_dotlist_prefix():
    assert main_entry._has_opt(["a=1", "logging.folder=/tmp/out"], "logging.folder=") is True
    assert main_entry._has_opt(["a=1"], "logging.folder=") is False


def test_orchestrate_pipeline_artifacts_runs_both_build_stages(monkeypatch, tmp_path: Path):
    calls: list[list[str]] = []

    class _Ok:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def _fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        return _Ok()

    monkeypatch.setattr(main_entry.subprocess, "run", _fake_run)

    out = main_entry.orchestrate_pipeline_artifacts(
        profile_config=str(tmp_path / "profile.yaml"),
        manifest_csv=str(tmp_path / "manifest.csv"),
        output_folder=str(tmp_path / "out"),
    )

    assert len(calls) == 2
    assert calls[0][1].endswith("scripts/build_slide_metadata_index_from_manifest.py")
    assert calls[1][1].endswith("scripts/build_valid_context_anchor_catalog.py")
    assert out["slide_metadata_parquet"].endswith("indexes/slide_metadata.parquet")
    assert out["anchor_catalog_manifest"].endswith("indexes/anchor_catalog_manifest.json")
    assert out["training_output_folder"].endswith("/out")


def test_run_checked_raises_with_command_output_on_failure(monkeypatch):
    class _Fail:
        returncode = 9
        stdout = "x"
        stderr = "y"

    monkeypatch.setattr(main_entry.subprocess, "run", lambda *args, **kwargs: _Fail())

    try:
        main_entry._run_checked(["echo", "hello"])
        raise AssertionError("Expected SystemExit for failing command")
    except SystemExit as exc:
        text = str(exc)
        assert "Pipeline stage failed" in text
        assert "stdout" in text and "stderr" in text


def test_resolve_reserved_tuning_device_token_returns_none_when_tuning_disabled(monkeypatch):
    monkeypatch.setattr(
        main_entry,
        "load_training_config",
        lambda **_kwargs: {"tuning": {"enable": False}},
    )
    token = main_entry._resolve_reserved_tuning_device_token(
        profile_config="profile.yaml",
        run_config="run.yaml",
        opts=[],
        visible_devices=["0", "1"],
    )
    assert token is None


def test_resolve_reserved_tuning_device_token_validates_visibility(monkeypatch):
    monkeypatch.setattr(
        main_entry,
        "load_training_config",
        lambda **_kwargs: {
            "tuning": {"enable": True, "execution": {"mode": "async", "device": "cuda:2"}}
        },
    )
    try:
        main_entry._resolve_reserved_tuning_device_token(
            profile_config="profile.yaml",
            run_config="run.yaml",
            opts=[],
            visible_devices=["0", "1"],
        )
        raise AssertionError("Expected SystemExit for non-visible dedicated tuning GPU")
    except SystemExit as exc:
        assert "not visible" in str(exc)


def test_resolve_reserved_tuning_device_token_auto_picks_last_visible(monkeypatch):
    monkeypatch.setattr(
        main_entry,
        "load_training_config",
        lambda **_kwargs: {"tuning": {"enable": True, "execution": {"mode": "async", "device": "auto"}}},
    )
    token = main_entry._resolve_reserved_tuning_device_token(
        profile_config="profile.yaml",
        run_config="run.yaml",
        opts=[],
        visible_devices=["0", "1", "2"],
    )
    assert token == "2"


def test_process_main_remaps_rank0_tuning_device_to_local_cuda1(monkeypatch):
    captured_opts = {}

    class _FakeLogger:
        def info(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(main_entry, "setup_logging", lambda **_: _FakeLogger())
    monkeypatch.setattr(main_entry, "init_distributed", lambda **_kwargs: (1, 0))

    def _fake_load_training_config(**kwargs):
        captured_opts["opts"] = list(kwargs.get("opts") or [])
        return {"ok": True}

    monkeypatch.setattr(main_entry, "load_training_config", _fake_load_training_config)
    monkeypatch.setattr(main_entry, "app_main", lambda **_kwargs: None)

    main_entry.process_main(
        rank=0,
        profile_config="profile.yaml",
        run_config="run.yaml",
        opts=["x=1", "tuning.execution.device=cuda:7"],
        world_size=1,
        visible_devices=["0"],
        master_addr=None,
        master_port=None,
        tuning_device_token="7",
    )

    assert "tuning.execution.device=cuda:1" in captured_opts["opts"]

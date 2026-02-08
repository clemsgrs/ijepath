import main as main_entry
import torch


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
    monkeypatch.setattr(main_entry, "init_distributed", lambda rank_and_world_size: (1, 0))

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


def test_resolve_checkpoint_every_epochs_defaults_and_validates():
    from ijepath.train_cross_resolution_jepa import resolve_checkpoint_every_epochs

    assert resolve_checkpoint_every_epochs({}) == 50
    assert resolve_checkpoint_every_epochs({"checkpoint_every_epochs": 3}) == 3

    try:
        resolve_checkpoint_every_epochs({"checkpoint_every_epochs": 0})
        raise AssertionError("Expected ValueError for non-positive checkpoint frequency")
    except ValueError:
        pass


def test_build_epoch_train_results_uses_standardized_throughput_keys():
    from ijepath.train_cross_resolution_jepa import build_epoch_train_results

    payload = build_epoch_train_results(
        loss_avg=0.3,
        loss_min=0.2,
        loss_max=0.5,
        mask_a=16.0,
        mask_b=64.0,
        iter_time_ms=12.0,
        epoch_time_s=8.0,
        images_seen=256,
        images_per_sec=32.0,
        iterations_per_sec=2.0,
        lr=1e-4,
        wd=0.04,
        global_batch_size=32,
    )

    assert payload["images_per_sec"] == 32.0
    assert payload["iterations_per_sec"] == 2.0
    assert "images_per_s" not in payload
    assert "iters_per_s" not in payload


def test_train_step_csv_schema_is_standardized():
    from ijepath.train_cross_resolution_jepa import get_train_step_csv_columns

    columns = get_train_step_csv_columns()
    headers = [header for _, header in columns]

    assert headers == [
        "epoch",
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
        epoch=2,
        iteration=12,
        loss_avg=0.3456,
        context_keep_tokens=18.0,
        target_predict_tokens=64.0,
        weight_decay=0.04,
        learning_rate=2e-4,
        max_memory_mb=1536.0,
        iteration_time_ms=11.2,
    )

    assert line.startswith("epoch=2 iteration=12")
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
        epoch=3,
        iteration=7,
        first_layer_grad_norm=1.23e-2,
        last_layer_grad_norm=4.56e-2,
        grad_min=7.89e-4,
        grad_max=9.99e-1,
    )

    assert line.startswith("epoch=3 iteration=7")
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
        iterations_per_epoch=1,
        start_lr=1e-5,
        ref_lr=1e-4,
        warmup=0.1,
        num_epochs=1,
        use_bfloat16=True,
    )
    assert scaler is None


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

        def join(self):
            joins.append(self._args[0])

    monkeypatch.setattr(main_entry.mp, "Process", _FakeProcess)

    main_entry.launch_worker_processes(
        profile_config="profile.yaml",
        run_config="run.yaml",
        opts=["x=1"],
        visible_devices=["cpu", "1"],
    )

    assert len(starts) == 2
    assert len(joins) == 2
    assert set(joins) == {0, 1}


def test_launch_worker_processes_raises_on_failed_child(monkeypatch):
    class _FakeProcess:
        def __init__(self, target, args):
            self._target = target
            self._args = args
            self.exitcode = 0 if args[0] == 0 else 2

        def start(self):
            return None

        def join(self):
            return None

    monkeypatch.setattr(main_entry.mp, "Process", _FakeProcess)

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

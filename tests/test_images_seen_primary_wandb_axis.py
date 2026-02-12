from ijepath.log.tracker import log_images_seen_dict, update_log_dict


def test_wandb_images_seen_logging_uses_images_seen_step(monkeypatch):
    calls = []
    defined = []

    class _FakeRun:
        @staticmethod
        def define_metric(*args, **kwargs):
            defined.append((args, kwargs))

    class _FakeWandb:
        run = _FakeRun()

        @staticmethod
        def log(payload, step=None):
            calls.append((payload, step))

        @staticmethod
        def define_metric(*args, **kwargs):
            defined.append((args, kwargs))

    monkeypatch.setattr("ijepath.log.tracker.wandb", _FakeWandb)

    log_dict = {"images_seen": 128}
    update_log_dict("train", {"loss": 0.25, "wd": 0.04}, log_dict, step="images_seen")
    update_log_dict("train", {"lr": 1e-4}, log_dict, step="images_seen")
    log_images_seen_dict(log_dict, images_seen=128)

    assert len(calls) == 1
    payload, step = calls[0]
    assert step == 128
    assert payload["images_seen"] == 128
    assert payload["train/loss"] == 0.25
    assert payload["train/wd"] == 0.04
    assert payload["train/lr"] == 1e-4


def test_tune_metrics_can_bind_to_images_seen_axis(monkeypatch):
    calls = []
    defined = []

    class _FakeWandb:
        @staticmethod
        def log(payload, step=None):
            calls.append((payload, step))

        @staticmethod
        def define_metric(*args, **kwargs):
            defined.append((args, kwargs))

    monkeypatch.setattr("ijepath.log.tracker.wandb", _FakeWandb)

    log_payload = {"images_seen": 5000}
    update_log_dict(
        "tune",
        {"ri": 0.42, "tune_index": 7},
        log_payload,
        step="images_seen",
    )
    log_images_seen_dict(log_payload, images_seen=5000)

    assert any(args[0] == "tune/ri" and kwargs.get("step_metric") == "images_seen" for args, kwargs in defined)
    assert any(
        args[0] == "tune/tune_index" and kwargs.get("step_metric") == "images_seen"
        for args, kwargs in defined
    )
    payload, step = calls[0]
    assert payload["images_seen"] == 5000
    assert step == 5000

from ijepath.log.tracker import log_images_seen_dict, update_log_dict


def test_wandb_images_seen_logging_uses_images_seen_step_and_single_payload(monkeypatch):
    calls = []

    class _FakeWandb:
        @staticmethod
        def log(payload, step=None):
            calls.append((payload, step))

        @staticmethod
        def define_metric(*args, **kwargs):
            return None

    monkeypatch.setattr("ijepath.log.tracker.wandb", _FakeWandb)

    log_dict = {"images_seen": 3}
    update_log_dict("train", {"loss": 0.25, "wd": 0.04}, log_dict, step="images_seen")
    update_log_dict("train", {"lr": 1e-4}, log_dict, step="images_seen")
    log_images_seen_dict(log_dict, images_seen=3)

    assert len(calls) == 1
    payload, step = calls[0]
    assert step == 3
    assert payload["images_seen"] == 3
    assert payload["train/loss"] == 0.25
    assert payload["train/wd"] == 0.04
    assert payload["train/lr"] == 1e-4


def test_wandb_pass_loss_metrics_bind_to_images_seen_step(monkeypatch):
    calls = []

    class _FakeWandb:
        @staticmethod
        def log(payload, step=None):
            calls.append((payload, step))

        @staticmethod
        def define_metric(*args, **kwargs):
            return None

    monkeypatch.setattr("ijepath.log.tracker.wandb", _FakeWandb)

    log_dict = {"images_seen": 6400, "pass_index": 2}
    update_log_dict(
        "train",
        {"loss_pass_avg": 0.1234, "loss_pass_min": 0.101, "loss_pass_max": 0.165},
        log_dict,
        step="images_seen",
    )
    log_images_seen_dict(log_dict, images_seen=6400)

    assert len(calls) == 1
    payload, step = calls[0]
    assert step == 6400
    assert payload["images_seen"] == 6400
    assert payload["pass_index"] == 2
    assert payload["train/loss_pass_avg"] == 0.1234
    assert payload["train/loss_pass_min"] == 0.101
    assert payload["train/loss_pass_max"] == 0.165

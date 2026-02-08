from ijepath.log.tracker import log_epoch_dict, update_log_dict


def test_wandb_epoch_logging_uses_epoch_step_and_single_payload(monkeypatch):
    calls = []

    class _FakeWandb:
        @staticmethod
        def log(payload, step=None):
            calls.append((payload, step))

        @staticmethod
        def define_metric(*args, **kwargs):
            return None

    monkeypatch.setattr("ijepath.log.tracker.wandb", _FakeWandb)

    log_dict = {"epoch": 3}
    update_log_dict("train", {"loss": 0.25, "wd": 0.04}, log_dict, step="epoch")
    update_log_dict("train", {"lr": 1e-4}, log_dict, step="epoch")
    log_epoch_dict(log_dict, epoch=3)

    assert len(calls) == 1
    payload, step = calls[0]
    assert step == 3
    assert payload["epoch"] == 3
    assert payload["train/loss"] == 0.25
    assert payload["train/wd"] == 0.04
    assert payload["train/lr"] == 1e-4


from ijepath.log import tracker
from ijepath.train_cross_resolution_jepa import resolve_wandb_exit_code_from_outcome


def test_finish_wandb_forwards_explicit_exit_code(monkeypatch):
    calls: list[dict] = []

    class _FakeWandb:
        run = object()

        @staticmethod
        def finish(**kwargs):
            calls.append(dict(kwargs))

    monkeypatch.setattr(tracker, "wandb", _FakeWandb)

    tracker.finish_wandb(exit_code=2)

    assert calls == [{"exit_code": 2}]


def test_resolve_wandb_exit_code_finished_when_training_completed():
    assert resolve_wandb_exit_code_from_outcome(
        training_completed_successfully=True,
        fatal_training_exception=None,
    ) == 0


def test_resolve_wandb_exit_code_failed_on_runtime_error():
    assert resolve_wandb_exit_code_from_outcome(
        training_completed_successfully=False,
        fatal_training_exception=RuntimeError("boom"),
    ) == 1


def test_resolve_wandb_exit_code_failed_on_keyboard_interrupt():
    assert resolve_wandb_exit_code_from_outcome(
        training_completed_successfully=False,
        fatal_training_exception=KeyboardInterrupt(),
    ) == 1


def test_resolve_wandb_exit_code_finished_even_if_teardown_error_logged():
    assert resolve_wandb_exit_code_from_outcome(
        training_completed_successfully=True,
        fatal_training_exception=RuntimeError("teardown failure"),
    ) == 0

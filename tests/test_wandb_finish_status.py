from ijepath.log import tracker
from ijepath.train_cross_resolution_jepa import resolve_uncaught_exception_exit_code


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


def test_resolve_uncaught_exception_exit_code_runtime_error():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        assert resolve_uncaught_exception_exit_code() == 1


def test_resolve_uncaught_exception_exit_code_system_exit():
    try:
        raise SystemExit(7)
    except SystemExit:
        assert resolve_uncaught_exception_exit_code() == 7

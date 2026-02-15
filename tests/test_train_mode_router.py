import types

import pytest

import ijepath.train as train_router


def _dummy_args(mode: str) -> dict:
    return {
        "pretraining": {"mode": mode},
    }


def test_router_dispatches_to_cross_resolution(monkeypatch):
    calls = []

    def _fake_main(*, args, resume_preempt, distributed_state):
        calls.append((args, resume_preempt, distributed_state))
        return "cross"

    fake_module = types.SimpleNamespace(main=_fake_main)
    monkeypatch.setattr(train_router, "_load_cross_resolution_trainer", lambda: fake_module)

    out = train_router.main(args=_dummy_args("cross_resolution"), resume_preempt=True, distributed_state=(0, 1))
    assert out == "cross"
    assert calls[0][1] is True
    assert calls[0][2] == (0, 1)


def test_router_dispatches_to_canonical(monkeypatch):
    calls = []

    def _fake_main(*, args, resume_preempt, distributed_state):
        calls.append((args, resume_preempt, distributed_state))
        return "canonical"

    fake_module = types.SimpleNamespace(main=_fake_main)
    monkeypatch.setattr(train_router, "_load_canonical_trainer", lambda: fake_module)

    out = train_router.main(args=_dummy_args("canonical"), resume_preempt=False, distributed_state=None)
    assert out == "canonical"
    assert calls[0][1] is False


def test_router_rejects_missing_mode():
    with pytest.raises(ValueError, match="pretraining.mode"):
        train_router.main(args={}, resume_preempt=False, distributed_state=None)


def test_router_rejects_unknown_mode():
    with pytest.raises(ValueError, match="pretraining.mode"):
        train_router.main(args=_dummy_args("foo"), resume_preempt=False, distributed_state=None)

import pytest

from ijepath.utils import distributed


def test_init_distributed_single_process_skips_process_group(monkeypatch):
    init_calls: list[dict] = []
    monkeypatch.setattr(distributed.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        distributed.dist,
        "init_process_group",
        lambda **kwargs: init_calls.append(dict(kwargs)),
    )

    world_size, rank = distributed.init_distributed(rank_and_world_size=(0, 1))

    assert (world_size, rank) == (1, 0)
    assert init_calls == []


def test_init_distributed_requires_master_port_for_multi_process(monkeypatch):
    monkeypatch.setattr(distributed.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(distributed.torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("MASTER_PORT", raising=False)

    with pytest.raises(ValueError, match="MASTER_PORT"):
        distributed.init_distributed(
            rank_and_world_size=(0, 2),
            master_addr="127.0.0.1",
            port=None,
        )


def test_init_distributed_rejects_invalid_master_port(monkeypatch):
    monkeypatch.setattr(distributed.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(distributed.torch.cuda, "is_available", lambda: True)

    with pytest.raises(ValueError, match="MASTER_PORT"):
        distributed.init_distributed(
            rank_and_world_size=(0, 2),
            master_addr="127.0.0.1",
            port="bad-port",
        )


def test_init_distributed_raises_on_process_group_failure(monkeypatch):
    monkeypatch.setattr(distributed.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(distributed.torch.cuda, "is_available", lambda: True)

    def _raise_init_process_group(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(distributed.dist, "init_process_group", _raise_init_process_group)

    with pytest.raises(RuntimeError, match="Failed to initialize distributed process group"):
        distributed.init_distributed(
            rank_and_world_size=(0, 2),
            master_addr="127.0.0.1",
            port=29599,
        )


def test_init_distributed_returns_existing_context_when_already_initialized(monkeypatch):
    monkeypatch.setattr(distributed.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(distributed.dist, "get_rank", lambda: 2)

    world_size, rank = distributed.init_distributed(rank_and_world_size=(0, 1))

    assert (world_size, rank) == (4, 2)

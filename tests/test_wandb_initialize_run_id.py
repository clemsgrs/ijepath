from ijepath.log import tracker


def test_initialize_wandb_uses_run_id_without_forced_resume(monkeypatch):
    init_calls: list[dict] = []

    class _FakeRun:
        def define_metric(self, *_args, **_kwargs):
            return None

    class _FakeWandb:
        @staticmethod
        def login(**_kwargs):
            return None

        @staticmethod
        def init(**kwargs):
            init_calls.append(dict(kwargs))
            return _FakeRun()

    monkeypatch.setattr(tracker, "wandb", _FakeWandb)
    monkeypatch.setattr(tracker.os, "environ", {})

    tracker.initialize_wandb(
        {
            "logging": {"write_tag": "unit"},
            "wandb": {"enable": True, "project": "ijepath", "run_id": "abc123"},
        }
    )

    assert len(init_calls) == 1
    assert init_calls[0]["id"] == "abc123"
    assert "resume" not in init_calls[0]


def test_initialize_wandb_resume_id_takes_precedence(monkeypatch):
    init_calls: list[dict] = []

    class _FakeRun:
        def define_metric(self, *_args, **_kwargs):
            return None

    class _FakeWandb:
        @staticmethod
        def login(**_kwargs):
            return None

        @staticmethod
        def init(**kwargs):
            init_calls.append(dict(kwargs))
            return _FakeRun()

    monkeypatch.setattr(tracker, "wandb", _FakeWandb)
    monkeypatch.setattr(tracker.os, "environ", {})

    tracker.initialize_wandb(
        {
            "logging": {"write_tag": "unit"},
            "wandb": {
                "enable": True,
                "project": "ijepath",
                "run_id": "generated-id",
                "resume_id": "resume-id",
            },
        }
    )

    assert len(init_calls) == 1
    assert init_calls[0]["id"] == "resume-id"
    assert init_calls[0]["resume"] == "must"

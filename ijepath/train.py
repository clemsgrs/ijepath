from __future__ import annotations


def _load_cross_resolution_trainer():
    from ijepath import train_cross_resolution_jepa as trainer_module

    return trainer_module


def _load_canonical_trainer():
    from ijepath import train_canonical_jepa as trainer_module

    return trainer_module


def _resolve_pretraining_mode(args: dict) -> str:
    pretraining_cfg = dict(args.get("pretraining", {}) or {})
    mode_raw = pretraining_cfg.get("mode", None)
    if mode_raw is None or not str(mode_raw).strip():
        raise ValueError("Missing required config value: pretraining.mode")

    mode = str(mode_raw).strip().lower()
    if mode not in {"canonical", "cross_resolution"}:
        raise ValueError(
            "pretraining.mode must be one of {'canonical', 'cross_resolution'}, "
            f"got {mode_raw!r}"
        )
    return mode


def main(
    args,
    resume_preempt: bool = False,
    distributed_state: tuple[int, int] | None = None,
):
    mode = _resolve_pretraining_mode(args=args)
    trainer = (
        _load_canonical_trainer()
        if mode == "canonical"
        else _load_cross_resolution_trainer()
    )

    return trainer.main(
        args=args,
        resume_preempt=resume_preempt,
        distributed_state=distributed_state,
    )

from ijepath.train_cross_resolution_jepa import main as train_main


def main(
    args,
    resume_preempt: bool = False,
    distributed_state: tuple[int, int] | None = None,
):
    return train_main(
        args=args,
        resume_preempt=resume_preempt,
        distributed_state=distributed_state,
    )

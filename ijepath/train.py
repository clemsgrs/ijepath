from ijepath.train_cross_resolution_jepa import main as train_main


def main(args, resume_preempt: bool = False):
    return train_main(args=args, resume_preempt=resume_preempt)

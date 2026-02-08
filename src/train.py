from src.train_pathology_cross_resolution_jepa import main as pathology_main


def main(args, resume_preempt: bool = False):
    return pathology_main(args=args, resume_preempt=resume_preempt)

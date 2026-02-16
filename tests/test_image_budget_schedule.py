from ijepath.train_cross_resolution_jepa import (
    compute_schedule_total_steps,
    compute_total_steps,
    crossed_image_thresholds,
)


def test_compute_total_steps_uses_ceiling():
    assert compute_total_steps(total_images_budget=10, global_batch_size=4) == 3
    assert compute_total_steps(total_images_budget=8, global_batch_size=4) == 2


def test_crossed_image_thresholds_handles_multi_cross():
    crossed, next_threshold = crossed_image_thresholds(
        prev_images_seen=900_000,
        new_images_seen=2_300_000,
        next_threshold=1_000_000,
        interval=1_000_000,
    )
    assert crossed == [1_000_000, 2_000_000]
    assert next_threshold == 3_000_000


def test_compute_schedule_total_steps_scales_from_total_steps():
    assert compute_schedule_total_steps(total_steps=100, ipe_scale=1.25) == 125
    # Ensure scheduler horizon never falls below optimization steps.
    assert compute_schedule_total_steps(total_steps=100, ipe_scale=0.5) == 100

import pytest

from ijepath.train_cross_resolution_jepa import (
    compute_anchor_pass_budget,
    resolve_total_images_budget,
)


def test_compute_anchor_pass_budget_requires_positive_values():
    with pytest.raises(ValueError, match="anchor_count"):
        compute_anchor_pass_budget(
            anchor_count=0,
            total_images_budget=1000,
            interval_images=100,
            run_baseline_at_zero=True,
        )


def test_resolve_total_images_budget_requires_positive_value():
    optimization_cfg = {"total_images_budget": 1000}
    assert resolve_total_images_budget(optimization_cfg) == 1000

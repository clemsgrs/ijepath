from ijepath.train_cross_resolution_jepa import compute_anchor_pass_budget


def test_anchor_pass_budget_computation():
    budget = compute_anchor_pass_budget(
        anchor_count=400,
        total_images_budget=1000,
        interval_images=200,
        run_baseline_at_zero=True,
    )

    assert budget["anchor_count"] == 400
    assert budget["anchor_passes_total"] == 2.5
    assert budget["coverage_first_pass"] == 1.0
    assert budget["mean_anchor_reuse"] == 1.5
    assert budget["expected_tune_events"] == 6


def test_anchor_pass_budget_partial_coverage():
    budget = compute_anchor_pass_budget(
        anchor_count=1000,
        total_images_budget=400,
        interval_images=200,
        run_baseline_at_zero=False,
    )

    assert budget["anchor_passes_total"] == 0.4
    assert budget["coverage_first_pass"] == 0.4
    assert budget["mean_anchor_reuse"] == 0.0
    assert budget["expected_tune_events"] == 2

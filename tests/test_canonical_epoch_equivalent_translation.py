import pytest

from ijepath.train_canonical_jepa import build_epoch_equivalent_schedule_summary


def test_build_epoch_equivalent_schedule_summary_derives_recommendations():
    summary = build_epoch_equivalent_schedule_summary(
        optimization_cfg={
            "total_images_budget": 1000,
            "warmup": 0.1,
            "epochs_equivalent": 2.0,
            "warmup_epochs": 0.5,
        },
        anchor_count=400,
    )

    assert summary["recommended_total_images_budget"] == 800
    assert summary["implied_epochs_from_budget"] == pytest.approx(2.5)
    assert summary["recommended_warmup_fraction"] == pytest.approx(0.25)
    assert summary["warmup_fraction_matches_recommendation"] is False


def test_build_epoch_equivalent_schedule_summary_handles_missing_epoch_inputs():
    summary = build_epoch_equivalent_schedule_summary(
        optimization_cfg={
            "total_images_budget": 1000,
            "warmup": 0.1,
        },
        anchor_count=400,
    )

    assert summary["recommended_total_images_budget"] is None
    assert summary["recommended_warmup_fraction"] is None
    assert summary["implied_epochs_from_budget"] == pytest.approx(2.5)


def test_build_epoch_equivalent_schedule_summary_validates_positive_values():
    with pytest.raises(ValueError, match="anchor_count"):
        build_epoch_equivalent_schedule_summary(
            optimization_cfg={"total_images_budget": 1000, "warmup": 0.1},
            anchor_count=0,
        )

    with pytest.raises(ValueError, match="warmup_epochs"):
        build_epoch_equivalent_schedule_summary(
            optimization_cfg={
                "total_images_budget": 1000,
                "warmup": 0.1,
                "epochs_equivalent": 2.0,
                "warmup_epochs": 0.0,
            },
            anchor_count=400,
        )

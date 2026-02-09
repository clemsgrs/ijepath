from types import SimpleNamespace

import pytest
import torch

from ijepath.datasets.cross_resolution_loader_factory import (
    validate_dataset_collator_size_alignment,
)
from ijepath.train_cross_resolution_jepa import validate_encoder_input_sizes


def test_validate_encoder_input_sizes_accepts_matching_tensors():
    context_images = torch.rand(2, 3, 224, 224)
    target_images = torch.rand(2, 4, 3, 112, 112)

    validate_encoder_input_sizes(
        context_images=context_images,
        target_images=target_images,
        expected_context_input_size_px=224,
        expected_target_input_size_px=112,
        patch_size=16,
    )


def test_validate_encoder_input_sizes_raises_on_mismatch():
    context_images = torch.rand(2, 3, 224, 224)
    target_images = torch.rand(2, 4, 3, 112, 112)

    with pytest.raises(ValueError, match="Context image size mismatch"):
        validate_encoder_input_sizes(
            context_images=context_images,
            target_images=target_images,
            expected_context_input_size_px=208,
            expected_target_input_size_px=112,
            patch_size=16,
        )


def test_validate_dataset_collator_size_alignment_raises_on_mismatch():
    dataset = SimpleNamespace(
        context_size_requested_px=224,
        target_size_requested_px=112,
    )

    with pytest.raises(ValueError, match="Context input size drift"):
        validate_dataset_collator_size_alignment(
            dataset=dataset,
            context_input_size_px=208,
            target_input_size_px=112,
            patch_size=16,
        )

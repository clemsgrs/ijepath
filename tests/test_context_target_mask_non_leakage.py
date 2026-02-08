import torch

from ijepath.masks.context_target_footprint_mask_collator import ContextTargetFootprintMaskCollator


def test_pathology_context_target_mask_non_leakage():
    collator = ContextTargetFootprintMaskCollator(
        input_size=224,
        patch_size=16,
        nenc=1,
        min_keep=4,
    )

    context = torch.zeros(3, 224, 224)
    targets = torch.zeros(4, 3, 224, 224)
    boxes = torch.tensor(
        [
            [32.0, 32.0, 96.0, 96.0],
            [96.0, 32.0, 160.0, 96.0],
            [32.0, 96.0, 96.0, 160.0],
            [96.0, 96.0, 160.0, 160.0],
        ],
        dtype=torch.float32,
    )

    batch = [
        {
            "context_image": context,
            "target_images": targets,
            "target_boxes_in_context_pixels": boxes,
            "sample_metadata": {"slide_id": "A", "anchor_id": "A_0"},
        }
    ]

    collated_batch, masks_enc, masks_pred = collator(batch)

    assert collated_batch["context_images"].shape == (1, 3, 224, 224)
    assert collated_batch["target_images"].shape == (1, 4, 3, 224, 224)
    assert len(masks_enc) == 1
    assert len(masks_pred) == 4

    enc_set = set(masks_enc[0][0].tolist())
    pred_union = set()
    for k in range(len(masks_pred)):
        pred_union.update(masks_pred[k][0].tolist())

    assert pred_union, "Predictor footprint should not be empty"
    assert enc_set.isdisjoint(pred_union), "Target footprints leaked into encoder-visible context"

import torch

from ijepath.helper import init_model
from ijepath.masks.context_target_footprint_mask_collator import ContextTargetFootprintMaskCollator
from ijepath.train_cross_resolution_jepa import (
    flatten_teacher_targets_for_predictor_order,
    pool_predictor_tokens,
)


def test_training_step_smoke():
    device = torch.device("cpu")
    context_input_size_px = 224
    target_input_size_px = 112

    encoder, predictor = init_model(
        device=device,
        patch_size=16,
        architecture="vit_tiny",
        crop_size=context_input_size_px,
        pred_depth=3,
        pred_emb_dim=192,
    )
    target_encoder, _ = init_model(
        device=device,
        patch_size=16,
        architecture="vit_tiny",
        crop_size=target_input_size_px,
        pred_depth=3,
        pred_emb_dim=192,
        init_predictor=False,
    )
    for p in target_encoder.parameters():
        p.requires_grad = False

    collator = ContextTargetFootprintMaskCollator(
        input_size=context_input_size_px,
        patch_size=16,
        nenc=1,
        min_keep=4,
    )

    batch = []
    for i in range(2):
        batch.append(
            {
                "context_image": torch.rand(3, context_input_size_px, context_input_size_px),
                "target_images": torch.rand(4, 3, target_input_size_px, target_input_size_px),
                "target_boxes_in_context_pixels": torch.tensor(
                    [
                        [32.0, 32.0, 96.0, 96.0],
                        [96.0, 32.0, 160.0, 96.0],
                        [32.0, 96.0, 96.0, 160.0],
                        [96.0, 96.0, 160.0, 160.0],
                    ],
                    dtype=torch.float32,
                ),
                "sample_metadata": {"slide_id": "S", "anchor_id": f"A_{i}"},
            }
        )

    collated_batch, masks_enc, masks_pred = collator(batch)
    context = collated_batch["context_images"]
    targets = collated_batch["target_images"]

    bsz, ntargets, c, h, w = targets.shape

    z = encoder(context, masks_enc)
    z = predictor(z, masks_enc, masks_pred)
    z = pool_predictor_tokens(z)

    with torch.no_grad():
        target_tokens = target_encoder(targets.view(bsz * ntargets, c, h, w))
        teacher = target_tokens.mean(dim=1).view(bsz, ntargets, -1)

    teacher_flat = flatten_teacher_targets_for_predictor_order(
        teacher=teacher,
        batch_size=bsz,
        num_pred_masks=len(masks_pred),
        num_enc_masks=len(masks_enc),
    )

    loss = torch.nn.functional.smooth_l1_loss(z, teacher_flat)
    assert torch.isfinite(loss)

    loss.backward()
    grad_norm = encoder.patch_embed.proj.weight.grad.abs().mean().item()
    assert grad_norm > 0.0

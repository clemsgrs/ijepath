import torch
import torch.nn.functional as F

from ijepath.helper import init_model
from ijepath.masks.multiblock import MaskCollator
from ijepath.masks.utils import apply_masks
from ijepath.train_canonical_jepa import flatten_teacher_tokens_for_predictor_order


def test_canonical_training_step_smoke():
    device = torch.device("cpu")
    input_size = 224

    encoder, predictor = init_model(
        device=device,
        patch_size=16,
        architecture="vit_tiny",
        crop_size=input_size,
        pred_depth=3,
        pred_emb_dim=192,
    )
    target_encoder, _ = init_model(
        device=device,
        patch_size=16,
        architecture="vit_tiny",
        crop_size=input_size,
        pred_depth=3,
        pred_emb_dim=192,
        init_predictor=False,
    )
    for p in target_encoder.parameters():
        p.requires_grad = False

    collator = MaskCollator(
        input_size=input_size,
        patch_size=16,
        nenc=1,
        npred=4,
        min_keep=4,
    )

    batch = [torch.rand(3, input_size, input_size), torch.rand(3, input_size, input_size)]
    images, masks_enc, masks_pred = collator(batch)

    z = encoder(images, masks_enc)
    z = predictor(z, masks_enc, masks_pred)

    with torch.no_grad():
        h = target_encoder(images)
        h = F.layer_norm(h, (h.size(-1),))
        h = apply_masks(h, masks_pred)

    h_flat = flatten_teacher_tokens_for_predictor_order(
        teacher=h,
        batch_size=int(images.shape[0]),
        num_pred_masks=len(masks_pred),
        num_enc_masks=len(masks_enc),
    )

    loss = F.smooth_l1_loss(z, h_flat)
    assert torch.isfinite(loss)

    loss.backward()
    grad_norm = encoder.patch_embed.proj.weight.grad.abs().mean().item()
    assert grad_norm > 0.0

import torch

from ijepath.train_canonical_jepa import flatten_teacher_tokens_for_predictor_order


def test_flatten_teacher_tokens_for_predictor_order_repeats_per_encoder_mask():
    batch_size = 2
    num_pred_masks = 3
    num_enc_masks = 2
    num_tokens = 4
    embed_dim = 5

    teacher = torch.arange(
        batch_size * num_pred_masks * num_tokens * embed_dim,
        dtype=torch.float32,
    ).view(batch_size * num_pred_masks, num_tokens, embed_dim)

    flattened = flatten_teacher_tokens_for_predictor_order(
        teacher=teacher,
        batch_size=batch_size,
        num_pred_masks=num_pred_masks,
        num_enc_masks=num_enc_masks,
    )

    assert flattened.shape == (batch_size * num_pred_masks * num_enc_masks, num_tokens, embed_dim)

    # Predictor order repeats each pred-mask block for each encoder mask.
    assert torch.equal(flattened[0:batch_size], teacher[0:batch_size])
    assert torch.equal(flattened[batch_size : 2 * batch_size], teacher[0:batch_size])
    assert torch.equal(flattened[2 * batch_size : 3 * batch_size], teacher[batch_size : 2 * batch_size])

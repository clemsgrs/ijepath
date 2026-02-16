import torch

from ijepath.models.vision_transformer import VisionTransformer


def _make_model() -> VisionTransformer:
    return VisionTransformer(
        img_size=[224],
        patch_size=16,
        embed_dim=384,
        depth=1,
        num_heads=6,
        predictor_depth=1,
    )


def test_canonical_vit_keeps_224_patch_count_without_off_by_one():
    model = _make_model().eval()
    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 14 * 14, 384)


def test_canonical_vit_interpolates_pos_embed_for_square_256_input():
    model = _make_model().eval()
    x = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 16 * 16, 384)


def test_interpolate_pos_encoding_target_length_matches_square_grid():
    model = _make_model().eval()
    tokens_256 = torch.zeros(1, 16 * 16, 384)

    with torch.no_grad():
        pos = model.interpolate_pos_encoding(tokens_256, model.pos_embed)

    assert pos.shape == (1, 16 * 16, 384)

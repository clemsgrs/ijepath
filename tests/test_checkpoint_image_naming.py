from ijepath.train_cross_resolution_jepa import checkpoint_name_for_images


def test_checkpoint_name_for_images_uses_stable_suffix():
    assert checkpoint_name_for_images(tag="jepa", images_seen=1_000_000) == "jepa-img1000000.pth.tar"
    assert checkpoint_name_for_images(tag="run", images_seen=42) == "run-img42.pth.tar"

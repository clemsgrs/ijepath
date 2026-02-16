import pickle

import torch

from ijepath.datasets import canonical_loader_factory as clf


class _FakeDataset:
    def __init__(self, *args, **kwargs):
        self.crop_size_px = int(kwargs["crop_size_px"])

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        raise IndexError(idx)


def test_canonical_loader_collate_fn_is_picklable(monkeypatch):
    monkeypatch.setattr(clf, "CanonicalWSIDataset", _FakeDataset)

    _dataset, loader, _sampler = clf.make_canonical_loader(
        batch_size=2,
        pin_mem=False,
        num_workers=1,
        world_size=1,
        rank=0,
        drop_last=False,
        anchor_catalog_manifest="dummy.json",
        patch_size=16,
        input_mpp=0.5,
        source_tile_size_px=256,
        crop_size_px=224,
        transform_preset="official_ijepa",
        crop_scale=(0.3, 1.0),
        use_horizontal_flip=True,
        horizontal_flip_prob=0.5,
        use_color_distortion=False,
        color_jitter_strength=0.0,
        use_gaussian_blur=False,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        num_enc_masks=1,
        num_pred_masks=4,
        min_keep=16,
        allow_overlap=False,
        seed=0,
        spacing_tolerance=0.05,
    )

    payload = pickle.dumps(loader.collate_fn)
    assert isinstance(payload, bytes)
    assert len(payload) > 0


def test_canonical_loader_uses_configured_mask_collator_scales(monkeypatch):
    monkeypatch.setattr(clf, "CanonicalWSIDataset", _FakeDataset)
    captured = {}
    enc_mask_scale = (0.81, 0.97)
    pred_mask_scale = (0.13, 0.21)
    aspect_ratio = (0.61, 1.42)

    class _DummyMaskCollator:
        def __init__(self, **kwargs):
            captured["kwargs"] = dict(kwargs)

        def __call__(self, images):
            tensor = torch.stack(images, dim=0)
            masks_enc = [torch.zeros((tensor.shape[0], 1), dtype=torch.long)]
            masks_pred = [torch.zeros((tensor.shape[0], 1), dtype=torch.long)]
            return tensor, masks_enc, masks_pred

    monkeypatch.setattr(clf, "MaskCollator", _DummyMaskCollator)

    _dataset, loader, _sampler = clf.make_canonical_loader(
        batch_size=2,
        pin_mem=False,
        num_workers=0,
        world_size=1,
        rank=0,
        drop_last=False,
        anchor_catalog_manifest="dummy.json",
        patch_size=16,
        input_mpp=0.5,
        source_tile_size_px=256,
        crop_size_px=224,
        transform_preset="official_ijepa",
        crop_scale=(0.3, 1.0),
        use_horizontal_flip=True,
        horizontal_flip_prob=0.5,
        use_color_distortion=False,
        color_jitter_strength=0.0,
        use_gaussian_blur=False,
        enc_mask_scale=enc_mask_scale,
        pred_mask_scale=pred_mask_scale,
        aspect_ratio=aspect_ratio,
        num_enc_masks=1,
        num_pred_masks=4,
        min_keep=16,
        allow_overlap=False,
        seed=0,
        spacing_tolerance=0.05,
    )

    collate = loader.collate_fn
    batch = [
        {"image": torch.zeros((3, 224, 224)), "sample_metadata": {"anchor_id": "a0"}},
        {"image": torch.zeros((3, 224, 224)), "sample_metadata": {"anchor_id": "a1"}},
    ]
    collated_batch, masks_enc, masks_pred = collate(batch)
    assert collated_batch["image"].shape == (2, 3, 224, 224)
    assert len(masks_enc) == 1
    assert len(masks_pred) == 1

    kwargs = captured["kwargs"]
    assert kwargs["enc_mask_scale"] == enc_mask_scale
    assert kwargs["pred_mask_scale"] == pred_mask_scale
    assert kwargs["aspect_ratio"] == aspect_ratio


def test_canonical_loader_uses_configured_allow_overlap(monkeypatch):
    monkeypatch.setattr(clf, "CanonicalWSIDataset", _FakeDataset)
    captured = {}

    class _DummyMaskCollator:
        def __init__(self, **kwargs):
            captured["kwargs"] = dict(kwargs)

        def __call__(self, images):
            tensor = torch.stack(images, dim=0)
            masks_enc = [torch.zeros((tensor.shape[0], 1), dtype=torch.long)]
            masks_pred = [torch.zeros((tensor.shape[0], 1), dtype=torch.long)]
            return tensor, masks_enc, masks_pred

    monkeypatch.setattr(clf, "MaskCollator", _DummyMaskCollator)

    _dataset, loader, _sampler = clf.make_canonical_loader(
        batch_size=2,
        pin_mem=False,
        num_workers=0,
        world_size=1,
        rank=0,
        drop_last=False,
        anchor_catalog_manifest="dummy.json",
        patch_size=16,
        input_mpp=0.5,
        source_tile_size_px=256,
        crop_size_px=224,
        transform_preset="official_ijepa",
        crop_scale=(0.3, 1.0),
        use_horizontal_flip=True,
        horizontal_flip_prob=0.5,
        use_color_distortion=False,
        color_jitter_strength=0.0,
        use_gaussian_blur=False,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        num_enc_masks=1,
        num_pred_masks=4,
        min_keep=16,
        allow_overlap=True,
        seed=0,
        spacing_tolerance=0.05,
    )

    collate = loader.collate_fn
    batch = [
        {"image": torch.zeros((3, 224, 224)), "sample_metadata": {"anchor_id": "a0"}},
        {"image": torch.zeros((3, 224, 224)), "sample_metadata": {"anchor_id": "a1"}},
    ]
    _collated_batch, _masks_enc, _masks_pred = collate(batch)

    kwargs = captured["kwargs"]
    assert kwargs["allow_overlap"] is True

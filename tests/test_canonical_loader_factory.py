import pickle

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
        num_enc_masks=1,
        num_pred_masks=4,
        min_keep=16,
        seed=0,
        spacing_tolerance=0.05,
    )

    payload = pickle.dumps(loader.collate_fn)
    assert isinstance(payload, bytes)
    assert len(payload) > 0

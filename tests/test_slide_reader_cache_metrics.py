import json
from pathlib import Path

from ijepath.datasets import cross_resolution_wsi_dataset as ds_mod
from ijepath.datasets.cross_resolution_wsi_dataset import CrossResolutionWSIDataset


def _write_manifest(path: Path) -> Path:
    manifest = {
        "schema_version": 1,
        "profile": {
            "context_mpp": 1.0,
            "target_mpp": 0.5,
            "context_fov_um": 512.0,
            "target_fov_um": 128.0,
            "targets_per_context": 4,
        },
        "total_anchors": 1,
        "stratum_counts": {"unknown": 1},
        "anchor_shards": [
            {
                "path": "/tmp/nonexistent.parquet",
                "rows": 1,
                "stratum_counts": {"unknown": 1},
            }
        ],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_reader_cache_event_and_counters(monkeypatch, tmp_path: Path):
    manifest = _write_manifest(tmp_path / "manifest.json")
    dataset = CrossResolutionWSIDataset(
        anchor_catalog_manifest=str(manifest),
        context_mpp=1.0,
        target_mpp=0.5,
        context_fov_um=512.0,
        target_fov_um=128.0,
        patch_size=16,
        targets_per_context=4,
        seed=0,
        max_open_slides_per_worker=1,
    )

    class _DummyReader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(ds_mod, "WholeSlideDataReaderAdapter", _DummyReader)

    row_a = {"slide_id": "a", "wsi_path": "a.svs", "mask_path": None}
    row_b = {"slide_id": "b", "wsi_path": "b.svs", "mask_path": None}

    dataset._get_reader(row_a)
    assert dataset._last_reader_cache_event["event"] == "miss"
    assert dataset._last_reader_cache_event["evicted"] == 0

    dataset._get_reader(row_a)
    assert dataset._last_reader_cache_event["event"] == "hit"

    dataset._get_reader(row_b)
    assert dataset._last_reader_cache_event["event"] == "miss"
    assert dataset._last_reader_cache_event["evicted"] == 1
    assert dataset._last_reader_cache_event["open_slides"] == 1
    assert dataset._reader_cache_hits == 1
    assert dataset._reader_cache_misses == 2
    assert dataset._reader_cache_evictions == 1

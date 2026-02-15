import importlib.util
import json
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from ijepath.datasets.canonical_wsi_dataset import CanonicalWSIDataset

if importlib.util.find_spec("pyarrow") is None:
    pytest.skip("pyarrow is required for parquet pipeline tests", allow_module_level=True)


def _write_anchor_manifest(path: Path, rows: list[dict]) -> None:
    shard_path = path.with_suffix(".parquet")
    normalized_rows = []
    for row in rows:
        row = dict(row)
        row.setdefault("stratum_id", "unknown")
        row.setdefault("profile_id", "canonical_test")
        normalized_rows.append(row)

    table = pa.Table.from_pylist(normalized_rows)
    pq.write_table(table, str(shard_path))

    stratum_counts: dict[str, int] = {}
    for row in normalized_rows:
        stratum = str(row.get("stratum_id", "unknown"))
        stratum_counts[stratum] = int(stratum_counts.get(stratum, 0) + 1)

    manifest = {
        "schema_version": 1,
        "profile": {
            "context_mpp": 0.5,
            "target_mpp": 0.5,
            "context_fov_um": 128.0,
            "target_fov_um": 128.0,
            "targets_per_context": 4,
        },
        "total_anchors": len(normalized_rows),
        "stratum_counts": stratum_counts,
        "anchor_shards": [
            {
                "path": str(shard_path.resolve()),
                "rows": len(normalized_rows),
                "stratum_counts": stratum_counts,
            }
        ],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_canonical_dataset_shapes(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    fixture_wsi = repo_root / "data/test-fixtures/test-wsi.tif"
    fixture_mask = repo_root / "data/test-fixtures/test-mask.tif"
    if not fixture_wsi.exists() or not fixture_mask.exists():
        pytest.skip("WSI fixture not available")

    manifest_path = tmp_path / "anchors_manifest.json"
    _write_anchor_manifest(
        manifest_path,
        [
            {
                "anchor_id": "test-slide_0000000",
                "slide_id": "test-slide",
                "wsi_path": str(fixture_wsi),
                "mask_path": str(fixture_mask),
                "center_x_level0": 2032,
                "center_y_level0": 2032,
                "wsi_level0_spacing_mpp": 0.25200000393750005,
                "stratum_id": "unknown",
            }
        ],
    )

    dataset = CanonicalWSIDataset(
        anchor_catalog_manifest=str(manifest_path),
        input_mpp=0.5,
        source_tile_size_px=256,
        crop_size_px=224,
        patch_size=16,
        seed=0,
        spacing_tolerance=0.05,
        backend="openslide",
        transform_preset="official_ijepa",
    )

    sample = dataset[0]
    assert sample["image"].shape == (3, 224, 224)
    assert sample["sample_metadata"]["slide_id"] == "test-slide"
    assert sample["sample_metadata"]["anchor_id"] == "test-slide_0000000"
    assert sample["sample_metadata"]["model_crop_size_px"] == 224
    assert sample["sample_metadata"]["source_tile_size_px_requested"] == 256

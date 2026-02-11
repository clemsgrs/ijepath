import importlib.util
import json
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from ijepath.datasets.cross_resolution_wsi_dataset import CrossResolutionWSIDataset

if importlib.util.find_spec("pyarrow") is None:
    pytest.skip("pyarrow is required for parquet pipeline tests", allow_module_level=True)


def _write_anchor_csv(path: Path):
    rows = []
    for i in range(3):
        rows.append(
            {
                "anchor_id": f"a{i}",
                "slide_id": f"slide{i}",
                "wsi_path": "/tmp/none.tif",
                "mask_path": "",
                "center_x_level0": 0,
                "center_y_level0": 0,
                "wsi_level0_spacing_mpp": 0.25,
                "target_margin_um": 8.0,
                "stratum_id": "unknown",
                "profile_id": "test_profile",
            }
        )

    shard_path = path.with_suffix(".parquet")
    pq.write_table(pa.Table.from_pylist(rows), str(shard_path))
    manifest = {
        "schema_version": 1,
        "profile": {
            "context_mpp": 1.0,
            "target_mpp": 0.5,
            "context_fov_um": 64.0,
            "target_fov_um": 16.0,
            "targets_per_context": 4,
        },
        "total_anchors": len(rows),
        "stratum_counts": {"unknown": len(rows)},
        "anchor_shards": [
            {
                "path": str(shard_path.resolve()),
                "rows": len(rows),
                "stratum_counts": {"unknown": len(rows)},
            }
        ],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_dataset_len_matches_anchor_count_and_pass_seed_changes(tmp_path: Path):
    csv_path = tmp_path / "anchors.csv"
    _write_anchor_csv(csv_path)

    dataset = CrossResolutionWSIDataset(
        anchor_catalog_manifest=str(csv_path),
        context_mpp=1.0,
        target_mpp=0.5,
        context_fov_um=64.0,
        target_fov_um=16.0,
        patch_size=16,
        targets_per_context=4,
        seed=7,
    )

    assert len(dataset) == 3

    s0 = dataset._rng_seed_for(index=1, anchor_attempt=0)
    dataset.set_pass_index(1)
    s1 = dataset._rng_seed_for(index=1, anchor_attempt=0)
    assert s1 != s0

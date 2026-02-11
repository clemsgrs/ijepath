import json
from pathlib import Path

import numpy as np

from ijepath.datasets.cross_resolution_wsi_dataset import CrossResolutionWSIDataset


def _write_manifest(path: Path, stratum_counts: dict[str, int]) -> Path:
    manifest = {
        "schema_version": 1,
        "profile": {
            "context_mpp": 1.0,
            "target_mpp": 0.5,
            "context_fov_um": 512.0,
            "target_fov_um": 128.0,
            "targets_per_context": 4,
        },
        "total_anchors": int(sum(stratum_counts.values())),
        "stratum_counts": stratum_counts,
        "anchor_shards": [
            {
                "path": "/tmp/nonexistent.parquet",
                "rows": int(sum(stratum_counts.values())),
                "stratum_counts": stratum_counts,
            }
        ],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_stratified_inverse_frequency_weights(tmp_path: Path):
    manifest = _write_manifest(tmp_path / "manifest.json", {"a": 100, "b": 25})
    dataset = CrossResolutionWSIDataset(
        anchor_catalog_manifest=str(manifest),
        context_mpp=1.0,
        target_mpp=0.5,
        context_fov_um=512.0,
        target_fov_um=128.0,
        patch_size=16,
        targets_per_context=4,
        seed=0,
        sampling_strategy="stratified_weighted",
        sampling_stratum_weights="inverse_frequency",
    )

    strata, weights = dataset._resolve_strata_and_weights()
    assert strata == ["a", "b"]
    assert np.isclose(float(weights.sum()), 1.0)
    assert float(weights[1]) > float(weights[0])


def test_anchor_partition_is_deterministic(tmp_path: Path):
    manifest = _write_manifest(tmp_path / "manifest.json", {"unknown": 10})
    dataset = CrossResolutionWSIDataset(
        anchor_catalog_manifest=str(manifest),
        context_mpp=1.0,
        target_mpp=0.5,
        context_fov_um=512.0,
        target_fov_um=128.0,
        patch_size=16,
        targets_per_context=4,
        seed=0,
    )

    part_a = dataset._partition_for_anchor("anchor-1", total_partitions=17)
    part_b = dataset._partition_for_anchor("anchor-1", total_partitions=17)
    assert part_a == part_b
    assert 0 <= part_a < 17

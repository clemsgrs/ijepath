import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

if importlib.util.find_spec("pyarrow") is None:
    pytest.skip("pyarrow is required for parquet pipeline tests", allow_module_level=True)


def test_pathology_valid_context_anchor_catalog_builder_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    manifest = tmp_path / "manifest.csv"
    slide_index = tmp_path / "slide_metadata.parquet"
    report_csv = tmp_path / "slide_metadata_build_report.csv"
    profile_yaml = tmp_path / "profile.yaml"
    anchor_manifest = tmp_path / "anchor_catalog_manifest.json"

    wsi_path = repo_root / "data/tcga-prad/wsi/TCGA-HC-8257.tif"
    mask_path = repo_root / "data/tcga-prad/tissue-masks/TCGA-HC-8257.tif"
    if not wsi_path.exists() or not mask_path.exists():
        pytest.skip("WSI test data is not available")

    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["slide_id", "wsi_path", "mask_path"])
        writer.writeheader()
        writer.writerow(
            {
                "slide_id": "TCGA-HC-8257",
                "wsi_path": str(wsi_path),
                "mask_path": str(mask_path),
            }
        )

    build_index_script = repo_root / "scripts/build_slide_metadata_index_from_manifest.py"
    build_index = subprocess.run(
        [
            sys.executable,
            str(build_index_script),
            "--manifest",
            str(manifest),
            "--output",
            str(slide_index),
            "--report",
            str(report_csv),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert build_index.returncode == 0, build_index.stderr

    profile_yaml.write_text(
        "\n".join(
            [
                "profile_id: ctx1p0_tgt0p5_fov512um_k4",
                "context_mpp: 1.0",
                "target_mpp: 0.5",
                "context_fov_um: 512.0",
                "target_fov_um: 128.0",
                "targets_per_context: 4",
                "min_tissue_fraction: 0.2",
                "anchor_stride_fraction: 0.5",
                "target_margin_um: 16.0",
                "spacing_tolerance: 0.05",
                "seed: 0",
            ]
        ),
        encoding="utf-8",
    )

    build_anchors_script = repo_root / "scripts/build_valid_context_anchor_catalog.py"
    build_anchors = subprocess.run(
        [
            sys.executable,
            str(build_anchors_script),
            "--slide-index",
            str(slide_index),
            "--profile",
            str(profile_yaml),
            "--output",
            str(anchor_manifest),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert build_anchors.returncode == 0, build_anchors.stderr
    assert anchor_manifest.exists()

    manifest = json.loads(anchor_manifest.read_text(encoding="utf-8"))
    assert int(manifest["total_anchors"]) > 0, "Expected at least one valid anchor"
    assert manifest["profile"]["profile_id"] == "ctx1p0_tgt0p5_fov512um_k4"

    import pyarrow.parquet as pq

    first_shard = Path(manifest["anchor_shards"][0]["path"])
    rows = pq.read_table(str(first_shard)).to_pylist()
    row = dict(rows[0])
    assert row["slide_id"] == "TCGA-HC-8257"
    assert row["profile_id"] == "ctx1p0_tgt0p5_fov512um_k4"
    assert float(row["context_mpp"]) == 1.0
    assert float(row["target_mpp"]) == 0.5
    assert float(row["tissue_fraction"]) >= 0.2
    assert int(row["is_in_bounds"]) == 1

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

if importlib.util.find_spec("pyarrow") is None:
    pytest.skip("pyarrow is required for parquet pipeline tests", allow_module_level=True)


def test_pathology_slide_metadata_index_builder_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    manifest = tmp_path / "manifest.csv"
    output_parquet = tmp_path / "slide_metadata.parquet"
    report_csv = tmp_path / "slide_metadata_build_report.csv"

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

    script = repo_root / "scripts/build_slide_metadata_index_from_manifest.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest),
            "--output",
            str(output_parquet),
            "--report",
            str(report_csv),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output_parquet.exists()
    assert report_csv.exists()

    import pyarrow.parquet as pq

    rows = pq.read_table(str(output_parquet)).to_pylist()
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "ok"
    assert row["slide_id"] == "TCGA-HC-8257"
    assert row["wsi_level0_spacing_mpp"] > 0
    assert row["mask_level0_spacing_mpp"] > 0
    assert row["mask_to_wsi_scale_x"] > 0
    assert row["mask_to_wsi_scale_y"] > 0

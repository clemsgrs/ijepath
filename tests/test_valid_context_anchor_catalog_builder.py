import csv
import subprocess
import sys
from pathlib import Path


def test_pathology_valid_context_anchor_catalog_builder_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    manifest = tmp_path / "manifest.csv"
    slide_index = tmp_path / "slide_metadata_index.jsonl"
    report_csv = tmp_path / "slide_metadata_build_report.csv"
    profile_yaml = tmp_path / "profile.yaml"
    anchors_csv = tmp_path / "anchors.csv"

    wsi_path = repo_root / "data/tcga-prad/wsi/TCGA-HC-8257.tif"
    mask_path = repo_root / "data/tcga-prad/tissue-masks/TCGA-HC-8257.tif"

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
            str(anchors_csv),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert build_anchors.returncode == 0, build_anchors.stderr
    assert anchors_csv.exists()

    with anchors_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows, "Expected at least one valid anchor"
    row = rows[0]
    assert row["slide_id"] == "TCGA-HC-8257"
    assert row["profile_id"] == "ctx1p0_tgt0p5_fov512um_k4"
    assert float(row["context_mpp"]) == 1.0
    assert float(row["target_mpp"]) == 0.5
    assert float(row["tissue_fraction"]) >= 0.2
    assert row["is_in_bounds"] == "1"

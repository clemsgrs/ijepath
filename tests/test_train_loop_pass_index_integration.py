import csv
from pathlib import Path

from ijepath.datasets.cross_resolution_wsi_dataset import CrossResolutionWSIDataset


def _write_anchor_csv(path: Path):
    fieldnames = [
        "anchor_id",
        "slide_id",
        "wsi_path",
        "mask_path",
        "center_x_level0",
        "center_y_level0",
        "wsi_level0_spacing_mpp",
        "target_margin_um",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(3):
            writer.writerow(
                {
                    "anchor_id": f"a{i}",
                    "slide_id": f"slide{i}",
                    "wsi_path": "/tmp/none.tif",
                    "mask_path": "",
                    "center_x_level0": 0,
                    "center_y_level0": 0,
                    "wsi_level0_spacing_mpp": 0.25,
                    "target_margin_um": 8.0,
                }
            )


def test_dataset_len_matches_anchor_count_and_pass_seed_changes(tmp_path: Path):
    csv_path = tmp_path / "anchors.csv"
    _write_anchor_csv(csv_path)

    dataset = CrossResolutionWSIDataset(
        anchor_catalog_csv=str(csv_path),
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

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ijepath.eval.pathorob.allocations import CAMELYON_ALLOCATIONS
from ijepath.eval.pathorob.splits import cramers_v_from_counts, generate_apd_splits


def test_camelyon_paper_allocations_match_target_v_tolerance():
    for v_target, alloc in CAMELYON_ALLOCATIONS.items():
        v = cramers_v_from_counts(alloc)
        assert abs(v - v_target) <= 0.02


def test_generate_apd_splits_is_deterministic_and_leak_free():
    rows = []
    sample_idx = 0
    for label in ["normal", "tumor"]:
        for center in ["RUMC", "UMCU", "CWZ"]:
            for slide in range(8):
                slide_id = f"{center}_{label}_{slide}"
                for _ in range(4):
                    rows.append(
                        {
                            "sample_id": f"s{sample_idx}",
                            "image_path": f"/tmp/fake/{sample_idx}.png",
                            "label": label,
                            "medical_center": center,
                            "slide_id": slide_id,
                        }
                    )
                    sample_idx += 1
    df = pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        out1 = generate_apd_splits(
            df=df,
            output_dir=Path(d1),
            dataset_name="camelyon",
            repetitions=2,
            correlation_levels=[0.0, 0.5, 1.0],
            id_centers=["RUMC", "UMCU"],
            ood_centers=["CWZ"],
            id_test_fraction=0.2,
            seed=123,
            mode="custom",
        )
        out2 = generate_apd_splits(
            df=df,
            output_dir=Path(d2),
            dataset_name="camelyon",
            repetitions=2,
            correlation_levels=[0.0, 0.5, 1.0],
            id_centers=["RUMC", "UMCU"],
            ood_centers=["CWZ"],
            id_test_fraction=0.2,
            seed=123,
            mode="custom",
        )

        assert len(out1) == len(out2)
        for a, b in zip(out1, out2):
            assert a[["sample_id", "partition", "rep", "split_id"]].equals(
                b[["sample_id", "partition", "rep", "split_id"]]
            )
            train_slides = set(a[a["partition"] == "train"]["slide_id"].tolist())
            id_slides = set(a[a["partition"] == "id_test"]["slide_id"].tolist())
            assert len(train_slides.intersection(id_slides)) == 0


def test_generate_apd_splits_camelyon_tiny_paper_mode_falls_back_to_custom_without_levels(caplog):
    rows = []
    sample_idx = 0
    for label in ["normal", "tumor"]:
        for center in ["RUMC", "UMCU", "CWZ", "RST", "LPON"]:
            for slide in range(4):
                slide_id = f"{center}_{label}_{slide}"
                for _ in range(2):
                    rows.append(
                        {
                            "sample_id": f"s{sample_idx}",
                            "image_path": f"/tmp/fake/{sample_idx}.png",
                            "label": label,
                            "medical_center": center,
                            "slide_id": slide_id,
                        }
                    )
                    sample_idx += 1
    df = pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as d:
        with caplog.at_level(logging.WARNING, logger="ijepath"):
            out = generate_apd_splits(
                df=df,
                output_dir=Path(d),
                dataset_name="camelyon_tiny",
                repetitions=2,
                correlation_levels=[],
                id_centers=["RUMC", "UMCU"],
                ood_centers=["CWZ", "RST", "LPON"],
                id_test_fraction=0.2,
                seed=123,
                mode="paper",
            )

    assert len(out) == 0
    assert any("Paper mode is unavailable for dataset=camelyon_tiny" in rec.message for rec in caplog.records)


def test_generate_apd_splits_paper_mode_warns_when_rescaling(caplog):
    rows = []
    sample_idx = 0
    for label in ["normal", "tumor"]:
        for center in ["RUMC", "UMCU", "CWZ", "RST", "LPON"]:
            for slide in range(3):
                slide_id = f"{center}_{label}_{slide}"
                rows.append(
                    {
                        "sample_id": f"s{sample_idx}",
                        "image_path": f"/tmp/fake/{sample_idx}.png",
                        "label": label,
                        "medical_center": center,
                        "slide_id": slide_id,
                    }
                )
                sample_idx += 1
    df = pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as d:
        with caplog.at_level(logging.WARNING, logger="ijepath"):
            out = generate_apd_splits(
                df=df,
                output_dir=Path(d),
                dataset_name="camelyon",
                repetitions=1,
                correlation_levels=[],
                id_centers=["RUMC", "UMCU"],
                ood_centers=["CWZ", "RST", "LPON"],
                id_test_fraction=0.2,
                seed=123,
                mode="paper",
            )

    assert len(out) == 8
    assert any("Rescaling paper allocations" in rec.message for rec in caplog.records)

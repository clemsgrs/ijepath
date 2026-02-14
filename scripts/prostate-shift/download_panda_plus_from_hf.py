#!/usr/bin/env python3
"""Download andreasveit/panda-plus from Hugging Face and write ijepath manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download PANDA-PLUS dataset and build manifest."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/panda-plus"),
        help="Destination folder for images/manifest.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="andreasveit/panda-plus",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="baseline",
        help="Hugging Face split name.",
    )
    parser.add_argument(
        "--image-column",
        type=str,
        default="image",
        help="Column containing image objects.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Column containing class label.",
    )
    parser.add_argument(
        "--slide-id-column",
        type=str,
        default="slide_id",
        help="Column containing slide id.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    output_dir = args.output_dir.resolve()
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {args.repo_id} [{args.split}] ...")
    dataset = load_dataset(args.repo_id, split=args.split)
    print(f"Loaded {len(dataset)} samples.")

    records: list[dict[str, str]] = []
    for idx, sample in enumerate(dataset):
        image = sample[args.image_column]
        filename = f"{idx:07d}.png"
        image_path = images_dir / filename
        image.save(image_path)

        records.append(
            {
                "sample_id": f"panda_plus_{idx:09d}",
                "image_path": str(image_path.resolve()),
                "label": str(sample[args.label_column]),
                "medical_center": "RUMC",
                "slide_id": str(sample[args.slide_id_column]),
            }
        )

    manifest = pd.DataFrame.from_records(records)
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    metadata = {
        "source_repo": str(args.repo_id),
        "split": str(args.split),
        "total_samples": int(len(manifest)),
        "centers": sorted(manifest["medical_center"].unique().tolist()),
        "labels": sorted(manifest["label"].unique().tolist()),
        "samples_per_center": {
            k: int(v) for k, v in manifest["medical_center"].value_counts().sort_index().items()
        },
        "samples_per_label": {
            k: int(v) for k, v in manifest["label"].value_counts().sort_index().items()
        },
        "slides": sorted(manifest["slide_id"].unique().tolist()),
        "samples_per_slide": {
            k: int(v) for k, v in manifest["slide_id"].value_counts().sort_index().items()
        },
        "columns": list(manifest.columns),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote metadata: {metadata_path}")
    print("Manifest columns: sample_id,image_path,label,medical_center,slide_id")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

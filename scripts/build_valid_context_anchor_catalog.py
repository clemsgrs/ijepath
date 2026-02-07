#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.wsi_readers.wholeslidedata_reader_adapter import (
    WholeSlideDataReaderAdapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a profile-specific context-anchor catalog.")
    parser.add_argument("--slide-index", required=True, type=str, help="Slide metadata JSONL path")
    parser.add_argument("--profile", required=True, type=str, help="Sampling profile YAML path")
    parser.add_argument("--output", required=True, type=str, help="Output anchor CSV path")
    parser.add_argument("--backend", type=str, default="openslide", help="wholeslidedata backend")
    parser.add_argument("--qc-overlays-dir", type=str, default=None, help="Optional QC overlay output directory")
    parser.add_argument("--qc-per-slide", type=int, default=32, help="Number of anchors to draw in QC image")
    return parser.parse_args()


def load_slide_index(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def load_profile(path: Path):
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    required_keys = [
        "profile_id",
        "context_mpp",
        "target_mpp",
        "context_fov_um",
        "target_fov_um",
        "targets_per_context",
        "min_tissue_fraction",
        "anchor_stride_fraction",
        "target_margin_um",
        "spacing_tolerance",
    ]
    for key in required_keys:
        if key not in profile:
            raise ValueError(f"Missing required profile key: {key}")
    return profile


def compute_tissue_fraction(mask_patch: np.ndarray) -> float:
    if mask_patch.ndim == 3:
        mask_patch = mask_patch[..., 0]
    tissue = (mask_patch > 0).astype(np.uint8)
    return float(tissue.mean())


def write_qc_overlay(
    qc_path: Path,
    mask_shape: tuple[int, int],
    anchors_mask: list[tuple[int, int]],
    context_size_mask_px: int,
    max_points: int,
):
    h, w = int(mask_shape[1]), int(mask_shape[0])
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, (x, y) in enumerate(anchors_mask[:max_points]):
        color = (60, 180, 75) if idx % 2 == 0 else (255, 140, 0)
        half = context_size_mask_px // 2
        x0, y0 = max(0, x - half), max(0, y - half)
        x1, y1 = min(w - 1, x + half), min(h - 1, y + half)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 1)
        cv2.circle(canvas, (x, y), 1, (255, 255, 255), -1)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(qc_path), canvas)


def main() -> int:
    args = parse_args()
    slide_index_path = Path(args.slide_index).resolve()
    profile_path = Path(args.profile).resolve()
    output_path = Path(args.output).resolve()

    slides = load_slide_index(slide_index_path)
    profile = load_profile(profile_path)

    context_mpp = float(profile["context_mpp"])
    target_mpp = float(profile["target_mpp"])
    context_fov_um = float(profile["context_fov_um"])
    target_fov_um = float(profile["target_fov_um"])
    targets_per_context = int(profile["targets_per_context"])
    min_tissue_fraction = float(profile["min_tissue_fraction"])
    anchor_stride_fraction = float(profile["anchor_stride_fraction"])
    target_margin_um = float(profile["target_margin_um"])
    spacing_tolerance = float(profile["spacing_tolerance"])
    profile_id = str(profile["profile_id"])

    context_size_px = int(round(context_fov_um / context_mpp))
    target_size_context_px = int(round(target_fov_um / context_mpp))
    target_margin_context_px = max(int(round(target_margin_um / context_mpp)), target_size_context_px // 2)
    if context_size_px <= 2 * target_margin_context_px:
        raise ValueError("Invalid profile: context FOV too small for target size + margin constraints")

    anchor_rows = []
    build_report_rows = []

    for slide in slides:
        if slide.get("status") != "ok":
            build_report_rows.append(
                {
                    "slide_id": slide.get("slide_id"),
                    "status": "skipped",
                    "reason": "slide_status_not_ok",
                    "anchors": 0,
                }
            )
            continue
        if not slide.get("mask_path"):
            build_report_rows.append(
                {
                    "slide_id": slide.get("slide_id"),
                    "status": "skipped",
                    "reason": "missing_mask_path",
                    "anchors": 0,
                }
            )
            continue

        adapter = WholeSlideDataReaderAdapter(
            wsi_path=slide["wsi_path"],
            mask_path=slide["mask_path"],
            backend=args.backend,
        )

        mask_spacing0 = float(slide["mask_level0_spacing_mpp"])
        mask_w = int(slide["mask_level0_width"])
        mask_h = int(slide["mask_level0_height"])
        mask_to_wsi_x = float(slide["mask_to_wsi_scale_x"])
        mask_to_wsi_y = float(slide["mask_to_wsi_scale_y"])

        context_size_mask_px = max(1, int(round(context_fov_um / mask_spacing0)))
        half_mask = context_size_mask_px // 2
        stride_mask = max(1, int(round(context_size_mask_px * anchor_stride_fraction)))

        slide_anchor_count = 0
        slide_anchor_centers_mask = []
        for cy_mask in range(half_mask, max(half_mask + 1, mask_h - half_mask), stride_mask):
            for cx_mask in range(half_mask, max(half_mask + 1, mask_w - half_mask), stride_mask):
                x0 = cx_mask - half_mask
                y0 = cy_mask - half_mask
                x1 = x0 + context_size_mask_px
                y1 = y0 + context_size_mask_px
                if x0 < 0 or y0 < 0 or x1 > mask_w or y1 > mask_h:
                    continue

                mask_patch = adapter.get_patch_by_center_level0(
                    center_x_level0=int(cx_mask),
                    center_y_level0=int(cy_mask),
                    width_pixels_at_spacing=context_size_mask_px,
                    height_pixels_at_spacing=context_size_mask_px,
                    spacing_mpp=mask_spacing0,
                    use_mask=True,
                )
                tissue_fraction = compute_tissue_fraction(mask_patch)
                if tissue_fraction < min_tissue_fraction:
                    continue

                center_x_level0 = int(round(cx_mask * mask_to_wsi_x))
                center_y_level0 = int(round(cy_mask * mask_to_wsi_y))
                in_bounds = adapter.level0_center_in_bounds(
                    center_x_level0=center_x_level0,
                    center_y_level0=center_y_level0,
                    size_pixels_at_spacing=context_size_px,
                    spacing_mpp=context_mpp,
                )
                if not in_bounds:
                    continue

                anchor_id = f"{slide['slide_id']}_{slide_anchor_count:07d}"
                anchor_rows.append(
                    {
                        "anchor_id": anchor_id,
                        "slide_id": slide["slide_id"],
                        "profile_id": profile_id,
                        "wsi_path": slide["wsi_path"],
                        "mask_path": slide["mask_path"],
                        "center_x_level0": center_x_level0,
                        "center_y_level0": center_y_level0,
                        "center_x_mask": cx_mask,
                        "center_y_mask": cy_mask,
                        "tissue_fraction": round(tissue_fraction, 6),
                        "is_in_bounds": 1,
                        "wsi_level0_spacing_mpp": slide["wsi_level0_spacing_mpp"],
                        "mask_level0_spacing_mpp": slide["mask_level0_spacing_mpp"],
                        "context_mpp": context_mpp,
                        "target_mpp": target_mpp,
                        "context_fov_um": context_fov_um,
                        "target_fov_um": target_fov_um,
                        "targets_per_context": targets_per_context,
                        "min_tissue_fraction": min_tissue_fraction,
                        "target_margin_um": target_margin_um,
                        "target_margin_context_px": target_margin_context_px,
                        "context_size_px": context_size_px,
                        "target_size_context_px": target_size_context_px,
                        "spacing_tolerance": spacing_tolerance,
                    }
                )
                slide_anchor_centers_mask.append((cx_mask, cy_mask))
                slide_anchor_count += 1

        build_report_rows.append(
            {
                "slide_id": slide["slide_id"],
                "status": "ok" if slide_anchor_count > 0 else "failed",
                "reason": "" if slide_anchor_count > 0 else "no_valid_anchors",
                "anchors": slide_anchor_count,
            }
        )

        if args.qc_overlays_dir:
            qc_path = Path(args.qc_overlays_dir).resolve() / f"{slide['slide_id']}.png"
            write_qc_overlay(
                qc_path=qc_path,
                mask_shape=(mask_w, mask_h),
                anchors_mask=slide_anchor_centers_mask,
                context_size_mask_px=context_size_mask_px,
                max_points=args.qc_per_slide,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "anchor_id",
        "slide_id",
        "profile_id",
        "wsi_path",
        "mask_path",
        "center_x_level0",
        "center_y_level0",
        "center_x_mask",
        "center_y_mask",
        "tissue_fraction",
        "is_in_bounds",
        "wsi_level0_spacing_mpp",
        "mask_level0_spacing_mpp",
        "context_mpp",
        "target_mpp",
        "context_fov_um",
        "target_fov_um",
        "targets_per_context",
        "min_tissue_fraction",
        "target_margin_um",
        "target_margin_context_px",
        "context_size_px",
        "target_size_context_px",
        "spacing_tolerance",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in anchor_rows:
            writer.writerow(row)

    report_path = output_path.with_name("anchor_catalog_build_report.csv")
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["slide_id", "status", "reason", "anchors"])
        writer.writeheader()
        for row in build_report_rows:
            writer.writerow(row)

    print(f"wrote_anchor_catalog={output_path}")
    print(f"wrote_report={report_path}")
    print(f"anchors_total={len(anchor_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import tqdm
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ijepath.datasets.wsi_readers.wholeslidedata_reader_adapter import (
    WholeSlideDataReaderAdapter,
)
from ijepath.utils.parquet import require_pyarrow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a profile-specific context-anchor catalog (Parquet shards + manifest).")
    parser.add_argument("--slide-index", required=True, type=str, help="Slide metadata Parquet path")
    parser.add_argument("--profile", required=True, type=str, help="Sampling profile YAML path")
    parser.add_argument("--output", required=True, type=str, help="Output anchor manifest JSON path")
    parser.add_argument("--backend", type=str, default="asap", help="wholeslidedata backend")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1), help="Parallel slide workers")
    parser.add_argument("--rows-per-shard", type=int, default=100_000, help="Max rows per anchor shard")
    parser.add_argument("--stratum-key", type=str, default="organ", help="Slide metadata column used for stratified sampling")
    parser.add_argument("--qc-overlays-dir", type=str, default=None, help="Optional QC overlay output directory")
    parser.add_argument("--qc-per-slide", type=int, default=32, help="Number of anchors to draw in QC image")
    return parser.parse_args()


def load_slide_index(path: Path) -> list[dict]:
    _, _, ds = require_pyarrow()
    table = ds.dataset(str(path), format="parquet").to_table()
    return [dict(row) for row in table.to_pylist()]


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


def _write_parquet_rows(rows: list[dict], output_path: Path) -> int:
    pa, pq, _ = require_pyarrow()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(output_path))
    return int(table.num_rows)


def _sanitize_slide_id(slide_id: str) -> str:
    return str(slide_id).replace("/", "_").replace(" ", "_")


def _process_slide_worker(
    slide: dict,
    profile: dict,
    backend: str,
    anchors_dir: str,
    rows_per_shard: int,
    stratum_key: str,
    qc_overlays_dir: str | None,
    qc_per_slide: int,
) -> dict:
    slide_id = str(slide.get("slide_id"))
    if slide.get("status") != "ok":
        return {
            "slide_id": slide_id,
            "status": "skipped",
            "reason": "slide_status_not_ok",
            "anchors": 0,
            "parts": [],
            "stratum_counts": {},
        }
    if not slide.get("mask_path"):
        return {
            "slide_id": slide_id,
            "status": "skipped",
            "reason": "missing_mask_path",
            "anchors": 0,
            "parts": [],
            "stratum_counts": {},
        }

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
        return {
            "slide_id": slide_id,
            "status": "failed",
            "reason": "invalid_profile_geometry",
            "anchors": 0,
            "parts": [],
            "stratum_counts": {},
        }

    adapter = WholeSlideDataReaderAdapter(
        wsi_path=str(slide["wsi_path"]),
        mask_path=str(slide["mask_path"]),
        backend=backend,
    )

    mask_spacing0 = float(slide["mask_level0_spacing_mpp"])
    mask_w = int(slide["mask_level0_width"])
    mask_h = int(slide["mask_level0_height"])
    mask_to_wsi_x = float(slide["mask_to_wsi_scale_x"])
    mask_to_wsi_y = float(slide["mask_to_wsi_scale_y"])

    stratum_value = str(slide.get(stratum_key) or "unknown")

    context_size_mask_px = max(1, int(round(context_fov_um / mask_spacing0)))
    half_mask = context_size_mask_px // 2
    stride_mask = max(1, int(round(context_size_mask_px * anchor_stride_fraction)))

    anchors_dir_path = Path(anchors_dir)
    slide_prefix = _sanitize_slide_id(slide_id)

    anchor_count = 0
    part_idx = 0
    rows_buffer: list[dict] = []
    parts: list[dict] = []
    qc_points: list[tuple[int, int]] = []

    def flush_rows_buffer() -> None:
        nonlocal part_idx, rows_buffer
        if not rows_buffer:
            return
        part_path = anchors_dir_path / f"part-{slide_prefix}-{part_idx:06d}.parquet"
        row_count = _write_parquet_rows(rows_buffer, part_path)
        parts.append(
            {
                "path": str(part_path.resolve()),
                "rows": int(row_count),
                "stratum_counts": {stratum_value: int(row_count)},
            }
        )
        part_idx += 1
        rows_buffer = []

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

            anchor_id = f"{slide_id}_{anchor_count:07d}"
            rows_buffer.append(
                {
                    "anchor_id": anchor_id,
                    "slide_id": slide_id,
                    "profile_id": profile_id,
                    "stratum_id": stratum_value,
                    "wsi_path": str(slide["wsi_path"]),
                    "mask_path": str(slide["mask_path"]),
                    "center_x_level0": int(center_x_level0),
                    "center_y_level0": int(center_y_level0),
                    "center_x_mask": int(cx_mask),
                    "center_y_mask": int(cy_mask),
                    "tissue_fraction": float(round(tissue_fraction, 6)),
                    "is_in_bounds": 1,
                    "wsi_level0_spacing_mpp": float(slide["wsi_level0_spacing_mpp"]),
                    "context_mpp": context_mpp,
                    "target_mpp": target_mpp,
                    "context_fov_um": context_fov_um,
                    "target_fov_um": target_fov_um,
                    "targets_per_context": targets_per_context,
                    "target_margin_um": target_margin_um,
                    "target_margin_context_px": target_margin_context_px,
                    "context_size_px": context_size_px,
                    "target_size_context_px": target_size_context_px,
                    "spacing_tolerance": spacing_tolerance,
                }
            )
            if len(qc_points) < max(0, int(qc_per_slide)):
                qc_points.append((cx_mask, cy_mask))

            anchor_count += 1
            if len(rows_buffer) >= int(rows_per_shard):
                flush_rows_buffer()

    flush_rows_buffer()

    if qc_overlays_dir:
        qc_path = Path(qc_overlays_dir).resolve() / f"{slide_prefix}.png"
        write_qc_overlay(
            qc_path=qc_path,
            mask_shape=(mask_w, mask_h),
            anchors_mask=qc_points,
            context_size_mask_px=context_size_mask_px,
            max_points=qc_per_slide,
        )

    return {
        "slide_id": slide_id,
        "status": "ok" if anchor_count > 0 else "failed",
        "reason": "" if anchor_count > 0 else "no_valid_anchors",
        "anchors": int(anchor_count),
        "parts": parts,
        "stratum_counts": {stratum_value: int(anchor_count)} if anchor_count > 0 else {},
    }


def main() -> int:
    args = parse_args()
    slide_index_path = Path(args.slide_index).resolve()
    profile_path = Path(args.profile).resolve()
    output_path = Path(args.output).resolve()

    slides = load_slide_index(slide_index_path)
    profile = load_profile(profile_path)

    anchors_dir = output_path.parent / "anchors"
    anchors_dir.mkdir(parents=True, exist_ok=True)

    build_report_rows: list[dict] = []
    all_parts: list[dict] = []
    all_stratum_counts: dict[str, int] = {}

    workers = max(1, int(args.workers))
    def _accumulate_result(result: dict) -> None:
        build_report_rows.append(
            {
                "slide_id": result["slide_id"],
                "status": result["status"],
                "reason": result["reason"],
                "anchors": int(result["anchors"]),
            }
        )
        all_parts.extend(result.get("parts", []))
        for stratum, count in result.get("stratum_counts", {}).items():
            all_stratum_counts[str(stratum)] = int(all_stratum_counts.get(str(stratum), 0) + int(count))

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _process_slide_worker,
                    slide,
                    profile,
                    args.backend,
                    str(anchors_dir),
                    int(args.rows_per_shard),
                    str(args.stratum_key),
                    args.qc_overlays_dir,
                    int(args.qc_per_slide),
                )
                for slide in slides
            ]

            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing slides", unit="slide"):
                _accumulate_result(future.result())
    except PermissionError:
        for slide in tqdm.tqdm(slides, total=len(slides), desc="Processing slides", unit="slide"):
            _accumulate_result(
                _process_slide_worker(
                    slide=slide,
                    profile=profile,
                    backend=args.backend,
                    anchors_dir=str(anchors_dir),
                    rows_per_shard=int(args.rows_per_shard),
                    stratum_key=str(args.stratum_key),
                    qc_overlays_dir=args.qc_overlays_dir,
                    qc_per_slide=int(args.qc_per_slide),
                )
            )

    total_anchors = int(sum(int(x.get("rows", 0)) for x in all_parts))

    manifest = {
        "schema_version": 1,
        "profile": profile,
        "slide_index_path": str(slide_index_path),
        "sampling_stratum_key": str(args.stratum_key),
        "total_anchors": total_anchors,
        "stratum_counts": all_stratum_counts,
        "anchor_shards": all_parts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_path = output_path.with_name("anchor_catalog_build_report.csv")
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["slide_id", "status", "reason", "anchors"])
        writer.writeheader()
        for row in sorted(build_report_rows, key=lambda x: str(x["slide_id"])):
            writer.writerow(row)

    print(f"wrote_anchor_manifest={output_path}")
    print(f"wrote_report={report_path}")
    print(f"slides total={len(slides):,}")
    print(f"anchors total={total_anchors:,}")
    print(f"shards total={len(all_parts):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

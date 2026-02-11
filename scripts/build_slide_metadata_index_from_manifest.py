#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ijepath.datasets.wsi_readers.wholeslidedata_reader_adapter import WholeSlideDataReaderAdapter
from ijepath.utils.parquet import require_pyarrow


CORE_MANIFEST_COLUMNS = {"slide_id", "wsi_path", "slide_path", "mask_path"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build slide metadata index from a WSI+mask manifest.")
    parser.add_argument("--manifest", required=True, type=str, help="CSV with slide_id,wsi_path,mask_path")
    parser.add_argument("--output", required=True, type=str, help="Output Parquet path")
    parser.add_argument("--report", type=str, default=None, help="Optional process report CSV path")
    parser.add_argument("--backend", type=str, default="asap", help="wholeslidedata backend")
    parser.add_argument("--resume", action="store_true", help="Reuse existing rows from output parquet")
    return parser.parse_args()


def load_existing_rows(output_path: Path) -> dict[str, dict]:
    if not output_path.exists():
        return {}
    _, _, ds = require_pyarrow()
    table = ds.dataset(str(output_path), format="parquet").to_table()
    rows = table.to_pylist()
    return {str(row["slide_id"]): dict(row) for row in rows}


def read_manifest(manifest_path: Path) -> list[dict]:
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest is missing headers: {manifest_path}")
        rows: list[dict] = []
        for row in reader:
            wsi_path = row.get("wsi_path") or row.get("slide_path")
            if not wsi_path:
                raise ValueError("Manifest must include `wsi_path` or `slide_path` column")
            slide_id = row.get("slide_id") or Path(wsi_path).stem
            metadata = {
                str(k): v
                for k, v in row.items()
                if str(k) not in CORE_MANIFEST_COLUMNS and v not in (None, "")
            }
            rows.append(
                {
                    "slide_id": slide_id,
                    "wsi_path": str(Path(wsi_path).resolve()),
                    "mask_path": str(Path(row["mask_path"]).resolve()) if row.get("mask_path") else None,
                    "metadata": metadata,
                }
            )
        return rows


def build_slide_row(slide_id: str, wsi_path: str, mask_path: str | None, backend: str, metadata: dict):
    result = {
        "slide_id": slide_id,
        "wsi_path": wsi_path,
        "mask_path": mask_path,
        "status": "ok",
        "error": None,
        "metadata": dict(metadata),
    }

    try:
        adapter = WholeSlideDataReaderAdapter(
            wsi_path=wsi_path,
            mask_path=mask_path,
            backend=backend,
        )
        wsi_shapes = adapter.wsi_shapes
        wsi_spacings = adapter.wsi_spacings

        result.update(
            {
                "wsi_level0_width": int(wsi_shapes[0][0]),
                "wsi_level0_height": int(wsi_shapes[0][1]),
                "wsi_level0_spacing_mpp": float(wsi_spacings[0]),
                "wsi_shapes": [[int(w), int(h)] for (w, h) in wsi_shapes],
                "wsi_spacings_mpp": [float(s) for s in wsi_spacings],
            }
        )

        if mask_path is not None:
            mask_shapes = adapter.mask_shapes
            mask_spacings = adapter.mask_spacings
            if mask_shapes is None or mask_spacings is None:
                raise RuntimeError("Mask path provided but mask metadata is unavailable")

            result.update(
                {
                    "mask_level0_width": int(mask_shapes[0][0]),
                    "mask_level0_height": int(mask_shapes[0][1]),
                    "mask_level0_spacing_mpp": float(mask_spacings[0]),
                    "mask_shapes": [[int(w), int(h)] for (w, h) in mask_shapes],
                    "mask_spacings_mpp": [float(s) for s in mask_spacings],
                    "mask_to_wsi_scale_x": float(wsi_shapes[0][0] / mask_shapes[0][0]),
                    "mask_to_wsi_scale_y": float(wsi_shapes[0][1] / mask_shapes[0][1]),
                }
            )
        else:
            result.update(
                {
                    "mask_level0_width": None,
                    "mask_level0_height": None,
                    "mask_level0_spacing_mpp": None,
                    "mask_shapes": None,
                    "mask_spacings_mpp": None,
                    "mask_to_wsi_scale_x": None,
                    "mask_to_wsi_scale_y": None,
                }
            )

    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    # Flatten manifest metadata for downstream stratification (organ/site/...)
    for key, value in result.pop("metadata", {}).items():
        result[str(key)] = value

    return result


def write_parquet(path: Path, rows: list[dict]) -> None:
    pa, pq, _ = require_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(path))


def write_report(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["slide_id", "wsi_path", "mask_path", "status", "error"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "slide_id": row.get("slide_id"),
                    "wsi_path": row.get("wsi_path"),
                    "mask_path": row.get("mask_path"),
                    "status": row.get("status"),
                    "error": row.get("error"),
                }
            )


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve() if args.report else output_path.with_name("slide_metadata_build_report.csv")

    manifest_rows = read_manifest(manifest_path)
    existing = load_existing_rows(output_path) if args.resume else {}

    final_rows = []
    with tqdm.tqdm(
        manifest_rows,
        desc="Processing slides",
        unit="slide",
        leave=True,
    ) as t:
        for row in t:
            slide_id = row["slide_id"]
            if slide_id in existing:
                final_rows.append(existing[slide_id])
                continue
            final_rows.append(
                build_slide_row(
                    slide_id=slide_id,
                    wsi_path=row["wsi_path"],
                    mask_path=row["mask_path"],
                    backend=args.backend,
                    metadata=row.get("metadata", {}),
                )
            )

    write_parquet(output_path, final_rows)
    write_report(report_path, final_rows)

    ok_count = sum(1 for row in final_rows if row.get("status") == "ok")
    fail_count = len(final_rows) - ok_count
    print(f"wrote_index={output_path}")
    print(f"wrote_report={report_path}")
    print(f"slides_ok={ok_count} slides_failed={fail_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

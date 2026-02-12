#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark parquet indexing/loading/training throughput.")
    parser.add_argument("--mode", choices=["index", "load", "train"], required=True)
    parser.add_argument("--manifest", type=str, default=None, help="Slide manifest CSV")
    parser.add_argument("--profile", type=str, default="configs/profiles/ctx1p0_tgt0p5_fov512um_k4.yaml")
    parser.add_argument("--slide-metadata", type=str, default="/tmp/slide_metadata.parquet")
    parser.add_argument("--anchor-manifest", type=str, default="/tmp/anchor_catalog_manifest.json")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--load-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _ru_maxrss_mb() -> float:
    # macOS reports bytes, Linux reports KB. Heuristic keeps output usable cross-platform.
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if rss > 10_000_000:
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def run_cmd(cmd: list[str], cwd: Path) -> tuple[float, float]:
    start = time.perf_counter()
    completed = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return elapsed, _ru_maxrss_mb()


def benchmark_index(args: argparse.Namespace, repo_root: Path) -> None:
    if not args.manifest:
        raise ValueError("--manifest is required for --mode index")

    slide_cmd = [
        args.python_exe,
        str(repo_root / "scripts/build_slide_metadata_index_from_manifest.py"),
        "--manifest",
        str(args.manifest),
        "--output",
        str(args.slide_metadata),
    ]
    t_slide, rss_slide = run_cmd(slide_cmd, cwd=repo_root)

    anchor_cmd = [
        args.python_exe,
        str(repo_root / "scripts/build_valid_context_anchor_catalog.py"),
        "--slide-index",
        str(args.slide_metadata),
        "--profile",
        str(args.profile),
        "--output",
        str(args.anchor_manifest),
    ]
    t_anchor, rss_anchor = run_cmd(anchor_cmd, cwd=repo_root)

    manifest = json.loads(Path(args.anchor_manifest).read_text(encoding="utf-8"))
    total_anchors = int(manifest.get("total_anchors", 0))
    slides = int(sum(1 for _ in open(args.manifest, "r", encoding="utf-8")) - 1)

    print(f"index.slides={slides}")
    print(f"index.anchors={total_anchors}")
    print(f"index.slide_stage_s={t_slide:.3f}")
    print(f"index.anchor_stage_s={t_anchor:.3f}")
    print(f"index.total_s={t_slide + t_anchor:.3f}")
    print(f"index.slides_per_s={slides / max(1e-9, t_slide):.3f}")
    print(f"index.anchors_per_s={total_anchors / max(1e-9, t_anchor):.3f}")
    print(f"index.peak_rss_mb={max(rss_slide, rss_anchor):.2f}")


def benchmark_load(args: argparse.Namespace, repo_root: Path) -> None:
    from ijepath.datasets.cross_resolution_loader_factory import make_cross_resolution_loader

    dataset, loader, _ = make_cross_resolution_loader(
        batch_size=int(args.batch_size),
        pin_mem=False,
        num_workers=int(args.num_workers),
        world_size=1,
        rank=0,
        drop_last=True,
        anchor_catalog_manifest=str(args.anchor_manifest),
        patch_size=16,
        context_mpp=1.0,
        target_mpp=0.5,
        context_fov_um=512.0,
        target_fov_um=128.0,
        targets_per_context=4,
        seed=0,
        spacing_tolerance=0.05,
        min_target_tissue_fraction=0.25,
        insufficient_target_policy="skip_anchor",
        min_target_tissue_fraction_floor=None,
        min_target_tissue_fraction_step=0.05,
        min_keep=16,
        num_enc_masks=1,
        backend="asap",
        align_targets_to_patch_grid=False,
    )

    start = time.perf_counter()
    examples = 0
    it = iter(loader)
    for _ in range(int(args.load_steps)):
        batch_data, _masks_enc, _masks_pred = next(it)
        examples += int(batch_data["context_images"].shape[0])
    elapsed = time.perf_counter() - start

    print(f"load.steps={int(args.load_steps)}")
    print(f"load.examples={examples}")
    print(f"load.elapsed_s={elapsed:.3f}")
    print(f"load.examples_per_s={examples / max(1e-9, elapsed):.3f}")
    print(f"load.dataset_len={len(dataset)}")
    print(f"load.peak_rss_mb={_ru_maxrss_mb():.2f}")


def benchmark_train(args: argparse.Namespace, repo_root: Path) -> None:
    cmd = [
        args.python_exe,
        str(repo_root / "main.py"),
        "--profile-config",
        str(repo_root / "configs/profiles/ctx1p0_tgt0p5_fov512um_k4.yaml"),
        "--run-config",
        str(repo_root / "configs/runs/smoke.yaml"),
        f"data.slide_manifest_csv={args.manifest or ''}",
        f"data.slide_metadata_parquet={args.slide_metadata}",
        f"data.anchor_catalog_manifest={args.anchor_manifest}",
        f"data.batch_size_per_gpu={args.batch_size}",
        f"data.num_workers={args.num_workers}",
        "optimization.total_images_budget=256",
        "output.root=/tmp/ijepath-benchmark-train",
        "logging.write_tag=benchmark",
    ]
    elapsed, rss = run_cmd(cmd, cwd=repo_root)
    print(f"train.elapsed_s={elapsed:.3f}")
    print(f"train.peak_rss_mb={rss:.2f}")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if args.mode == "index":
        benchmark_index(args, repo_root)
    elif args.mode == "load":
        benchmark_load(args, repo_root)
    elif args.mode == "train":
        benchmark_train(args, repo_root)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate CAMELYON PathoROB APD splits using the ijepath split builder.

Paper-faithful mode uses the hard-coded allocation tables implemented in
`ijepath.eval.pathorob.allocations`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ijepath.eval.pathorob.datasets import ensure_required_columns
from ijepath.eval.pathorob.splits import generate_apd_splits

CAMELYON_ID_CENTERS = ["RUMC", "UMCU"]
CAMELYON_OOD_CENTERS = ["CWZ", "RST", "LPON"]


def _parse_levels(text: str) -> list[float]:
    if not text.strip():
        return []
    out = []
    for chunk in text.split(","):
        value = float(chunk.strip())
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Correlation level must be in [0, 1], got {value}")
        out.append(value)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate CAMELYON APD splits for PathoROB")
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("data/pathorob/camelyon/manifest.csv"),
        help="Path to CAMELYON manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pathorob/camelyon/splits"),
        help="Directory where generated split CSVs are written.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of random repetitions.",
    )
    parser.add_argument(
        "--id-test-fraction",
        type=float,
        default=0.2,
        help="Fraction of ID slides held out for ID test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic split generation.",
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "custom"],
        default="paper",
        help="Split generation mode.",
    )
    parser.add_argument(
        "--correlation-levels",
        type=str,
        default="",
        help="Comma-separated levels for custom mode (e.g. '0.0,0.5,1.0').",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.repetitions <= 0:
        raise ValueError("--repetitions must be > 0")
    if not (0.0 < float(args.id_test_fraction) < 1.0):
        raise ValueError("--id-test-fraction must be in (0, 1)")

    if not args.manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {args.manifest_csv}")

    print(f"Loading CAMELYON manifest: {args.manifest_csv}")
    df = pd.read_csv(args.manifest_csv)
    ensure_required_columns(df, source=str(args.manifest_csv))
    if "sample_id" not in df.columns:
        df = df.copy()
        df["sample_id"] = [f"camelyon_{i:09d}" for i in range(len(df))]

    levels = _parse_levels(args.correlation_levels)
    if args.mode == "custom" and not levels:
        raise ValueError("--correlation-levels is required when --mode=custom")

    print("Generating splits with configuration:")
    print(f"  mode={args.mode}")
    print(f"  repetitions={args.repetitions}")
    print(f"  id_test_fraction={args.id_test_fraction}")
    print(f"  seed={args.seed}")
    print(f"  output_dir={args.output_dir}")

    splits = generate_apd_splits(
        df=df,
        output_dir=args.output_dir,
        dataset_name="camelyon",
        repetitions=int(args.repetitions),
        correlation_levels=levels,
        id_centers=CAMELYON_ID_CENTERS,
        ood_centers=CAMELYON_OOD_CENTERS,
        id_test_fraction=float(args.id_test_fraction),
        seed=int(args.seed),
        mode=args.mode,
    )

    print(f"Generated {len(splits)} split tables.")
    if splits:
        first = splits[0]
        print("Example split summary (first split):")
        for partition in ["train", "id_test", "ood_test"]:
            count = int((first["partition"] == partition).sum())
            print(f"  {partition}: {count}")
        print(
            "  cramers_v_realized="
            f"{float(first['cramers_v_realized'].iloc[0]):.4f}"
        )
    print(f"Saved split CSV files under: {args.output_dir / 'camelyon'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

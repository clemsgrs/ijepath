import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

if importlib.util.find_spec("pyarrow") is None:
    pytest.skip("pyarrow is required for parquet pipeline tests", allow_module_level=True)


def _run(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )


@pytest.mark.integration
def test_end_to_end_fixture_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    python_exe = sys.executable
    assert Path(python_exe).exists(), f"python executable not found: {python_exe}"

    fixture_wsi = repo_root / "data/test-fixtures/test-wsi.tif"
    fixture_mask = repo_root / "data/test-fixtures/test-mask.tif"
    assert fixture_wsi.exists(), f"missing fixture: {fixture_wsi}"
    assert fixture_mask.exists(), f"missing fixture: {fixture_mask}"

    work_dir = tmp_path / "pipeline"
    manifests_dir = work_dir / "manifests"
    indexes_dir = work_dir / "indexes"
    outputs_dir = work_dir / "outputs"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    indexes_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv = manifests_dir / "test_slide_with_mask.csv"
    manifest_csv.write_text(
        "slide_id,wsi_path,mask_path\n"
        f"test-slide,{fixture_wsi},{fixture_mask}\n",
        encoding="utf-8",
    )

    slide_index_jsonl = indexes_dir / "slide_metadata.parquet"
    slide_report_csv = indexes_dir / "slide_metadata_build_report.csv"
    anchor_catalog_manifest = indexes_dir / "anchor_catalog_manifest.json"
    anchor_report_csv = indexes_dir / "anchor_catalog_build_report.csv"

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    env["CUDA_VISIBLE_DEVICES"] = ""

    _run(
        [
            python_exe,
            str(repo_root / "scripts/build_slide_metadata_index_from_manifest.py"),
            "--manifest",
            str(manifest_csv),
            "--output",
            str(slide_index_jsonl),
            "--report",
            str(slide_report_csv),
        ],
        env=env,
    )
    assert slide_index_jsonl.exists(), "slide index was not created"
    assert slide_report_csv.exists(), "slide index report was not created"

    with slide_report_csv.open("r", newline="", encoding="utf-8") as f:
        report_rows = list(csv.DictReader(f))
    assert len(report_rows) == 1
    assert report_rows[0]["status"] == "ok"

    _run(
        [
            python_exe,
            str(repo_root / "scripts/build_valid_context_anchor_catalog.py"),
            "--slide-index",
            str(slide_index_jsonl),
            "--profile",
            str(repo_root / "configs/profiles/ctx1p0_tgt0p5_fov512um_k4.yaml"),
            "--output",
            str(anchor_catalog_manifest),
        ],
        env=env,
    )
    assert anchor_catalog_manifest.exists(), "anchor catalog was not created"
    assert anchor_report_csv.exists(), "anchor report was not created"

    manifest = json.loads(anchor_catalog_manifest.read_text(encoding="utf-8"))
    assert int(manifest["total_anchors"]) > 0
    assert str(manifest["profile"]["profile_id"]) == "ctx1p0_tgt0p5_fov512um_k4"

    train_out_dir = outputs_dir / "test-fixture-smoke"
    write_tag = "test-fixture-smoke"
    _run(
        [
            python_exe,
            str(repo_root / "main.py"),
            "--profile-config",
            str(repo_root / "configs/profiles/ctx1p0_tgt0p5_fov512um_k4.yaml"),
            "--run-config",
            str(repo_root / "configs/runs/smoke.yaml"),
            f"data.slide_manifest_csv={manifest_csv}",
            f"data.slide_metadata_parquet={slide_index_jsonl}",
            f"data.anchor_catalog_manifest={anchor_catalog_manifest}",
            "data.num_workers=0",
            "data.batch_size_per_gpu=2",
            f"logging.folder={train_out_dir}",
            f"logging.write_tag={write_tag}",
        ],
        env=env,
    )

    params_path = train_out_dir / "params-ijepa.yaml"
    log_path = train_out_dir / f"{write_tag}_r0.csv"
    ckpt_path = train_out_dir / f"{write_tag}-latest.pth.tar"
    assert params_path.exists(), "training params file missing"
    assert log_path.exists(), "training CSV log missing"
    assert ckpt_path.exists(), "latest checkpoint missing"

    with log_path.open("r", newline="", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    assert len(train_rows) > 0
    expected_columns = {
        "images_seen",
        "pass_index",
        "iteration",
        "loss",
        "context_keep_tokens",
        "target_predict_tokens",
        "iteration_time_ms",
        "learning_rate",
        "weight_decay",
    }
    assert expected_columns.issubset(set(train_rows[0].keys()))
    losses = [float(r["loss"]) for r in train_rows]
    assert all(loss == loss and loss != float("inf") and loss != float("-inf") for loss in losses)

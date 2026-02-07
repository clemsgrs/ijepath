import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


def _load_preview_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts/preview_context_targets.py"
    spec = importlib.util.spec_from_file_location("preview_context_targets", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_ijepath_animation_frames_and_write_gif(tmp_path):
    mod = _load_preview_module()

    thumbnail = np.full((540, 900, 3), 212, dtype=np.uint8)
    thumbnail[120:420, 240:700, :] = np.array([196, 180, 168], dtype=np.uint8)
    anchor_box = np.array([420.0, 220.0, 540.0, 340.0], dtype=np.float32)

    context = np.full((224, 224, 3), 184, dtype=np.uint8)
    boxes = np.array(
        [
            [36.0, 36.0, 92.0, 92.0],
            [132.0, 40.0, 190.0, 98.0],
            [44.0, 130.0, 102.0, 188.0],
            [128.0, 128.0, 190.0, 190.0],
        ],
        dtype=np.float32,
    )
    targets = [
        np.full((224, 224, 3), np.array([150 + i * 20, 120 + i * 10, 100], dtype=np.uint8), dtype=np.uint8)
        for i in range(4)
    ]

    sample_metadata = {
        "slide_id": "S1",
        "anchor_id": "A1",
        "output_context_mpp": 1.0,
        "output_target_mpp": 0.5,
    }

    frames, durations = mod.build_ijepath_animation_frames(
        thumbnail_rgb=thumbnail,
        anchor_box_thumbnail_xyxy=anchor_box,
        context_rgb=context,
        boxes_xyxy=boxes,
        target_images_rgb=targets,
        sample_metadata=sample_metadata,
        context_display_size=400,
        target_tile_size=140,
    )

    assert len(frames) == 21
    assert len(durations) == 21

    h, w = frames[0].shape[:2]
    assert h > 0 and w > 0
    for frame in frames:
        assert frame.dtype == np.uint8
        assert frame.shape == (h, w, 3)

    out_gif = tmp_path / "preview.gif"
    mod.write_gif(frames_rgb=frames, durations_ms=durations, output_path=out_gif)
    assert out_gif.exists()
    assert out_gif.stat().st_size > 0

    with Image.open(out_gif) as img:
        assert getattr(img, "n_frames", 1) == len(frames)
        assert img.size == (w, h)


def test_anchor_display_name_formatting():
    mod = _load_preview_module()
    assert mod._format_anchor_display_name("TCGA-HC-8257_0000000") == "A001"
    assert mod._format_anchor_display_name("TCGA-HC-8257_0000012") == "A013"
    assert mod._format_anchor_display_name("AnchorX") == "AnchorX"


def test_build_output_stem_uses_clean_anchor_display_name():
    mod = _load_preview_module()
    stem = mod.build_output_stem(
        sample_index=0,
        slide_id="TCGA-HC-8257",
        anchor_id="TCGA-HC-8257_0000000",
    )
    assert stem == "s000_TCGA-HC-8257_A001"
    assert stem.count("TCGA-HC-8257") == 1


def test_write_final_png_scales_output(tmp_path):
    mod = _load_preview_module()
    frame = np.full((120, 200, 3), 180, dtype=np.uint8)
    out_png = tmp_path / "final.png"

    mod.write_final_png(frame_rgb=frame, output_path=out_png, scale=2.0)
    assert out_png.exists()

    with Image.open(out_png) as img:
        assert img.size == (400, 240)


def test_make_zoomed_four_panel_figure_shape():
    mod = _load_preview_module()
    images = [
        np.full((180, 180, 3), 90, dtype=np.uint8),
        np.full((180, 180, 3), 120, dtype=np.uint8),
        np.full((180, 180, 3), 150, dtype=np.uint8),
        np.full((180, 180, 3), 180, dtype=np.uint8),
    ]
    titles = [
        "2) Context (1.00 mpp)",
        "3) Sampling valid targets",
        "4) Targets (0.50 mpp)",
        "5) Predictor input (per target)",
    ]
    figure = mod.make_zoomed_four_panel_figure(
        step_images_rgb=images,
        slot_titles=titles,
        sample_metadata={"slide_id": "S1", "anchor_id": "A1"},
    )
    assert figure.dtype == np.uint8
    assert figure.shape[0] >= 1100
    assert figure.shape[1] >= 1600

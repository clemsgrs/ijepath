#!/usr/bin/env python3
import argparse
import csv
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ijepath.datasets.cross_resolution_wsi_dataset import CrossResolutionWSIDataset
from ijepath.datasets.wsi_readers.wholeslidedata_reader_adapter import (
    WholeSlideDataReaderAdapter,
    spacing_pixels_to_level0_pixels,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TARGET_COLORS = [
    (233, 81, 86),
    (56, 161, 105),
    (59, 130, 246),
    (245, 158, 11),
    (217, 70, 239),
    (20, 184, 166),
    (239, 68, 68),
    (14, 165, 233),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview context and target crops for pathology JEPA samples.")
    parser.add_argument("--anchor-catalog", required=True, type=str, help="Path to anchor catalog CSV")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory to write previews")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples to visualize")
    parser.add_argument("--context-mpp", type=float, default=None)
    parser.add_argument("--target-mpp", type=float, default=None)
    parser.add_argument("--context-fov-um", type=float, default=None)
    parser.add_argument("--target-fov-um", type=float, default=None)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--targets-per-context", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spacing-tolerance", type=float, default=0.05)
    parser.add_argument("--wsi-backend", type=str, default="asap")
    parser.add_argument("--context-display-size", type=int, default=560)
    parser.add_argument("--target-tile-size", type=int, default=180)
    parser.add_argument("--thumbnail-max-side", type=int, default=1024)
    parser.add_argument(
        "--final-png-scale",
        type=float,
        default=2.0,
        help="Scale factor for final all-steps PNG (set <=0 to disable).",
    )
    parser.add_argument(
        "--num-zoomed-previews",
        type=int,
        default=2,
        help="Number of first samples for which to emit zoomed 4-panel static previews.",
    )
    parser.add_argument(
        "--zoomed-preview-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to zoomed 4-panel previews.",
    )
    return parser.parse_args()


def read_profile_from_anchor_catalog(anchor_catalog: Path) -> dict:
    with anchor_catalog.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first = next(reader)
    return {
        "context_mpp": float(first["context_mpp"]),
        "target_mpp": float(first["target_mpp"]),
        "context_fov_um": float(first["context_fov_um"]),
        "target_fov_um": float(first["target_fov_um"]),
        "targets_per_context": int(first["targets_per_context"]),
    }


def read_anchor_rows(anchor_catalog: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with anchor_catalog.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[str(row["anchor_id"])] = dict(row)
    return rows


def tensor_to_rgb_uint8(tensor_image: np.ndarray) -> np.ndarray:
    array = tensor_image.detach().cpu().numpy().transpose(1, 2, 0)
    array = (array * IMAGENET_STD[None, None, :]) + IMAGENET_MEAN[None, None, :]
    array = np.clip(array, 0.0, 1.0)
    return (array * 255.0).astype(np.uint8)


def array_to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[2] > 3:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.6,
    thickness: int = 1,
):
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_target_boxes(
    image_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    fill_alpha: float = 0.2,
) -> np.ndarray:
    out = image_rgb.copy()
    overlay = out.copy()
    for idx, box in enumerate(boxes_xyxy):
        color = TARGET_COLORS[idx % len(TARGET_COLORS)]
        x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
    out = cv2.addWeighted(overlay, fill_alpha, out, 1.0 - fill_alpha, 0)

    for idx, box in enumerate(boxes_xyxy):
        color = TARGET_COLORS[idx % len(TARGET_COLORS)]
        x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 3)
        label_bg_x1 = min(out.shape[1] - 1, x0 + 36)
        label_bg_y1 = min(out.shape[0] - 1, y0 + 22)
        cv2.rectangle(out, (x0, y0), (label_bg_x1, label_bg_y1), color, -1)
        put_text(out, f"T{idx + 1}", (x0 + 5, y0 + 16), (255, 255, 255), scale=0.52, thickness=1)
    return out


def scale_boxes(boxes_xyxy: np.ndarray, scale: float) -> np.ndarray:
    return boxes_xyxy * float(scale)


def safe_clip_box(box_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in box_xyxy.tolist()]
    x0 = min(max(x0, 0.0), float(width - 1))
    y0 = min(max(y0, 0.0), float(height - 1))
    x1 = min(max(x1, x0 + 1.0), float(width))
    y1 = min(max(y1, y0 + 1.0), float(height))
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _draw_anchor_on_thumbnail(
    thumbnail_rgb: np.ndarray,
    anchor_box_xyxy: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    out = thumbnail_rgb.copy()
    box = safe_clip_box(anchor_box_xyxy, width=out.shape[1], height=out.shape[0])
    x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
    cv2.rectangle(out, (x0, y0), (x1, y1), color, 1)
    return out


def _compute_anchor_box_in_thumbnail(
    anchor_row: dict,
    thumbnail_width: int,
    thumbnail_height: int,
    level0_width: int,
    level0_height: int,
    context_mpp: float | None = None,
    context_fov_um: float | None = None,
) -> np.ndarray:
    center_x_level0 = int(float(anchor_row["center_x_level0"]))
    center_y_level0 = int(float(anchor_row["center_y_level0"]))
    eff_context_mpp = float(anchor_row["context_mpp"]) if context_mpp is None else float(context_mpp)
    eff_context_fov_um = float(anchor_row["context_fov_um"]) if context_fov_um is None else float(context_fov_um)
    context_size_px = max(1, int(round(eff_context_fov_um / eff_context_mpp)))
    wsi_level0_spacing_mpp = float(anchor_row["wsi_level0_spacing_mpp"])

    context_size_level0 = spacing_pixels_to_level0_pixels(
        size_pixels_at_spacing=context_size_px,
        spacing=eff_context_mpp,
        spacing_at_level0=wsi_level0_spacing_mpp,
    )
    x0_l0 = center_x_level0 - context_size_level0 // 2
    y0_l0 = center_y_level0 - context_size_level0 // 2
    x1_l0 = x0_l0 + context_size_level0
    y1_l0 = y0_l0 + context_size_level0

    sx = float(thumbnail_width) / float(level0_width)
    sy = float(thumbnail_height) / float(level0_height)
    box = np.array(
        [x0_l0 * sx, y0_l0 * sy, x1_l0 * sx, y1_l0 * sy],
        dtype=np.float32,
    )
    return safe_clip_box(box, width=thumbnail_width, height=thumbnail_height)


def _thumbnail_from_wsi_row(
    anchor_row: dict,
    backend: str,
    max_side: int,
    thumbnail_cache: dict[str, dict],
    context_mpp: float | None = None,
    context_fov_um: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    slide_id = str(anchor_row["slide_id"])
    cached = thumbnail_cache.get(slide_id)
    if cached is None:
        reader = WholeSlideDataReaderAdapter(
            wsi_path=anchor_row["wsi_path"],
            mask_path=None,
            backend=backend,
        )
        level = len(reader.wsi_spacings) - 1
        spacing = float(reader.wsi_spacings[level])
        level_shape = reader.wsi_shapes[level]
        width = int(level_shape[0])
        height = int(level_shape[1])
        patch = reader.wsi.get_patch(0, 0, width, height, spacing=spacing, center=False)
        thumbnail = array_to_rgb_uint8(np.asarray(patch))
        geometry = reader.geometry

        max_side = max(64, int(max_side))
        h, w = thumbnail.shape[:2]
        scale = min(1.0, float(max_side) / float(max(h, w)))
        if scale < 1.0:
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            thumbnail = cv2.resize(thumbnail, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cached = {
            "thumbnail": thumbnail,
            "level0_width": int(geometry.level0_width),
            "level0_height": int(geometry.level0_height),
        }
        thumbnail_cache[slide_id] = cached

    anchor_box = _compute_anchor_box_in_thumbnail(
        anchor_row=anchor_row,
        thumbnail_width=cached["thumbnail"].shape[1],
        thumbnail_height=cached["thumbnail"].shape[0],
        level0_width=cached["level0_width"],
        level0_height=cached["level0_height"],
        context_mpp=context_mpp,
        context_fov_um=context_fov_um,
    )
    return cached["thumbnail"], anchor_box


def _make_predictor_single_query_view(
    context_rgb: np.ndarray,
    target_box_xyxy: np.ndarray,
) -> np.ndarray:
    out = context_rgb.copy()
    x0, y0, x1, y1 = [int(round(v)) for v in target_box_xyxy.tolist()]
    x0 = max(0, min(out.shape[1] - 1, x0))
    y0 = max(0, min(out.shape[0] - 1, y0))
    x1 = max(x0 + 1, min(out.shape[1], x1))
    y1 = max(y0 + 1, min(out.shape[0], y1))

    fill_x0 = min(x1, x0 + 3)
    fill_y0 = min(y1, y0 + 3)
    fill_x1 = max(fill_x0 + 1, x1 - 2)
    fill_y1 = max(fill_y0 + 1, y1 - 2)
    # Opaque black fill for explicit masked-target cue.
    out[fill_y0:fill_y1, fill_x0:fill_x1] = 0
    return out


def _make_predictor_query_grid(
    context_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    tile_size: int,
) -> np.ndarray:
    n = len(boxes_xyxy)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    gap = 12

    grid_h = rows * tile_size + (rows - 1) * gap
    grid_w = cols * tile_size + (cols - 1) * gap
    grid = np.full((grid_h, grid_w, 3), 246, dtype=np.uint8)

    for idx, box in enumerate(boxes_xyxy):
        color = TARGET_COLORS[idx % len(TARGET_COLORS)]
        view = _make_predictor_single_query_view(context_rgb=context_rgb, target_box_xyxy=box)
        tile = cv2.resize(view, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        r = idx // cols
        c = idx % cols
        y0 = r * (tile_size + gap)
        x0 = c * (tile_size + gap)
        grid[y0 : y0 + tile_size, x0 : x0 + tile_size] = tile
        cv2.rectangle(grid, (x0, y0), (x0 + tile_size, y0 + tile_size), color, 4)
        cv2.rectangle(grid, (x0, y0), (x0 + 54, y0 + 22), color, -1)
        put_text(grid, f"T{idx + 1}", (x0 + 5, y0 + 16), (255, 255, 255), scale=0.45, thickness=1)

    return grid


def _format_anchor_display_name(anchor_id: str) -> str:
    match = re.search(r"_(\d+)$", str(anchor_id))
    if match:
        return f"A{int(match.group(1)) + 1:03d}"
    return str(anchor_id)


def _sanitize_filename_token(token: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(token).strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "x"


def build_output_stem(sample_index: int, slide_id: str, anchor_id: str) -> str:
    slide_token = _sanitize_filename_token(slide_id)
    anchor_token = _sanitize_filename_token(_format_anchor_display_name(anchor_id))
    return f"s{int(sample_index):03d}_{slide_token}_{anchor_token}"


def _fit_image_with_transform(
    image_rgb: np.ndarray,
    box_w: int,
    box_h: int,
    bg_color: int = 248,
) -> tuple[np.ndarray, int, int, float]:
    canvas = np.full((box_h, box_w, 3), bg_color, dtype=np.uint8)
    h, w = image_rgb.shape[:2]
    if h <= 0 or w <= 0:
        return canvas, 0, 0, 1.0
    scale = min(float(box_w) / float(w), float(box_h) / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    ox = (box_w - new_w) // 2
    oy = (box_h - new_h) // 2
    canvas[oy : oy + new_h, ox : ox + new_w] = resized
    return canvas, ox, oy, scale


def _compose_flow_frame(
    wsi_with_anchor_rgb: np.ndarray,
    step_images_rgb: list[np.ndarray],
    reveal_alphas: list[float],
    slot_titles: list[str],
    sample_metadata: dict,
) -> np.ndarray:
    frame_w = 1536
    frame_h = 900

    gradient = np.linspace(246, 236, frame_h, dtype=np.uint8)[:, None]
    frame = np.stack(
        [
            np.repeat(gradient, frame_w, axis=1),
            np.repeat(gradient, frame_w, axis=1),
            np.repeat(gradient, frame_w, axis=1),
        ],
        axis=2,
    )

    margin = 24
    left_x = margin
    left_y = 92
    left_w = 700
    left_h = 780

    slot_w = 372
    slot_h = 376
    gap_x = 16
    gap_y = 28
    right_x0 = 752
    right_y0 = 92

    slot_positions = [
        (right_x0, right_y0),  # Context
        (right_x0 + slot_w + gap_x, right_y0),  # Target boxes
        (right_x0, right_y0 + slot_h + gap_y),  # Target crops
        (right_x0 + slot_w + gap_x, right_y0 + slot_h + gap_y),  # Predictor
    ]

    left_placeholder = np.full((left_h, left_w, 3), 246, dtype=np.uint8)
    cv2.rectangle(left_placeholder, (0, 0), (left_w - 1, left_h - 1), (214, 214, 214), 1)
    put_text(left_placeholder, "1) WSI + anchor", (12, 24), (58, 58, 58), scale=0.50, thickness=1)
    left_content, _, _, _ = _fit_image_with_transform(
        image_rgb=wsi_with_anchor_rgb,
        box_w=left_w - 8,
        box_h=left_h - 42,
    )
    left_placeholder[34 : 34 + left_content.shape[0], 4 : 4 + left_content.shape[1]] = left_content
    frame[left_y : left_y + left_h, left_x : left_x + left_w] = left_placeholder

    for idx, (sx, sy) in enumerate(slot_positions):
        placeholder = np.full((slot_h, slot_w, 3), 246, dtype=np.uint8)
        cv2.rectangle(placeholder, (0, 0), (slot_w - 1, slot_h - 1), (222, 222, 222), 1)
        put_text(placeholder, slot_titles[idx], (12, 24), (126, 126, 126), scale=0.50, thickness=1)

        alpha = float(max(0.0, min(1.0, reveal_alphas[idx])))
        if alpha > 0.0:
            panel, _, _, _ = _fit_image_with_transform(step_images_rgb[idx], box_w=slot_w - 8, box_h=slot_h - 42)
            panel_full = np.full((slot_h, slot_w, 3), 246, dtype=np.uint8)
            panel_full[34 : 34 + panel.shape[0], 4 : 4 + panel.shape[1]] = panel
            cv2.rectangle(panel_full, (0, 0), (slot_w - 1, slot_h - 1), (214, 214, 214), 1)
            put_text(panel_full, slot_titles[idx], (12, 22), (58, 58, 58), scale=0.50, thickness=1)
            if alpha < 1.0:
                panel_full = cv2.addWeighted(panel_full, alpha, placeholder, 1.0 - alpha, 0)
            frame[sy : sy + slot_h, sx : sx + slot_w] = panel_full
        else:
            frame[sy : sy + slot_h, sx : sx + slot_w] = placeholder

    anchor_name = _format_anchor_display_name(sample_metadata["anchor_id"])
    put_text(
        frame,
        "Cross-resolution context-to-target correspondence",
        (margin, 40),
        (24, 24, 24),
        scale=0.72,
        thickness=2,
    )
    put_text(
        frame,
        f"slide = {sample_metadata['slide_id']}",
        (margin, 62),
        (80, 80, 80),
        scale=0.46,
        thickness=1,
    )
    put_text(
        frame,
        f"anchor = {anchor_name}",
        (margin, 74),
        (80, 80, 80),
        scale=0.46,
        thickness=1,
    )
    return frame


def build_flow_step_views(
    context_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    target_images_rgb: list[np.ndarray],
    sample_metadata: dict,
    context_display_size: int,
    target_tile_size: int,
) -> tuple[list[np.ndarray], list[str]]:
    context_size = max(360, int(round(context_display_size * 0.78)))
    context = cv2.resize(context_rgb, (context_size, context_size), interpolation=cv2.INTER_LINEAR)
    scale = float(context_size) / float(context_rgb.shape[0])
    boxes_scaled = scale_boxes(boxes_xyxy, scale)
    context_with_box = context.copy()
    cv2.rectangle(
        context_with_box,
        (2, 2),
        (context_with_box.shape[1] - 3, context_with_box.shape[0] - 3),
        (0, 0, 0),
        2,
    )
    targets_overlay = draw_target_boxes(context.copy(), boxes_scaled, fill_alpha=0.25)
    target_grid, _, _ = make_target_grid(target_images_rgb, tile_size=max(105, min(target_tile_size, 176)))
    predictor_grid = _make_predictor_query_grid(
        context_rgb=context,
        boxes_xyxy=boxes_scaled,
        tile_size=max(130, min(188, context_size // 2)),
    )

    out_ctx = float(sample_metadata["effective_context_mpp"])
    out_tgt = float(sample_metadata["effective_target_mpp"])
    right_views = [
        context_with_box,
        targets_overlay,
        target_grid,
        predictor_grid,
    ]
    slot_titles = [
        f"2) Context ({out_ctx:.2f} mpp)",
        "3) Sampling valid targets",
        f"4) Targets ({out_tgt:.2f} mpp)",
        "5) Predictor input (masked target)",
    ]
    return right_views, slot_titles


def make_zoomed_four_panel_figure(
    step_images_rgb: list[np.ndarray],
    slot_titles: list[str],
    sample_metadata: dict,
) -> np.ndarray:
    if len(step_images_rgb) != 4 or len(slot_titles) != 4:
        raise ValueError("step_images_rgb and slot_titles must both have length 4")

    panel_w = 1080
    panel_h = 820
    gap_x = 28
    gap_y = 28
    margin = 26
    title_h = 88

    fig_w = margin * 2 + panel_w * 2 + gap_x
    fig_h = title_h + panel_h * 2 + gap_y + margin
    fig = np.full((fig_h, fig_w, 3), 246, dtype=np.uint8)

    anchor_name = _format_anchor_display_name(sample_metadata["anchor_id"])
    put_text(
        fig,
        "Cross-resolution context-to-target correspondence",
        (margin, 40),
        (24, 24, 24),
        scale=0.82,
        thickness=2,
    )
    put_text(
        fig,
        f"slide = {sample_metadata['slide_id']}    anchor = {anchor_name}",
        (margin, 66),
        (88, 88, 88),
        scale=0.5,
        thickness=1,
    )

    positions = [
        (margin, title_h),
        (margin + panel_w + gap_x, title_h),
        (margin, title_h + panel_h + gap_y),
        (margin + panel_w + gap_x, title_h + panel_h + gap_y),
    ]

    for idx, (x0, y0) in enumerate(positions):
        panel = np.full((panel_h, panel_w, 3), 246, dtype=np.uint8)
        cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (214, 214, 214), 1)
        put_text(panel, slot_titles[idx], (18, 30), (58, 58, 58), scale=0.62, thickness=1)
        image, _, _, _ = _fit_image_with_transform(
            image_rgb=step_images_rgb[idx],
            box_w=panel_w - 16,
            box_h=panel_h - 54,
        )
        panel[44 : 44 + image.shape[0], 8 : 8 + image.shape[1]] = image
        fig[y0 : y0 + panel_h, x0 : x0 + panel_w] = panel
    return fig


def build_ijepath_animation_frames(
    thumbnail_rgb: np.ndarray,
    anchor_box_thumbnail_xyxy: np.ndarray,
    context_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    target_images_rgb: list[np.ndarray],
    sample_metadata: dict,
    context_display_size: int,
    target_tile_size: int,
) -> tuple[list[np.ndarray], list[int]]:
    frames: list[np.ndarray] = []
    durations_ms: list[int] = []

    wsi_with_anchor = _draw_anchor_on_thumbnail(thumbnail_rgb, anchor_box_thumbnail_xyxy, color=(0, 0, 0))
    right_views, slot_titles = build_flow_step_views(
        context_rgb=context_rgb,
        boxes_xyxy=boxes_xyxy,
        target_images_rgb=target_images_rgb,
        sample_metadata=sample_metadata,
        context_display_size=context_display_size,
        target_tile_size=target_tile_size,
    )
    reveal_alphas = [0.0, 0.0, 0.0, 0.0]
    frames.append(
        _compose_flow_frame(
            wsi_with_anchor_rgb=wsi_with_anchor,
            step_images_rgb=right_views,
            reveal_alphas=reveal_alphas,
            slot_titles=slot_titles,
            sample_metadata=sample_metadata,
        )
    )
    durations_ms.append(1000)

    blend_schedule = [0.18, 0.38, 0.58, 0.78, 1.0]
    for slot_idx in range(len(right_views)):
        for alpha in blend_schedule:
            current = list(reveal_alphas)
            current[slot_idx] = alpha
            frames.append(
                _compose_flow_frame(
                    wsi_with_anchor_rgb=wsi_with_anchor,
                    step_images_rgb=right_views,
                    reveal_alphas=current,
                    slot_titles=slot_titles,
                    sample_metadata=sample_metadata,
                )
            )
            durations_ms.append(1100 if alpha == 1.0 else 120)
        reveal_alphas[slot_idx] = 1.0
    return frames, durations_ms


def write_gif(frames_rgb: list[np.ndarray], durations_ms: list[int], output_path: Path) -> None:
    if not frames_rgb:
        raise ValueError("frames_rgb must be non-empty")
    if len(frames_rgb) != len(durations_ms):
        raise ValueError("durations_ms length must match frames_rgb length")
    pil_frames = [Image.fromarray(frame.astype(np.uint8), mode="RGB") for frame in frames_rgb]
    pil_frames[0].save(
        str(output_path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=[int(max(20, d)) for d in durations_ms],
        loop=0,
        optimize=False,
    )


def write_final_png(frame_rgb: np.ndarray, output_path: Path, scale: float = 2.0) -> None:
    if scale <= 0:
        raise ValueError("scale must be > 0")
    out = frame_rgb
    if abs(float(scale) - 1.0) > 1e-6:
        new_w = max(1, int(round(frame_rgb.shape[1] * float(scale))))
        new_h = max(1, int(round(frame_rgb.shape[0] * float(scale))))
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        out = cv2.resize(frame_rgb, (new_w, new_h), interpolation=interp)
    cv2.imwrite(str(output_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def make_target_grid(target_images_rgb: list[np.ndarray], tile_size: int) -> tuple[np.ndarray, int, int]:
    n = len(target_images_rgb)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    gap = 12

    grid_h = rows * tile_size + (rows - 1) * gap
    grid_w = cols * tile_size + (cols - 1) * gap
    grid = np.full((grid_h, grid_w, 3), 248, dtype=np.uint8)

    for idx, image in enumerate(target_images_rgb):
        color = TARGET_COLORS[idx % len(TARGET_COLORS)]
        tile = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        r = idx // cols
        c = idx % cols
        y0 = r * (tile_size + gap)
        x0 = c * (tile_size + gap)
        grid[y0 : y0 + tile_size, x0 : x0 + tile_size] = tile
        cv2.rectangle(grid, (x0, y0), (x0 + tile_size, y0 + tile_size), color, 4)
        cv2.rectangle(grid, (x0, y0), (x0 + 42, y0 + 22), color, -1)
        put_text(grid, f"T{idx + 1}", (x0 + 6, y0 + 16), (255, 255, 255), scale=0.52, thickness=1)

    return grid, rows, cols


def main() -> int:
    args = parse_args()
    anchor_catalog = Path(args.anchor_catalog).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = read_profile_from_anchor_catalog(anchor_catalog)
    anchor_rows = read_anchor_rows(anchor_catalog)
    context_mpp = args.context_mpp if args.context_mpp is not None else profile["context_mpp"]
    target_mpp = args.target_mpp if args.target_mpp is not None else profile["target_mpp"]
    context_fov_um = args.context_fov_um if args.context_fov_um is not None else profile["context_fov_um"]
    target_fov_um = args.target_fov_um if args.target_fov_um is not None else profile["target_fov_um"]
    targets_per_context = args.targets_per_context if args.targets_per_context is not None else profile["targets_per_context"]

    dataset = CrossResolutionWSIDataset(
        anchor_catalog_csv=str(anchor_catalog),
        context_mpp=context_mpp,
        target_mpp=target_mpp,
        context_fov_um=context_fov_um,
        target_fov_um=target_fov_um,
        patch_size=int(args.patch_size),
        targets_per_context=targets_per_context,
        seed=args.seed,
        spacing_tolerance=args.spacing_tolerance,
        backend=args.wsi_backend,
    )

    thumbnail_cache: dict[str, dict] = {}
    limit = min(args.num_samples, len(dataset))
    for i in range(limit):
        sample = dataset[i]
        context_rgb = tensor_to_rgb_uint8(sample["context_image"])
        target_rgbs = [tensor_to_rgb_uint8(t) for t in sample["target_images"]]
        boxes = sample["target_boxes_in_context_pixels"].detach().cpu().numpy()
        sample_metadata = sample["sample_metadata"]

        anchor_id = sample_metadata["anchor_id"]
        slide_id = sample_metadata["slide_id"]
        anchor_row = anchor_rows.get(str(anchor_id))
        if anchor_row is None:
            raise KeyError(f"anchor_id={anchor_id} not found in anchor catalog {anchor_catalog}")

        thumbnail_rgb, anchor_box_thumb = _thumbnail_from_wsi_row(
            anchor_row=anchor_row,
            backend=args.wsi_backend,
            max_side=args.thumbnail_max_side,
            thumbnail_cache=thumbnail_cache,
            context_mpp=float(sample_metadata["effective_context_mpp"]),
            context_fov_um=context_fov_um,
        )

        frames_rgb, durations_ms = build_ijepath_animation_frames(
            thumbnail_rgb=thumbnail_rgb,
            anchor_box_thumbnail_xyxy=anchor_box_thumb,
            context_rgb=context_rgb,
            boxes_xyxy=boxes,
            target_images_rgb=target_rgbs,
            sample_metadata=sample_metadata,
            context_display_size=args.context_display_size,
            target_tile_size=max(130, int(args.target_tile_size * 0.68)),
        )
        right_views, slot_titles = build_flow_step_views(
            context_rgb=context_rgb,
            boxes_xyxy=boxes,
            target_images_rgb=target_rgbs,
            sample_metadata=sample_metadata,
            context_display_size=args.context_display_size,
            target_tile_size=max(130, int(args.target_tile_size * 0.68)),
        )

        output_stem = build_output_stem(
            sample_index=i,
            slide_id=slide_id,
            anchor_id=anchor_id,
        )
        out_preview = output_dir / f"preview_{output_stem}.png"
        out_ijepa = output_dir / f"ijepa_{output_stem}.gif"
        out_ijepa_final = output_dir / f"ijepa_{output_stem}_all_steps.png"
        out_zoom4 = output_dir / f"zoom4_{output_stem}.png"

        # Plain static preview now matches the final flow layout.
        write_final_png(
            frame_rgb=frames_rgb[-1],
            output_path=out_preview,
            scale=1.0,
        )
        write_gif(frames_rgb=frames_rgb, durations_ms=durations_ms, output_path=out_ijepa)
        if args.final_png_scale > 0:
            write_final_png(
                frame_rgb=frames_rgb[-1],
                output_path=out_ijepa_final,
                scale=float(args.final_png_scale),
            )
        if i < max(0, int(args.num_zoomed_previews)):
            zoom4 = make_zoomed_four_panel_figure(
                step_images_rgb=right_views,
                slot_titles=slot_titles,
                sample_metadata=sample_metadata,
            )
            write_final_png(
                frame_rgb=zoom4,
                output_path=out_zoom4,
                scale=float(args.zoomed_preview_scale),
            )

    print(f"wrote_previews_dir={output_dir}")
    figure_types = "preview_png,ijepa_gif"
    if args.final_png_scale > 0:
        figure_types += ",ijepa_final_png"
    if args.num_zoomed_previews > 0:
        figure_types += ",zoom4_png"
    print(f"figure_types={figure_types}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

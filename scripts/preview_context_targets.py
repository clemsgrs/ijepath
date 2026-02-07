#!/usr/bin/env python3
import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.pathology_cross_resolution_wsi_dataset import PathologyCrossResolutionWSIDataset

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
    parser.add_argument("--crop-size", type=int, default=224, help="Model crop size")
    parser.add_argument("--context-mpp", type=float, default=None)
    parser.add_argument("--target-mpp", type=float, default=None)
    parser.add_argument("--context-fov-um", type=float, default=None)
    parser.add_argument("--target-fov-um", type=float, default=None)
    parser.add_argument("--targets-per-context", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spacing-tolerance", type=float, default=0.05)
    parser.add_argument("--wsi-backend", type=str, default="openslide")
    parser.add_argument("--context-display-size", type=int, default=560)
    parser.add_argument("--target-tile-size", type=int, default=180)
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


def tensor_to_rgb_uint8(tensor_image: np.ndarray) -> np.ndarray:
    array = tensor_image.detach().cpu().numpy().transpose(1, 2, 0)
    array = (array * IMAGENET_STD[None, None, :]) + IMAGENET_MEAN[None, None, :]
    array = np.clip(array, 0.0, 1.0)
    return (array * 255.0).astype(np.uint8)


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


def wrap_text(text: str, max_width: int, scale: float, thickness: int) -> list[str]:
    if not text:
        return []
    words = text.split()
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        (w, _), _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if w <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_wrapped_header(
    canvas: np.ndarray,
    title: str,
    subtitle: str,
    x: int,
    y: int,
    max_width: int,
) -> int:
    title_scale = 0.72
    sub_scale = 0.5
    title_thick = 2
    sub_thick = 1
    line_gap = 8

    title_lines = wrap_text(title, max_width=max_width, scale=title_scale, thickness=title_thick)
    sub_lines = wrap_text(subtitle, max_width=max_width, scale=sub_scale, thickness=sub_thick)

    cursor_y = y
    for line in title_lines:
        put_text(canvas, line, (x, cursor_y), (18, 18, 18), scale=title_scale, thickness=title_thick)
        (_, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_thick)
        cursor_y += h + line_gap

    for line in sub_lines:
        put_text(canvas, line, (x, cursor_y), (85, 85, 85), scale=sub_scale, thickness=sub_thick)
        (_, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, sub_thick)
        cursor_y += h + 5

    return cursor_y


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


def safe_crop(image: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy.tolist()]
    x0 = max(0, min(image.shape[1] - 1, x0))
    y0 = max(0, min(image.shape[0] - 1, y0))
    x1 = max(x0 + 1, min(image.shape[1], x1))
    y1 = max(y0 + 1, min(image.shape[0], y1))
    return image[y0:y1, x0:x1]


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


def mask_regions_for_context_encoder(image_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    for idx, box in enumerate(boxes_xyxy):
        color = TARGET_COLORS[idx % len(TARGET_COLORS)]
        x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]

        x0 = max(0, min(out.shape[1] - 1, x0))
        y0 = max(0, min(out.shape[0] - 1, y0))
        x1 = max(x0 + 1, min(out.shape[1], x1))
        y1 = max(y0 + 1, min(out.shape[0], y1))

        # Gray-out hidden target footprint without hatch artifacts.
        patch = out[y0:y1, x0:x1]
        muted = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        muted = np.repeat(muted[..., None], 3, axis=2)
        muted = cv2.addWeighted(muted, 0.72, np.full_like(muted, 214), 0.28, 0)
        out[y0:y1, x0:x1] = muted

        cv2.rectangle(out, (x0, y0), (x1, y1), color, 3)
        put_text(out, f"T{idx + 1}", (x0 + 4, y0 + 16), color, scale=0.52, thickness=2)

    return out


def draw_query_markers(image_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    for idx, box in enumerate(boxes_xyxy):
        color = TARGET_COLORS[idx % len(TARGET_COLORS)]
        x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 3)
        cv2.circle(out, (cx, cy), 14, color, -1)
        put_text(out, f"Q{idx + 1}", (cx - 10, cy + 6), (255, 255, 255), scale=0.5, thickness=1)
    return out


def draw_block_title(block: np.ndarray, title: str, subtitle: str, margin_x: int = 12, margin_top: int = 28) -> int:
    max_width = block.shape[1] - 2 * margin_x
    end_y = draw_wrapped_header(
        canvas=block,
        title=title,
        subtitle=subtitle,
        x=margin_x,
        y=margin_top,
        max_width=max_width,
    )
    return end_y


def add_header(canvas: np.ndarray, sample_metadata: dict, title: str):
    slide_id = sample_metadata["slide_id"]
    anchor_id = sample_metadata["anchor_id"]
    out_ctx = sample_metadata.get("output_context_mpp", sample_metadata.get("requested_context_mpp"))
    out_tgt = sample_metadata.get("output_target_mpp", sample_metadata.get("requested_target_mpp"))
    src_ctx = sample_metadata.get("source_context_mpp", sample_metadata.get("effective_context_mpp"))
    src_tgt = sample_metadata.get("source_target_mpp", sample_metadata.get("effective_target_mpp"))

    put_text(canvas, title, (14, 30), (20, 20, 20), scale=0.95, thickness=2)
    put_text(canvas, f"slide={slide_id}  anchor={anchor_id}", (14, 54), (50, 50, 50), scale=0.55, thickness=1)
    put_text(
        canvas,
        f"output mpp: context {out_ctx:.3f}, target {out_tgt:.3f} | source-read mpp: context {src_ctx:.3f}, target {src_tgt:.3f}",
        (14, 72),
        (80, 80, 80),
        scale=0.48,
        thickness=1,
    )


def make_pair_figure(
    context_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    target_images_rgb: list[np.ndarray],
    sample_metadata: dict,
    context_display_size: int,
    target_tile_size: int,
) -> np.ndarray:
    context = cv2.resize(context_rgb, (context_display_size, context_display_size), interpolation=cv2.INTER_LINEAR)
    scale = float(context_display_size) / float(context_rgb.shape[0])
    boxes_scaled = scale_boxes(boxes_xyxy, scale)
    context_annot = draw_target_boxes(context, boxes_scaled, fill_alpha=0.2)

    target_grid, _, _ = make_target_grid(target_images_rgb, tile_size=target_tile_size)

    left_header_h = 92
    left_panel = np.full((left_header_h + context_annot.shape[0], context_annot.shape[1], 3), 244, dtype=np.uint8)
    draw_block_title(
        left_panel,
        "Context at lower resolution",
        "Colored footprints define each target region",
        margin_top=28,
    )
    left_panel[left_header_h:, :] = context_annot

    right_header_h = 92
    right_panel = np.full((right_header_h + target_grid.shape[0], target_grid.shape[1], 3), 244, dtype=np.uint8)
    draw_block_title(
        right_panel,
        "Matching high-resolution targets",
        "Same color means same spatial region",
        margin_top=28,
    )
    right_panel[right_header_h:, :] = target_grid

    gap = 24
    body_h = max(left_panel.shape[0], right_panel.shape[0])
    body_w = left_panel.shape[1] + gap + right_panel.shape[1]
    body = np.full((body_h, body_w, 3), 236, dtype=np.uint8)
    body[: left_panel.shape[0], : left_panel.shape[1]] = left_panel
    body[: right_panel.shape[0], left_panel.shape[1] + gap :] = right_panel

    top_h = 82
    fig = np.full((top_h + body_h, body_w, 3), 250, dtype=np.uint8)
    fig[top_h:, :] = body
    add_header(fig, sample_metadata, "Cross-Resolution Context/Target Correspondence")
    return fig


def make_targets_grid_2x2(target_images_rgb: list[np.ndarray], tile_size: int) -> np.ndarray:
    rows = 2
    cols = 2
    gap = 12
    grid_h = rows * tile_size + (rows - 1) * gap
    grid_w = cols * tile_size + (cols - 1) * gap
    grid = np.full((grid_h, grid_w, 3), 248, dtype=np.uint8)

    limit = min(len(target_images_rgb), rows * cols)
    for idx in range(limit):
        image = target_images_rgb[idx]
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

    return grid


def make_ijepa_figure(
    context_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    target_images_rgb: list[np.ndarray],
    sample_metadata: dict,
    context_display_size: int,
    target_tile_size: int,
) -> np.ndarray:
    ijepa_context_size = max(360, int(round(context_display_size * 0.68)))
    context = cv2.resize(context_rgb, (ijepa_context_size, ijepa_context_size), interpolation=cv2.INTER_LINEAR)
    scale = float(ijepa_context_size) / float(context_rgb.shape[0])
    boxes_scaled = scale_boxes(boxes_xyxy, scale)

    plain_context = context.copy()
    target_overlay = draw_target_boxes(context.copy(), boxes_scaled, fill_alpha=0.2)
    predictor_view = draw_query_markers(mask_regions_for_context_encoder(context.copy(), boxes_scaled), boxes_scaled)
    target_tile_size = max(120, min(target_tile_size, (ijepa_context_size - 12) // 2))
    target_grid = make_targets_grid_2x2(target_images_rgb, tile_size=target_tile_size)

    block_pad = 10
    min_header_h = 94

    def make_block(img: np.ndarray, title: str, subtitle: str) -> np.ndarray:
        block = np.full((min_header_h + img.shape[0] + block_pad, img.shape[1], 3), 245, dtype=np.uint8)
        title_end_y = draw_block_title(block, title, subtitle, margin_top=28)
        img_y = max(min_header_h, title_end_y + 8)
        if img_y + img.shape[0] + block_pad > block.shape[0]:
            new_h = img_y + img.shape[0] + block_pad
            expanded = np.full((new_h, img.shape[1], 3), 245, dtype=np.uint8)
            expanded[: block.shape[0], :] = block
            block = expanded
        block[img_y : img_y + img.shape[0], :] = img
        return block

    block_a = make_block(
        plain_context,
        "1) Context Crop",
        "Raw low-resolution context region",
    )
    block_b = make_block(
        target_overlay,
        "2) Target Footprints",
        "Prediction targets in context (T1..Tk)",
    )
    block_c = make_block(
        target_grid,
        "3) Target Encoder",
        "High-resolution target crops (2x2 grid)",
    )
    block_d = make_block(
        predictor_view,
        "4) Predictor",
        "Visible context + query tokens (Q1..Qk)",
    )

    block_w = max(block_a.shape[1], block_b.shape[1], block_c.shape[1], block_d.shape[1])
    block_h = max(block_a.shape[0], block_b.shape[0], block_c.shape[0], block_d.shape[0])

    def pad_block(block: np.ndarray) -> np.ndarray:
        out = np.full((block_h, block_w, 3), 245, dtype=np.uint8)
        x_margin = 10
        y_margin = 10
        x = x_margin
        y = y_margin
        max_w = block_w - x_margin
        max_h = block_h - y_margin
        h = min(block.shape[0], max_h)
        w = min(block.shape[1], max_w)
        block_crop = block[:h, :w]
        out[y : y + h, x : x + w] = block_crop
        cv2.rectangle(out, (0, 0), (block_w - 1, block_h - 1), (222, 222, 222), 1)
        return out

    block_a = pad_block(block_a)
    block_b = pad_block(block_b)
    block_c = pad_block(block_c)
    block_d = pad_block(block_d)

    block_gap_x = 28
    block_gap_y = 24
    body_h = block_h * 2 + block_gap_y
    body_w = block_w * 2 + block_gap_x
    body = np.full((body_h, body_w, 3), 236, dtype=np.uint8)

    x_left = 0
    x_right = block_w + block_gap_x
    y_top = 0
    y_bottom = block_h + block_gap_y

    body[y_top : y_top + block_h, x_left : x_left + block_w] = block_a
    body[y_top : y_top + block_h, x_right : x_right + block_w] = block_b
    body[y_bottom : y_bottom + block_h, x_left : x_left + block_w] = block_c
    body[y_bottom : y_bottom + block_h, x_right : x_right + block_w] = block_d

    # Flow hints: 1->2, 2->4, 3->4.
    arrow_color = (72, 72, 72)
    cv2.arrowedLine(
        body,
        (x_left + block_w + 6, y_top + int(0.35 * block_h)),
        (x_right - 8, y_top + int(0.35 * block_h)),
        arrow_color,
        2,
        tipLength=0.04,
    )
    cv2.arrowedLine(
        body,
        (x_right + int(0.5 * block_w), y_top + block_h + 6),
        (x_right + int(0.5 * block_w), y_bottom - 8),
        arrow_color,
        2,
        tipLength=0.04,
    )
    cv2.arrowedLine(
        body,
        (x_left + block_w + 6, y_bottom + int(0.5 * block_h)),
        (x_right - 8, y_bottom + int(0.5 * block_h)),
        arrow_color,
        2,
        tipLength=0.04,
    )

    top_h = 82
    fig = np.full((top_h + body_h, body_w, 3), 250, dtype=np.uint8)
    fig[top_h:, :] = body
    add_header(fig, sample_metadata, "I-JEPA Data Flow (Cross-Resolution Pathology)")
    return fig


def main() -> int:
    args = parse_args()
    anchor_catalog = Path(args.anchor_catalog).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = read_profile_from_anchor_catalog(anchor_catalog)
    context_mpp = args.context_mpp if args.context_mpp is not None else profile["context_mpp"]
    target_mpp = args.target_mpp if args.target_mpp is not None else profile["target_mpp"]
    context_fov_um = args.context_fov_um if args.context_fov_um is not None else profile["context_fov_um"]
    target_fov_um = args.target_fov_um if args.target_fov_um is not None else profile["target_fov_um"]
    targets_per_context = args.targets_per_context if args.targets_per_context is not None else profile["targets_per_context"]

    dataset = PathologyCrossResolutionWSIDataset(
        anchor_catalog_csv=str(anchor_catalog),
        crop_size=args.crop_size,
        context_mpp=context_mpp,
        target_mpp=target_mpp,
        context_fov_um=context_fov_um,
        target_fov_um=target_fov_um,
        targets_per_context=targets_per_context,
        seed=args.seed,
        spacing_tolerance=args.spacing_tolerance,
        backend=args.wsi_backend,
        samples_per_epoch=max(1, args.num_samples),
    )

    limit = min(args.num_samples, len(dataset))
    for i in range(limit):
        sample = dataset[i]
        context_rgb = tensor_to_rgb_uint8(sample["context_image"])
        target_rgbs = [tensor_to_rgb_uint8(t) for t in sample["target_images"]]
        boxes = sample["target_boxes_in_context_pixels"].detach().cpu().numpy()

        pair_fig = make_pair_figure(
            context_rgb=context_rgb,
            boxes_xyxy=boxes,
            target_images_rgb=target_rgbs,
            sample_metadata=sample["sample_metadata"],
            context_display_size=args.context_display_size,
            target_tile_size=args.target_tile_size,
        )
        ijepa_fig = make_ijepa_figure(
            context_rgb=context_rgb,
            boxes_xyxy=boxes,
            target_images_rgb=target_rgbs,
            sample_metadata=sample["sample_metadata"],
            context_display_size=args.context_display_size,
            target_tile_size=max(130, int(args.target_tile_size * 0.68)),
        )

        anchor_id = sample["sample_metadata"]["anchor_id"]
        slide_id = sample["sample_metadata"]["slide_id"]

        out_pair = output_dir / f"pair_{i:03d}_{slide_id}_{anchor_id}.png"
        out_ijepa = output_dir / f"ijepa_{i:03d}_{slide_id}_{anchor_id}.png"

        cv2.imwrite(str(out_pair), cv2.cvtColor(pair_fig, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_ijepa), cv2.cvtColor(ijepa_fig, cv2.COLOR_RGB2BGR))

    print(f"wrote_previews_dir={output_dir}")
    print("figure_types=pair,ijepa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Utilities for Stage C target-footprint diagnostics plots.

This module keeps plotting logic out of the notebook so the notebook stays readable
while preserving a versioned source of truth for figure generation.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CONTINUOUS_FILL = "#86efac"
CONTINUOUS_EDGE = "#059669"
COVER_MISS = "#ef4444"  # inside target content, absent in predictor map
COVER_EXTRA = "#f59e0b"  # outside target content, present in predictor map


def _xyxy_vals(box_xyxy_pixels: Sequence[float]) -> tuple[float, float, float, float]:
    vals = box_xyxy_pixels.tolist() if hasattr(box_xyxy_pixels, "tolist") else box_xyxy_pixels
    x0, y0, x1, y1 = [float(v) for v in vals]
    return x0, y0, x1, y1


def continuous_token_rect(
    box_xyxy_pixels: Sequence[float], patch_size: int | float
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = _xyxy_vals(box_xyxy_pixels)
    patch = float(patch_size)
    return (
        x0 / patch - 0.5,
        y0 / patch - 0.5,
        x1 / patch - 0.5,
        y1 / patch - 0.5,
    )


def draw_continuous_extent(
    ax,
    box_xyxy_pixels: Sequence[float],
    patch_size: int | float,
    fill_alpha: float = 0.24,
    edge_alpha: float = 0.95,
) -> None:
    tx0, ty0, tx1, ty1 = continuous_token_rect(box_xyxy_pixels, patch_size)
    tw = tx1 - tx0
    th = ty1 - ty0

    if fill_alpha > 0:
        ax.add_patch(
            plt.Rectangle(
                (tx0, ty0),
                tw,
                th,
                fill=True,
                facecolor=CONTINUOUS_FILL,
                edgecolor="none",
                alpha=fill_alpha,
            )
        )

    ax.add_patch(
        plt.Rectangle(
            (tx0, ty0),
            tw,
            th,
            fill=False,
            edgecolor=CONTINUOUS_EDGE,
            linewidth=1.8,
            linestyle="-",
            alpha=edge_alpha,
        )
    )


def style_token_axis(ax, title: str, grid_h: int, grid_w: int) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("token x")
    ax.set_ylabel("token y")
    ax.set_xlim(-0.5, grid_w - 0.5)
    ax.set_ylim(grid_h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#ffffff")
    ax.set_xticks(np.arange(0, grid_w, 4))
    ax.set_yticks(np.arange(0, grid_h, 4))
    ax.tick_params(labelsize=7)
    ax.set_xticks(np.arange(-0.5, grid_w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_h, 1), minor=True)
    ax.grid(which="minor", color="#000000", alpha=0.05, linewidth=0.35)


def token_map_dims(token_map: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(token_map > 0.5)
    if ys.size == 0 or xs.size == 0:
        return 0, 0
    h_tok = int(ys.max() - ys.min() + 1)
    w_tok = int(xs.max() - xs.min() + 1)
    return w_tok, h_tok


def annotate_box_dims(
    ax,
    meta: dict,
    token_map: np.ndarray,
    label_prefix: str = "R",
    edgecolor: str = "#111827",
    textcolor: str = "#111827",
) -> None:
    w_tok, h_tok = token_map_dims(token_map)

    # Keep label just above the token box while right-aligning to box right edge.
    x_right = float(meta["px1"]) - 0.5
    y_text = max(0.35, float(meta["py0"]) - 0.72)

    ax.text(
        x_right,
        y_text,
        f"{label_prefix}:{w_tok}x{h_tok}",
        ha="right",
        va="bottom",
        fontsize=7,
        color=textcolor,
        clip_on=True,
        zorder=7,
        bbox=dict(
            boxstyle="round,pad=0.14",
            facecolor="white",
            edgecolor=edgecolor,
            alpha=0.95,
        ),
    )


def _fmt_dim(v: float) -> str:
    vr = round(float(v))
    if abs(float(v) - vr) < 1e-6:
        return str(int(vr))
    return f"{float(v):.1f}"


def annotate_continuous_dims(
    ax, box_xyxy_pixels: Sequence[float], patch_size: int | float
) -> None:
    tx0, ty0, tx1, ty1 = continuous_token_rect(box_xyxy_pixels, patch_size)
    w_tok = tx1 - tx0
    h_tok = ty1 - ty0

    x_text = tx0 + 0.25
    y_text = ty0 + 0.32

    ax.text(
        x_text,
        y_text,
        f"C:{_fmt_dim(w_tok)}x{_fmt_dim(h_tok)}",
        ha="left",
        va="top",
        fontsize=7,
        color="#065f46",
        bbox=dict(
            boxstyle="round,pad=0.14",
            facecolor="#ecfdf5",
            edgecolor=CONTINUOUS_EDGE,
            alpha=0.95,
        ),
    )


def token_map_rect(token_map: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(token_map > 0.5)
    if ys.size == 0 or xs.size == 0:
        return None
    return (
        float(xs.min()) - 0.5,
        float(ys.min()) - 0.5,
        float(xs.max()) + 0.5,
        float(ys.max()) + 0.5,
    )


def draw_token_map_box(
    ax,
    token_map: np.ndarray,
    color: str = "#111827",
    linewidth: float = 2.0,
    linestyle: str = "-",
) -> None:
    rect = token_map_rect(token_map)
    if rect is None:
        return
    x0, y0, x1, y1 = rect
    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )
    )


def _cell_rect_intersection(
    rect_a: tuple[float, float, float, float],
    rect_b: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    ax0, ay0, ax1, ay1 = rect_a
    bx0, by0, bx1, by1 = rect_b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return (ix0, iy0, ix1, iy1)


def _add_rect_patch(
    ax, rect: tuple[float, float, float, float], facecolor: str, alpha: float = 0.80
) -> None:
    x0, y0, x1, y1 = rect
    if (x1 - x0) <= 1e-6 or (y1 - y0) <= 1e-6:
        return
    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=True,
            facecolor=facecolor,
            edgecolor="none",
            alpha=alpha,
        )
    )


def draw_continuous_mismatch_regions(
    ax,
    token_map: np.ndarray,
    box_xyxy_pixels: Sequence[float],
    patch_size: int | float,
) -> list[int]:
    cont_rect = continuous_token_rect(box_xyxy_pixels, patch_size)
    h, w = token_map.shape
    present: list[int] = []
    has_missing = False
    has_extra = False

    for py in range(h):
        for px in range(w):
            cell = (float(px) - 0.5, float(py) - 0.5, float(px) + 0.5, float(py) + 0.5)
            inter = _cell_rect_intersection(cell, cont_rect)
            token_present = bool(token_map[py, px] > 0.5)

            if not token_present:
                if inter is not None:
                    _add_rect_patch(ax, inter, COVER_MISS)
                    has_missing = True
                continue

            if inter is None:
                _add_rect_patch(ax, cell, COVER_EXTRA)
                has_extra = True
                continue

            cx0, cy0, cx1, cy1 = cell
            ix0, iy0, ix1, iy1 = inter

            _add_rect_patch(ax, (cx0, cy0, ix0, cy1), COVER_EXTRA)
            _add_rect_patch(ax, (ix1, cy0, cx1, cy1), COVER_EXTRA)
            _add_rect_patch(ax, (ix0, cy0, ix1, iy0), COVER_EXTRA)
            _add_rect_patch(ax, (ix0, iy1, ix1, cy1), COVER_EXTRA)
            if (ix0 > cx0) or (ix1 < cx1) or (iy0 > cy0) or (iy1 < cy1):
                has_extra = True

    if has_missing:
        present.append(-1)
    if has_extra:
        present.append(1)
    return present


def coverage_mismatch_legend_handles(present_values: Iterable[int]) -> list[Patch]:
    present = set(present_values)
    handles: list[Patch] = []
    if -1 in present:
        handles.append(
            Patch(
                facecolor=COVER_MISS,
                edgecolor="none",
                label="C \\ T: inside target, absent in predictor",
            )
        )
    if 1 in present:
        handles.append(
            Patch(
                facecolor=COVER_EXTRA,
                edgecolor="none",
                label="T \\ C: outside target, present in predictor",
            )
        )
    return handles


def select_focus_target(strategy_visuals: dict) -> int:
    non_aligned_losses = [
        int(m.sum()) for m in strategy_visuals["non_aligned"]["truncation_loss_per_target"]
    ]
    return int(np.argmax(non_aligned_losses)) if non_aligned_losses else 0


def print_focus_target_summary(
    strategy_visuals: dict,
    focus_target_idx: int,
    colors: Sequence[str] | None = None,
) -> None:
    focus_color = colors[focus_target_idx % len(colors)] if colors else "n/a"
    print(f"Focus target index: T{focus_target_idx + 1} (color={focus_color})")
    for name in ["non_aligned", "aligned"]:
        v = strategy_visuals[name]
        meta = v["raw_target_meta"][focus_target_idx]
        raw_tokens = int(v["raw_target_maps"][focus_target_idx].sum())
        collated_tokens = int(v["collated_target_maps"][focus_target_idx].sum())
        dropped = raw_tokens - collated_tokens
        print(
            f"  {name}: mod16={meta['box_mod_patch']} raw={raw_tokens} collated={collated_tokens} dropped={dropped}"
        )


def print_token_accounting(strategy_visuals: dict) -> None:
    print("Per-target token accounting:")
    for target_idx in range(len(strategy_visuals["non_aligned"]["raw_target_maps"])):
        print(f"  T{target_idx + 1}")
        for name in ["non_aligned", "aligned"]:
            v = strategy_visuals[name]
            meta = v["raw_target_meta"][target_idx]
            raw_tokens = int(v["raw_target_maps"][target_idx].sum())
            collated_tokens = int(v["collated_target_maps"][target_idx].sum())
            dropped = raw_tokens - collated_tokens
            print(
                f"    {name}: mod16={meta['box_mod_patch']} raw={raw_tokens} collated={collated_tokens} dropped={dropped}"
            )

    print("Union accounting:")
    for name in ["non_aligned", "aligned"]:
        v = strategy_visuals[name]
        raw_union_tokens = int(v["raw_union"].sum())
        collated_union_tokens = int(v["collated_union"].sum())
        dropped_union = int(v["truncation_loss_union"].sum())
        print(
            f"  {name}: raw_union={raw_union_tokens} collated_union={collated_union_tokens} dropped_union={dropped_union}"
        )


def plot_stagec_focus_figure(
    strategy_visuals: dict,
    patch_size: int,
    colors: Sequence[str] | None = None,
    strategy_titles: dict[str, str] | None = None,
    focus_target_idx: int | None = None,
):
    strategy_titles = strategy_titles or {
        "non_aligned": "Non-aligned",
        "aligned": "Patch-aligned",
    }
    focus_target_idx = (
        select_focus_target(strategy_visuals)
        if focus_target_idx is None
        else int(focus_target_idx)
    )

    grid_h = int(strategy_visuals["non_aligned"]["grid_h"])
    grid_w = int(strategy_visuals["non_aligned"]["grid_w"])

    raw_legend = [
        Patch(facecolor=CONTINUOUS_FILL, edgecolor=CONTINUOUS_EDGE, label="C: target content"),
        Line2D([0], [0], color="#111827", lw=2.0, linestyle="-", label="R: raw predictor footprint"),
    ]
    post_legend = [
        Patch(facecolor=CONTINUOUS_FILL, edgecolor=CONTINUOUS_EDGE, label="C: target content"),
        Line2D(
            [0],
            [0],
            color="#111827",
            lw=2.0,
            linestyle="-",
            label="T: post-collator predictor footprint",
        ),
    ]
    coverage_ref_legend = [
        Patch(facecolor=CONTINUOUS_FILL, edgecolor=CONTINUOUS_EDGE, label="C: target content"),
        Line2D(
            [0],
            [0],
            color="#111827",
            lw=2.0,
            linestyle="-",
            label="T: post-collator predictor footprint",
        ),
    ]

    fig_target, axes_target = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    strategy_order = ["aligned", "non_aligned"]

    for row, name in enumerate(strategy_order):
        v = strategy_visuals[name]
        meta = v["raw_target_meta"][focus_target_idx]
        raw_map = v["raw_target_maps"][focus_target_idx]
        post_map = v["collated_target_maps"][focus_target_idx]
        box = v["boxes0"][focus_target_idx]

        ax_raw = axes_target[row, 0]
        draw_continuous_extent(ax_raw, box, patch_size=patch_size, fill_alpha=0.26)
        draw_token_map_box(ax_raw, raw_map, color="#111827", linewidth=1.8, linestyle="-")
        style_token_axis(ax_raw, f"{strategy_titles[name]}\nRaw footprint (R vs C)", grid_h, grid_w)
        annotate_box_dims(
            ax_raw,
            meta,
            raw_map,
            label_prefix="R",
            edgecolor="#111827",
            textcolor="#111827",
        )
        annotate_continuous_dims(ax_raw, box, patch_size)
        ax_raw.legend(handles=raw_legend, loc="upper left", fontsize=7.0, framealpha=0.92)

        ax_post = axes_target[row, 1]
        draw_continuous_extent(ax_post, box, patch_size=patch_size, fill_alpha=0.26)
        draw_token_map_box(ax_post, post_map, color="#111827", linewidth=1.8, linestyle="-")
        style_token_axis(
            ax_post,
            f"{strategy_titles[name]}\nPost-collator footprint (T vs C)",
            grid_h,
            grid_w,
        )
        annotate_box_dims(
            ax_post,
            meta,
            post_map,
            label_prefix="T",
            edgecolor="#111827",
            textcolor="#111827",
        )
        annotate_continuous_dims(ax_post, box, patch_size)
        ax_post.legend(handles=post_legend, loc="upper left", fontsize=7.0, framealpha=0.92)

        token_map = post_map > 0.5
        ax_delta = axes_target[row, 2]
        draw_continuous_extent(
            ax_delta,
            box,
            patch_size=patch_size,
            fill_alpha=0.26,
            edge_alpha=0.95,
        )
        present = draw_continuous_mismatch_regions(ax_delta, token_map, box, patch_size)
        draw_token_map_box(
            ax_delta,
            token_map,
            color="#111827",
            linewidth=2.0,
            linestyle="-",
        )
        title_suffix = " (exact match)" if len(present) == 0 else ""
        style_token_axis(
            ax_delta,
            f"{strategy_titles[name]}\nCoverage mismatch (T vs C){title_suffix}",
            grid_h,
            grid_w,
        )

        cont_rect = continuous_token_rect(box, patch_size)
        tok_rect = token_map_rect(token_map.astype(np.float32))
        if tok_rect is None:
            ux0, uy0, ux1, uy1 = cont_rect
        else:
            ux0 = min(cont_rect[0], tok_rect[0])
            uy0 = min(cont_rect[1], tok_rect[1])
            ux1 = max(cont_rect[2], tok_rect[2])
            uy1 = max(cont_rect[3], tok_rect[3])

        margin = 3.5
        x0 = max(-0.5, ux0 - margin)
        x1 = min(grid_w - 0.5, ux1 + margin)
        y0 = max(-0.5, uy0 - margin)
        y1 = min(grid_h - 0.5, uy1 + margin)
        ax_delta.set_xlim(x0, x1)
        ax_delta.set_ylim(y1, y0)

        ax_delta.legend(
            handles=coverage_ref_legend + coverage_mismatch_legend_handles(present),
            loc="upper left",
            fontsize=6.6,
            framealpha=0.92,
        )

    return fig_target, axes_target


def draw_context_targets(sample: dict, title: str, colors: Sequence[str], tensor_to_rgb_uint8) -> None:
    context_rgb = tensor_to_rgb_uint8(sample["context_image"])
    boxes = sample["target_boxes_in_context_pixels"].detach().cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(context_rgb)
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box.tolist()
        color = colors[i % len(colors)]
        ax.add_patch(
            plt.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                linewidth=2.0,
                color=color,
            )
        )
        ax.text(
            x0 + 2,
            y0 + 10,
            f"T{i+1}",
            color="white",
            fontsize=9,
            bbox=dict(facecolor=color, alpha=0.9, pad=1.5),
        )
    ax.set_title(title)
    ax.axis("off")
    plt.show()


def show_target_tiles(sample: dict, tensor_to_rgb_uint8) -> None:
    targets = sample["target_images"]
    k = targets.shape[0]
    fig, axes = plt.subplots(1, k, figsize=(3 * k, 3))
    if k == 1:
        axes = [axes]
    for i in range(k):
        axes[i].imshow(tensor_to_rgb_uint8(targets[i]))
        axes[i].set_title(f"T{i+1} target crop")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


def show_predictor_proxy(sample: dict, tensor_to_rgb_uint8) -> None:
    context_rgb = tensor_to_rgb_uint8(sample["context_image"])
    boxes = sample["target_boxes_in_context_pixels"].detach().cpu().numpy()
    k = boxes.shape[0]

    fig, axes = plt.subplots(1, k, figsize=(3 * k, 3))
    if k == 1:
        axes = [axes]
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
        masked = context_rgb.copy()
        masked[y0:y1, x0:x1] = 0
        axes[i].imshow(masked)
        axes[i].set_title(f"Predictor query T{i+1}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


def box_to_token_map(box_xyxy: np.ndarray, grid_h: int, grid_w: int, patch_size: int):
    x0, y0, x1, y1 = [float(v) for v in box_xyxy.tolist()]
    px0 = max(0, min(grid_w, int(np.floor(x0 / patch_size))))
    py0 = max(0, min(grid_h, int(np.floor(y0 / patch_size))))
    px1 = max(0, min(grid_w, int(np.ceil(x1 / patch_size))))
    py1 = max(0, min(grid_h, int(np.ceil(y1 / patch_size))))

    if px1 <= px0:
        px1 = min(grid_w, px0 + 1)
    if py1 <= py0:
        py1 = min(grid_h, py0 + 1)

    token_map = np.zeros((grid_h, grid_w), dtype=np.float32)
    token_map[py0:py1, px0:px1] = 1.0
    return token_map, {
        "px0": px0,
        "py0": py0,
        "px1": px1,
        "py1": py1,
        "token_count": int((py1 - py0) * (px1 - px0)),
    }


def token_ids_to_map(token_ids, grid_h: int, grid_w: int):
    out = np.zeros(grid_h * grid_w, dtype=np.float32)
    out[token_ids.detach().cpu().numpy()] = 1.0
    return out.reshape(grid_h, grid_w)


def make_strategy_visual_payload(batch_data_local: dict, masks_pred_local, patch_size: int):
    context0 = batch_data_local["context_images"][0]
    boxes0 = batch_data_local["target_boxes_in_context_pixels"][0].detach().cpu().numpy()

    grid_h = int(context0.shape[-2]) // patch_size
    grid_w = int(context0.shape[-1]) // patch_size

    raw_target_maps = []
    raw_target_meta = []
    for box in boxes0:
        m, meta = box_to_token_map(box, grid_h=grid_h, grid_w=grid_w, patch_size=patch_size)
        meta["box_mod_patch"] = [int(v) % patch_size for v in box.tolist()]
        raw_target_maps.append(m)
        raw_target_meta.append(meta)

    collated_target_maps = [
        token_ids_to_map(pm[0], grid_h=grid_h, grid_w=grid_w) for pm in masks_pred_local
    ]

    raw_union = np.clip(np.sum(raw_target_maps, axis=0), 0.0, 1.0).astype(np.float32)
    collated_union = np.clip(np.sum(collated_target_maps, axis=0), 0.0, 1.0).astype(np.float32)

    truncation_loss_per_target = [
        np.clip(raw_target_maps[i] - collated_target_maps[i], 0.0, 1.0).astype(np.float32)
        for i in range(len(raw_target_maps))
    ]
    truncation_loss_union = np.clip(raw_union - collated_union, 0.0, 1.0).astype(np.float32)

    return {
        "boxes0": boxes0,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "raw_target_maps": raw_target_maps,
        "raw_target_meta": raw_target_meta,
        "collated_target_maps": collated_target_maps,
        "raw_union": raw_union,
        "collated_union": collated_union,
        "truncation_loss_per_target": truncation_loss_per_target,
        "truncation_loss_union": truncation_loss_union,
    }


def build_strategy_visuals(strategy_batches: dict, patch_size: int) -> dict:
    return {
        name: make_strategy_visual_payload(payload["batch_data"], payload["masks_pred"], patch_size)
        for name, payload in strategy_batches.items()
    }

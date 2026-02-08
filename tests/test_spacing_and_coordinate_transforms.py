import math

from ijepath.datasets.wsi_readers.wholeslidedata_reader_adapter import (
    context_box_from_center_level0,
    level0_pixels_to_spacing_pixels,
    spacing_pixels_to_level0_pixels,
)


def test_pathology_spacing_roundtrip_pixels():
    spacing_at_level0 = 0.25
    target_spacing = 1.0
    spacing_pixels = 512

    lv0_pixels = spacing_pixels_to_level0_pixels(
        size_pixels_at_spacing=spacing_pixels,
        spacing=target_spacing,
        spacing_at_level0=spacing_at_level0,
    )
    recovered = level0_pixels_to_spacing_pixels(
        size_pixels_at_level0=lv0_pixels,
        spacing=target_spacing,
        spacing_at_level0=spacing_at_level0,
    )

    assert lv0_pixels == 2048
    assert recovered == spacing_pixels


def test_pathology_context_box_from_center_level0():
    center_x = 5000
    center_y = 7000
    spacing_at_level0 = 0.25
    context_spacing = 1.0
    context_size_pixels = 512

    x0, y0, x1, y1 = context_box_from_center_level0(
        center_x_level0=center_x,
        center_y_level0=center_y,
        size_pixels_at_spacing=context_size_pixels,
        spacing=context_spacing,
        spacing_at_level0=spacing_at_level0,
    )

    expected_half = spacing_pixels_to_level0_pixels(
        size_pixels_at_spacing=context_size_pixels,
        spacing=context_spacing,
        spacing_at_level0=spacing_at_level0,
    ) // 2

    assert x0 == center_x - expected_half
    assert y0 == center_y - expected_half
    assert x1 == center_x + expected_half
    assert y1 == center_y + expected_half
    assert math.isclose((x1 - x0), 2048, rel_tol=0, abs_tol=0)

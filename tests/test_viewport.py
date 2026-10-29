"""Regression tests for precision-preserving viewport navigation."""

from decimal import Decimal, localcontext

import pytest

from fractalis.viewport import Viewport, requires_deep_precision


def test_zoom_preserves_long_decimal_centre():
    view = Viewport(
        "-0.7436438870371510000000000000000000001",
        "0.1318259042053300000000000000000000002",
        "1e-40",
    )

    zoomed = view.zoom("0.5")

    assert zoomed.center_x == view.center_x
    assert zoomed.center_y == view.center_y
    assert Decimal(zoomed.span_x) == Decimal("5e-41")


def test_zoom_out_restores_span():
    view = Viewport("-0.743643887037151", "0.1", "1e-80")

    assert view.zoom("0.5").zoom("2").span_x == view.span_x


def test_relative_pan_preserves_tiny_offset():
    view = Viewport("-0.743643887037151", "0", "1e-40")
    moved = view.pan("1e-42", "-2e-42")

    assert Decimal(moved.center_x) - Decimal(view.center_x) == Decimal("1e-42")
    assert Decimal(moved.center_y) == Decimal("-2e-42")


def test_vertical_span_uses_image_aspect_ratio():
    view = Viewport("-0.5", "0", "3.2")

    # Compare both calculations at the same precision. Decimal division
    # is rounded according to the active context, so the default 28-digit
    # result must not be compared exactly with a 50-digit result.
    with localcontext() as ctx:
        ctx.prec = 50
        expected = Decimal("3.2") * Decimal(32) / Decimal(48)
        actual = Decimal(view.vertical_span(48, 32))
        assert abs(actual - expected) < Decimal("1e-49")


def test_deep_engine_selection_accounts_for_centre_scale():
    assert requires_deep_precision(
        "-0.743643887037151", "0", "1e-20", 800
    )
    assert not requires_deep_precision("-0.5", "0", "3.2", 800)


def test_float_coordinates_are_rejected():
    with pytest.raises(ValueError, match="decimal string"):
        Viewport(-0.5, "0", "1e-20")

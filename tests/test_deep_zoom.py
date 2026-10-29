"""Regression tests for the arbitrary-precision reference engine."""

import numpy as np
import pytest

from fractalis.deep_zoom import deep_coordinate_grid, deep_mandelbrot


def test_deep_coordinates_preserve_tiny_differences():
    xs, _ = deep_coordinate_grid(
        3, 1, "-0.5", "0", "1e-40"
    )

    assert xs[0] != xs[1]
    assert xs[1] != xs[2]
    assert abs(xs[1] - xs[0]) < 1e-30


def test_known_parameter_one_escapes_on_iteration_three():
    result = deep_mandelbrot(
        width=2,
        height=1,
        max_iterations=10,
        center_x="1.5",
        center_y="0",
        span_x="1",
    )

    # The first sample is c=1; its orbit escapes at iteration three.
    assert result.escaped[0, 0]
    assert result.escape_iterations[0, 0] == 3


def test_zero_parameter_does_not_escape():
    result = deep_mandelbrot(
        width=1,
        height=1,
        max_iterations=20,
        center_x="0",
        center_y="0",
        span_x="1e-40",
    )

    assert not result.escaped[0, 0]
    assert result.escape_iterations[0, 0] == 0
    assert np.isnan(result.smooth_iterations[0, 0])


def test_result_arrays_are_consistent():
    result = deep_mandelbrot(
        width=12,
        height=8,
        max_iterations=40,
    )

    np.testing.assert_array_equal(
        result.escaped,
        result.escape_iterations > 0,
    )
    assert np.all(np.isfinite(result.smooth_iterations[result.escaped]))
    assert np.all(np.isnan(result.smooth_iterations[~result.escaped]))


def test_large_reference_render_is_rejected():
    with pytest.raises(ValueError, match="8192 pixels"):
        deep_mandelbrot(width=100, height=100)


def test_invalid_span_is_rejected():
    with pytest.raises(ValueError, match="span_x"):
        deep_mandelbrot(span_x="0")

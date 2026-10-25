
"""Mathematical reference tests for the FRACTALIS core engine."""

import numpy as np
import pytest

from fractalis.core import complex_grid, mandelbrot


# ============================================================
# SECTION 1 — Test the complex-plane coordinate system
# ============================================================

def test_complex_grid_orientation():
    """Verify corners and center of a symmetric coordinate grid."""

    grid = complex_grid(
        width=3,
        height=3,
        x_min=-1.0,
        x_max=1.0,
        y_min=-1.0,
        y_max=1.0,
    )

    # The first image row represents the largest imaginary coordinate.
    assert grid[0, 0] == -1.0 + 1.0j

    # The final row represents the smallest imaginary coordinate.
    assert grid[-1, -1] == 1.0 - 1.0j

    # The center sample represents the complex origin.
    assert grid[1, 1] == 0.0 + 0.0j


# ============================================================
# SECTION 2 — Test known orbits
# ============================================================

def test_zero_parameter_never_escapes():
    """For c=0, the orbit remains at z=0 forever."""

    result = mandelbrot(
        width=1,
        height=1,
        max_iterations=30,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
    )

    assert not result.escaped[0, 0]
    assert result.escape_iterations[0, 0] == 0
    assert np.isnan(result.smooth_iterations[0, 0])


def test_minus_one_parameter_never_escapes():
    """For c=-1, the orbit alternates between 0 and -1."""

    result = mandelbrot(
        width=2,
        height=2,
        max_iterations=30,
        x_min=-1.0,
        x_max=0.0,
        y_min=-1.0,
        y_max=0.0,
    )

    # The top-left pixel samples c = -1 + 0j.
    assert not result.escaped[0, 0]
    assert result.escape_iterations[0, 0] == 0

def test_parameter_one_escapes_on_iteration_three():
    """For c=1, the orbit 0 -> 1 -> 2 -> 5 escapes at n=3."""

    result = mandelbrot(
        width=2,
        height=2,
        max_iterations=10,
        x_min=1.0,
        x_max=2.0,
        y_min=-1.0,
        y_max=0.0,
    )

    # The top-left pixel samples c = x_min + y_max*j = 1 + 0j.
    assert result.escaped[0, 0]
    assert result.escape_iterations[0, 0] == 3


# ============================================================
# SECTION 3 — Verify consistency of the numerical fields
# ============================================================

def test_escape_mask_matches_iteration_counts():
    """An observed escape must have a positive iteration count."""

    result = mandelbrot(width=80, height=60, max_iterations=100)

    np.testing.assert_array_equal(
        result.escaped,
        result.escape_iterations > 0,
    )


def test_smooth_values_are_finite_only_for_escaped_points():
    """Smooth coloring values are defined only for observed escapes."""

    result = mandelbrot(width=80, height=60, max_iterations=100)

    assert np.all(np.isfinite(result.smooth_iterations[result.escaped]))
    assert np.all(np.isnan(result.smooth_iterations[~result.escaped]))


# ============================================================
# SECTION 4 — Verify input validation
# ============================================================

@pytest.mark.parametrize(
    "overrides",
    [
        {"width": 0},
        {"height": -1},
        {"max_iterations": 0},
        {"width": 1.5},
        {"x_min": 1.0, "x_max": -1.0},
        {"y_min": 2.0, "y_max": 2.0},
    ],
)
def test_invalid_parameters_raise_value_error(overrides):
    """Invalid numerical settings must fail with a clear exception."""

    parameters = {
        "width": 100,
        "height": 80,
        "max_iterations": 50,
        "x_min": -2.0,
        "x_max": 1.0,
        "y_min": -1.0,
        "y_max": 1.0,
    }

    parameters.update(overrides)

    with pytest.raises(ValueError):
        mandelbrot(**parameters)

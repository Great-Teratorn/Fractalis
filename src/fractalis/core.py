
"""
FRACTALIS — Core Mathematical Engine
====================================

This module calculates the Mandelbrot escape-time field.

Mathematical definition
----------------------
For each complex parameter c, define the orbit:

    z_0 = 0
    z_(n+1) = z_n**2 + c

The Mandelbrot set consists of parameters c for which this orbit
remains bounded.

For the quadratic map, if an orbit reaches |z| > 2, it escapes
to infinity. This gives us a rigorous way to classify escaped
points without needing to iterate them forever.

Important numerical distinction
--------------------------------
A point that does not escape within our finite iteration budget
is classified as "not observed to escape". We do not claim that
finite computation proves its orbit is bounded forever.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


# ============================================================
# SECTION 1 — Define the result data structure
# ============================================================
# We return several arrays because different stages of FRACTALIS
# will need different mathematical properties of the same orbit.
#
# Using a dataclass makes the output easier to understand and
# safer to extend than returning an undocumented tuple.

@dataclass(frozen=True)
class FractalResult:
    """
    Store the numerical results of a Mandelbrot calculation.

    Attributes
    ----------
    escape_iterations:
        First iteration at which each orbit exceeded the escape
        radius. Zero means that no escape was observed.

    escaped:
        Boolean mask identifying points that actually escaped.

    smooth_iterations:
        Continuous escape-time estimates for escaped points.
        Non-escaped points contain NaN.

    max_iterations:
        The maximum iteration budget used by the calculation.
    """

    escape_iterations: NDArray[np.int32]
    escaped: NDArray[np.bool_]
    smooth_iterations: NDArray[np.float64]
    max_iterations: int


# ============================================================
# SECTION 2 — Validate all user-controlled parameters
# ============================================================
# Catch invalid inputs before allocating arrays or performing
# expensive calculations. This is particularly important when
# users eventually request very large images or deep zooms.

def _validate_parameters(
    width: int,
    height: int,
    max_iterations: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    """Validate dimensions, iteration budget and complex-plane bounds."""

    # Image dimensions and iteration counts must be positive integers.
    # Python treats bool as a subclass of int, so reject it explicitly.
    for name, value in (
        ("width", width),
        ("height", height),
        ("max_iterations", max_iterations),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be an integer.")

        if value < 1:
            raise ValueError(f"{name} must be positive.")

    # Coordinates must be finite real numbers.
    bounds = (x_min, x_max, y_min, y_max)

    if not all(np.isfinite(value) for value in bounds):
        raise ValueError("Coordinate bounds must all be finite.")

    # Each interval must have positive width.
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min.")

    if y_max <= y_min:
        raise ValueError("y_max must be greater than y_min.")


# ============================================================
# SECTION 3 — Map pixels to complex numbers
# ============================================================
# Every pixel represents a sample point c = x + iy.
#
# Array convention:
#   - Columns progress from left to right.
#   - Rows progress from top to bottom.
#   - The first row corresponds to y_max.
#
# Both ends of each coordinate interval are included. If an axis
# contains only one pixel, that pixel samples the minimum bound.

def complex_grid(
    width: int,
    height: int,
    x_min: float = -2.2,
    x_max: float = 1.0,
    y_min: float = -1.2,
    y_max: float = 1.2,
) -> NDArray[np.complex128]:
    """
    Construct a 2-D array of complex-plane sample coordinates.

    Returns an array with shape (height, width).
    """

    # Reuse the same validation rules as the main algorithm.
    _validate_parameters(
        width,
        height,
        1,
        x_min,
        x_max,
        y_min,
        y_max,
    )

    # Generate the real coordinates from the left edge to the right.
    real = np.linspace(x_min, x_max, width, dtype=np.float64)

    # Reverse the vertical direction to match image coordinates.
    imag = np.linspace(y_max, y_min, height, dtype=np.float64)

    # Broadcasting combines the 1-D arrays into a complex 2-D grid.
    # No explicit nested Python loops are needed.
    grid = real[np.newaxis, :] + 1j * imag[:, np.newaxis]

    return np.asarray(grid, dtype=np.complex128)


# ============================================================
# SECTION 4 — Calculate Mandelbrot escape times
# ============================================================

def mandelbrot(
    width: int = 1200,
    height: int = 800,
    max_iterations: int = 500,
    x_min: float = -2.2,
    x_max: float = 1.0,
    y_min: float = -1.2,
    y_max: float = 1.2,
) -> FractalResult:
    """
    Calculate the Mandelbrot escape-time field for a rectangular region.

    Parameters
    ----------
    width, height:
        Number of horizontal and vertical sample points.

    max_iterations:
        Maximum number of recurrence steps for each orbit.

    x_min, x_max, y_min, y_max:
        Rectangular region of the complex plane.

    Returns
    -------
    FractalResult
        Iteration counts, escape mask and smooth escape values.
    """

    # --------------------------------------------------------
    # 4.1 Validate the requested calculation.
    # --------------------------------------------------------
    _validate_parameters(
        width,
        height,
        max_iterations,
        x_min,
        x_max,
        y_min,
        y_max,
    )

    # --------------------------------------------------------
    # 4.2 Assign a complex parameter to every pixel.
    # --------------------------------------------------------
    c = complex_grid(
        width,
        height,
        x_min,
        x_max,
        y_min,
        y_max,
    )

    # --------------------------------------------------------
    # 4.3 Initialize the orbit z_0 = 0.
    # --------------------------------------------------------
    # Each array position represents an independent orbit.
    z = np.zeros(c.shape, dtype=np.complex128)

    # Zero is a sentinel indicating that escape has not been observed.
    escape_iterations = np.zeros(c.shape, dtype=np.int32)

    # Every point begins active. Once it escapes, it is deactivated.
    active = np.ones(c.shape, dtype=np.bool_)

    # --------------------------------------------------------
    # 4.4 Apply the recurrence z_(n+1) = z_n**2 + c.
    # --------------------------------------------------------
    for iteration in range(1, max_iterations + 1):

        # If every point has escaped, there is nothing left to compute.
        if not np.any(active):
            break

        # Extract only the currently active orbits.
        active_indices = np.flatnonzero(active)
        z_active = z.flat[active_indices]
        c_active = c.flat[active_indices]

        # Perform one iteration of the quadratic polynomial.
        z_active = z_active * z_active + c_active

        # Write the new orbit values back into the full array.
        z.flat[active_indices] = z_active

        # ----------------------------------------------------
        # 4.5 Apply the mathematically justified escape criterion.
        # ----------------------------------------------------
        # |z| > 2 is equivalent to Re(z)^2 + Im(z)^2 > 4.
        # Squared magnitude avoids calculating a square root.
        magnitude_squared = (
            z_active.real * z_active.real
            + z_active.imag * z_active.imag
        )

        # Identify the points that escaped during this iteration.
        escaped_now = magnitude_squared > 4.0

        # Record their first escape iteration.
        newly_escaped_indices = active_indices[escaped_now]
        escape_iterations.flat[newly_escaped_indices] = iteration

        # These points must never be iterated again.
        active.flat[newly_escaped_indices] = False

    # --------------------------------------------------------
    # 4.6 Distinguish observed escape from non-escape.
    # --------------------------------------------------------
    # A point can remain active when we exhaust the iteration budget.
    # That does not prove the orbit remains bounded forever.
    escaped = ~active

    # --------------------------------------------------------
    # 4.7 Calculate smooth escape-time estimates.
    # --------------------------------------------------------
    # Integer iteration counts produce visible bands when colored.
    #
    # For escaped points, a common smooth approximation is:
    #
    # mu = n + 1 - log2(log(|z_n|))
    #
    # This is useful for visual interpolation. It is not an exact
    # escape time or an exact distance to the fractal boundary.
    smooth_iterations = np.full(
        c.shape,
        np.nan,
        dtype=np.float64,
    )

    # Only evaluate logarithms for points that have actually escaped.
    escaped_magnitudes = np.abs(z[escaped])

    # Every selected magnitude exceeds 2, so the nested logarithm
    # is mathematically defined for these finite values.
    smooth_iterations[escaped] = (
        escape_iterations[escaped].astype(np.float64)
        + 1.0
        - np.log2(np.log(escaped_magnitudes))
    )

    # --------------------------------------------------------
    # 4.8 Package the results for the rendering and analysis layers.
    # --------------------------------------------------------
    return FractalResult(
        escape_iterations=escape_iterations,
        escaped=escaped,
        smooth_iterations=smooth_iterations,
        max_iterations=max_iterations,
    )

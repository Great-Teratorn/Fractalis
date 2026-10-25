
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
from numba import njit
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



# ============================================================
# SECTION 4 — Compiled Mandelbrot calculation kernel
# ============================================================
# Numba compiles this ordinary Python function into machine code.
# cache=True stores compiled specializations on disk when supported,
# reducing startup compilation work in subsequent application runs.
#
# IMPORTANT:
# - Keep this function focused on numerical work.
# - Do not put Streamlit, Plotly, or session-state operations here.
# - The public mandelbrot() function below remains the validated API.

@njit(cache=True)
def _mandelbrot_kernel(real, imag, max_iterations):
    """Calculate escape counts and smooth values using compiled loops."""

    # The coordinate arrays define the exact pixel sampling:
    # real runs from left to right; imag runs from top to bottom.
    height = imag.size
    width = real.size

    # Zero means that the orbit did not escape within the iteration
    # budget. This preserves the existing public result convention.
    escape_iterations = np.zeros((height, width), dtype=np.int32)

    # Non-escaped pixels retain NaN for smooth colouring.
    smooth_iterations = np.full(
        (height, width), np.nan, dtype=np.float64
    )

    # Visit each pixel directly. Unlike the previous implementation,
    # this does not repeatedly allocate active-index arrays or copy
    # the active portions of the complex orbit array.
    for row in range(height):
        ci = imag[row]

        for column in range(width):
            cr = real[column]

            # Start the Mandelbrot orbit at z = 0 + 0i.
            zr = 0.0
            zi = 0.0

            # Iterate this pixel until it escapes or reaches the
            # configured maximum iteration count.
            for iteration in range(1, max_iterations + 1):

                # Complex squaring:
                # (zr + zi*i)^2 = (zr^2 - zi^2) + (2*zr*zi)*i
                #
                # Save the old real component before updating either
                # component. Both updates must use the old orbit.
                old_zr = zr
                old_zi = zi

                zr = old_zr * old_zr - old_zi * old_zi + cr
                zi = 2.0 * old_zr * old_zi + ci

                # A point escapes when |z| > 2, equivalently when
                # zr^2 + zi^2 > 4. Squared magnitude avoids a
                # square root for every iteration.
                magnitude_squared = zr * zr + zi * zi

                if magnitude_squared > 4.0:
                    # Record the FIRST iteration at which escape
                    # occurs, matching the original engine.
                    escape_iterations[row, column] = iteration

                    # Smooth escape-time colouring reduces the
                    # visible bands produced by integer counts.
                    #
                    # mu = n + 1 - log2(log(|z|))
                    #
                    # The square root recovers |z| from its squared
                    # magnitude. This formula is only evaluated for
                    # points that actually escaped.
                    smooth_iterations[row, column] = (
                        iteration
                        + 1.0
                        - np.log(np.log(np.sqrt(magnitude_squared)))
                        / np.log(2.0)
                    )

                    # Escaped orbits never need another iteration.
                    break

    # Return plain NumPy arrays; the wrapper constructs the public
    # FractalResult object and its boolean escape mask.
    return escape_iterations, smooth_iterations



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

    The public interface is intentionally unchanged. Existing callers,
    including the Streamlit explorer and its cache, need no modifications.
    """

    # Validate dimensions, iteration count, and coordinate bounds before
    # allocating arrays or invoking the compiled numerical kernel.
    _validate_parameters(
        width, height, max_iterations,
        x_min, x_max, y_min, y_max,
    )

    # Use the same inclusive coordinate sampling as complex_grid().
    # This preserves the location represented by every pixel.
    real = np.linspace(x_min, x_max, width, dtype=np.float64)
    imag = np.linspace(y_max, y_min, height, dtype=np.float64)

    # Run the compiled calculation. The very first invocation may take
    # longer because Numba must compile this function for these input
    # types; later invocations can reuse the compiled version.
    escape_iterations, smooth_iterations = _mandelbrot_kernel(
        real, imag, max_iterations
    )

    # The original engine considered a point escaped exactly when its
    # recorded iteration count was positive. Preserve that convention.
    escaped = escape_iterations > 0

    return FractalResult(
        escape_iterations=escape_iterations,
        escaped=escaped,
        smooth_iterations=smooth_iterations,
        max_iterations=max_iterations,
    )

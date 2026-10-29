"""Reference arbitrary-precision Mandelbrot engine for deep zooms.

Coordinates are accepted as decimal strings so callers can preserve
precision beyond ordinary Python floats. This implementation prioritizes
numerical correctness over speed and is intended for modest test images.
"""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from math import ceil, log10

import mpmath as mp
import numpy as np
from numba import njit
from numpy.typing import NDArray


@dataclass(frozen=True)
class DeepFractalResult:
    """Escape-time fields produced by the deep-zoom reference engine."""

    escape_iterations: NDArray[np.int32]
    escaped: NDArray[np.bool_]
    smooth_iterations: NDArray[np.float64]
    max_iterations: int


def _decimal_value(name: str, value: str) -> Decimal:
    """Validate a decimal-string parameter without converting through float."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be supplied as a decimal string.")

    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError):
        raise ValueError(f"{name} must be a valid decimal number.") from None

    if not result.is_finite():
        raise ValueError(f"{name} must be finite.")

    return result


def _working_precision(center_x: str, center_y: str, span_x: str) -> int:
    """Choose precision from both the absolute centre and requested span.

    The span determines how many decimal places are needed to distinguish
    neighbouring samples. The centre determines how many significant digits
    must survive when tiny offsets are added to a nonzero coordinate.
    Guard digits protect intermediate coordinate and orbit calculations.
    """
    cx = _decimal_value("center_x", center_x)
    cy = _decimal_value("center_y", center_y)
    span = _decimal_value("span_x", span_x)

    if span <= 0:
        raise ValueError("span_x must be positive.")

    # Account for long decimal centres as well as a tiny viewport around
    # a centre whose magnitude is much larger than the viewport span.
    centre_digits = max(
        len(cx.as_tuple().digits),
        len(cy.as_tuple().digits),
        1,
    )
    separation_digits = max(
        0,
        cx.adjusted() - span.adjusted() if cx else 0,
        cy.adjusted() - span.adjusted() if cy else 0,
    )

    return max(50, centre_digits + 30, separation_digits + 40)


def deep_coordinate_grid(
    width: int,
    height: int,
    center_x: str,
    center_y: str,
    span_x: str,
) -> tuple[list[mp.mpf], list[mp.mpf]]:
    """Return high-precision x coordinates and top-to-bottom y coordinates.

    Width and height greater than one include both ends of each interval.
    A one-pixel dimension samples the centre of that interval.
    """
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    cx_d = _decimal_value("center_x", center_x)
    cy_d = _decimal_value("center_y", center_y)
    span_d = _decimal_value("span_x", span_x)

    if span_d <= 0:
        raise ValueError("span_x must be positive.")

    # Derive precision from both the centre's significant digits and
    # the span. A tiny span around a long decimal centre needs enough
    # working digits to retain the distinction between adjacent samples.
    precision = _working_precision(center_x, center_y, span_x)

    with mp.workdps(precision):
        cx = mp.mpf(str(cx_d))
        cy = mp.mpf(str(cy_d))
        sx = mp.mpf(str(span_d))
        sy = sx * height / width

        def axis(center: mp.mpf, span: mp.mpf, count: int, reverse=False):
            if count == 1:
                return [center]

            low = center - span / 2
            step = span / (count - 1)
            values = [low + i * step for i in range(count)]
            return list(reversed(values)) if reverse else values

        return axis(cx, sx, width), axis(cy, sy, height, reverse=True)



@njit(cache=True)
def _fast_orbit_kernel(xs, ys, max_iterations):
    """Compiled escape-time calculation for safely representable coordinates."""
    height = len(ys)
    width = len(xs)
    counts = np.zeros((height, width), dtype=np.int32)
    smooth = np.full((height, width), np.nan, dtype=np.float64)

    log_two = np.log(2.0)

    for row in range(height):
        ci = ys[row]
        for column in range(width):
            cr = xs[column]
            zr = 0.0
            zi = 0.0

            for iteration in range(1, max_iterations + 1):
                old_zr = zr
                old_zi = zi
                zr = old_zr * old_zr - old_zi * old_zi + cr
                zi = 2.0 * old_zr * old_zi + ci
                magnitude_squared = zr * zr + zi * zi

                if magnitude_squared > 4.0:
                    counts[row, column] = iteration
                    magnitude = np.sqrt(magnitude_squared)
                    smooth[row, column] = (
                        iteration + 1.0
                        - np.log(np.log(magnitude)) / log_two
                    )
                    break

    return counts, smooth


def deep_mandelbrot(
    width: int = 160,
    height: int = 108,
    max_iterations: int = 300,
    *,
    center_x: str = "-0.5",
    center_y: str = "0",
    span_x: str = "3.2",
) -> DeepFractalResult:
    """Calculate a modest Mandelbrot image using arbitrary precision.

    The centre and span must be decimal strings. Output colours are
    floating-point estimates, but orbit calculations use mpmath precision.
    """
    for name, value in (
        ("width", width),
        ("height", height),
        ("max_iterations", max_iterations),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    # Validate coordinates before applying the rendering-size limit,
    # so invalid parameters always receive the appropriate error.
    xs, ys = deep_coordinate_grid(
        width, height, center_x, center_y, span_x
    )

    # Avoid accidentally launching a very expensive pure-Python calculation.
    if width * height > 8192:
        raise ValueError(
            "Reference deep-zoom engine is limited to 8192 pixels per call."
        )

    # Only use the compiled float64 kernel when the viewport has enough
    # separation between adjacent samples to remain numerically meaningful.
    span_decimal = _decimal_value("span_x", span_x)
    precision = _working_precision(center_x, center_y, span_x)

    safe_float_view = (
        span_decimal >= Decimal("1e-12")
        and abs(_decimal_value("center_x", center_x)) < Decimal("1e6")
        and abs(_decimal_value("center_y", center_y)) < Decimal("1e6")
    )

    if safe_float_view:
        xs_float = np.asarray([float(x) for x in xs], dtype=np.float64)
        ys_float = np.asarray([float(y) for y in ys], dtype=np.float64)
        counts, smooth = _fast_orbit_kernel(
            xs_float, ys_float, max_iterations
        )
    else:
        counts = np.zeros((height, width), dtype=np.int32)
        smooth = np.full((height, width), np.nan, dtype=np.float64)

        with mp.workdps(precision):
            four = mp.mpf(4)
            log_two = mp.log(2)

            for row, ci in enumerate(ys):
                for column, cr in enumerate(xs):
                    zr = mp.mpf(0)
                    zi = mp.mpf(0)

                    for iteration in range(1, max_iterations + 1):
                        old_zr = zr
                        old_zi = zi
                        zr = old_zr * old_zr - old_zi * old_zi + cr
                        zi = 2 * old_zr * old_zi + ci
                        magnitude_squared = zr * zr + zi * zi

                        if magnitude_squared > four:
                            counts[row, column] = iteration
                            magnitude = mp.sqrt(magnitude_squared)
                            smooth[row, column] = float(
                                iteration + 1
                                - mp.log(mp.log(magnitude)) / log_two
                            )
                            break

    return DeepFractalResult(
        escape_iterations=counts,
        escaped=counts > 0,
        smooth_iterations=smooth,
        max_iterations=max_iterations,
    )

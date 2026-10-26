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

    # Base precision on the requested scale, with guard digits for
    # arithmetic around the centre. Never convert the coordinates to float.
    precision = max(50, ceil(-span_d.adjusted()) + 30 if span_d < 1 else 50)

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
    if width * height > 4096:
        raise ValueError(
            "Reference deep-zoom engine is limited to 4096 pixels per call."
        )

    precision = max(
        50,
        ceil(-_decimal_value("span_x", span_x).adjusted()) + 30
        if _decimal_value("span_x", span_x) < 1
        else 50,
    )

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
                        value = (
                            iteration + 1
                            - mp.log(mp.log(magnitude)) / log_two
                        )
                        smooth[row, column] = float(value)
                        break

    return DeepFractalResult(
        escape_iterations=counts,
        escaped=counts > 0,
        smooth_iterations=smooth,
        max_iterations=max_iterations,
    )

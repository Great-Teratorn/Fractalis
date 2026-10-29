"""Precision-preserving viewport arithmetic for FRACTALIS.

The viewport stores its centre and horizontal span as decimal strings.
This avoids converting deep coordinates through binary floating point
during navigation, history operations, or engine selection.

Plotly offsets are relative to the viewport centre in deep mode. They
must be converted to Decimal from their textual representation before
being added to the canonical centre.
"""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext


@dataclass(frozen=True)
class Viewport:
    """An immutable, precision-preserving rectangle in the complex plane."""

    center_x: str = "-0.5"
    center_y: str = "0"
    span_x: str = "3.2"

    def __post_init__(self):
        # Normalize and validate values without ever routing through float.
        for name in ("center_x", "center_y", "span_x"):
            value = _decimal(getattr(self, name), name)
            if name == "span_x" and value <= 0:
                raise ValueError("span_x must be positive")
            object.__setattr__(self, name, _format_decimal(value))

    @property
    def center(self):
        """Return the canonical centre as Decimal values."""
        return Decimal(self.center_x), Decimal(self.center_y)

    @property
    def span(self):
        """Return the canonical horizontal span as a Decimal."""
        return Decimal(self.span_x)

    def zoom(self, factor: str) -> "Viewport":
        """Scale the viewport around its existing centre."""
        factor_value = _decimal(factor, "factor")
        if factor_value <= 0:
            raise ValueError("factor must be positive")

        with localcontext() as ctx:
            ctx.prec = _precision(self.center_x, self.center_y,
                                  self.span_x, factor)
            new_span = self.span * factor_value

        return Viewport(self.center_x, self.center_y,
                        _format_decimal(new_span))

    def pan(self, offset_x: str, offset_y: str) -> "Viewport":
        """Move the centre by relative complex-plane offsets."""
        dx = _decimal(offset_x, "offset_x")
        dy = _decimal(offset_y, "offset_y")

        with localcontext() as ctx:
            ctx.prec = _precision(self.center_x, self.center_y,
                                  self.span_x, offset_x, offset_y)
            cx = Decimal(self.center_x) + dx
            cy = Decimal(self.center_y) + dy

        return Viewport(_format_decimal(cx), _format_decimal(cy),
                        self.span_x)

    def with_center(self, center_x: str, center_y: str) -> "Viewport":
        """Change the centre while retaining the current horizontal span."""
        return Viewport(center_x, center_y, self.span_x)

    def vertical_span(self, width: int, height: int) -> str:
        """Derive vertical span from the image's exact aspect ratio."""
        if width < 1 or height < 1:
            raise ValueError("width and height must be positive")

        with localcontext() as ctx:
            ctx.prec = _precision(self.span_x)
            return _format_decimal(self.span * Decimal(height) / Decimal(width))


def _decimal(value, name: str) -> Decimal:
    """Accept decimal strings or Decimal values, never binary floats."""
    if isinstance(value, float) or not isinstance(value, (str, Decimal)):
        raise ValueError(f"{name} must be a decimal string or Decimal")
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError):
        raise ValueError(f"{name} must be a valid decimal number") from None
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def _format_decimal(value: Decimal) -> str:
    """Format decimals consistently without losing significant digits.

    Fixed-point formatting keeps coordinates readable and avoids exponent
    notation in the UI. Remove only fractional trailing zeroes so that
    mathematically identical spans have identical canonical strings.
    """
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def _precision(*values: str) -> int:
    """Choose working precision from the smallest scale and largest centre.

    Significant centre digits matter as much as viewport span: adding a
    tiny offset to a long decimal centre needs guard digits to preserve
    that offset. The additional margin protects chained navigation.
    """
    decimals = [Decimal(value) for value in values]
    smallest_exponent = min(
        (value.adjusted() if value else 0) for value in decimals
    )
    largest_digits = max(
        len(value.as_tuple().digits) for value in decimals
    )
    return max(50, largest_digits + max(0, -smallest_exponent) + 20)


def requires_deep_precision(
    center_x: str,
    center_y: str,
    span_x: str,
    width: int,
) -> bool:
    """Return whether float64 spacing is too coarse for adjacent samples.

    Compare the viewport's pixel spacing with the ULP scale of its
    absolute centre. This is a conservative engine-selection heuristic,
    not a guarantee that the standard orbit calculation is accurate.
    """
    if width < 2:
        raise ValueError("width must be at least 2")

    cx = _decimal(center_x, "center_x")
    cy = _decimal(center_y, "center_y")
    span = _decimal(span_x, "span_x")
    if span <= 0:
        raise ValueError("span_x must be positive")

    # Decimal exponents provide a conservative estimate of the scale
    # at which binary float coordinates lose distinguishable spacing.
    centre_scale = max(abs(cx), abs(cy))
    if centre_scale == 0:
        return False

    # float64 has about 16 decimal significant digits. Trigger deep mode
    # before pixel spacing approaches the centre's representable spacing.
    pixel_spacing = span / Decimal(width - 1)
    float_spacing_estimate = centre_scale * Decimal("2.220446049250313e-16")
    return pixel_spacing <= float_spacing_estimate

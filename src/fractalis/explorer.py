
"""
FRACTALIS — Interactive Mandelbrot Explorer
============================================

This module adds an interactive exploration layer around the existing
mathematical engine in core.py.

Design principles:
1. Keep the numerical engine separate from the user interface.
2. Treat every view as a rectangle in the complex plane.
3. Make zooming a change of coordinates, not a stretched image.
4. Cache calculations so returning to a previous view is inexpensive.
5. Explain numerical limitations instead of claiming literal infinity.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import plotly.graph_objects as go


# Use Plotly's established scientific colour maps directly.
# These are separate, predefined gradients rather than approximations.


import plotly.express as px

import plotly.colors as plotly_colors

FRACTALIS_PALETTES = {
    "magma": [
        [0.00, "#10072b"],
        [0.16, "#35106b"],
        [0.32, "#612078"],
        [0.48, "#a52f83"],
        [0.64, "#e65383"],
        [0.80, "#ff9368"],
        [0.92, "#ffd080"],
        [1.00, "#fff4c2"],
    ],
    "inferno": [
        [0.00, "#000000"],
        [0.16, "#16051c"],
        [0.32, "#50102f"],
        [0.48, "#9c1830"],
        [0.64, "#e33325"],
        [0.80, "#ff7017"],
        [0.92, "#ffc126"],
        [1.00, "#fff95e"],
    ],
    # Aurora: cool teal, blue and violet tones with a bright mint highlight.
    # These stops are interpolated by Plotly to create continuous gradients.
    "aurora": [
        [0.00, "#07111f"],
        [0.16, "#123e59"],
        [0.32, "#087e8b"],
        [0.48, "#35b7a5"],
        [0.64, "#7778c8"],
        [0.82, "#c18bdf"],
        [1.00, "#e5fff4"],
    ],
    # Electric: high-contrast navy, blue, magenta and gold.
    # This offers a warmer, more dramatic alternative to the existing palettes.
    "electric": [
        [0.00, "#050510"],
        [0.16, "#151a70"],
        [0.32, "#164ee8"],
        [0.48, "#00bcd4"],
        [0.64, "#a52aff"],
        [0.82, "#ff4f9a"],
        [1.00, "#fff0a6"],
    ],
}

import streamlit as st
from streamlit_plotly_events2 import plotly_events

from fractalis.core import mandelbrot
from fractalis.deep_zoom import deep_mandelbrot
from fractalis.viewport import Viewport


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
# Streamlit uses this information to configure the browser tab
# and the overall layout of the application.

st.set_page_config(
    page_title="FRACTALIS | Explorer",
    page_icon="🌀",
    layout="wide",
)


# ------------------------------------------------------------
# VISUAL IDENTITY
# ------------------------------------------------------------
# Keep the interface dark so the fractal's colours stand out.
# CSS is presentation only; it does not affect the mathematics.

st.markdown(
    """
    <style>
/* Fractalis: readable buttons before hover. */
div.stButton > button {
    background: linear-gradient(135deg, #211a35, #302044) !important;
    color: #f5efff !important;
    border: 1px solid #7255a4 !important;
    font-weight: 600 !important;
    transition: background 180ms ease, border-color 180ms ease,
                box-shadow 180ms ease, transform 180ms ease !important;
}
div.stButton > button p {
    color: #f5efff !important;
}
div.stButton > button:hover {
    color: #ffffff !important;
    border-color: #c6a7ff !important;
    box-shadow: 0 0 16px rgba(166, 115, 255, 0.28) !important;
    transform: translateY(-1px);
}
div.stButton > button:hover p {
    color: #ffffff !important;
}

    .stApp {
        background:
            radial-gradient(ellipse at 50% 0%, #211438 0%, #090911 65%);
        color: #f4efff;
    }
    .block-container {
        max-width: 1600px;
        padding-top: 2rem;
    }
    h1 {
        letter-spacing: .18em;
        font-weight: 800 !important;
    }
    [data-testid="stMetric"] {
        background: #151323;
        border: 1px solid #38264d;
        border-radius: 12px;
        padding: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# MATHEMATICAL CALCULATION CACHE
# ------------------------------------------------------------
# Streamlit caches results by function arguments.
# Revisiting exactly the same region with the same resolution
# and iteration count can reuse the previous calculation.

@st.cache_data(max_entries=12, show_spinner=False)
def calculate_view(
    width,
    height,
    iterations,
    x_min,
    x_max,
    y_min,
    y_max,
):
    """Calculate one rectangular region of the Mandelbrot plane."""

    return mandelbrot(
        width=width,
        height=height,
        max_iterations=iterations,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


@st.cache_data(max_entries=12, show_spinner=False)
def calculate_deep_view(width, height, iterations, center_x, center_y, span_x):
    """Cache a deep render by its exact decimal-string viewport.

    Keeping the coordinate arguments as strings prevents Streamlit's cache
    key from being based on rounded float approximations.
    """
    return deep_mandelbrot(
        width=width,
        height=height,
        max_iterations=iterations,
        center_x=center_x,
        center_y=center_y,
        span_x=span_x,
    )


# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
# Session state survives Streamlit reruns.
# Without it, clicking a button would reset the current viewpoint.

DEFAULTS = {
    "center_x": -0.5,
    "center_y": 0.0,
    "span_x": 3.2,
    "zoom_history": [],
    "selected_palette": "magma",
    "render_mode": "Standard",
    "deep_center_x": "-0.5",
    "deep_center_y": "0",
    "deep_span_x": "3.2",
}

for key, default_value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# Avoid viewport widths where float64 coordinate precision becomes unreliable.
MIN_SPAN_X = 1e-12


# ------------------------------------------------------------
# NAVIGATION FUNCTIONS
# ------------------------------------------------------------

def _deep_view():
    """Read the canonical arbitrary-precision viewport from session state."""
    return Viewport(
        st.session_state.deep_center_x,
        st.session_state.deep_center_y,
        st.session_state.deep_span_x,
    )


def _store_deep_view(view):
    """Persist a viewport without converting its coordinates to floats."""
    st.session_state.deep_center_x = view.center_x
    st.session_state.deep_center_y = view.center_y
    st.session_state.deep_span_x = view.span_x


def save_view():
    """Save the active viewport, retaining exact deep-mode coordinates."""
    if st.session_state.render_mode == "Arbitrary precision":
        view = _deep_view()
        st.session_state.zoom_history.append(
            ("deep", view.center_x, view.center_y, view.span_x)
        )
    else:
        st.session_state.zoom_history.append(
            ("standard", st.session_state.center_x,
             st.session_state.center_y, st.session_state.span_x)
        )


def zoom(factor):
    """Zoom around the current centre using the active numerical model."""
    if st.session_state.render_mode == "Arbitrary precision":
        save_view()
        # String factors keep navigation arithmetic out of binary floats.
        factor_text = "0.5" if factor < 1 else "2"
        _store_deep_view(_deep_view().zoom(factor_text))
        return

    new_span = st.session_state.span_x * factor
    if new_span < MIN_SPAN_X:
        st.warning(
            "The standard renderer has reached its reliable coordinate "
            "limit. Switch to Arbitrary precision to explore further."
        )
        return
    save_view()
    st.session_state.span_x = new_span


def reset_view():
    """Reset both engine viewports and clear navigation history."""
    st.session_state.center_x = -0.5
    st.session_state.center_y = 0.0
    st.session_state.span_x = 3.2

    st.session_state.deep_center_x = "-0.5"
    st.session_state.deep_center_y = "0"
    st.session_state.deep_span_x = "3.2"

    st.session_state.zoom_history.clear()


# ------------------------------------------------------------
# APPLICATION HEADER
# ------------------------------------------------------------

st.title("F R A C T A L I S")
st.caption("MATHEMATICAL EXPLORATION · COMPLEX DYNAMICS · INFINITE DETAIL")

st.markdown(
    "Navigate the boundary between order and chaos. "
    "Every new view is calculated from the underlying equation."
)


# ------------------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------------------

with st.sidebar:
    st.header("EXPLORATION")

    st.selectbox(
        "Numerical engine",
        ["Standard", "Arbitrary precision"],
        key="render_mode",
        help=(
            "Standard mode uses the fast NumPy engine. "
            "Arbitrary precision uses mpmath and decimal-string "
            "coordinates for deep exploration."
        ),
    )

       # Expose every built-in Plotly colour scale and its reverse.
    # Keep FRACTALIS's custom palettes available as well.
    builtin_palettes = px.colors.named_colorscales()

    palette_options = sorted(
        set(
            builtin_palettes
            + [f"{name}_r" for name in builtin_palettes]
            + list(FRACTALIS_PALETTES.keys())
        )
    )

    st.selectbox(
        "Colour palette",
        palette_options,
        key="selected_palette",
        help="Choose from the full colour collection. Scales ending in _r run in reverse.",
    )

    # Optional lighting effect derived from the calculated fractal.
    # It changes display colouring only; it does not recalculate or
    # modify the underlying complex coordinates or escape-time results.
    boundary_shading = st.checkbox(
        "Boundary shading",
        value=False,
        help=(
            "Add subtle contrast around intricate fractal boundaries. "
            "Disable it to restore the original smooth colour rendering."
        ),
    )

    # Higher sample counts can reveal finer structures at the same
    # viewport size. They increase calculation time and memory usage,
    # so retain 800 as the default for responsive everyday exploration.
    resolution = st.select_slider(
        "Horizontal resolution",
        options=[400, 600, 800, 1000, 1200, 1600],
        value=800,
    )

    iterations = st.slider(
        "Maximum iterations",
        min_value=50,
        max_value=2000,
        value=300,
        step=50,
    )

    st.divider()

    st.markdown("**PRECISION NOTE**")
    st.caption(
        "The current engine uses complex128 floating-point arithmetic. "
        "At extreme magnifications, distinct coordinates eventually "
        "become indistinguishable. Arbitrary-precision deep zoom is a "
        "separate numerical-engine upgrade."
    )


# ------------------------------------------------------------
# NAVIGATION TOOLBAR
# ------------------------------------------------------------
# These buttons change the complex-plane viewport.
# The next Streamlit rerun calculates the newly requested region.

left, middle, right, reset = st.columns([1, 1, 1, 2])

with left:
    if st.button("＋ Zoom in", use_container_width=True):
        zoom(0.5)

with middle:
    if st.button("－ Zoom out", use_container_width=True):
        zoom(2.0)

with right:
    if st.button("↶ Back", use_container_width=True):
        if st.session_state.zoom_history:
            previous = st.session_state.zoom_history.pop()

            # New history entries are tagged so standard and deep views
            # retain their original representations.
            if len(previous) == 4 and previous[0] == "deep":
                _, cx, cy, span = previous
                _store_deep_view(Viewport(cx, cy, span))
            elif len(previous) == 4 and previous[0] == "standard":
                _, cx, cy, span = previous
                st.session_state.center_x = cx
                st.session_state.center_y = cy
                st.session_state.span_x = span
            else:
                # Accept older three-value history entries from sessions
                # created before this integration patch.
                (
                    st.session_state.center_x,
                    st.session_state.center_y,
                    st.session_state.span_x,
                ) = previous


with reset:
    if st.button("⌂ Reset universe", use_container_width=True):
        reset_view()


# ------------------------------------------------------------
# MAP THE CURRENT VIEW TO COMPLEX-PLANE COORDINATES
# ------------------------------------------------------------

deep_mode = st.session_state.render_mode == "Arbitrary precision"

if deep_mode:
    width, height = 64, 64
    deep_center_x = st.session_state.deep_center_x
    deep_center_y = st.session_state.deep_center_y
    deep_span_x = st.session_state.deep_span_x
    effective_iterations = iterations

    with st.spinner("Calculating with arbitrary precision..."):
        result = calculate_deep_view(
            width,
            height,
            effective_iterations,
            deep_center_x,
            deep_center_y,
            deep_span_x,
        )

    # Plot normalized local coordinates: absolute coordinates at deep
    # zoom scales cannot safely be represented by browser-side numbers.
    x_values = np.linspace(-0.5, 0.5, width)
    y_half_span = height / (2.0 * width)
    y_values = np.linspace(y_half_span, -y_half_span, height)

    center_x, center_y, span_x = -0.5, 0.0, 1.0

else:
    width = resolution
    height = max(1, round(width * 0.68))

    center_x = st.session_state.center_x
    center_y = st.session_state.center_y
    span_x = max(MIN_SPAN_X, st.session_state.span_x)
    st.session_state.span_x = span_x

    if not np.isfinite(span_x) or span_x <= 0:
        st.error("The requested zoom is outside the supported numerical range.")
        st.stop()

    base_magnification = 3.2 / span_x
    zoom_levels = max(
        0.0, float(np.log2(max(1.0, base_magnification)))
    )
    iteration_multiplier = min(1.5, 1.0 + 0.10 * zoom_levels)
    effective_iterations = min(
        4000, max(iterations, round(iterations * iteration_multiplier))
    )

    span_y = span_x * height / width
    x_min = center_x - span_x / 2
    x_max = center_x + span_x / 2
    y_min = center_y - span_y / 2
    y_max = center_y + span_y / 2

    with st.spinner("Calculating the complex plane..."):
        result = calculate_view(
            width, height, effective_iterations,
            x_min, x_max, y_min, y_max,
        )

    x_values = np.linspace(x_min, x_max, width)
    y_values = np.linspace(y_max, y_min, height)


# BUILD AND DISPLAY THE FRACTAL

# Escaped points receive their smooth escape-time colour.
# Points that did not escape are shown against the dark background.
# Plotly only needs display precision for the final colour field.
# Keep the underlying Mandelbrot calculation and complex-plane
# coordinates at float64/complex128 precision for accurate zooming.
#
# float32 halves the colour array's memory footprint and, in our
# benchmark, substantially reduced Plotly figure creation time and
# JSON payload size. NaN values are preserved for non-escaped points.
colour_field = np.where(
    result.escaped,
    result.smooth_iterations,
    np.nan,
).astype(np.float32)

# ------------------------------------------------------------
# OPTIONAL BOUNDARY SHADING
# ------------------------------------------------------------
# Estimate local changes in the smooth escape-time field using
# neighbouring pixels. Larger changes often occur near intricate
# boundaries, where a little extra contrast can reveal structure.
#
# The field is normalized only for this visual effect. The original
# escape iterations, smooth values, and coordinate precision remain
# untouched. This is a display enhancement, not a distance estimate.
if boundary_shading:
    # Only escaped points have valid smooth escape-time values.
    valid = result.escaped & np.isfinite(colour_field)

    if np.any(valid):
        # Normalize valid colour values to a consistent 0-to-1 range.
        # Handle a constant field safely to avoid division by zero.
        low = float(np.min(colour_field[valid]))
        high = float(np.max(colour_field[valid]))
        span = high - low

        if span > 0.0:
            normalized = (colour_field - low) / span
        else:
            normalized = np.zeros_like(colour_field, dtype=np.float32)

        # NaNs cannot be passed through np.gradient reliably. Fill
        # non-escaped pixels temporarily, then restore their NaNs
        # after shading so the dark interior background is preserved.
        gradient_input = np.where(valid, normalized, 0.0).astype(
            np.float32
        )

        # Estimate horizontal and vertical changes in the colour field.
        # These arrays are display data, not additional fractal orbits.
        gradient_y, gradient_x = np.gradient(gradient_input)

        # Gradient magnitude indicates how quickly the field changes.
        # Normalize conservatively and clamp the lighting effect so
        # strong edges gain contrast without overwhelming the palette.
        edge_strength = np.sqrt(
            gradient_x * gradient_x + gradient_y * gradient_y
        )
        edge_strength = np.clip(edge_strength * 4.0, 0.0, 1.0)

        # Darken the steepest transitions by at most 18 percent.
        # This creates subtle boundary contrast without fake geometry.
        shaded = normalized * (1.0 - 0.18 * edge_strength)

        # Keep the display in a stable range across different views.
        # Restore NaN for all points that did not escape.
        colour_field = np.where(valid, shaded, np.nan).astype(np.float32)


# Construct the Plotly figure from the actual calculated data.
fig = go.Figure(
    data=[
        go.Heatmap(
            x=x_values,
            y=y_values,
            z=colour_field,
            zsmooth="fast",
            zmin=0.0 if boundary_shading else None,
            zmax=1.0 if boundary_shading else None,
            autocolorscale=False,
            colorscale=FRACTALIS_PALETTES.get(
                st.session_state.selected_palette,
                st.session_state.selected_palette,
            ),
            hovertemplate=(
                "Re(c): %{x:.12g}<br>"
                "Im(c): %{y:.12g}<br>"
                "Smooth escape value: %{z:.5f}"
                "<extra></extra>"
            ),
            colorbar={
                "title": {
                    "text": "Smooth<br>escape time",
                    "font": {"color": "#d8cbed"},
                },
                "tickfont": {"color": "#d8cbed"},
            },
            connectgaps=False,
            name="Escape-time field",
        )
    ]
)


# Style the graph and keep equal scales on both complex-plane axes.
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#090911",
    plot_bgcolor="#080810",
    height=700,
    margin={"l": 55, "r": 35, "t": 25, "b": 55},

    

    xaxis={
        "title": "Local real offset" if deep_mode else "Real axis · Re(c)",
        "gridcolor": "#242033",
        "zerolinecolor": "#65577c",
        "constrain": "domain",
    },
    yaxis={
        "title": "Local imaginary offset" if deep_mode else "Imaginary axis · Im(c)",
        "gridcolor": "#242033",
        "zerolinecolor": "#65577c",
        "scaleanchor": "x",
        "scaleratio": 1,
    },
    
)

# ------------------------------------------------------------
# INTERACTIVE FRACTAL DISPLAY
# ------------------------------------------------------------
# Use the verified updated Plotly event component as the only
# renderer. Do not also call st.plotly_chart() for this figure:
# rendering it twice would create duplicate charts and state.
#
# The key changes when the viewpoint changes. This gives each new
# view a fresh component identity and prevents the previous click
# from being replayed after the next zoom.

# Keep the interactive component identity stable across zooms.
# This gives Streamlit a chance to update the existing chart instead
# of destroying it and mounting a brand-new component each time.
#
# Because the component may retain its previous click event, the
# handler below also records handled coordinates to prevent replay.
canvas_key = (
    f"mandelbrot_click_canvas_"
    f"{st.session_state.get('_canvas_generation', 0)}"
)

# Render the completed figure directly, exactly once.
# Avoiding a temporary preview and placeholder replacement prevents
# the extra chart lifecycle that caused the visible transition.
# Keep the component key stable so existing zoom-event handling works.
clicked_points = plotly_events(
    fig,
    click_event=True,
    select_event=False,
    hover_event=False,
    override_height=700,
    override_width="100%",
    key=canvas_key,
)

# ------------------------------------------------------------
# Deep-mode axes are relative offsets, not absolute complex coordinates.
# Preserve the returned click event and translate it with Decimal arithmetic
# in the handler below. Never convert the absolute deep centre to float.

# CLICK-TO-ZOOM: MAP THE CLICK TO THE COMPLEX PLANE
# ------------------------------------------------------------
# Heatmap clicks return data coordinates. The x coordinate is the
# real part of c, and the y coordinate is its imaginary part.
#
# Recalculate the fractal around the clicked location at twice
# the magnification. Save the previous view so the Back button
# can restore the previous centre and scale.



if deep_mode and clicked_points:
    selected = clicked_points[0]
    selected_x = selected.get("x")
    selected_y = selected.get("y")

    if selected_x is not None and selected_y is not None:
        # Plotly's browser values are relative offsets. Convert their decimal
        # text, multiply by the exact span, then add to the exact centre.
        # This avoids ever constructing an absolute centre as a float.
        try:
            from decimal import Decimal, localcontext

            dx_fraction = Decimal(str(selected_x))
            dy_fraction = Decimal(str(selected_y))
            current = _deep_view()

            with localcontext() as ctx:
                ctx.prec = max(
                    80,
                    len(current.center_x) + len(current.span_x) + 30,
                    len(current.center_y) + len(current.span_x) + 30,
                )
                offset_x = dx_fraction * Decimal(current.span_x)
                offset_y = dy_fraction * Decimal(current.span_x)

            # A stable component may replay the previous click after reruns.
            # Record the event before changing state to prevent duplicate zooms.
            click_signature = (float(selected_x), float(selected_y))
            last_handled_click = st.session_state.get("_last_handled_click")

            if (
                click_signature != last_handled_click
                and (offset_x != 0 or offset_y != 0)
            ):
                st.session_state["_last_handled_click"] = click_signature
                save_view()
                moved = current.pan(str(offset_x), str(offset_y)).zoom("0.5")
                _store_deep_view(moved)
                st.rerun()
        except (ValueError, ArithmeticError):
            st.warning("That deep-zoom click could not be interpreted safely.")

elif clicked_points:
    selected = clicked_points[0]

    selected_x = selected.get("x")
    selected_y = selected.get("y")

    if selected_x is not None and selected_y is not None:
        selected_x = float(selected_x)
        selected_y = float(selected_y)

        # Reject invalid coordinates before changing session state.
        if np.isfinite(selected_x) and np.isfinite(selected_y):
            # Remember the most recently handled click coordinates.
            # A stable component key can preserve an old event during a
            # rerun, so ignore an event that we have already processed.
            click_signature = (selected_x, selected_y)
            last_handled_click = st.session_state.get(
                "_last_handled_click"
            )

            # Avoid both replaying the same click and zooming into the
            # current centre, which would not change the visible region.
            if (
                click_signature != last_handled_click
                and (selected_x != center_x or selected_y != center_y)
                and span_x > MIN_SPAN_X
            ):
                # Record the event BEFORE changing the view. This ensures
                # a rerun cannot process the same event a second time.
                st.session_state["_last_handled_click"] = click_signature

                save_view()
                st.session_state.center_x = selected_x
                st.session_state.center_y = selected_y
                st.session_state.span_x *= 0.5

                # Recalculate at the new coordinates.
                st.rerun()

# MATHEMATICAL STATUS
# ------------------------------------------------------------

metric_a, metric_b, metric_c = st.columns(3)

metric_a.metric(
    "Magnification",
    f"{3.2 / span_x:.2e}×" if not deep_mode else f"{3.2 / float(deep_span_x):.2e}×",
)

metric_b.metric(
    "Observed escape",
    f"{100 * result.escaped.mean():.2f}%",
)

metric_c.metric(
    "Iteration budget",
    f"{effective_iterations:,}",
    delta=(
        f"+{effective_iterations - iterations:,} adaptive"
        if effective_iterations > iterations
        else "Base budget"
    ),
)

if deep_mode:
    st.caption(
        f"Arbitrary-precision centre: {deep_center_x} "
        f"{deep_center_y}i · Horizontal span: {deep_span_x}"
    )
else:
    st.caption(
        f"Current centre: {center_x:.12g} "
        f"{center_y:+.12g}i · "
        f"Horizontal span: {span_x:.6g}"
    )

st.info(
    "Black areas indicate points that did not escape within the "
    "iteration budget. A finite calculation alone does not prove "
    "that an orbit remains bounded forever."
)

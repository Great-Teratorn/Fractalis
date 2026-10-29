
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
}

for key, default_value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# Avoid viewport widths where float64 coordinate precision becomes unreliable.
MIN_SPAN_X = 1e-12


# ------------------------------------------------------------
# NAVIGATION FUNCTIONS
# ------------------------------------------------------------

def save_view():
    """
    Save the current viewpoint before changing it.

    Each history entry contains the real centre coordinates and
    horizontal span, allowing us to return to the exact prior view.
    """

    st.session_state.zoom_history.append(
        (
            st.session_state.center_x,
            st.session_state.center_y,
            st.session_state.span_x,
        )
    )


def zoom(factor):
    """
    Zoom around the current centre.

    A factor below 1 zooms in.
    A factor above 1 zooms out.

    This changes the coordinate window itself. It does not merely
    enlarge an existing raster image.
    """

    new_span = st.session_state.span_x * factor
    if new_span < MIN_SPAN_X:
        st.warning("Maximum reliable zoom reached for the current numerical engine.")
        return
    save_view()
    st.session_state.span_x = new_span


def reset_view():
    """Return to the standard full Mandelbrot view."""

    st.session_state.center_x = -0.5
    st.session_state.center_y = 0.0
    st.session_state.span_x = 3.2
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
        "Colour palette",
        [
            "magma", "inferno", "viridis", "plasma", "turbo", "twilight",
            "aurora", "electric",
        ],
        key="selected_palette",
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
# The aspect ratio is preserved by deriving the vertical span
# from the horizontal span and the requested image dimensions.

width = resolution
height = max(1, round(width * 0.68))

center_x = st.session_state.center_x
center_y = st.session_state.center_y
span_x = st.session_state.span_x
if span_x < MIN_SPAN_X:
    span_x = MIN_SPAN_X
    st.session_state.span_x = span_x

# ------------------------------------------------------------
# ADAPTIVE ITERATION BUDGET
# ------------------------------------------------------------
# The user's slider defines the base iteration budget. At the default
# view, we use it unchanged so the initial render stays just as fast.
#
# As magnification increases, points near intricate boundaries often
# need more recurrence steps before their orbits escape. Gradually
# increasing the budget helps reveal these details at deeper zooms.
#
# Every doubling of magnification adds 0.25 to the multiplier, up to
# 2.5x. The cap prevents the adaptive budget from growing without limit.
# The effective value is passed into calculate_view(), so Streamlit's
# cache continues to distinguish results calculated at different budgets.
base_magnification = 3.2 / span_x
zoom_levels = max(0.0, float(np.log2(max(1.0, base_magnification))))

iteration_multiplier = min(
    1.5,
    1.0 + 0.10 * zoom_levels,
)

# Never exceed 4,000 iterations automatically, even if the slider's
# base value is already high. The user's slider remains the minimum.
effective_iterations = min(
    4000,
    max(iterations, round(iterations * iteration_multiplier)),
)

# Avoid requesting a zero-width region from floating-point underflow.
if not np.isfinite(span_x) or span_x <= 0:
    st.error("The requested zoom is outside the supported numerical range.")
    st.stop()

span_y = span_x * height / width

x_min = center_x - span_x / 2
x_max = center_x + span_x / 2
y_min = center_y - span_y / 2
y_max = center_y + span_y / 2


# ------------------------------------------------------------
# CALCULATE THE VIEW
# ------------------------------------------------------------
# The existing Mandelbrot engine does the actual mathematics.
# The interface never substitutes decorative graphics for data.

with st.spinner("Calculating the complex plane…"):
    result = calculate_view(
        width,
        height,
        effective_iterations,
        x_min,
        x_max,
        y_min,
        y_max,
    )


# ------------------------------------------------------------

# ------------------------------------------------------------

# ------------------------------------------------------------

# ------------------------------------------------------------

# ------------------------------------------------------------
# BUILD AND DISPLAY THE FRACTAL
# ------------------------------------------------------------
# Convert the current complex-plane region into pixel coordinates.
x_values = np.linspace(x_min, x_max, width)
y_values = np.linspace(y_max, y_min, height)

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
        "title": "Real axis · Re(c)",
        "gridcolor": "#242033",
        "zerolinecolor": "#65577c",
        "constrain": "domain",
    },
    yaxis={
        "title": "Imaginary axis · Im(c)",
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
canvas_key = "mandelbrot_click_canvas"

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
# CLICK-TO-ZOOM: MAP THE CLICK TO THE COMPLEX PLANE
# ------------------------------------------------------------
# Heatmap clicks return data coordinates. The x coordinate is the
# real part of c, and the y coordinate is its imaginary part.
#
# Recalculate the fractal around the clicked location at twice
# the magnification. Save the previous view so the Back button
# can restore the previous centre and scale.

if (
    isinstance(clicked_points, dict)
    and clicked_points.get("event") == "relayout"
):
    data = clicked_points.get("data", {})

    def get_axis_range(axis):
        pair = data.get(f"{axis}.range")
        if pair is None:
            low = data.get(f"{axis}.range[0]")
            high = data.get(f"{axis}.range[1]")
            if low is not None and high is not None:
                pair = [low, high]
        if pair is None or len(pair) != 2:
            return None
        try:
            values = sorted((float(pair[0]), float(pair[1])))
        except (TypeError, ValueError):
            return None
        if not all(np.isfinite(v) for v in values):
            return None
        return values if values[1] > values[0] else None

    xr = get_axis_range("xaxis")
    yr = get_axis_range("yaxis")

    if xr is not None and yr is not None:
        x0, x1 = xr
        y0, y1 = yr
        new_span_x = x1 - x0
        signature = (x0, x1, y0, y1)

        # Only treat a genuine inward rectangular zoom as a new view.
        if (
            MIN_SPAN_X <= new_span_x < span_x
            and signature != st.session_state.get("_last_handled_box")
        ):
            st.session_state["_last_handled_box"] = signature
            save_view()
            # Keep the viewport's aspect ratio consistent with the
            # numerical image dimensions. Plotly's reported y range
            # can differ from the intended range after axis constraints.
            st.session_state.center_x = (x0 + x1) / 2
            st.session_state.center_y = (y0 + y1) / 2
            st.session_state.span_x = new_span_x
            st.rerun()

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
    f"{3.2 / span_x:.2e}×",
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

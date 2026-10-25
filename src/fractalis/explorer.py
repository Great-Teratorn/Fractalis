
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

    save_view()
    st.session_state.span_x *= factor


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
        ["magma", "inferno", "viridis", "plasma", "turbo", "twilight"],
        key="selected_palette",
    )

    resolution = st.select_slider(
        "Horizontal resolution",
        options=[400, 600, 800, 1000, 1200],
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
        iterations,
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
colour_field = np.where(
    result.escaped,
    result.smooth_iterations,
    np.nan,
)

# Construct the Plotly figure from the actual calculated data.
fig = go.Figure(
    data=[
        go.Heatmap(
            x=x_values,
            y=y_values,
            z=colour_field,
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

canvas_key = (
    f"mandelbrot_click_{center_x:.12g}_"
    f"{center_y:.12g}_{span_x:.12g}"
)

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

if clicked_points:
    selected = clicked_points[0]

    selected_x = selected.get("x")
    selected_y = selected.get("y")

    if selected_x is not None and selected_y is not None:
        selected_x = float(selected_x)
        selected_y = float(selected_y)

        # Reject invalid coordinates before changing session state.
        if np.isfinite(selected_x) and np.isfinite(selected_y):
            # Avoid redundant reruns for a click exactly at the centre.
            if selected_x != center_x or selected_y != center_y:
                save_view()

                st.session_state.center_x = selected_x
                st.session_state.center_y = selected_y
                st.session_state.span_x *= 0.5

                # Trigger a fresh calculation at the new coordinates.
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
    f"{iterations:,}",
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

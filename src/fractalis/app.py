"""Fractalis — interactive fractal explorer."""

from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from fractalis.core import mandelbrot

st.set_page_config(
    page_title="FRACTALIS",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background: radial-gradient(ellipse at 50% 0%, #17152e 0%, #090a12 65%);
    color: #f4f1ff;
}
.block-container { max-width: 1500px; padding-top: 2rem; }
h1 {
    letter-spacing: .20em;
    font-weight: 800 !important;
}
.fractal-subtitle {
    color: #a8a4c8;
    letter-spacing: .12em;
    text-transform: uppercase;
    font-size: .8rem;
}
[data-testid="stMetric"] {
    background: #151525;
    border: 1px solid #302849;
    padding: 16px;
    border-radius: 12px;
}
.stButton button, .stDownloadButton button {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("F R A C T A L I S")
st.markdown(
    '<p class="fractal-subtitle">Explore the infinite · A mathematical universe</p>',
    unsafe_allow_html=True,
)

PRESETS = {
    "Mandelbrot set": (-0.5, 0.0, 1.7),
    "Seahorse valley": (-0.75, 0.1, 0.12),
    "Elephant valley": (0.28, 0.008, 0.08),
    "Triple spiral": (-0.088, 0.654, 0.025),
    "Deep boundary": (-0.235125, 0.827215, 0.008),
}

with st.sidebar:
    st.header("✳ Control room")
    preset = st.selectbox("Destination", list(PRESETS))
    default_x, default_y, default_span = PRESETS[preset]

    st.caption("Change the coordinates to explore the complex plane.")
    center_x = st.number_input(
        "Real coordinate (X)", value=float(default_x),
        format="%.8f", step=0.01,
        help="The horizontal position in the complex plane.",
    )
    center_y = st.number_input(
        "Imaginary coordinate (Y)", value=float(default_y),
        format="%.8f", step=0.01,
        help="The vertical position in the complex plane.",
    )
    span = st.number_input(
        "Horizontal view width", min_value=1e-9,
        max_value=4.0, value=float(default_span),
        format="%.8f",
        help="Smaller values zoom further into the fractal.",
    )

    width = st.select_slider(
        "Render quality",
        options=[500, 700, 900, 1100, 1400],
        value=900,
    )
    iterations = st.slider(
        "Iteration depth", min_value=50, max_value=1500,
        value=300, step=50,
    )
    palette = st.selectbox(
        "Colour universe",
        ["magma", "inferno", "twilight", "turbo", "icefire", "hot"],
        index=0,
    )
    render = st.button("✨ Generate fractal", type="primary", use_container_width=True)

    st.divider()
    st.caption("FRACTALIS · Built with Python + NumPy")

# A preset selection updates the default view. Keep manually edited
# coordinates intact until the user chooses another preset.
if "last_preset" not in st.session_state:
    st.session_state.last_preset = preset
if st.session_state.last_preset != preset:
    x, y, s = PRESETS[preset]
    st.session_state.last_preset = preset
    st.session_state.center_x = float(x)
    st.session_state.center_y = float(y)
    st.session_state.span = float(s)
    st.rerun()

# Initialise form values after preset changes.
st.session_state.setdefault("center_x", float(center_x))
st.session_state.setdefault("center_y", float(center_y))
st.session_state.setdefault("span", float(span))

# Cache expensive calculations so changing unrelated UI elements
# doesn't recompute an identical fractal.
@st.cache_data(show_spinner=False, max_entries=8)
def calculate_fractal(width, height, iterations, x_min, x_max, y_min, y_max):
    return mandelbrot(
        width=width,
        height=height,
        max_iterations=iterations,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

height = max(1, round(width * 0.68))
x_min = center_x - span / 2
x_max = center_x + span / 2
vertical_span = span * height / width
y_min = center_y - vertical_span / 2
y_max = center_y + vertical_span / 2

if render or "fractal_result" not in st.session_state:
    with st.spinner("Calculating the complex universe…"):
        result = calculate_fractal(
            width, height, iterations,
            x_min, x_max, y_min, y_max,
        )
        st.session_state.fractal_result = result
        st.session_state.fractal_settings = (
            width, height, iterations,
            x_min, x_max, y_min, y_max, palette,
        )

result = st.session_state.fractal_result
settings = st.session_state.fractal_settings

# If the user changes controls, regenerate on the next button press.
if settings[-1] != palette:
    st.info("Colour palette changed. Press **Generate fractal** to apply it.")
    active_palette = settings[-1]
else:
    active_palette = palette

col1, col2, col3 = st.columns(3)
col1.metric("Resolution", f"{settings[0]} × {settings[1]}")
col2.metric("Iteration depth", f"{settings[2]:,}")
col3.metric("Observed escape", f"{100 * result.escaped.mean():.1f}%")

fig, ax = plt.subplots(figsize=(13, 8), facecolor="#090a12")
ax.set_facecolor("#05050b")
cmap = plt.get_cmap(active_palette).copy()
cmap.set_bad("#05050b")

field = np.ma.masked_invalid(result.smooth_iterations)
image = ax.imshow(
    field,
    extent=(settings[3], settings[4], settings[5], settings[6]),
    origin="upper",
    cmap=cmap,
    interpolation="bilinear",
    aspect="equal",
)
ax.set_xlabel("Real axis", color="#aaa5c5")
ax.set_ylabel("Imaginary axis", color="#aaa5c5")
ax.tick_params(colors="#aaa5c5", labelsize=8)
for spine in ax.spines.values():
    spine.set_color("#302849")
fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Smooth escape time")
fig.axes[-1].yaxis.label.set_color("#aaa5c5")
fig.axes[-1].tick_params(colors="#aaa5c5")
fig.tight_layout(pad=2)

st.pyplot(fig, use_container_width=True)
st.caption(
    "Black regions did not escape within the selected iteration budget; "
    "this finite calculation does not prove every such orbit is bounded forever."
)

buffer = BytesIO()
fig.savefig(buffer, format="png", dpi=180, facecolor=fig.get_facecolor(),
            bbox_inches="tight")
plt.close(fig)

st.download_button(
    "⬇ Download fractal as PNG",
    data=buffer.getvalue(),
    file_name="fractalis.png",
    mime="image/png",
    type="primary",
)

with st.expander("About this calculation"):
    st.markdown("""
    Fractalis iterates the quadratic map

    \\[
    z_{n+1} = z_n^2 + c, \\qquad z_0 = 0
    \\]

    Each pixel represents a complex parameter \\(c\\). Colours show
    smooth escape-time estimates; black marks points that did not
    escape during the available iterations.
    """)

"""Centralized visual design for the dashboard.

A single source of truth for colors, the Plotly chart template and the CSS that
gives the Streamlit app a cohesive look. The palette is a validated,
colorblind-safe categorical set (blue / orange / aqua ...) with a reserved
status palette (good / warning / critical) that is never reused for series.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --- Palette -------------------------------------------------------------
# Categorical series colors, assigned in fixed order (never cycled past the set).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]

# Reserved status colors — meaning, not identity.
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"

# Sequential blue ramp for magnitude encodings.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

# Chart chrome & ink.
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'


def register_plotly_template() -> str:
    """Register and return the name of the shared Plotly template."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=FONT_FAMILY, color=INK_SECONDARY, size=13),
        title=dict(font=dict(color=INK, size=17), x=0.01, xanchor="left"),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=SERIES,
        margin=dict(l=56, r=24, t=52, b=44),
        xaxis=dict(
            gridcolor=GRID, zerolinecolor=GRID, linecolor=AXIS,
            tickcolor=AXIS, tickfont=dict(color=INK_MUTED), automargin=True,
        ),
        yaxis=dict(
            gridcolor=GRID, zerolinecolor=GRID, linecolor=AXIS,
            tickcolor=AXIS, tickfont=dict(color=INK_MUTED), automargin=True,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color=INK_SECONDARY),
        ),
        hoverlabel=dict(font=dict(family=FONT_FAMILY, size=12)),
    )
    pio.templates["inventory"] = template
    return "inventory"


def style_figure(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    """Apply the shared template and sensible defaults to a figure."""
    fig.update_layout(template="inventory", height=height)
    if title is not None:
        fig.update_layout(title=title)
    return fig


CSS = f"""
<style>
    :root {{
        --series-1: {SERIES[0]};
        --ink: {INK};
        --ink-secondary: {INK_SECONDARY};
        --surface: {SURFACE};
        --page: {PAGE};
        --grid: {GRID};
    }}
    .stApp {{ background: {PAGE}; }}
    .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}

    h1, h2, h3 {{ color: {INK}; font-family: {FONT_FAMILY}; letter-spacing: -0.01em; }}
    h1 {{ font-weight: 680; }}

    /* Hero header */
    .app-hero {{
        background: linear-gradient(135deg, {SERIES[0]} 0%, #1c5cab 100%);
        color: #fff; padding: 1.5rem 1.75rem; border-radius: 16px;
        margin-bottom: 1.4rem;
    }}
    .app-hero h1 {{ color: #fff; margin: 0; font-size: 1.7rem; }}
    .app-hero p {{ color: rgba(255,255,255,0.86); margin: .35rem 0 0; font-size: .95rem; }}

    /* Metric cards */
    [data-testid="stMetric"] {{
        background: {SURFACE};
        border: 1px solid {GRID};
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 2px rgba(11,11,11,0.04);
    }}
    [data-testid="stMetricLabel"] p {{
        color: {INK_MUTED}; font-size: .8rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: .04em;
    }}
    [data-testid="stMetricValue"] {{ color: {INK}; font-weight: 680; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: .35rem; border-bottom: 1px solid {GRID}; }}
    .stTabs [data-baseweb="tab"] {{
        font-weight: 600; color: {INK_MUTED};
        padding: .5rem 1rem; border-radius: 8px 8px 0 0;
    }}
    .stTabs [aria-selected="true"] {{ color: {SERIES[0]}; }}

    /* Buttons */
    .stButton > button {{
        border-radius: 9px; border: 1px solid {GRID}; font-weight: 600;
    }}
    .stButton > button[kind="primary"] {{
        background: {SERIES[0]}; border-color: {SERIES[0]};
    }}

    section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {GRID}; }}

    /* Cards for grouped content */
    .surface-card {{
        background: {SURFACE}; border: 1px solid {GRID}; border-radius: 12px;
        padding: 1.1rem 1.25rem; margin-bottom: 1rem;
    }}
    .caption-muted {{ color: {INK_MUTED}; font-size: .86rem; }}
</style>
"""

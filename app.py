"""
====================================================
App    : app.py
Project: SkyForecast — Airline Passenger Forecasting
Purpose: Streamlit front-end for the RNN/LSTM forecaster
====================================================
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import DataLoader
from src.evaluation import Evaluator
from src.forecast import Forecaster

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "lstm_model.keras"
SCALER_PATH = ROOT / "models" / "scaler.pkl"

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="SkyForecast · Passenger Forecast Console",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# PALETTE — "instrument panel" theme
# deep midnight surface + amber / teal dial glow
# ==================================================
BG_DEEP      = "#0A0F1A"
PANEL        = "#111A2B"
PANEL_HI     = "#152238"
BORDER       = "#223250"
INK          = "#E8EEF7"
INK_MUTE     = "#7C8CA6"
AMBER        = "#F2A63D"
AMBER_DIM    = "#8A5F22"
TEAL         = "#38B6A8"
TEAL_DIM     = "#1F6B63"
DANGER       = "#E2635B"
GOOD         = "#4FAE7C"

# ==================================================
# GLOBAL STYLE
# ==================================================
st.html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root{{
    --bg-deep:{BG_DEEP}; --panel:{PANEL}; --panel-hi:{PANEL_HI}; --border:{BORDER};
    --ink:{INK}; --ink-mute:{INK_MUTE};
    --amber:{AMBER}; --amber-dim:{AMBER_DIM};
    --teal:{TEAL}; --teal-dim:{TEAL_DIM};
    --danger:{DANGER}; --good:{GOOD};
}}

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}}

.stApp {{
    background:
        radial-gradient(1200px 500px at 15% -10%, rgba(56,182,168,0.08), transparent 60%),
        radial-gradient(1000px 500px at 100% 0%, rgba(242,166,61,0.07), transparent 55%),
        var(--bg-deep);
}}

h1, h2, h3, h4 {{
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--ink) !important;
    letter-spacing: 0.2px;
}}

code, .stMetricValue, div[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
}}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"]{{
    background: linear-gradient(180deg, #0D1424 0%, #080C15 100%);
    border-right: 1px solid var(--border);
}}
section[data-testid="stSidebar"] * {{ color: var(--ink) !important; }}
section[data-testid="stSidebar"] .caption-mute {{ color: var(--ink-mute) !important; }}
section[data-testid="stSidebar"] hr {{ border-top: 1px solid var(--border); }}

/* ---------- TABS ---------- */
.stTabs [data-baseweb="tab-list"]{{ gap: 4px; background: transparent; }}
.stTabs [data-baseweb="tab"]{{
    background: var(--panel);
    border: 1px solid var(--border);
    border-bottom: none;
    border-radius: 10px 10px 0 0;
    padding: 10px 18px;
    color: var(--ink-mute);
    font-weight: 500;
}}
.stTabs [aria-selected="true"]{{
    background: var(--panel-hi) !important;
    color: var(--amber) !important;
}}

/* ---------- BUTTON ---------- */
div.stButton > button:first-child{{
    background: linear-gradient(120deg, var(--amber) 0%, #D98A22 100%);
    color: #14100A;
    border: none;
    width: 100%;
    border-radius: 10px;
    height: 3.1em;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.4px;
    box-shadow: 0 8px 22px rgba(242,166,61,0.22);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
div.stButton > button:first-child:hover{{
    transform: translateY(-2px);
    box-shadow: 0 12px 26px rgba(242,166,61,0.32);
}}

/* ---------- DATAFRAMES / CHARTS ---------- */
div[data-testid="stDataFrame"], .stPlotlyChart{{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 6px;
}}

hr{{ border: none; border-top: 1px solid var(--border); margin: 28px 0; }}

.section-badge{{
    display:inline-block;
    background: var(--panel);
    color: var(--teal);
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 10px;
    border: 1px solid var(--border);
}}

.footnote {{ color: var(--ink-mute); font-size: 0.82rem; }}

/* ---------- GAUGE PANEL ---------- */
.gauge-row {{ display:flex; gap:18px; flex-wrap:wrap; }}
.gauge-card {{
    flex:1; min-width:200px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 18px 16px;
    text-align:center;
}}
.gauge-ring {{
    width:118px; height:118px; margin:0 auto 10px;
    border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    position:relative;
}}
.gauge-ring::before{{
    content:"";
    position:absolute; inset:10px;
    border-radius:50%;
    background: var(--panel);
}}
.gauge-value {{
    position:relative; z-index:2;
    font-family:'JetBrains Mono', monospace;
    font-weight:600; font-size:1.28rem; color:var(--ink);
    line-height:1.1;
}}
.gauge-delta {{
    position:relative; z-index:2;
    font-family:'JetBrains Mono', monospace;
    font-size:0.72rem; margin-top:2px;
}}
.gauge-label {{
    font-size:0.82rem; color:var(--ink-mute); margin-top:6px;
    text-transform:uppercase; letter-spacing:0.6px;
}}

iframe {{ border: none !important; }}
</style>
""")


# ==================================================
# COMPONENT: HERO
# ==================================================
def render_hero(df, pass_col):
    vals = df[pass_col].values[-48:]
    vmin, vmax = float(vals.min()), float(vals.max())
    span = (vmax - vmin) or 1.0

    w, h, pad = 760, 120, 8
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        x = pad + (w - 2 * pad) * (i / max(n - 1, 1))
        y = pad + (h - 2 * pad) * (1 - (v - vmin) / span)
        pts.append(f"{x:.1f},{y:.1f}")
    path_d = "M " + " L ".join(pts)

    st.html(f"""
    <div style="
        background:linear-gradient(135deg, {PANEL_HI} 0%, {PANEL} 100%);
        border:1px solid {BORDER}; border-radius:20px;
        padding:28px 32px; margin-bottom:8px;
        display:flex; align-items:center; justify-content:space-between; gap:24px;
        flex-wrap:wrap;
    ">
      <div style="min-width:260px;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:1.5px;
                    color:{TEAL}; text-transform:uppercase; margin-bottom:10px;">
          ● LIVE MODEL · LSTM SEQUENCE-TO-ONE
        </div>
        <h1 style="font-size:2.1rem; margin:0 0 8px;">SkyForecast Console</h1>
        <p style="color:{INK_MUTE}; max-width:440px; line-height:1.55; margin:0; font-size:0.98rem;">
          A recurrent network trained on decades of monthly airline traffic,
          forecasting the months ahead from the last 12 months of signal.
        </p>
      </div>
      <div style="flex:1; min-width:280px; max-width:420px;">
        <svg viewBox="0 0 {w} {h}" style="width:100%; height:auto; display:block;">
          <defs>
            <linearGradient id="sparkfade" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="{AMBER}" stop-opacity="0.28"/>
              <stop offset="100%" stop-color="{AMBER}" stop-opacity="0"/>
            </linearGradient>
          </defs>
          <path d="{path_d} L {w-pad},{h-pad} L {pad},{h-pad} Z" fill="url(#sparkfade)" stroke="none"/>
          <path d="{path_d}" fill="none" stroke="{AMBER}" stroke-width="2.4"
                stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <div style="font-family:'JetBrains Mono',monospace; font-size:10px; color:{INK_MUTE};
                    text-align:right; margin-top:2px; letter-spacing:0.5px;">
          LAST {len(vals)} MONTHS · RAW SIGNAL
        </div>
      </div>
    </div>
    """)


def render_gauges(items):
    """
    items: list of dicts with keys label, value, unit(str), accent(hex),
           pct(0-100 fill), delta(optional float %)
    """
    cards = []
    for it in items:
        pct = max(0, min(100, it.get("pct", 60)))
        accent = it.get("accent", AMBER)
        delta = it.get("delta")
        delta_html = ""
        if delta is not None:
            color = GOOD if delta >= 0 else DANGER
            arrow = "▲" if delta >= 0 else "▼"
            delta_html = f'<div class="gauge-delta" style="color:{color};">{arrow} {abs(delta):.1f}%</div>'

        ring_bg = (
            f"conic-gradient({accent} {pct * 3.6:.1f}deg, {BORDER} {pct * 3.6:.1f}deg)"
        )
        cards.append(f"""
        <div class="gauge-card">
          <div class="gauge-ring" style="background:{ring_bg};">
            <div class="gauge-value">{it['value']}</div>
          </div>
          {delta_html}
          <div class="gauge-label">{it['label']}</div>
        </div>
        """)
    st.html(f'<div class="gauge-row">{"".join(cards)}</div>')


def plotly_dark_layout(fig):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor=PANEL,
        paper_bgcolor=PANEL,
        font=dict(family="Inter, sans-serif", color=INK),
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(showgrid=False, color=INK_MUTE),
        yaxis=dict(gridcolor=BORDER, color=INK_MUTE),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ==================================================
# DATA
# ==================================================
try:
    loader = DataLoader()
    df = loader.load_data()
except FileNotFoundError as e:
    st.error(f"**Couldn't load the dataset.**\n\n{e}")
    st.stop()

PASS_COL = "Passengers" if "Passengers" in df.columns else "passengers"

# ==================================================
# HERO
# ==================================================
render_hero(df, PASS_COL)

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("### ⚙️ Forecast Settings")
    future_months = st.slider("Forecast Horizon (Months)", 1, 24, 12)
    st.markdown(
        '<div class="footnote">Longer horizons widen the approximate '
        'uncertainty band in the combined projection chart.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### 📊 Dataset")
    st.markdown(
        f'<div class="footnote">{len(df):,} monthly observations · '
        f'{df.index.min().strftime("%b %Y")} → {df.index.max().strftime("%b %Y")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### 🧭 Pipeline")
    for i, step in enumerate(
        ["Load data", "Scale (MinMax)", "Sequence (12mo window)", "LSTM inference", "Inverse-scale forecast"], 1
    ):
        st.markdown(f'<div class="footnote">{i:02d} · {step}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Model: LSTM · pre-trained, loaded for inference only")

# ==================================================
# Guard: model artifacts present?
# ==================================================
missing = [p.name for p in (MODEL_PATH, SCALER_PATH) if not p.exists()]
if missing:
    st.warning(
        f"**Model file(s) not found in `models/`:** {', '.join(missing)}.\n\n"
        "This app needs `models/lstm_model.keras` and `models/scaler.pkl` "
        "committed to the repo (or run `python -m src.train` to generate them). "
        "The dataset explorer below still works without them."
    )

# ==================================================
# TABS — MODEL PERFORMANCE / EDA
# ==================================================
st.markdown('<div class="section-badge">Model Diagnostics</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🚀 Model Performance", "🔎 Exploratory Data Analysis"])

with tab1:
    st.subheader("Model Accuracy Metrics")
    if missing:
        st.info("Add the model + scaler files to see live accuracy metrics here.")
    else:
        with st.spinner("Scoring held-out predictions..."):
            try:
                mae, mse, rmse = Evaluator().evaluate()
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                mae = mse = rmse = None

        if mae is not None:
            render_gauges([
                {"label": "Mean Absolute Error", "value": f"{mae:.2f}", "accent": AMBER, "pct": 70},
                {"label": "Mean Squared Error", "value": f"{mse:.2f}", "accent": TEAL, "pct": 55},
                {"label": "Root Mean Squared Error", "value": f"{rmse:.2f}", "accent": GOOD, "pct": 65},
            ])

with tab2:
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.subheader("Raw Data")
        st.dataframe(df, height=380, use_container_width=True)

    with col_b:
        st.subheader("Historical Trend")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=df.index, y=df[PASS_COL],
            mode="lines",
            line=dict(color=AMBER, width=2.4),
            fill="tozeroy",
            fillcolor="rgba(242,166,61,0.12)",
            name="Passengers",
        ))
        st.plotly_chart(plotly_dark_layout(fig_hist), use_container_width=True)

# ==================================================
# FORECAST SECTION
# ==================================================
st.markdown("---")
st.markdown('<div class="section-badge">Prediction Engine</div>', unsafe_allow_html=True)
st.header("🔮 Generate Future Forecast")

run = st.button("📡  Run LSTM Model", disabled=bool(missing))
if missing:
    st.caption("Button disabled until the model files are available.")

if run:
    progress = st.progress(0, text="Warming up the LSTM...")
    for pct, label in [
        (25, "Loading pre-trained weights..."),
        (55, "Reading temporal patterns..."),
        (80, "Projecting future passenger volume..."),
        (100, "Finalizing forecast..."),
    ]:
        time.sleep(0.2)
        progress.progress(pct, text=label)
    time.sleep(0.15)
    progress.empty()

    try:
        forecaster = Forecaster()
        future = forecaster.forecast(future_months)
    except Exception as e:
        st.error(f"Forecast failed: {e}")
        st.stop()

    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=future_months,
        freq="MS",
    )

    predicted = np.asarray(future).flatten()

    # Illustrative uncertainty band, scaled by a fixed heuristic and
    # growing with horizon. Not a statistically calibrated interval.
    horizon_idx = np.arange(1, future_months + 1)
    base_spread = predicted.std() * 0.08 + 5
    band_width = base_spread * (1 + 0.08 * horizon_idx)
    upper = predicted + band_width
    lower = np.clip(predicted - band_width, a_min=0, a_max=None)

    forecast_df = pd.DataFrame({
        "Month": future_dates,
        "Predicted Passengers": predicted.round(1),
        "Lower Bound (approx.)": lower.round(1),
        "Upper Bound (approx.)": upper.round(1),
    })

    st.success(f"Forecast generated for the next {future_months} month(s). 📡")

    last_known = float(df[PASS_COL].iloc[-1])
    growth_pct = ((predicted[-1] - last_known) / last_known) * 100

    render_gauges([
        {"label": "Last Known Volume", "value": f"{last_known:,.0f}", "accent": TEAL, "pct": 50},
        {
            "label": f'Predicted ({future_dates[-1].strftime("%b %Y")})',
            "value": f"{predicted[-1]:,.0f}", "accent": AMBER, "pct": 70, "delta": growth_pct,
        },
        {"label": "Avg. Predicted Volume", "value": f"{predicted.mean():,.0f}", "accent": GOOD, "pct": 60},
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.subheader("Forecasted Values")
        st.dataframe(forecast_df, use_container_width=True, height=360)

        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name="forecast_results.csv",
            mime="text/csv",
        )

    with res_col2:
        st.subheader("Combined Projection")

        fig_combined = go.Figure()

        fig_combined.add_trace(go.Scatter(
            x=list(forecast_df["Month"]) + list(forecast_df["Month"][::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself",
            fillcolor="rgba(242,166,61,0.14)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="Approx. uncertainty",
            showlegend=True,
        ))

        fig_combined.add_trace(go.Scatter(
            x=df.index, y=df[PASS_COL],
            name="Historical",
            line=dict(color=TEAL, width=2.2),
        ))

        fig_combined.add_trace(go.Scatter(
            x=forecast_df["Month"], y=forecast_df["Predicted Passengers"],
            name="Forecast",
            line=dict(color=AMBER, width=3, dash="dot"),
            marker=dict(size=6, color=AMBER),
            mode="lines+markers",
        ))

        fig_combined.update_layout(hovermode="x unified")
        st.plotly_chart(plotly_dark_layout(fig_combined), use_container_width=True)

    st.markdown(
        '<p class="footnote">The shaded band is an illustrative uncertainty range '
        "— it widens with the forecast horizon and is not a formally calibrated "
        "confidence interval.</p>",
        unsafe_allow_html=True,
    )
else:
    st.info("Set your horizon in the sidebar, then click **Run LSTM Model** to generate a forecast.")

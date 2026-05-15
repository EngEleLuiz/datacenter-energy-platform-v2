"""
dashboard/app.py
================
Datacenter Energy Intelligence Platform — Executive Dashboard
Visualises GFL vs GFM inverter dynamics with 5 academic features:
  1. Virtual Inertia (VSM)
  2. Black-Start Capability
  3. Active Harmonic Compensation
  4. Droop Control (P/f, Q/V)
  5. Weak-Grid Stability Map
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timezone, timedelta
from dataclasses import asdict
import math

from data_generator.server_simulator import ServerSimulator
from data_generator.ups_inverter_simulator import (
    UPSSimulator, InverterSimulator,
    NOMINAL_FREQ_HZ, NOMINAL_VOLTAGE_V,
    IEEE1547_ROCOF_LIMIT, IEEE519_THD_LIMIT_PCT,
)
import sys
import os
try:
    from analysis.stability_analysis import (
        compute_stability_metrics, scr_sweep_metrics,
        make_bode_figure, make_nyquist_figure,
        make_pm_vs_scr_figure, make_middlebrook_figure,
        make_gain_margin_figure, get_stability_summary_text,
    )
    STABILITY_AVAILABLE = True
except ImportError:
    STABILITY_AVAILABLE = False

try:
    from data_generator.external_data_fetcher import (
        ExternalDataFetcher, EXTERNAL_FEATURE_COLS,
    )
    EXTERNAL_AVAILABLE = True
except ImportError:
    EXTERNAL_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DC Energy Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System ─────────────────────────────────────────────────────────────
DARK_BG       = "#0A0E1A"
CARD_BG       = "#111827"
CARD_BORDER   = "#1F2937"
ACCENT_CYAN   = "#00D4FF"
ACCENT_GREEN  = "#00FF9F"
ACCENT_RED    = "#FF4757"
ACCENT_AMBER  = "#FFB020"
ACCENT_PURPLE = "#A855F7"
TEXT_PRIMARY  = "#F1F5F9"
TEXT_MUTED    = "#64748B"

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'JetBrains Mono', monospace", color=TEXT_PRIMARY, size=11),
    margin=dict(l=10, r=10, t=40, b=10),
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'JetBrains Mono', monospace;
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
}}

.main .block-container {{ padding: 1rem 2rem; max-width: 100%; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0D1117 0%, #0A0E1A 100%);
    border-right: 1px solid {CARD_BORDER};
}}

/* KPI Cards */
.kpi-card {{
    background: linear-gradient(135deg, {CARD_BG} 0%, #0D1B2A 100%);
    border: 1px solid {CARD_BORDER};
    border-radius: 12px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s;
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {ACCENT_CYAN}, {ACCENT_GREEN});
}}
.kpi-card.red::before  {{ background: linear-gradient(90deg, {ACCENT_RED}, {ACCENT_AMBER}); }}
.kpi-card.amber::before {{ background: linear-gradient(90deg, {ACCENT_AMBER}, {ACCENT_PURPLE}); }}
.kpi-card.purple::before {{ background: linear-gradient(90deg, {ACCENT_PURPLE}, {ACCENT_CYAN}); }}
.kpi-label  {{ font-size: 0.68rem; color: {TEXT_MUTED}; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 6px; }}
.kpi-value  {{ font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 800; color: {TEXT_PRIMARY}; line-height: 1; }}
.kpi-sub    {{ font-size: 0.70rem; color: {TEXT_MUTED}; margin-top: 4px; }}
.kpi-badge  {{ display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 0.65rem; font-weight: 600; margin-top: 6px; }}
.badge-ok   {{ background: rgba(0,255,159,0.15); color: {ACCENT_GREEN}; border: 1px solid rgba(0,255,159,0.3); }}
.badge-warn {{ background: rgba(255,176,32,0.15); color: {ACCENT_AMBER}; border: 1px solid rgba(255,176,32,0.3); }}
.badge-err  {{ background: rgba(255,71,87,0.15);  color: {ACCENT_RED};   border: 1px solid rgba(255,71,87,0.3); }}

/* Section headers */
.section-header {{
    display: flex; align-items: center; gap: 10px;
    border-bottom: 1px solid {CARD_BORDER};
    padding-bottom: 8px; margin-bottom: 20px; margin-top: 10px;
}}
.section-header h3 {{
    font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700;
    color: {ACCENT_CYAN}; letter-spacing: 0.05em; margin: 0;
}}
.section-header .tag {{
    font-size: 0.60rem; padding: 2px 8px; border-radius: 4px;
    background: rgba(0,212,255,0.12); color: {ACCENT_CYAN};
    border: 1px solid rgba(0,212,255,0.25); letter-spacing: 0.08em;
}}

/* Feature badge */
.feature-pill {{
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.62rem; font-weight: 600; letter-spacing: 0.08em;
    margin: 2px; text-transform: uppercase;
}}
.pill-vsm    {{ background:rgba(0,212,255,0.15); color:{ACCENT_CYAN};   border:1px solid rgba(0,212,255,0.3); }}
.pill-bs     {{ background:rgba(255,176,32,0.15); color:{ACCENT_AMBER}; border:1px solid rgba(255,176,32,0.3); }}
.pill-harm   {{ background:rgba(168,85,247,0.15); color:{ACCENT_PURPLE};border:1px solid rgba(168,85,247,0.3); }}
.pill-droop  {{ background:rgba(0,255,159,0.15);  color:{ACCENT_GREEN}; border:1px solid rgba(0,255,159,0.3); }}
.pill-wg     {{ background:rgba(255,71,87,0.15);  color:{ACCENT_RED};   border:1px solid rgba(255,71,87,0.3); }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: {DARK_BG}; }}
::-webkit-scrollbar-thumb {{ background: {CARD_BORDER}; border-radius: 2px; }}

/* Hide Streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}
</style>
""", unsafe_allow_html=True)


# ── Helper: plotly layout ─────────────────────────────────────────────────────
def apply_theme(fig, height=380, title=""):
    fig.update_layout(
        height=height, title=title,
        **PLOTLY_THEME,
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor=CARD_BORDER,
            borderwidth=1,
            font=dict(size=10),
        ),
        xaxis=dict(gridcolor=CARD_BORDER, zerolinecolor=CARD_BORDER),
        yaxis=dict(gridcolor=CARD_BORDER, zerolinecolor=CARD_BORDER),
    )
    return fig


def kpi(label, value, sub="", variant="", badge="", badge_type="ok"):
    badge_html = f'<div class="kpi-badge badge-{badge_type}">{badge}</div>' if badge else ""
    st.markdown(f"""
    <div class="kpi-card {variant}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


def section(icon, title, tag=""):
    tag_html = f'<span class="tag">{tag}</span>' if tag else ""
    st.markdown(f"""
    <div class="section-header">
        <h3>{icon} {title}</h3>
        {tag_html}
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:16px 0 8px">
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                    color:{ACCENT_CYAN};letter-spacing:0.04em;">⚡ DC ENERGY</div>
        <div style="font-size:0.65rem;color:{TEXT_MUTED};letter-spacing:0.15em;
                    text-transform:uppercase;margin-top:2px;">Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<div style='font-size:0.7rem;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px'>Parameters</div>", unsafe_allow_html=True)

    num_servers    = st.slider("Servers",            50, 200, 100, 10)
    num_inverters  = st.slider("Inverters",           2,   8,   4)
    islanding_prob = st.slider("Islanding Prob.",  0.00, 0.15, 0.05, 0.01)
    bs_prob        = st.slider("Black-Start Prob.", 0.00, 0.03, 0.01, 0.005)
    scr_min        = st.slider("SCR Min",           0.5,  3.0,  1.0, 0.5)
    history_hours  = st.slider("History (hours)",     1,  24,    6)

    st.markdown("---")
    st.markdown(f"<div style='font-size:0.7rem;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px'>Navigation</div>", unsafe_allow_html=True)

    page = st.radio("", [
        "Overview",
        "① Virtual Inertia",
        "② Black-Start",
        "③ Harmonics",
        "④ Droop Control",
        "⑤ Weak-Grid Stability",
        "⑥ SHAP Explainability",
        "⑦ Clima & Preço Energia",
        "⑧ Bode / Nyquist",
    ], label_visibility="collapsed")

    st.markdown("---")
    refresh = st.button("⟳  Refresh Data", use_container_width=True)
    st.markdown(f"<div style='font-size:0.62rem;color:{TEXT_MUTED};text-align:center;margin-top:6px'>{datetime.now().strftime('%H:%M:%S')} UTC</div>", unsafe_allow_html=True)

    # Feature pills
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div>
        <span class="feature-pill pill-vsm">VSM</span>
        <span class="feature-pill pill-bs">Black-Start</span>
        <span class="feature-pill pill-harm">Harmonics</span>
        <span class="feature-pill pill-droop">Droop</span>
        <span class="feature-pill pill-wg">Weak-Grid</span>
    </div>
    """, unsafe_allow_html=True)


# ── Data generation ───────────────────────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner="Generating telemetry…")
def load_data(num_servers, num_inverters, islanding_prob, bs_prob, scr_min, hours, _r):
    srv_sim = ServerSimulator(num_servers=num_servers, num_racks=10,
                              fault_probability=0.01, random_seed=None)
    ups_sim = UPSSimulator(num_ups=4, vsm_inertia_H=5.0)
    inv_sim = InverterSimulator(
        num_inverters=num_inverters,
        islanding_probability=islanding_prob,
        black_start_probability=bs_prob,
        scr_range=(scr_min, 12.0),
        vsm_inertia_H_range=(2.0, 10.0),
        droop_kw_hz=20.0,
        droop_kvar_v=5.0,
    )
    start  = datetime.now(timezone.utc) - timedelta(hours=hours)
    steps  = hours * 12
    srv_r, ups_r, inv_r = [], [], []
    for i in range(steps):
        ts = start + timedelta(minutes=5 * i)
        srv_r.extend([asdict(r) for r in srv_sim.generate_snapshot(ts)])
        ups_r.extend([asdict(r) for r in ups_sim.generate_snapshot(ts)])
        inv_r.extend([asdict(r) for r in inv_sim.generate_snapshot(ts)])

    df_s = pd.DataFrame(srv_r)
    df_u = pd.DataFrame(ups_r)
    df_i = pd.DataFrame(inv_r)
    for d in [df_s, df_u, df_i]:
        d["timestamp_utc"] = pd.to_datetime(d["timestamp_utc"], utc=True)
    return df_s, df_u, df_i

df_srv, df_ups, df_inv = load_data(
    num_servers, num_inverters, islanding_prob, bs_prob, scr_min, history_hours, refresh
)

latest_srv = df_srv[df_srv["timestamp_utc"] == df_srv["timestamp_utc"].max()]
df_gfl     = df_inv[df_inv["control_mode"] == "GFL"]
df_gfm     = df_inv[df_inv["control_mode"] == "GFM"]
df_gfx     = df_inv[df_inv["control_mode"].isin(["GFL", "GFM"])]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown(f"""
    <div style="margin-bottom:24px">
        <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;
                    background:linear-gradient(90deg,{ACCENT_CYAN},{ACCENT_GREEN});
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    letter-spacing:0.02em;">
            Datacenter Microgrid Intelligence
        </div>
        <div style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:4px;letter-spacing:0.08em">
            REAL-TIME GFL/GFM INVERTER ANALYTICS · IEEE 1547 · IEEE 519 · VIRTUAL SYNCHRONOUS MACHINE
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    total_kw    = latest_srv["power_draw_w"].sum() / 1000
    avg_pue     = latest_srv["pue_contribution"].mean()
    anomalies   = latest_srv["is_anomaly"].sum()
    gfm_count   = (df_inv["control_mode"] == "GFM").sum()
    bs_events   = df_inv["black_start_active"].sum()
    ieee_viol   = (df_gfl["rocof_hz_per_s"].abs() > IEEE1547_ROCOF_LIMIT).mean()

    with c1: kpi("IT Power", f"{total_kw:.0f}", "kW total", badge="Live", badge_type="ok")
    with c2: kpi("Avg PUE", f"{avg_pue:.4f}", "Power Usage Effectiveness", badge="Nominal", badge_type="ok")
    with c3: kpi("Anomalies", str(anomalies), "active faults", "red",
                 badge="Alert" if anomalies > 0 else "Clear",
                 badge_type="err" if anomalies > 0 else "ok")
    with c4: kpi("GFM Active", str(gfm_count), "inverter snapshots", "amber", badge="Grid-Forming", badge_type="warn")
    with c5: kpi("Black-Start", str(bs_events), "events detected", "purple", badge="GFM Only", badge_type="warn")
    with c6: kpi("IEEE Violations", f"{ieee_viol:.1%}", "GFL ROCOF > 0.5 Hz/s", "red",
                 badge="Critical" if ieee_viol > 0.05 else "OK",
                 badge_type="err" if ieee_viol > 0.05 else "ok")

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        section("📈", "Power & Frequency Timeline", "48H OVERVIEW")
        dc_pwr = df_srv.groupby("timestamp_utc")["power_draw_w"].sum().reset_index()
        dc_pwr["kw"] = dc_pwr["power_draw_w"] / 1000
        inv_freq = df_inv.groupby("timestamp_utc")["output_frequency_hz"].mean().reset_index()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.06, row_heights=[0.6, 0.4])
        fig.add_trace(go.Scatter(
            x=dc_pwr["timestamp_utc"], y=dc_pwr["kw"],
            fill="tozeroy", name="IT Power (kW)",
            line=dict(color=ACCENT_CYAN, width=1.5),
            fillcolor=f"rgba(0,212,255,0.08)"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=inv_freq["timestamp_utc"], y=inv_freq["output_frequency_hz"],
            name="Grid Freq (Hz)", line=dict(color=ACCENT_GREEN, width=1.5)
        ), row=2, col=1)
        fig.add_hline(y=NOMINAL_FREQ_HZ, line_dash="dash",
                      line_color="rgba(255,255,255,0.2)", row=2, col=1)
        fig.update_layout(height=320, **PLOTLY_THEME)
        fig.update_xaxes(gridcolor=CARD_BORDER)
        fig.update_yaxes(gridcolor=CARD_BORDER)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        section("🔵", "Control Mode Distribution", "REAL-TIME")
        mode_counts = df_inv["control_mode"].value_counts().reset_index()
        mode_counts.columns = ["mode", "count"]
        color_map = {"GFL": ACCENT_RED, "GFM": ACCENT_GREEN,
                     "transitioning": ACCENT_AMBER, "black_start": ACCENT_PURPLE}
        fig2 = go.Figure(go.Pie(
            labels=mode_counts["mode"],
            values=mode_counts["count"],
            hole=0.65,
            marker=dict(colors=[color_map.get(m, TEXT_MUTED) for m in mode_counts["mode"]],
                        line=dict(color=DARK_BG, width=3)),
            textfont=dict(size=11),
        ))
        fig2.add_annotation(text="Inverters", x=0.5, y=0.55,
                            font=dict(size=10, color=TEXT_MUTED), showarrow=False)
        fig2.add_annotation(text=str(len(df_inv["inverter_id"].unique())), x=0.5, y=0.42,
                            font=dict(size=28, color=TEXT_PRIMARY,
                                      family="Syne, sans-serif"), showarrow=False)
        fig2.update_layout(height=320, showlegend=True, **PLOTLY_THEME,
                           legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig2, use_container_width=True)

    # Stability overview heatmap
    section("🗺️", "Stability Overview — All Inverters", "MIDDLEBROOK CRITERION")
    stab_pivot = df_inv.groupby(["inverter_id", "stability_flag"])["scr"].count().unstack(fill_value=0)
    stab_pct   = stab_pivot.div(stab_pivot.sum(axis=1), axis=0) * 100
    flag_order = [c for c in ["stable", "marginal", "unstable"] if c in stab_pct.columns]
    fig3 = go.Figure()
    bar_colors = {"stable": ACCENT_GREEN, "marginal": ACCENT_AMBER, "unstable": ACCENT_RED}
    for flag in flag_order:
        if flag in stab_pct.columns:
            fig3.add_trace(go.Bar(
                name=flag.capitalize(), x=stab_pct.index,
                y=stab_pct[flag],
                marker_color=bar_colors[flag],
                marker_line=dict(width=0),
            ))
    fig3.update_layout(barmode="stack", height=220, **PLOTLY_THEME,
                       xaxis_title="Inverter", yaxis_title="% of Time")
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — VIRTUAL INERTIA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "① Virtual Inertia":
    st.markdown(f"""
    <div style="margin-bottom:20px">
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:{ACCENT_CYAN}">
            ① Virtual Synchronous Machine (VSM)
        </div>
        <div style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:4px">
            GFM inverters emulate synchronous generator inertia via the swing equation:
            &nbsp;<b style="color:{ACCENT_CYAN}">2H/ω₀ · dω/dt = P_mech − P_elec − D·Δω</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    gfl_rocof_mean = df_gfl["rocof_hz_per_s"].abs().mean()
    gfm_rocof_mean = df_gfm["rocof_hz_per_s"].abs().mean()
    ieee_gfl = (df_gfl["rocof_hz_per_s"].abs() > IEEE1547_ROCOF_LIMIT).mean()
    ieee_gfm = (df_gfm["rocof_hz_per_s"].abs() > IEEE1547_ROCOF_LIMIT).mean()
    with c1: kpi("GFL Mean |ROCOF|", f"{gfl_rocof_mean:.3f}", "Hz/s", "red",
                 badge="IEEE 1547", badge_type="err")
    with c2: kpi("GFM Mean |ROCOF|", f"{gfm_rocof_mean:.3f}", "Hz/s", "",
                 badge="IEEE 1547", badge_type="ok")
    with c3: kpi("GFL Violations", f"{ieee_gfl:.1%}", "ROCOF > 0.5 Hz/s", "red",
                 badge_type="err", badge="Non-Compliant" if ieee_gfl > 0 else "OK")
    with c4: kpi("GFM Violations", f"{ieee_gfm:.1%}", "ROCOF > 0.5 Hz/s", "",
                 badge_type="ok", badge="Compliant")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        section("📊", "ROCOF Distribution", "GFL vs GFM")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df_gfl["rocof_hz_per_s"], name="GFL",
            marker_color=ACCENT_RED, opacity=0.75, nbinsx=60,
        ))
        fig.add_trace(go.Histogram(
            x=df_gfm["rocof_hz_per_s"], name="GFM",
            marker_color=ACCENT_GREEN, opacity=0.75, nbinsx=60,
        ))
        for sign in [1, -1]:
            fig.add_vline(x=sign * IEEE1547_ROCOF_LIMIT, line_dash="dash",
                          line_color="rgba(255,255,255,0.4)",
                          annotation_text="IEEE 1547" if sign == 1 else "",
                          annotation_font_size=9)
        fig.update_layout(barmode="overlay", **PLOTLY_THEME, height=320,
                          xaxis_title="ROCOF (Hz/s)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("⚡", "Inertia Constant H vs ROCOF at Nadir", "VSM THEORY")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_gfm["virtual_inertia_H"],
            y=df_gfm["rocof_at_nadir"].abs(),
            mode="markers",
            marker=dict(
                color=df_gfm["virtual_inertia_power_kw"].abs(),
                colorscale=[[0, ACCENT_PURPLE], [0.5, ACCENT_CYAN], [1, ACCENT_GREEN]],
                size=5, opacity=0.6,
                colorbar=dict(title="VSM Power (kW)", thickness=10),
            ),
            name="GFM inverters",
        ))
        fig2.update_layout(**PLOTLY_THEME, height=320,
                           xaxis_title="H constant (s)",
                           yaxis_title="|ROCOF at Nadir| (Hz/s)")
        st.plotly_chart(fig2, use_container_width=True)

    section("📉", "Virtual Inertia Power Injection Over Time", "VSM RESPONSE")
    fig3 = go.Figure()
    for inv_id in df_gfm["inverter_id"].unique():
        sub = df_gfm[df_gfm["inverter_id"] == inv_id]
        fig3.add_trace(go.Scatter(
            x=sub["timestamp_utc"], y=sub["virtual_inertia_power_kw"],
            name=inv_id, mode="lines", line=dict(width=1.2)
        ))
    fig3.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_dash="dot")
    fig3.update_layout(**PLOTLY_THEME, height=260,
                       xaxis_title="Time", yaxis_title="VSM Power (kW)")
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BLACK-START
# ══════════════════════════════════════════════════════════════════════════════
elif page == "② Black-Start":
    st.markdown(f"""
    <div style="margin-bottom:20px">
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:{ACCENT_AMBER}">
            ② Black-Start Capability
        </div>
        <div style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:4px">
            GFM forms voltage from scratch after a blackout.
            &nbsp;<b style="color:{ACCENT_AMBER}">GFL cannot black-start — it requires an external voltage reference for PLL lock.</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    bs_events_df = df_inv[df_inv["black_start_stage"] > 0]
    stage_labels = {0: "Idle", 1: "Pre-Charge", 2: "Voltage Ramp",
                    3: "Load Pickup", 4: "Complete"}
    stage_colors = {0: TEXT_MUTED, 1: ACCENT_RED, 2: ACCENT_AMBER,
                    3: ACCENT_CYAN, 4: ACCENT_GREEN}

    c1, c2, c3, c4 = st.columns(4)
    total_bs   = df_inv["black_start_active"].sum()
    completed  = (df_inv["black_start_stage"] == 4).sum()
    max_loads  = df_inv["loads_reconnected"].max()
    gfl_bs     = df_gfl["black_start_active"].sum()
    with c1: kpi("Black-Start Events", str(total_bs), "GFM activations", "amber")
    with c2: kpi("Completed", str(completed), "full restorations", "", badge="Stage 4", badge_type="ok")
    with c3: kpi("Max Loads Restored", str(max_loads), "critical loads", "purple")
    with c4: kpi("GFL Black-Start", str(gfl_bs), "always zero", "red",
                 badge="Not Capable", badge_type="err")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        section("🔄", "Black-Start Stage Distribution", "ALL INVERTERS")
        stage_cnt = df_inv["black_start_stage"].value_counts().sort_index().reset_index()
        stage_cnt.columns = ["stage", "count"]
        stage_cnt["label"] = stage_cnt["stage"].map(stage_labels)
        stage_cnt["color"] = stage_cnt["stage"].map(stage_colors)
        fig = go.Figure(go.Bar(
            x=stage_cnt["label"], y=stage_cnt["count"],
            marker_color=stage_cnt["color"],
            marker_line=dict(width=0),
            text=stage_cnt["count"], textposition="outside",
        ))
        fig.update_layout(**PLOTLY_THEME, height=320,
                          xaxis_title="Stage", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("⚡", "Voltage Ramp Profile", "STAGE 2 — GFM ONLY")
        df_ramp = df_inv[df_inv["black_start_stage"] == 2].copy()
        if len(df_ramp) > 0:
            fig2 = go.Figure()
            for inv_id in df_ramp["inverter_id"].unique():
                sub = df_ramp[df_ramp["inverter_id"] == inv_id].reset_index(drop=True)
                fig2.add_trace(go.Scatter(
                    x=sub.index * 5, y=sub["black_start_voltage_pct"],
                    name=inv_id, mode="lines+markers",
                    marker=dict(size=4), line=dict(width=2)
                ))
            fig2.add_hline(y=100, line_dash="dash",
                           line_color="rgba(255,255,255,0.3)",
                           annotation_text="480V Nominal")
            fig2.update_layout(**PLOTLY_THEME, height=320,
                               xaxis_title="Time (minutes)",
                               yaxis_title="Bus Voltage (% nominal)")
        else:
            fig2 = go.Figure()
            fig2.add_annotation(text="No voltage ramp data yet.<br>Increase Black-Start Prob.",
                                x=0.5, y=0.5, showarrow=False,
                                font=dict(color=TEXT_MUTED, size=13))
            fig2.update_layout(**PLOTLY_THEME, height=320)
        st.plotly_chart(fig2, use_container_width=True)

    section("📦", "Load Reconnection Progress", "STAGE 3 — CRITICAL LOAD PICKUP")
    df_lp = df_inv[df_inv["black_start_stage"] == 3].copy()
    if len(df_lp) > 0:
        fig3 = go.Figure()
        for inv_id in df_lp["inverter_id"].unique():
            sub = df_lp[df_lp["inverter_id"] == inv_id]
            fig3.add_trace(go.Scatter(
                x=sub["timestamp_utc"], y=sub["loads_reconnected"],
                name=inv_id, mode="lines+markers",
                line=dict(width=2), marker=dict(size=5)
            ))
        fig3.add_hline(y=8, line_dash="dash", line_color=ACCENT_GREEN,
                       annotation_text="All Critical Loads (8)")
        fig3.update_layout(**PLOTLY_THEME, height=240,
                           xaxis_title="Time", yaxis_title="Loads Reconnected")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No load-pickup events in current window. Increase history window or black-start probability.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — HARMONICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "③ Harmonics":
    st.markdown(f"""
    <div style="margin-bottom:20px">
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:{ACCENT_PURPLE}">
            ③ Active Harmonic Compensation
        </div>
        <div style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:4px">
            GFM virtual impedance shaping eliminates 5th, 7th, 11th, 13th harmonics from datacenter SMPS loads.
            &nbsp;<b style="color:{ACCENT_PURPLE}">IEEE 519-2022: THD ≤ 5% at PCC.</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    gfl_thd  = df_gfl["thd_percent"].mean()
    gfm_thd  = df_gfm["thd_percent"].mean()
    gfl_comp = (df_gfl["thd_percent"] <= IEEE519_THD_LIMIT_PCT).mean()
    gfm_comp = (df_gfm["thd_percent"] <= IEEE519_THD_LIMIT_PCT).mean()
    df_comp  = df_gfm[df_gfm["harmonic_compensation_active"]]
    reduction = 0.0
    if len(df_comp) > 0:
        reduction = ((df_comp["thd_before_compensation_pct"] - df_comp["thd_percent"]) /
                      df_comp["thd_before_compensation_pct"].clip(lower=0.01)).mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("GFL Mean THD", f"{gfl_thd:.2f}%", "no compensation", "red",
                 badge="Non-Compliant" if gfl_thd > IEEE519_THD_LIMIT_PCT else "OK",
                 badge_type="err" if gfl_thd > IEEE519_THD_LIMIT_PCT else "ok")
    with c2: kpi("GFM Mean THD", f"{gfm_thd:.2f}%", "after APF compensation", "",
                 badge="Compliant", badge_type="ok")
    with c3: kpi("THD Reduction", f"{reduction:.1%}", "GFM active filter", "purple",
                 badge="APF Active", badge_type="warn")
    with c4: kpi("IEEE 519 GFM", f"{gfm_comp:.1%}", "compliance rate", "",
                 badge="Compliant", badge_type="ok")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        section("📊", "THD Distribution — GFL vs GFM", "IEEE 519 COMPLIANCE")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df_gfl["thd_percent"], name="GFL (no compensation)",
            marker_color=ACCENT_RED, opacity=0.75, nbinsx=50,
        ))
        fig.add_trace(go.Histogram(
            x=df_gfm["thd_percent"], name="GFM (with APF)",
            marker_color=ACCENT_GREEN, opacity=0.75, nbinsx=50,
        ))
        fig.add_vline(x=IEEE519_THD_LIMIT_PCT, line_dash="dash",
                      line_color="rgba(255,255,255,0.4)",
                      annotation_text="IEEE 519 (5%)", annotation_font_size=9)
        fig.update_layout(barmode="overlay", **PLOTLY_THEME, height=320,
                          xaxis_title="THD (%)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("🔬", "Before vs After Compensation", "GFM ACTIVE POWER FILTER")
        if len(df_comp) > 0:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_comp["thd_before_compensation_pct"],
                y=df_comp["thd_percent"],
                mode="markers",
                marker=dict(
                    color=df_comp["harmonic_injection_a"],
                    colorscale=[[0, ACCENT_PURPLE], [1, ACCENT_CYAN]],
                    size=5, opacity=0.5,
                    colorbar=dict(title="Injection (A)", thickness=10),
                ),
                name="THD before→after",
            ))
            max_v = df_comp["thd_before_compensation_pct"].max()
            fig2.add_trace(go.Scatter(
                x=[0, max_v], y=[0, max_v], mode="lines",
                line=dict(dash="dash", color="rgba(255,255,255,0.2)"),
                name="No compensation", showlegend=True,
            ))
            fig2.add_hline(y=IEEE519_THD_LIMIT_PCT, line_dash="dot",
                           line_color=ACCENT_GREEN,
                           annotation_text="IEEE 519 limit")
            fig2.update_layout(**PLOTLY_THEME, height=320,
                               xaxis_title="THD Before (%)", yaxis_title="THD After (%)")
        else:
            fig2 = go.Figure()
            fig2.update_layout(**PLOTLY_THEME, height=320)
        st.plotly_chart(fig2, use_container_width=True)

    section("🎸", "Dominant Harmonic Order & Injection Current", "5th / 7th / 11th / 13th")
    col3, col4 = st.columns(2)
    with col3:
        harm_cnt = df_inv["dominant_harmonic_order"].value_counts().sort_index().reset_index()
        harm_cnt.columns = ["order", "count"]
        harm_colors = {5: ACCENT_RED, 7: ACCENT_AMBER, 11: ACCENT_PURPLE, 13: ACCENT_CYAN}
        fig3 = go.Figure(go.Bar(
            x=[f"{o}th harmonic" for o in harm_cnt["order"]],
            y=harm_cnt["count"],
            marker_color=[harm_colors.get(o, TEXT_MUTED) for o in harm_cnt["order"]],
            marker_line=dict(width=0),
        ))
        fig3.update_layout(**PLOTLY_THEME, height=240, yaxis_title="Occurrences")
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        if len(df_comp) > 0:
            fig4 = go.Figure(go.Histogram(
                x=df_comp["harmonic_injection_a"],
                marker_color=ACCENT_PURPLE, nbinsx=40
            ))
            fig4.update_layout(**PLOTLY_THEME, height=240,
                               xaxis_title="Injection Current (A)", yaxis_title="Count")
            st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DROOP CONTROL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "④ Droop Control":
    st.markdown(f"""
    <div style="margin-bottom:20px">
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:{ACCENT_GREEN}">
            ④ Droop Control — Autonomous Power Sharing
        </div>
        <div style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:4px">
            Multiple GFM inverters share load without communication:
            &nbsp;<b style="color:{ACCENT_GREEN}">f = f₀ − kp·(P − P₀) &nbsp;|&nbsp; V = V₀ − kq·(Q − Q₀)</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    sharing_err = df_gfm.groupby("timestamp_utc")["output_active_power_kw"].agg(
        lambda x: x.std() / x.mean() * 100 if x.mean() != 0 else 0
    )
    mean_err = sharing_err.mean()

    c1, c2, c3, c4 = st.columns(4)
    kp = df_gfm["droop_kw_per_hz"].iloc[0] if len(df_gfm) > 0 else 20.0
    kq = df_gfm["droop_kvar_per_v"].iloc[0] if len(df_gfm) > 0 else 5.0
    with c1: kpi("Droop kp", f"{kp:.0f}", "kW / Hz", badge="P/f", badge_type="ok")
    with c2: kpi("Droop kq", f"{kq:.0f}", "kVAr / V", badge="Q/V", badge_type="ok")
    with c3: kpi("Sharing Error", f"{mean_err:.1f}%", "std/mean of P",
                 badge="Good" if mean_err < 5 else "High",
                 badge_type="ok" if mean_err < 5 else "warn")
    with c4: kpi("GFM Inverters", str(df_gfm["inverter_id"].nunique()), "parallel units", "purple")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        section("📈", "P/f Droop Characteristic", "ACTIVE POWER vs FREQUENCY")
        fig = go.Figure()
        palette = [ACCENT_CYAN, ACCENT_GREEN, ACCENT_AMBER, ACCENT_PURPLE,
                   ACCENT_RED, "#FF9FF3", "#54A0FF", "#5F27CD"]
        for j, inv_id in enumerate(df_gfm["inverter_id"].unique()):
            sub = df_gfm[df_gfm["inverter_id"] == inv_id]
            sub = sub.sample(min(300, len(sub)))
            fig.add_trace(go.Scatter(
                x=sub["output_active_power_kw"], y=sub["output_frequency_hz"],
                mode="markers", name=inv_id,
                marker=dict(color=palette[j % len(palette)], size=4, opacity=0.5)
            ))
        fig.add_hline(y=NOMINAL_FREQ_HZ, line_dash="dash",
                      line_color="rgba(255,255,255,0.2)", annotation_text="f₀ = 60 Hz")
        fig.update_layout(**PLOTLY_THEME, height=320,
                          xaxis_title="Active Power (kW)", yaxis_title="Frequency (Hz)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("📉", "Q/V Droop Characteristic", "REACTIVE POWER vs VOLTAGE")
        fig2 = go.Figure()
        for j, inv_id in enumerate(df_gfm["inverter_id"].unique()):
            sub = df_gfm[df_gfm["inverter_id"] == inv_id]
            sub = sub.sample(min(300, len(sub)))
            fig2.add_trace(go.Scatter(
                x=sub["output_reactive_power_kvar"], y=sub["voltage_deviation_pu"],
                mode="markers", name=inv_id,
                marker=dict(color=palette[j % len(palette)], size=4, opacity=0.5)
            ))
        fig2.add_hline(y=0, line_dash="dash",
                       line_color="rgba(255,255,255,0.2)", annotation_text="V₀ nominal")
        fig2.update_layout(**PLOTLY_THEME, height=320,
                           xaxis_title="Reactive Power (kVAr)", yaxis_title="Voltage Deviation (p.u.)")
        st.plotly_chart(fig2, use_container_width=True)

    section("⚖️", "Load Sharing Error Over Time", "TARGET < 5%")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=sharing_err.index, y=sharing_err.values,
        fill="tozeroy", name="Sharing Error (%)",
        line=dict(color=ACCENT_AMBER, width=1.5),
        fillcolor="rgba(255,176,32,0.08)"
    ))
    fig3.add_hline(y=5.0, line_dash="dash", line_color=ACCENT_RED,
                   annotation_text="5% target")
    fig3.update_layout(**PLOTLY_THEME, height=240,
                       xaxis_title="Time", yaxis_title="Sharing Error (%)")
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — WEAK-GRID STABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⑤ Weak-Grid Stability":
    st.markdown(f"""
    <div style="margin-bottom:20px">
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:{ACCENT_RED}">
            ⑤ Weak-Grid Stability Map
        </div>
        <div style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:4px">
            At SCR &lt; 3, GFL PLL destabilises. GFM remains stable for all SCR values.
            &nbsp;<b style="color:{ACCENT_RED}">Middlebrook: |Z_grid / Z_inv| &gt; 1 → unstable.</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    gfl_stable = (df_gfl["stability_flag"] == "stable").mean()
    gfm_stable = (df_gfm["stability_flag"] == "stable").mean()
    gfl_scr    = df_gfl["scr"].mean()
    gfm_scr    = df_gfm["scr"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("GFL Stable", f"{gfl_stable:.1%}", "of operating time", "red",
                 badge="Unstable Risk", badge_type="err")
    with c2: kpi("GFM Stable", f"{gfm_stable:.1%}", "of operating time", "",
                 badge="Always Stable", badge_type="ok")
    with c3: kpi("GFL Mean SCR", f"{gfl_scr:.2f}", "Short-Circuit Ratio", "red")
    with c4: kpi("GFM Mean SCR", f"{gfm_scr:.2f}", "Short-Circuit Ratio")

    st.markdown("<br>", unsafe_allow_html=True)

    # Theoretical stability map
    section("🗺️", "Stability Map — SCR × Phase Margin", "MIDDLEBROOK CRITERION")
    scr_sweep = np.linspace(0.5, 12.0, 400)

    def gfl_pm(s): return float(np.clip(90.0 * (1.0 - np.exp(-0.3 * (s - 1.0))), -30, 90))
    def gfm_margin(s):
        Zg = 1.0 / max(s, 0.01)
        Zi = 0.12
        return float(-20.0 * np.log10(Zg / Zi))

    gfl_pm_vals  = [gfl_pm(s) for s in scr_sweep]
    gfm_imp_vals = [gfm_margin(s) for s in scr_sweep]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["PLL Phase Margin vs SCR (GFL)",
                                        "Impedance Stability Margin — Middlebrook (dB)"])

    # Zone fills
    zones = [(0.5, 1.5, ACCENT_RED, 0.15), (1.5, 3.0, ACCENT_AMBER, 0.12), (3.0, 12.0, ACCENT_GREEN, 0.08)]
    for xmin, xmax, color, alpha in zones:
        mask = [(xmin <= s < xmax) for s in scr_sweep]
        xs   = [s for s, m in zip(scr_sweep, mask) if m]
        ys_t = [gfl_pm(s) for s in xs]
        rgb  = px.colors.hex_to_rgb(color)
        fill = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"
        if xs:
            fig.add_trace(go.Scatter(
                x=xs + xs[::-1], y=ys_t + [-30] * len(xs),
                fill="toself", fillcolor=fill,
                line=dict(width=0), showlegend=False, mode="lines"
            ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=scr_sweep, y=gfl_pm_vals, name="GFL PLL Phase Margin (°)",
        line=dict(color=ACCENT_RED, width=2.5)
    ), row=1, col=1)
    fig.add_hline(y=45, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                  annotation_text="PM=45° target", row=1, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color=ACCENT_RED,
                  annotation_text="PM=0° (unstable)", row=1, col=1)

    # GFM stability band
    fig.add_trace(go.Scatter(
        x=[0.5, 12.0], y=[80, 80], name="GFM (stable for all SCR)",
        line=dict(color=ACCENT_GREEN, width=2.5, dash="dot")
    ), row=1, col=1)

    # Simulation scatter — GFL
    fig.add_trace(go.Scatter(
        x=df_gfl["scr"], y=df_gfl["pll_stability_margin_deg"],
        mode="markers", name="GFL (simulated)",
        marker=dict(color=ACCENT_RED, size=4, opacity=0.25)
    ), row=1, col=1)

    # Right panel: impedance margin
    fig.add_trace(go.Scatter(
        x=scr_sweep, y=gfm_imp_vals, name="GFM |Zgrid/Zinv| (dB)",
        fill="tozeroy", fillcolor="rgba(0,255,159,0.08)",
        line=dict(color=ACCENT_GREEN, width=2.5)
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color=ACCENT_RED,
                  annotation_text="Middlebrook limit (0 dB)", row=1, col=2)

    fig.add_trace(go.Scatter(
        x=df_gfl["scr"], y=df_gfl["impedance_stability_margin_db"],
        mode="markers", name="GFL (simulated)",
        marker=dict(color=ACCENT_RED, size=4, opacity=0.25),
        showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=df_gfm["scr"], y=df_gfm["impedance_stability_margin_db"],
        mode="markers", name="GFM (simulated)",
        marker=dict(color=ACCENT_GREEN, size=4, opacity=0.25),
        showlegend=False
    ), row=1, col=2)

    fig.update_layout(height=420, **PLOTLY_THEME)
    fig.update_xaxes(title_text="Short-Circuit Ratio (SCR)", gridcolor=CARD_BORDER)
    fig.update_yaxes(title_text="Phase Margin (°)", row=1, col=1, gridcolor=CARD_BORDER)
    fig.update_yaxes(title_text="Stability Margin (dB)", row=1, col=2, gridcolor=CARD_BORDER)
    st.plotly_chart(fig, use_container_width=True)

    # Stability breakdown per inverter
    section("📋", "Stability Breakdown per Inverter", "SIMULATED DATA")
    stab_df = (df_inv.groupby(["inverter_id", "control_mode", "stability_flag"])
               .size().reset_index(name="count"))
    stab_pct = stab_df.copy()
    totals = stab_pct.groupby("inverter_id")["count"].transform("sum")
    stab_pct["pct"] = (stab_pct["count"] / totals * 100).round(1)
    sc = {f: c for f, c in
          [("stable", ACCENT_GREEN), ("marginal", ACCENT_AMBER), ("unstable", ACCENT_RED)]}
    fig2 = go.Figure()
    for flag in ["stable", "marginal", "unstable"]:
        sub = stab_pct[stab_pct["stability_flag"] == flag]
        if len(sub):
            fig2.add_trace(go.Bar(
                name=flag.capitalize(), x=sub["inverter_id"], y=sub["pct"],
                marker_color=sc[flag], marker_line=dict(width=0),
                text=sub["pct"].apply(lambda v: f"{v:.0f}%"),
                textposition="inside",
            ))
    fig2.update_layout(barmode="stack", **PLOTLY_THEME, height=240,
                       xaxis_title="Inverter", yaxis_title="% of Time")
    st.plotly_chart(fig2, use_container_width=True)
# ════════════════════════════════════════════════════════════════════════════
# PAGE ⑥ — SHAP EXPLAINABILITY
# ════════════════════════════════════════════════════════════════════════════
elif page == "⑥ SHAP Explainability":
    ACCENT_CYAN  = "#00D4FF"
    ACCENT_GREEN = "#00FF9F"
    ACCENT_RED   = "#FF4757"
    ACCENT_AMBER = "#FFB020"
    TEXT_MUTED   = "#64748B"
    CARD_BORDER  = "#1F2937"

    st.markdown(f"""
    <div style="margin-bottom:20px">
        <div style="font-family:\'Syne\',sans-serif;font-size:1.4rem;font-weight:800;color:{ACCENT_CYAN}">
            ⑥ SHAP Explainability — Why does the model flag an anomaly?
        </div>
        <div style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:4px">
            SHapley Additive exPlanations — each prediction explained by individual feature
            contributions, grounded in cooperative game theory (Shapley Values).
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        import shap
        import pickle, json
        SHAP_OK = True
    except ImportError:
        SHAP_OK = False

    if not SHAP_OK:
        st.error("⚠️  Package `shap` not installed. Run: `pip install shap`")
        st.code("pip install shap", language="bash")
        st.stop()

    @st.cache_resource(show_spinner="Loading model and computing SHAP values…")
    def load_shap_explainer():
        try:
            with open("ml/anomaly_model.pkl",  "rb") as f: model  = pickle.load(f)
            with open("ml/anomaly_scaler.pkl", "rb") as f: scaler = pickle.load(f)
            with open("ml/anomaly_features.json")    as f: feats  = json.load(f)
            feat_cols = feats if isinstance(feats, list) else feats.get("features", [])
            # IsolationForest: use shap.Explainer with masker for compatibility
            # TreeExplainer can work with IsolationForest in newer shap versions
            try:
                explainer = shap.TreeExplainer(model)
            except Exception:
                # Fallback: use KernelExplainer on a small background sample
                explainer = None
            return model, scaler, feat_cols, explainer
        except FileNotFoundError:
            return None, None, None, None

    model, scaler, feat_cols, shap_exp = load_shap_explainer()

    if model is None:
        st.warning("Models not found in ml/. Run notebook 02 first.")
        st.stop()

    @st.cache_data(ttl=60, show_spinner="Generating server snapshots…")
    def get_shap_data(_r):
        from data_generator.server_simulator import ServerSimulator
        import pandas as pd
        from dataclasses import asdict
        from datetime import datetime, timezone, timedelta

        sim  = ServerSimulator(num_servers=100, num_racks=10,
                               fault_probability=0.08, random_seed=None)
        rows = []
        for i in range(48):
            ts = datetime.now(timezone.utc) - timedelta(minutes=5 * i)
            rows.extend([asdict(r) for r in sim.generate_snapshot(ts)])
        return pd.DataFrame(rows)

    df_shap_raw = get_shap_data(refresh)

    available_feats = [f for f in feat_cols if f in df_shap_raw.columns]
    X_sample = df_shap_raw[available_feats].dropna()
    if len(X_sample) > 300:
        X_sample = X_sample.sample(300, random_state=42)

    X_scaled = scaler.transform(X_sample[available_feats])
    scores   = model.predict_proba(X_scaled)[:, 1]
    X_sample = X_sample.copy()
    X_sample["anomaly_score"] = scores
    X_sample["is_anomaly"]    = (scores >= 0.34).astype(int)

    # ── KPIs ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    n_anom = int((scores >= 0.34).sum())
    n_high = int((scores >= 0.65).sum())
    mean_sc = float(scores.mean())
    max_sc  = float(scores.max())

    with c1: kpi("Samples Analyzed",    str(len(X_sample)), "servers")
    with c2: kpi("Anomalies Detected",  str(n_anom), "threshold=0.34",
                 "red" if n_anom > 0 else "")
    with c3: kpi("Max Score",           f"{max_sc:.3f}", "most critical server",
                 "red" if max_sc > 0.65 else "amber")
    with c4: kpi("Mean Score",          f"{mean_sc:.3f}", "fleet average")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Global SHAP ───────────────────────────────────────────────────────
    section("📊", "Global Feature Importance", "MEAN |SHAP VALUE|")

    import hashlib
    hash_key = hashlib.md5(X_scaled.tobytes()).hexdigest()[:8]

    @st.cache_data(ttl=60, show_spinner="Computing SHAP values…")
    def compute_shap_global(key):
        if shap_exp is None:
            return None
        sv = shap_exp.shap_values(X_scaled)
        if isinstance(sv, list): sv = sv[1]
        return sv

    sv_all   = compute_shap_global(hash_key) if shap_exp is not None else None
    mean_abs = pd.Series(
        np.abs(sv_all).mean(axis=0),
        index=available_feats,
    ).sort_values(ascending=False)

    FEAT_LABELS = {
        "cpu_temp_c":         "CPU Temperature (°C)",
        "power_draw_w":       "Power Draw (W)",
        "cpu_utilization":    "CPU Utilization (%)",
        "memory_utilization": "Memory Utilization (%)",
        "cpu_temp_roll_mean": "CPU Temp — 10-sample Mean",
        "power_roll_std":     "Power Draw — Rolling Std",
        "cpu_util_roll_mean": "CPU Util — 10-sample Mean",
        "temp_zscore":        "CPU Temp — Z-Score",
    }

    col1, col2 = st.columns(2)
    with col1:
        top12  = mean_abs.head(12).sort_values()
        labels = [FEAT_LABELS.get(n, n) for n in top12.index]
        fig_imp = go.Figure(go.Bar(
            x=top12.values, y=labels, orientation="h",
            marker=dict(
                color=top12.values,
                colorscale=[[0,"#1A3A5C"],[0.5,"#2E75B6"],[1,"#00D4FF"]],
                line=dict(width=0),
            ),
            text=[f"{v:.4f}" for v in top12.values], textposition="outside",
        ))
        fig_imp.update_layout(
            height=400, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono,monospace", color="#F1F5F9", size=10),
            margin=dict(l=10,r=10,t=30,b=10), xaxis_title="Mean |SHAP|",
            title="Global Importance",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col2:
        # Simplified beeswarm (SHAP value scatter colored by feature value)
        top5 = mean_abs.head(5).index.tolist()
        fig_bee = go.Figure()
        for j, feat in enumerate(top5):
            fi      = available_feats.index(feat)
            sv_col  = sv_all[:, fi]
            raw_col = X_sample[feat].values[:len(sv_col)]
            raw_norm= (raw_col - raw_col.min()) / (raw_col.ptp() + 1e-9)
            fig_bee.add_trace(go.Scatter(
                x=sv_col,
                y=[FEAT_LABELS.get(feat, feat)] * len(sv_col)
                  + np.random.default_rng(j).normal(0, 0.06, len(sv_col)),
                mode="markers", showlegend=False,
                marker=dict(size=4, opacity=0.5, color=raw_norm,
                            colorscale=[[0,"#1A6FAF"],[1,"#FF4757"]]),
            ))
        fig_bee.add_vline(x=0, line_color="rgba(255,255,255,0.2)", line_dash="dot")
        fig_bee.update_layout(
            height=400, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono,monospace", color="#F1F5F9", size=10),
            margin=dict(l=10,r=10,t=30,b=10),
            xaxis_title="SHAP value", title="Beeswarm — Top 5 Features",
        )
        st.plotly_chart(fig_bee, use_container_width=True)

    # ── Individual Explanation ────────────────────────────────────────────
    section("💧", "Individual Explanation — Waterfall", "SELECT A SERVER")

    srv_scores = df_shap_raw.copy()
    # IsolationForest uses score_samples (lower=more anomalous); normalize to 0-1
    _raw = model.score_samples(scaler.transform(srv_scores[available_feats].fillna(0)))
    _min, _max = _raw.min(), _raw.max()
    srv_scores["anomaly_score"] = 1.0 - (_raw - _min) / (_max - _min + 1e-9)

    top_srvs = (srv_scores
                .groupby("server_id")["anomaly_score"]
                .max()
                .sort_values(ascending=False)
                .head(10))

    sel_srv  = st.selectbox("Server", top_srvs.index.tolist(),
                             format_func=lambda x: f"{x}  (score={top_srvs[x]:.3f})")

    srv_rows  = srv_scores[srv_scores["server_id"] == sel_srv]
    worst_row = srv_rows.loc[srv_rows["anomaly_score"].idxmax(), available_feats]

    if shap_exp is None:
        st.info("SHAP explainer unavailable for this model type. Showing score only.")
        sv_inst  = np.zeros(len(available_feats))
        base_v   = 0.0
    else:
        sv_inst = shap_exp.shap_values(scaler.transform(worst_row.values.reshape(1, -1)))
        if isinstance(sv_inst, list): sv_inst = sv_inst[1]
        sv_inst = sv_inst[0]
        base_v = (shap_exp.expected_value[1]
                  if isinstance(shap_exp.expected_value, (list, np.ndarray))
                  else shap_exp.expected_value)
    score_inst = float(srv_rows["anomaly_score"].max())

    sorted_idx  = np.argsort(np.abs(sv_inst))[::-1][:8]
    feat_names  = [FEAT_LABELS.get(available_feats[i], available_feats[i]) for i in sorted_idx]
    sv_top      = [float(sv_inst[i]) for i in sorted_idx]

    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"] + ["relative"] * len(feat_names) + ["total"],
        x=["Base"] + feat_names + ["Final Score"],
        y=[float(base_v)] + sv_top + [score_inst],
        connector={"line": {"color": "rgba(255,255,255,0.2)"}},
        increasing={"marker": {"color": ACCENT_RED}},
        decreasing={"marker": {"color": ACCENT_GREEN}},
        totals={"marker": {"color": ACCENT_CYAN}},
        text=[f"{v:.3f}" for v in [float(base_v)] + sv_top + [score_inst]],
        textposition="outside",
    ))
    fig_wf.add_hline(y=0.34, line_dash="dash", line_color=ACCENT_AMBER,
                     annotation_text="Threshold (0.34)")
    fig_wf.update_layout(
        height=360, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono,monospace", color="#F1F5F9", size=11),
        margin=dict(l=10,r=10,t=30,b=40), showlegend=False,
        title=f"SHAP Waterfall — {sel_srv} | Score = {score_inst:.3f}",
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    risk       = "High" if score_inst >= 0.65 else ("Medium" if score_inst >= 0.34 else "Low")
    risk_color = ACCENT_RED if risk=="High" else (ACCENT_AMBER if risk=="Medium" else ACCENT_GREEN)
    top_push   = [available_feats[i] for i in sorted_idx if sv_inst[i] > 0][:3]
    diag_text  = f"**{risk} risk** — main drivers: {', '.join(FEAT_LABELS.get(f,f) for f in top_push)}"
    st.markdown(
        f"<div style=\'padding:12px;background:rgba(0,0,0,0.3);border-left:3px solid {risk_color};"
        f"border-radius:4px;font-size:0.82rem\'>{diag_text}</div>",
        unsafe_allow_html=True
    )

# ════════════════════════════════════════════════════════════════════════════
# PAGE ⑦ — WEATHER & ENERGY PRICE
# ════════════════════════════════════════════════════════════════════════════
elif page == "⑦ Weather & Energy Price":
    ACCENT_AMBER  = "#FFB020"
    ACCENT_CYAN   = "#00D4FF"
    ACCENT_GREEN  = "#00FF9F"
    ACCENT_PURPLE = "#A855F7"
    ACCENT_RED    = "#FF4757"
    TEXT_MUTED    = "#64748B"
    CARD_BORDER   = "#1F2937"

    st.markdown(f"""
    <div style="margin-bottom:20px">
        <div style="font-family:\'Syne\',sans-serif;font-size:1.4rem;font-weight:800;color:{ACCENT_AMBER}">
            ⑦ External Data Fusion — Weather & Energy Price
        </div>
        <div style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:4px">
            Outdoor temperature, solar irradiance and energy spot price integrated
            into the multivariate LSTM PUE forecasting model.
            &nbsp;<b style="color:{ACCENT_AMBER}">Higher temperature = more cooling = higher PUE = higher cost.</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not EXTERNAL_AVAILABLE:
        st.error("Module external_data_fetcher not found. Place it in data_generator/.")
        st.stop()

    @st.cache_data(ttl=300, show_spinner="Fetching external data…")
    def load_external(_r):
        fetcher = ExternalDataFetcher(
            owm_api_key=os.getenv("OWM_API_KEY", ""),
            lat=-27.60, lon=-48.55,
        )
        return fetcher.get_merged_features(hours=48)

    df_ext = load_external(refresh)
    latest = df_ext.iloc[-1]

    # ── KPIs ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Outdoor Temp.",     f"{latest.get('temp_c',0):.1f}°C",  "Florianopolis")
    with c2: kpi("Humidity",          f"{latest.get('humidity_pct',0):.0f}%", "relative humidity")
    with c3: kpi("Solar Irradiance",  f"{latest.get('solar_ghi_wm2',0):.0f}",
                 "W/m² (GHI)",
                 badge="Solar Peak" if latest.get("solar_ghi_wm2",0)>500 else "Low",
                 badge_type="warn"  if latest.get("solar_ghi_wm2",0)>500 else "ok")
    with c4: kpi("Energy Price",      f"BRL {latest.get('price_brl_mwh',0):.0f}",
                 "per MWh (PLD mock)",
                 badge="Peak Hour" if latest.get("is_peak",0)>0.5 else "Off-peak",
                 badge_type="err"  if latest.get("is_peak",0)>0.5 else "ok")
    with c5: kpi("Cooling Factor",    f"{latest.get('cooling_load_factor',0):.2f}",
                 "0=min, 1=max",
                 "red" if latest.get("cooling_load_factor",0)>0.7 else "amber")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Time-series charts ────────────────────────────────────────────────
    section("🌡️", "Temperature & Energy Price — 48h", "TIME SERIES")
    fig_clim = make_subplots(rows=3, cols=1, shared_xaxes=True,
                              vertical_spacing=0.05, row_heights=[0.35, 0.35, 0.30])

    fig_clim.add_trace(go.Scatter(
        x=df_ext["timestamp_utc"], y=df_ext["temp_c"],
        name="Temperature (°C)", fill="tozeroy",
        line=dict(color=ACCENT_RED, width=1.5),
        fillcolor="rgba(255,71,87,0.08)",
    ), row=1, col=1)
    fig_clim.add_trace(go.Scatter(
        x=df_ext["timestamp_utc"], y=df_ext["humidity_pct"],
        name="Humidity (%)", line=dict(color=ACCENT_CYAN, width=1.2),
    ), row=1, col=1)
    fig_clim.add_trace(go.Scatter(
        x=df_ext["timestamp_utc"], y=df_ext["solar_ghi_wm2"],
        name="Solar Irradiance (W/m²)", fill="tozeroy",
        line=dict(color=ACCENT_AMBER, width=1.5),
        fillcolor="rgba(255,176,32,0.08)",
    ), row=2, col=1)
    fig_clim.add_trace(go.Scatter(
        x=df_ext["timestamp_utc"], y=df_ext["price_brl_mwh"],
        name="Energy Price (BRL/MWh)", fill="tozeroy",
        line=dict(color=ACCENT_GREEN, width=1.5),
        fillcolor="rgba(0,255,159,0.06)",
    ), row=3, col=1)

    fig_clim.update_layout(
        height=460, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono,monospace", color="#F1F5F9", size=10),
        margin=dict(l=10,r=10,t=20,b=10),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor=CARD_BORDER),
    )
    fig_clim.update_xaxes(gridcolor=CARD_BORDER)
    fig_clim.update_yaxes(gridcolor=CARD_BORDER)
    fig_clim.update_yaxes(title_text="Temp./Humidity", row=1, col=1)
    fig_clim.update_yaxes(title_text="W/m²",           row=2, col=1)
    fig_clim.update_yaxes(title_text="BRL/MWh",        row=3, col=1)
    st.plotly_chart(fig_clim, use_container_width=True)

    # ── Correlation analysis ──────────────────────────────────────────────
    section("🔗", "Correlation: Temperature × Cooling Load", "DEPENDENCY ANALYSIS")
    col1, col2 = st.columns(2)
    with col1:
        fig_corr = go.Figure(go.Scatter(
            x=df_ext["temp_c"], y=df_ext["cooling_load_factor"], mode="markers",
            marker=dict(
                color=df_ext["price_brl_mwh"],
                colorscale=[[0,ACCENT_CYAN],[0.5,ACCENT_AMBER],[1,ACCENT_RED]],
                size=5, opacity=0.5,
                colorbar=dict(title="BRL/MWh", thickness=10),
            ),
        ))
        fig_corr.update_layout(
            height=300, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono,monospace", color="#F1F5F9", size=10),
            margin=dict(l=10,r=10,t=30,b=10),
            xaxis_title="Outdoor Temperature (°C)",
            yaxis_title="Cooling Load Factor",
            title="Temp × Cooling (color = energy price)",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with col2:
        # Average hourly price profile
        df_ext["hour"] = df_ext["timestamp_utc"].dt.hour
        price_hourly   = df_ext.groupby("hour")["price_brl_mwh"].mean()
        fig_hp = go.Figure(go.Bar(
            x=price_hourly.index, y=price_hourly.values,
            marker=dict(
                color=price_hourly.values,
                colorscale=[[0,ACCENT_CYAN],[0.5,ACCENT_AMBER],[1,ACCENT_RED]],
                line=dict(width=0),
            ),
        ))
        fig_hp.add_hline(y=price_hourly.mean(), line_dash="dash",
                         line_color="rgba(255,255,255,0.3)",
                         annotation_text="Daily average")
        fig_hp.update_layout(
            height=300, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono,monospace", color="#F1F5F9", size=10),
            margin=dict(l=10,r=10,t=30,b=10),
            xaxis_title="Hour of day", yaxis_title="BRL/MWh",
            title="Average Hourly Energy Price Profile",
        )
        st.plotly_chart(fig_hp, use_container_width=True)

    # ── Cyclical features ─────────────────────────────────────────────────
    section("📐", "Cyclical Time Features for LSTM", "sin/cos ENCODING")
    fig_cyc = go.Figure()
    fig_cyc.add_trace(go.Scatter(
        x=df_ext["timestamp_utc"], y=df_ext["time_sin"],
        name="sin(hour)", line=dict(color=ACCENT_CYAN, width=1.5),
    ))
    fig_cyc.add_trace(go.Scatter(
        x=df_ext["timestamp_utc"], y=df_ext["time_cos"],
        name="cos(hour)", line=dict(color=ACCENT_PURPLE, width=1.5),
    ))
    fig_cyc.add_trace(go.Scatter(
        x=df_ext["timestamp_utc"], y=df_ext["is_peak"],
        name="Peak Hour Flag", line=dict(color=ACCENT_RED, width=1, dash="dot"),
    ))
    fig_cyc.update_layout(
        height=240, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono,monospace", color="#F1F5F9", size=10),
        margin=dict(l=10,r=10,t=20,b=10),
        xaxis_title="Time", yaxis_title="Value",
    )
    st.plotly_chart(fig_cyc, use_container_width=True)
# ════════════════════════════════════════════════════════════════════════════
# PAGE ⑧ — BODE / NYQUIST
# ════════════════════════════════════════════════════════════════════════════
elif page == "⑧ Bode / Nyquist":
    ACCENT_RED   = "#FF4757"
    ACCENT_GREEN = "#00FF9F"
    ACCENT_CYAN  = "#00D4FF"
    ACCENT_AMBER = "#FFB020"
    TEXT_MUTED   = "#64748B"
    CARD_BORDER  = "#1F2937"

    st.markdown(f"""
    <div style="margin-bottom:20px">
        <div style="font-family:\'Syne\',sans-serif;font-size:1.4rem;font-weight:800;color:{ACCENT_RED}">
            ⑧ Frequency-Domain Stability Analysis
        </div>
        <div style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:4px">
            Bode and Nyquist diagrams of the GFL PLL control loop and Middlebrook Criterion for GFM.
            &nbsp;<b style="color:{ACCENT_RED}">Based on classical control theory — phase margin and gain margin.</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not STABILITY_AVAILABLE:
        # ── Built-in stability analysis (no external module needed) ───────────
        import numpy as np

        def _pll_openloop(scr, kp=50.0, ki=2500.0, omega=None):
            """GFL PLL open-loop transfer function: sampled frequency response."""
            freqs = np.logspace(-1, 4, 500)
            s = 1j * 2 * np.pi * freqs
            # PLL compensator: (kp*s + ki) / s
            # Plant (grid): 1 / (scr * s)
            L = (kp * s + ki) / (s * scr * s)
            return freqs, L

        def _phase_margin(scr):
            freqs, L = _pll_openloop(scr)
            mag = np.abs(L)
            idx = np.argmin(np.abs(mag - 1.0))
            phase = np.angle(L[idx], deg=True)
            return phase + 180.0

        def _gain_margin_db(scr):
            freqs, L = _pll_openloop(scr)
            phase = np.angle(L, deg=True)
            idx = np.argmin(np.abs(phase + 180.0))
            return -20 * np.log10(np.abs(L[idx]) + 1e-9)

        scr_sweep = np.arange(0.5, 10.5, 0.5)
        pm_vals   = [_phase_margin(s) for s in scr_sweep]
        gm_vals   = [_gain_margin_db(s) for s in scr_sweep]

        # Status card
        pm_now = _phase_margin(scr_sel)
        gm_now = _gain_margin_db(scr_sel)
        if pm_now >= 45:
            status_label, status_color = "✅ STABLE", "#00FF9F"
        elif pm_now >= 20:
            status_label, status_color = "⚠️ MARGINAL", "#FFB020"
        else:
            status_label, status_color = "❌ UNSTABLE", "#FF4757"

        st.markdown(f"""
        <div style="padding:16px;background:rgba(0,0,0,0.3);border:1px solid {status_color};
                    border-radius:8px;margin-bottom:16px;display:flex;gap:24px;align-items:center">
            <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;color:{status_color}">
                {status_label}
            </div>
            <div>
                <div style="font-size:0.72rem;color:#94A3B8">
                    PM = {pm_now:.1f}° &nbsp;|&nbsp; GM = {gm_now:.1f} dB
                    &nbsp;|&nbsp; SCR = {scr_sel:.1f}
                </div>
                <div style="font-size:0.68rem;color:#64748B;margin-top:4px">
                    GFM: ✅ Stable for any SCR value (no PLL dependency)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Bode diagram
        section("📈", "Bode Diagram — Open-Loop PLL", "GFL vs GFM")
        freqs, L_gfl = _pll_openloop(scr_sel)
        mag_db  = 20 * np.log10(np.abs(L_gfl) + 1e-12)
        phase_d = np.angle(L_gfl, deg=True)

        fig_b = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              subplot_titles=("Magnitude (dB)", "Phase (°)"))
        fig_b.add_trace(go.Scatter(x=freqs, y=mag_db,  name="GFL",  line=dict(color="#00D4FF")), row=1, col=1)
        fig_b.add_trace(go.Scatter(x=freqs, y=phase_d, name="GFL",  line=dict(color="#00D4FF"), showlegend=False), row=2, col=1)
        # GFM is unconditionally stable — flat 90° phase margin approximation
        fig_b.add_trace(go.Scatter(x=freqs, y=[6.0]*len(freqs),   name="GFM (∞ SCR)", line=dict(color="#00FF9F", dash="dash")), row=1, col=1)
        fig_b.add_trace(go.Scatter(x=freqs, y=[-90.0]*len(freqs), name="GFM", line=dict(color="#00FF9F", dash="dash"), showlegend=False), row=2, col=1)
        fig_b.add_hline(y=0,    row=1, col=1, line_dash="dot", line_color="white", opacity=0.3)
        fig_b.add_hline(y=-180, row=2, col=1, line_dash="dot", line_color="#FF4757", opacity=0.6)
        fig_b.update_xaxes(type="log", title_text="Frequency (Hz)")
        fig_b.update_layout(height=420, template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="JetBrains Mono,monospace", color="#F1F5F9", size=10),
                            margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig_b, use_container_width=True)

        # Phase margin vs SCR
        section("📉", "Phase Margin vs SCR — GFL vs GFM", "STABILITY ENVELOPE")
        col_pm, col_gm = st.columns(2)
        with col_pm:
            fig_pm = go.Figure()
            fig_pm.add_trace(go.Scatter(x=scr_sweep, y=pm_vals, name="GFL PM",
                                        line=dict(color="#00D4FF"), fill="tozeroy",
                                        fillcolor="rgba(0,212,255,0.08)"))
            fig_pm.add_trace(go.Scatter(x=scr_sweep, y=[90.0]*len(scr_sweep), name="GFM PM",
                                        line=dict(color="#00FF9F", dash="dash")))
            fig_pm.add_hline(y=45, line_dash="dot", line_color="#FFB020",
                             annotation_text="45° margin (IEEE)")
            fig_pm.add_vline(x=scr_sel, line_dash="dot", line_color="#FFB020",
                             annotation_text=f"SCR={scr_sel:.1f}")
            fig_pm.update_layout(height=300, template="plotly_dark",
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 xaxis_title="SCR", yaxis_title="Phase Margin (°)",
                                 font=dict(family="JetBrains Mono,monospace", color="#F1F5F9", size=10),
                                 margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig_pm, use_container_width=True)
        with col_gm:
            fig_gm = go.Figure()
            fig_gm.add_trace(go.Scatter(x=scr_sweep, y=gm_vals, name="GFL GM",
                                        line=dict(color="#FF4757")))
            fig_gm.add_trace(go.Scatter(x=scr_sweep, y=[40.0]*len(scr_sweep), name="GFM GM",
                                        line=dict(color="#00FF9F", dash="dash")))
            fig_gm.add_hline(y=6, line_dash="dot", line_color="#FFB020",
                             annotation_text="6 dB minimum")
            fig_gm.add_vline(x=scr_sel, line_dash="dot", line_color="#FFB020")
            fig_gm.update_layout(height=300, template="plotly_dark",
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 xaxis_title="SCR", yaxis_title="Gain Margin (dB)",
                                 font=dict(family="JetBrains Mono,monospace", color="#F1F5F9", size=10),
                                 margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig_gm, use_container_width=True)

        # Summary table
        section("📋", "Summary Table — SCR × Stability", "COMPUTED VALUES")
        rows_data = []
        for s in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
            pm_s = _phase_margin(s)
            gm_s = _gain_margin_db(s)
            rows_data.append({
                "SCR": s,
                "PM GFL (°)":  f"{pm_s:.1f}",
                "GM GFL (dB)": f"{gm_s:.1f}",
                "GFL Status":  ("✅ Stable" if pm_s >= 45
                                else ("⚠️ Marginal" if pm_s >= 20 else "❌ Unstable")),
                "GFM Status":  "✅ Stable",
            })
        df_table = pd.DataFrame(rows_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True)
        st.stop()  # skip the rest of the page (which uses STABILITY_AVAILABLE functions)
    # End of built-in stability analysis
    # (full stability analysis available with analysis/stability_analysis.py)

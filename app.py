import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium
import difflib
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from html import escape
from urllib.parse import quote

from aqi_fetcher import fetch_aqi
from data_processing import process_data
from visualization import plot_pollutants
from model import (
    generate_dummy_data,
    train_model,
    predict_future,
    generate_7day_trend
)
from map_view import create_map


# Newer Forecast & Trend card renderer used by the live section below.
def render_forecast_card(title, data, x_col, y_col, line_color, file_name, chart_id):
    csv_href = "data:text/csv;charset=utf-8," + quote(data.to_csv(index=False))
    fig = px.line(data, x=x_col, y=y_col, markers=True)
    labels = [
        "" if pd.isna(value) else f"{value:.0f}" if isinstance(value, (int, float)) else str(value)
        for value in data[y_col]
    ]
    fig.update_traces(
        line_color=line_color,
        marker_color=line_color,
        line_width=3,
        marker_size=8,
        text=labels,
        textposition="top center",
        textfont=dict(color="#15264a", size=11, family="Poppins, sans-serif"),
        hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
    )

    xaxis_config = dict(title=x_col, showgrid=False, zeroline=False, fixedrange=True)
    if x_col.lower() == "hour" and len(data) == 6:
        hours = data[x_col].tolist()
        xaxis_config.update(
            tickmode="array",
            tickvals=hours,
            ticktext=["Now"] + [f"+{idx}h" for idx in range(1, len(hours))],
        )

    fig.update_layout(
        height=240,
        template="plotly_white",
        margin=dict(l=42, r=18, t=22, b=42),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#17254a", size=11),
        xaxis=xaxis_config,
        yaxis=dict(title=y_col, gridcolor="#e7eef8", zeroline=False, fixedrange=True),
        showlegend=False,
    )
    chart_html = fig.to_html(
        include_plotlyjs=True,
        full_html=False,
        config={"displayModeBar": False, "responsive": True},
        div_id=chart_id,
    )
    table_headers = "".join(f"<th>{escape(str(col))}</th>" for col in data.columns)
    table_rows = "".join(
        "<tr>" + "".join(
            f"<td>{escape(f'{row[col]:.1f}' if isinstance(row[col], float) else str(row[col]))}</td>"
            for col in data.columns
        ) + "</tr>"
        for _, row in data.iterrows()
    )

    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&display=swap');
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            background: transparent;
            font-family: 'Poppins', sans-serif;
        }}
        .forecast-ui-card {{
            background: rgba(255,255,255,0.98);
            border: 1px solid #dbe7f8;
            border-radius: 12px;
            box-shadow: 0 14px 34px rgba(42,70,120,0.08);
            padding: 14px 16px 12px;
            min-height: 318px;
            overflow: hidden;
        }}
        .forecast-ui-head {{
            display: grid;
            grid-template-columns: minmax(130px, 1fr) auto auto;
            align-items: center;
            gap: 14px;
            margin-bottom: 8px;
        }}
        .forecast-ui-title {{
            color: #15264a;
            font-size: 14px;
            font-weight: 900;
            white-space: nowrap;
        }}
        .forecast-toggle {{
            display: inline-flex;
            padding: 3px;
            border-radius: 9px;
            background: #eef4ff;
            border: 1px solid #dce8f8;
            color: #15264a;
            font-size: 11px;
            font-weight: 900;
        }}
        .forecast-toggle button {{
            border: 0;
            background: transparent;
            color: #15264a;
            cursor: pointer;
            font: inherit;
            min-width: 58px;
            padding: 6px 13px;
            border-radius: 7px;
            transition: background .18s ease, color .18s ease, box-shadow .18s ease;
        }}
        .forecast-toggle button.active {{
            color: #fff;
            background: linear-gradient(90deg, #2f9bff, #7357ff);
            box-shadow: 0 8px 16px rgba(80,95,255,0.22);
        }}
        .forecast-download {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 7px;
            min-height: 34px;
            padding: 0 13px;
            border-radius: 10px;
            border: 1px solid #d7e4f6;
            background: #fff;
            color: #15264a;
            font-size: 11px;
            font-weight: 900;
            text-decoration: none;
            box-shadow: 0 8px 18px rgba(42,70,120,0.06);
            white-space: nowrap;
        }}
        .forecast-download:hover {{
            border-color: #9fbbff;
            color: #3157ff;
        }}
        .forecast-download span {{
            font-size: 13px;
            line-height: 1;
        }}
        .forecast-panel {{
            min-height: 246px;
        }}
        .forecast-chart-panel .plotly-graph-div {{
            width: 100% !important;
        }}
        .forecast-table-panel {{
            display: none;
            padding-top: 12px;
        }}
        .forecast-table {{
            width: 100%;
            border-collapse: collapse;
            color: #15264a;
            font-size: 12px;
            overflow: hidden;
            border: 1px solid #e4edf8;
            border-radius: 10px;
        }}
        .forecast-table th,
        .forecast-table td {{
            border-bottom: 1px solid #e4edf8;
            padding: 10px 8px;
            text-align: center;
        }}
        .forecast-table th {{
            background: #f3f7ff;
            font-weight: 900;
        }}
        .forecast-table tbody tr:last-child td {{
            border-bottom: 0;
        }}
        .forecast-table tbody tr:hover {{
            background: #f7fbff;
        }}
        @media (max-width: 640px) {{
            .forecast-ui-head {{
                grid-template-columns: 1fr;
                justify-items: start;
                gap: 9px;
            }}
            .forecast-ui-card {{
                padding: 13px;
            }}
        }}
    </style>
    <div class="forecast-ui-card" id="{chart_id}-card">
        <div class="forecast-ui-head">
            <div class="forecast-ui-title">{escape(title)}</div>
            <div class="forecast-toggle" aria-label="Switch forecast view">
                <button type="button" class="active" data-mode="graph" aria-pressed="true">Graph</button>
                <button type="button" data-mode="table" aria-pressed="false">Table</button>
            </div>
            <a class="forecast-download" href="{csv_href}" download="{escape(file_name)}"><span>↓</span> Download CSV</a>
        </div>
        <div class="forecast-panel forecast-chart-panel">{chart_html}</div>
        <div class="forecast-panel forecast-table-panel">
            <table class="forecast-table">
                <thead><tr>{table_headers}</tr></thead>
                <tbody>{table_rows}</tbody>
            </table>
        </div>
    </div>
    <script>
        (() => {{
            const card = document.getElementById("{chart_id}-card");
            const graph = card.querySelector(".forecast-chart-panel");
            const table = card.querySelector(".forecast-table-panel");
            const buttons = card.querySelectorAll(".forecast-toggle button");
            buttons.forEach((button) => {{
                button.addEventListener("click", () => {{
                    const showTable = button.dataset.mode === "table";
                    graph.style.display = showTable ? "none" : "block";
                    table.style.display = showTable ? "block" : "none";
                    buttons.forEach((item) => {{
                        const active = item === button;
                        item.classList.toggle("active", active);
                        item.setAttribute("aria-pressed", active ? "true" : "false");
                    }});
                    if (!showTable && window.Plotly) {{
                        const plot = graph.querySelector(".plotly-graph-div");
                        if (plot) window.Plotly.Plots.resize(plot);
                    }}
                }});
            }});
        }})();
    </script>
    """


# =========================
# 🔗 PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AirFlow Intelligence",
    layout="wide",
    page_icon="🌀",
    initial_sidebar_state="expanded"
)


# =========================
# 📑 CITY LIST
# =========================
CITIES = [
    "Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore",
    "Pune", "Hyderabad", "Ahmedabad", "Jaipur",
    "New York", "London", "Paris", "Tokyo", "Dubai",
    "Los Angeles", "Chicago", "Toronto", "Berlin", "Sydney"
]


# =========================
# 😎 SMART CORRECTION
# =========================
def correct_city(user_input):
    user_input = user_input.strip().title()
    match = difflib.get_close_matches(user_input, CITIES, n=1, cutoff=0.6)
    if match:
        return match[0]
    return user_input


# =========================
# 💡 AQI HELPERS
# =========================
def get_aqi_status(aqi):
    if aqi <= 50:
        return "Good", "#00e5a0", "🟢", "rgba(0,229,160,0.12)", "rgba(0,229,160,0.35)"
    elif aqi <= 100:
        return "Moderate", "#f5c842", "🟡", "rgba(245,200,66,0.12)", "rgba(245,200,66,0.35)"
    elif aqi <= 150:
        return "Sensitive", "#ff8c42", "🟠", "rgba(255,140,66,0.12)", "rgba(255,140,66,0.35)"
    elif aqi <= 200:
        return "Unhealthy", "#ff4d6d", "🔴", "rgba(255,77,109,0.12)", "rgba(255,77,109,0.35)"
    else:
        return "Hazardous", "#c77dff", "🟣", "rgba(199,125,255,0.12)", "rgba(199,125,255,0.35)"


def health_advice(aqi):
    if aqi <= 50:
        return "Air quality is excellent. Perfect conditions for all outdoor activities."
    elif aqi <= 100:
        return "Air quality is acceptable. Unusually sensitive people should consider reducing prolonged outdoor exertion."
    elif aqi <= 150:
        return "Members of sensitive groups may experience health effects. Reduce prolonged or heavy outdoor exertion."
    elif aqi <= 200:
        return "Everyone may begin to experience health effects. Avoid prolonged outdoor exertion."
    else:
        return "Health warnings of emergency conditions. Entire population is likely to be affected. Stay indoors."


def aqi_risk_percent(aqi):
    return min(int((aqi / 500) * 100), 100)


def build_live_stream_table(df, api_data):
    iaqi = api_data.get("iaqi", {})

    def metric(column=None, api_key=None, default=0):
        if column and column in df:
            try:
                return float(df[column].iloc[0])
            except (TypeError, ValueError):
                pass
        try:
            return float(iaqi.get(api_key, {}).get("v", default))
        except (TypeError, ValueError):
            return float(default)

    current = {
        "AQI": metric("AQI"),
        "PM2.5": metric("PM2.5"),
        "PM10": metric("PM10"),
        "NO2": metric("NO2"),
        "O3": metric("O3"),
        "Temp": metric(api_key="t", default=28),
        "Humidity": metric(api_key="h", default=48),
    }

    hourly_offsets = [0, 1, 2, 3]
    variation = {
        "AQI": [0, -2, -5, -7],
        "PM2.5": [0, -3, -6, -8],
        "PM10": [0, -2, -4, -6],
        "NO2": [0, -0.4, -0.7, -0.9],
        "O3": [0, -0.3, -0.9, -3.0],
        "Temp": [0, 0, -1, -1],
        "Humidity": [0, -2, -3, -4],
    }
    now = datetime.now().replace(minute=0, second=0, microsecond=0)

    rows = []
    for idx, offset in enumerate(hourly_offsets):
        row_time = now - timedelta(hours=offset)
        rows.append({
            "Time": row_time.strftime("%I:%M %p"),
            "AQI": int(round(max(current["AQI"] + variation["AQI"][idx], 0))),
            "PM2.5 (µg/m³)": int(round(max(current["PM2.5"] + variation["PM2.5"][idx], 0))),
            "PM10 (µg/m³)": int(round(max(current["PM10"] + variation["PM10"][idx], 0))),
            "NO2 (µg/m³)": round(max(current["NO2"] + variation["NO2"][idx], 0), 1),
            "O3 (µg/m³)": round(max(current["O3"] + variation["O3"][idx], 0), 1),
            "Temp (°C)": int(round(current["Temp"] + variation["Temp"][idx])),
            "Humidity (%)": int(round(max(current["Humidity"] + variation["Humidity"][idx], 0))),
        })

    return pd.DataFrame(rows)


def render_live_stream_table(live_stream_df, city):
    csv_text = live_stream_df.to_csv(index=False)
    csv_href = "data:text/csv;charset=utf-8," + quote(csv_text)
    file_name = f"{city.lower().replace(' ', '_')}_aqi_data.csv"

    headers = "".join(f"<th>{col}</th>" for col in live_stream_df.columns)
    rows = []
    for _, row in live_stream_df.iterrows():
        cells = "".join(f"<td>{row[col]}</td>" for col in live_stream_df.columns)
        rows.append(f"<tr>{cells}</tr>")

    return f"""
    <div class="stream-table-card">
        <div class="stream-table-toolbar">
            <a class="stream-download-btn" href="{csv_href}" download="{file_name}">
                <span>↧</span> Download CSV
            </a>
        </div>
        <table class="stream-table">
            <thead><tr>{headers}</tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
    </div>
    """


def render_forecast_card_legacy(title, data, x_col, y_col, line_color, file_name, chart_id):
    csv_href = "data:text/csv;charset=utf-8," + quote(data.to_csv(index=False))
    fig = px.line(data, x=x_col, y=y_col, markers=True)
    fig.update_traces(line_color=line_color, marker_color=line_color, line_width=3, marker_size=8)
    fig.update_layout(
        height=240,
        template="plotly_white",
        margin=dict(l=42, r=18, t=8, b=42),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#17254a", size=11),
        xaxis=dict(title=x_col, showgrid=False, zeroline=False),
        yaxis=dict(title=y_col, gridcolor="#e7eef8", zeroline=False),
        showlegend=False,
    )
    chart_html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        config={"displayModeBar": False, "responsive": True},
        div_id=chart_id,
    )
    table_headers = "".join(f"<th>{col}</th>" for col in data.columns)
    table_rows = "".join(
        "<tr>" + "".join(f"<td>{row[col]:.1f}</td>" if isinstance(row[col], float) else f"<td>{row[col]}</td>" for col in data.columns) + "</tr>"
        for _, row in data.iterrows()
    )

    return f"""
    <div class="forecast-ui-card">
        <div class="forecast-ui-head">
            <div class="forecast-ui-title">{title}</div>
            <div class="forecast-toggle" aria-hidden="true">
                <span class="active">Graph</span><span>Table</span>
            </div>
            <a class="forecast-download" href="{csv_href}" download="{file_name}"><span>↧</span> Download CSV</a>
        </div>
        <div class="forecast-chart">{chart_html}</div>
        <details class="forecast-table-wrap">
            <summary>Table</summary>
            <table class="forecast-table">
                <thead><tr>{table_headers}</tr></thead>
                <tbody>{table_rows}</tbody>
            </table>
        </details>
    </div>
    """


# =========================
# 🎨 CSS
# =========================
def inject_css():
    st.markdown("""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&display=swap');

    *, *::before, *::after { box-sizing: border-box; }

    html, body, [class*="css"], [data-testid] {
        font-family: 'Poppins', sans-serif !important;
    }

    [data-testid="stLineChart"] {
        background: white !important;
        border-radius: 12px;
        padding: 10px;
    }

    .stApp {
        background-color: white;
        
        background-attachment: fixed;
        min-height: 100vh;
    }

    .stApp::before {
        content: '';
        position: fixed;
        top: 5%; right: 8%;
        width: 420px; height: 420px;
        background: radial-gradient(circle, rgba(0,200,255,0.09) 0%, transparent 68%);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        animation: blobDrift 20s ease-in-out infinite;
    }

    .stApp::after {
        content: '';
        position: fixed;
        bottom: 10%; left: 5%;
        width: 340px; height: 340px;
        background: radial-gradient(circle, rgba(120,40,255,0.08) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        animation: blobDrift 25s ease-in-out infinite reverse;
    }

    @keyframes blobDrift {
        0%, 100% { transform: translate(0,0) scale(1); }
        33%       { transform: translate(20px,-30px) scale(1.04); }
        66%       { transform: translate(-15px,20px) scale(0.96); }
    }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg,#00c8ff,#7b2fff);
        border-radius: 4px;
    }

    .block-container {
        padding: 0 2.5rem 5rem !important;
        max-width: 1380px !important;
    }

    /* Hide Streamlit's native sidebar toggle because the app uses one custom toggle on every screen */
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }

    /* ─── SIDEBAR TOGGLE BUTTON ─── */
    /* Visible on desktop, tablet, and mobile */
    .mobile-menu-btn {
        display: flex;
        /* Ensure it's always on top */
        position: fixed;
        top: 18px;
        left: 18px;
        z-index: 10001;
        background: rgba(255,255,255,0.92);
        border: 1px solid #dce8f8;
        border-radius: 10px;
        width: 40px;
        height: 40px;
        cursor: pointer;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        box-shadow: 0 10px 28px rgba(36,55,99,0.12);
        color: #4268d9;
        transition: left 0.3s cubic-bezier(.22,1,.36,1), transform 0.2s, box-shadow 0.2s, background 0.2s;
    }
    .mobile-menu-btn.sidebar-is-open { left: 298px; }
    .mobile-menu-btn:hover {
        transform: scale(1.07);
        background: #f6faff;
        box-shadow: 0 14px 34px rgba(74,113,230,0.18);
    }
    .mobile-menu-btn:active { transform: scale(0.95); }

    .top-control-bar {
        position: fixed;
        top: 14px;
        right: 24px;
        z-index: 10000;
        display: flex;
        align-items: center;
        gap: 14px;
        animation: fadeDown 0.7s ease both;
    }
    .top-icon-btn,
    .top-pill {
        border: 1px solid #dce8f8;
        background: rgba(255,255,255,0.92);
        color: #3157ff;
        box-shadow: 0 10px 26px rgba(36,55,99,0.10);
        backdrop-filter: blur(14px) saturate(1.18);
        -webkit-backdrop-filter: blur(14px) saturate(1.18);
    }
    .top-icon-btn {
        width: 40px;
        height: 40px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s, background 0.2s;
    }
    .top-icon-btn:hover {
        transform: translateY(-1px);
        border-color: #a8c1ff;
        box-shadow: 0 14px 30px rgba(70,103,224,0.16);
    }
    .top-icon-btn svg {
        width: 17px;
        height: 17px;
        stroke-width: 2;
    }
    .top-theme-toggle.is-dark .sun-icon { display: none; }
    .top-theme-toggle .moon-icon { display: none; }
    .top-theme-toggle.is-dark .moon-icon { display: block; }
    .top-pill {
        min-width: 50px;
        height: 40px;
        padding: 0 17px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: #fff;
        border: 0;
        font-size: 12px;
        font-weight: 900;
        letter-spacing: 0.4px;
        background: linear-gradient(135deg, #2f9bff 0%, #8639ff 100%);
        box-shadow: 0 13px 27px rgba(89,91,255,0.28);
    }
    .top-pill.af {
        min-width: 42px;
        padding: 0 13px;
        background: linear-gradient(135deg, #8b66ff 0%, #7138dd 100%);
    }

    body.airflow-dark .stApp {
        background: #071024 !important;
        color: #e8f0ff !important;
    }
    body.airflow-dark .stApp::before {
        background: radial-gradient(circle, rgba(0,200,255,0.13) 0%, transparent 68%) !important;
    }
    body.airflow-dark .stApp::after {
        background: radial-gradient(circle, rgba(123,47,255,0.12) 0%, transparent 70%) !important;
    }
    body.airflow-dark .top-icon-btn {
        background: rgba(12,22,45,0.86);
        border-color: rgba(125,156,215,0.28);
        color: #9ec4ff;
        box-shadow: 0 14px 34px rgba(0,0,0,0.28);
    }
    body.airflow-dark .navbar,
    body.airflow-dark .hero,
    body.airflow-dark .dashboard-heading,
    body.airflow-dark .numbered-section-heading,
    body.airflow-dark .sec-heading {
        color: #e8f0ff !important;
    }
    body.airflow-dark .hero-sub,
    body.airflow-dark .stat-label,
    body.airflow-dark .dash-label,
    body.airflow-dark .dash-sub,
    body.airflow-dark .location-coords,
    body.airflow-dark .aqi-caption {
        color: #aebfe0 !important;
    }
    body.airflow-dark .dash-card,
    body.airflow-dark .dashboard-map-card,
    body.airflow-dark .stream-table-card,
    body.airflow-dark .pollutant-chart-card,
    body.airflow-dark .advisory-wrap {
        background: rgba(12,22,45,0.84) !important;
        border-color: rgba(132,162,218,0.20) !important;
        box-shadow: 0 18px 44px rgba(0,0,0,0.26) !important;
    }
    body.airflow-dark .stream-table th,
    body.airflow-dark .stream-table td,
    body.airflow-dark .advisory-title,
    body.airflow-dark .location-name,
    body.airflow-dark .kpi-num {
        color: #e8f0ff !important;
    }
    body.airflow-dark [data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 88% 18%, rgba(36,78,130,0.38), transparent 30%),
            linear-gradient(180deg, #0b1429 0%, #101d35 52%, #0b1429 100%) !important;
        border-right-color: rgba(132,162,218,0.22) !important;
    }
    body.airflow-dark .sidebar-brand-name,
    body.airflow-dark .sidebar-brand-sub,
    body.airflow-dark .sidebar-section-title,
    body.airflow-dark .sidebar-nav-item,
    body.airflow-dark .sidebar-info-row,
    body.airflow-dark .status-title {
        color: #e8f0ff !important;
    }
    body.airflow-dark .sidebar-nav-item.active,
    body.airflow-dark .sidebar-nav-item:hover {
        background: rgba(37,56,94,0.72) !important;
    }
    body.airflow-dark .sidebar-status-card,
    body.airflow-dark [data-testid="stTextInput"] > div > div > input {
        background: rgba(12,22,45,0.82) !important;
        border-color: rgba(132,162,218,0.24) !important;
        color: #e8f0ff !important;
    }
    body.airflow-dark .status-subtitle,
    body.airflow-dark .stream-table th,
    body.airflow-dark .stream-table td {
        color: #aebfe0 !important;
    }

    /* Overlay for closing sidebar on mobile */
    .sidebar-overlay {
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.45);
        z-index: 9990;
        backdrop-filter: blur(2px);
        -webkit-backdrop-filter: blur(2px);
        transition: opacity 0.3s ease; /* Added transition for smoother fade */
    }
    .sidebar-overlay.active { display: block; }

    @media screen and (max-width: 768px) {
        .mobile-menu-btn.sidebar-is-open { left: min(calc(85vw - 58px), 292px); }
    }

    /* NAVBAR */
    .navbar {
    
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 20px 0 16px;
        margin-bottom: 4px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        animation: fadeDown 0.7s ease both;
    }
    .navbar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        transform: translateX(-44px);
    }
    .navbar-logo {
        width: 36px; height: 36px;
        background: linear-gradient(135deg, #00c8ff, #7b2fff);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 20px;
        box-shadow: 0 0 20px rgba(0,200,255,0.3);
    }
    .navbar-title {
        font-size: clamp(32px, 6vw, 44px) !important;
        font-weight: 700;
        color: #7b2fff;
        letter-spacing: -0.3px;
    }
    .navbar-title span {
        background: linear-gradient(90deg, #00c8ff, #7b2fff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .eyebrow {
        display: inline-flex; align-items: center; gap: 8px;
        background: rgba(0,200,255,0.08); border: 1px solid rgba(0,200,255,0.2);
        border-radius: 100px; padding: 5px 14px 5px 10px; margin-bottom: 22px;
        transform: translateY(12px);
    }
    .eyebrow-dot {
        width: 7px; height: 7px; background: #00c8ff; border-radius: 50%;
        box-shadow: 0 0 8px #00c8ff; animation: livePulse 2s ease infinite;
    }
    .navbar-links { display: flex; gap: 28px; align-items: center; }
    .navbar-link {
        font-size: 18px; display: none; font-weight: 500;
        color: rgba(200,220,255,0.5); text-decoration: none; transition: color 0.2s;
    }
    .navbar-link:hover { color: #00c8ff; }
    .navbar-cta {
        background: rgba(0,200,255,0.12); border: 1px solid rgba(0,200,255,0.3);
        color: #00c8ff !important; padding: 7px 20px; border-radius: 100px;
        font-size: 13px; font-weight: 600; transition: all 0.25s;
    }
    .navbar-cta:hover { background: rgba(0,200,255,0.2); box-shadow: 0 0 20px rgba(0,200,255,0.2); }

    /* HERO */
    .hero {
        padding: 56px 0 44px;
        animation: fadeUp 0.8s cubic-bezier(.22,1,.36,1) 0.1s both;
        background: rgba(0,200,255,0.10);
        padding: 3rem;
        border-radius: 1rem;
    }
    .eyebrow-text { font-size: 14px; font-weight: 600; color: #00c9ff; letter-spacing: 1.5px; text-transform: uppercase; }
    .hero-h1 {
        font-size: clamp(28px, 6vw, 64px) !important;
        font-weight: 800; line-height: 1.08; letter-spacing: -1.5px;
        color: rgba(0,200,255,0.9); margin-bottom: 6px;
    }
    .hero-h1 .accent {
        background: linear-gradient(135deg, #00c8ff 0%, #7b2fff 60%, #ff2fff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub { font-size: 16px; font-weight: 300; color: black; line-height: 1.75; max-width: 520px; margin-top: 14px; }
    .stat-strip { display: flex; gap: 6rem; margin-top: 36px; flex-wrap: wrap; }
    .stat-item { display: flex; flex-direction: column; }
    .stat-number { font-size: 34px; font-weight: 800; color: rgba(0,200,255,0.9); line-height: 1; }
    .stat-label { font-size: 11px; font-weight: 500; color: black; letter-spacing: 1px; text-transform: uppercase; margin-top: 3px; }
    .stat-divider { width: 1px; background: rgba(255,255,255,0.08); align-self: stretch; }

    /* SEC HEADING */
    .sec-heading {
        font-size: clamp(12px, 2vw, 16px) !important;
        font-weight: 600; color: #424242 !important; margin-top: 20px;
        letter-spacing: 2px; text-transform: uppercase; margin-bottom: 16px;
        display: flex; align-items: center; gap: 10px;
    }
    .sec-heading::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, rgba(0,200,255,0.2), transparent); }

    /* MAIN STATS CARD */
    .main-stats-card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 22px; padding: 30px; height: 100%;
        position: relative; overflow: hidden;
        transition: border-color 0.3s, box-shadow 0.3s;
        animation: fadeUp 0.8s cubic-bezier(.22,1,.36,1) 0.15s both;
    }
    .main-stats-card::before {
        content: ''; position: absolute; inset: 0;
        background: radial-gradient(ellipse 60% 50% at 80% 20%, rgba(0,200,255,0.05) 0%, transparent 70%);
        pointer-events: none;
    }
    .main-stats-card:hover { border-color: rgba(0,200,255,0.18); box-shadow: 0 0 40px rgba(0,200,255,0.06); }

    .card-badge {
        display: inline-flex; align-items: center; gap: 6px;
        font-size: 10px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase;
        padding: 4px 12px; border-radius: 100px; margin-bottom: 20px;
    }
    .card-title { font-size: 16px; font-weight: 700; color: #424242; margin-bottom: 6px; letter-spacing: -0.2px; }
    .advisory-title { font-size: 26px; font-weight: 800; color: #00c8ff; letter-spacing: -0.5px; margin-bottom: 10px; }
    .pollutant-info { font-size: 13px; color: rgba(180,200,255,0.45); font-weight: 400; }
    .pollutant-info strong { color: rgba(200,220,255,0.75); font-weight: 600; }

    /* RISK CARD */
    .risk-card {
        background: linear-gradient(135deg, rgba(0,200,255,0.08) 0%, rgba(123,47,255,0.08) 100%);
        border: 1px solid rgba(0,200,255,0.15); border-radius: 18px;
        padding: 24px; margin-top: 16px; position: relative; overflow: hidden;
        transition: box-shadow 0.3s;
    }
    .risk-card:hover { box-shadow: 0 0 30px rgba(0,200,255,0.08); }
    .risk-card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .risk-card-title { font-size: 14px; font-weight: 700; color: black; }
    .risk-details-btn {
        background: linear-gradient(90deg, #00c8ff, #7b2fff);
        color: #fff; font-size: 11px; font-weight: 700;
        padding: 5px 14px; border-radius: 100px; letter-spacing: 0.5px;
        box-shadow: 0 4px 16px rgba(0,200,255,0.25);
    }
    .risk-percent { font-size: 52px; font-weight: 900; line-height: 1; letter-spacing: -2px; color: red; margin-bottom: 6px; }
    .risk-desc { font-size: 12px; color: black; font-weight: 400; }
    .risk-bar-bg { height: 4px; background: rgba(255,255,255,0.07); border-radius: 4px; margin-top: 16px; overflow: hidden; }
    .risk-bar-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #00c8ff, #7b2fff, #ff2fff); }

    /* KPI ROW */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: clamp(10px, 2vw, 14px) !important;
        margin-bottom: 20px;
        animation: fadeUp 0.8s cubic-bezier(.22,1,.36,1) 0.2s both;
        justify-items: center;
    }
    .kpi-cell {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px; padding: 20px 22px; position: relative; overflow: hidden;
        transition: all 0.3s; width: 100%;
    }
    .kpi-cell::after {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #00c8ff, #7b2fff); opacity: 0; transition: opacity 0.3s;
    }
    .kpi-cell:hover { border-color: rgba(0,200,255,0.15); transform: translateY(-3px); box-shadow: 0 12px 36px rgba(0,0,0,0.35); }
    .kpi-cell:hover::after { opacity: 1; }
    .kpi-tag { font-size: 10px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: rgba(150,180,255,0.45); margin-bottom: 8px; }
    .kpi-num { font-size: 30px; font-weight: 800; color: #fff; letter-spacing: -0.5px; line-height: 1; }
    .kpi-meta { font-size: 11px; color: rgba(180,200,255,0.35); margin-top: 4px; }

    /* LIVE DASHBOARD LAYOUT */
    .dashboard-heading {
        font-size: 15px !important;
        font-weight: 800;
        color: #15264a !important;
        margin: 10px 0 14px;
        display: flex;
        align-items: center;
        gap: 14px;
        animation: fadeDown 0.7s ease both;
    }
    .dashboard-heading::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(102,134,194,0.28), rgba(102,134,194,0.08), transparent);
    }
    .dashboard-step {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: #6570ff;
        font-size: 15px;
        font-weight: 900;
        background: #fff;
        border: 1px solid #dce7fa;
        box-shadow: 0 8px 18px rgba(40, 72, 130, 0.08);
    }
    .numbered-section-heading {
        font-size: 15px !important;
        font-weight: 900;
        color: #15264a !important;
        margin: 18px 0 10px;
        display: flex;
        align-items: center;
        gap: 12px;
        animation: fadeDown 0.7s ease both;
    }
    .numbered-section-heading::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(102,134,194,0.30), rgba(102,134,194,0.08), transparent);
    }
    .section-step {
        width: 26px;
        height: 26px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: #6570ff;
        font-size: 14px;
        font-weight: 900;
        background: #fff;
        border: 1px solid #dce7fa;
        box-shadow: 0 8px 18px rgba(40, 72, 130, 0.08);
    }
    .data-card {
        background: rgba(255,255,255,0.96);
        border: 1px solid #dbe7f8;
        border-radius: 12px;
        box-shadow: 0 14px 34px rgba(42,70,120,0.06);
        padding: 14px;
        margin-bottom: 16px;
        overflow: hidden;
        animation: fadeUp 0.8s cubic-bezier(.22,1,.36,1) both;
    }
    .data-card [data-testid="stDataFrame"] {
        border: 0 !important;
    }
    .data-card [data-testid="stDownloadButton"] {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 8px;
    }
    .data-card [data-testid="stDownloadButton"] > button {
        width: auto !important;
        min-height: 36px !important;
        padding: 8px 14px !important;
        border-radius: 10px !important;
        background: #fff !important;
        border: 1px solid #dbe7f8 !important;
        color: #15264a !important;
        box-shadow: 0 8px 18px rgba(42,70,120,0.06) !important;
        font-size: 12px !important;
        font-weight: 900 !important;
    }
    .stream-table-card {
        position: relative;
        width: 100%;
        border: 2px solid #cddcf0;
        border-radius: 16px;
        background: #f5f5f5;
        box-shadow: 0 14px 34px rgba(42,70,120,0.10);
        overflow: hidden;
        padding-top: 0;
    }
    .stream-table-toolbar {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        min-height: 52px;
        padding: 10px 18px 6px;
        background: #f5f5f5;
        border-bottom: 1px solid #d8e3f2;
    }
    .stream-download-btn {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        min-height: 36px;
        padding: 0 15px;
        border-radius: 10px;
        border: 1px solid #d7e4f6;
        background: #fff;
        color: #15264a !important;
        font-size: 12px;
        font-weight: 900;
        text-decoration: none !important;
        box-shadow: 0 8px 18px rgba(42,70,120,0.07);
        transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
    }
    .stream-download-btn:hover {
        transform: translateY(-1px);
        border-color: #9fc0f7;
        box-shadow: 0 12px 22px rgba(42,70,120,0.10);
    }
    .stream-download-btn span {
        font-size: 15px;
        line-height: 1;
    }
    .stream-table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        color: #17254a;
        font-size: 13px;
        font-weight: 800;
    }
    .stream-table th,
    .stream-table td {
        text-align: center;
        padding: 13px 10px;
        border-bottom: 1px solid #e4edf8;
        white-space: nowrap;
    }
    .stream-table th {
        color: #18264c;
        font-size: 13px;
        font-weight: 900;
        line-height: 1.2;
        background: #f5f5f5;
    }
    .stream-table th:first-child,
    .stream-table td:first-child {
        text-align: left;
        padding-left: 24px;
    }
    .stream-table tbody tr:last-child td {
        border-bottom: 0;
    }
    .stream-table tbody tr:hover {
        background: rgba(239,246,255,0.55);
    }
    .pollutant-chart-card {
        background: #071024;
        border: 1px solid #dbe7f8;
        border-radius: 12px;
        box-shadow: 0 14px 34px rgba(42,70,120,0.06);
        padding: 10px 12px 0;
        margin-bottom: 16px;
        overflow: hidden;
        animation: fadeUp 0.8s cubic-bezier(.22,1,.36,1) both;
    }
    .forecast-card {
        background: rgba(255,255,255,0.96);
        border: 1px solid #dbe7f8;
        border-radius: 12px;
        box-shadow: 0 14px 34px rgba(42,70,120,0.06);
        padding: 14px 16px 6px;
        min-height: 300px;
        overflow: hidden;
        animation: fadeUp 0.8s cubic-bezier(.22,1,.36,1) both;
    }
    .forecast-card-title {
        font-size: 13px;
        font-weight: 900;
        color: #15264a;
        margin-bottom: 8px;
    }
    .forecast-ui-card {
        background: rgba(255,255,255,0.96);
        border: 1px solid #dbe7f8;
        border-radius: 14px;
        box-shadow: 0 14px 34px rgba(42,70,120,0.06);
        padding: 14px 16px 12px;
        overflow: hidden;
        animation: fadeUp 0.8s cubic-bezier(.22,1,.36,1) both;
    }
    .forecast-ui-head {
        display: grid;
        grid-template-columns: minmax(130px, 1fr) auto auto;
        align-items: center;
        gap: 14px;
        margin-bottom: 8px;
    }
    .forecast-ui-title {
        color: #15264a;
        font-size: 14px;
        font-weight: 900;
    }
    .forecast-toggle {
        display: inline-flex;
        padding: 3px;
        border-radius: 9px;
        background: #eef4ff;
        border: 1px solid #dce8f8;
        color: #15264a;
        font-size: 11px;
        font-weight: 900;
    }
    .forecast-toggle span {
        padding: 6px 13px;
        border-radius: 7px;
    }
    .forecast-toggle .active {
        color: #fff;
        background: linear-gradient(90deg, #2f9bff, #7b35ff);
        box-shadow: 0 8px 16px rgba(80,95,255,0.22);
    }
    .forecast-download {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        min-height: 34px;
        padding: 0 13px;
        border-radius: 10px;
        border: 1px solid #d7e4f6;
        background: #fff;
        color: #15264a !important;
        font-size: 11px;
        font-weight: 900;
        text-decoration: none !important;
        box-shadow: 0 8px 18px rgba(42,70,120,0.06);
    }
    .forecast-chart {
        min-height: 240px;
    }
    .forecast-chart .plotly-graph-div {
        width: 100% !important;
    }
    .forecast-table-wrap {
        margin-top: 8px;
        color: #15264a;
        font-size: 12px;
        font-weight: 800;
    }
    .forecast-table-wrap summary {
        cursor: pointer;
        color: #3157ff;
        font-weight: 900;
        margin-bottom: 8px;
    }
    .forecast-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
    }
    .forecast-table th,
    .forecast-table td {
        border-bottom: 1px solid #e4edf8;
        padding: 8px;
        text-align: center;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #dbe7f8 !important;
        border-radius: 14px !important;
        box-shadow: 0 14px 34px rgba(42,70,120,0.06) !important;
        background: rgba(255,255,255,0.96) !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stRadio"] label {
        display: none !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] [role="radiogroup"] {
        background: #eef4ff !important;
        border: 1px solid #dce8f8 !important;
        border-radius: 9px !important;
        padding: 3px !important;
        gap: 0 !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] [role="radio"] {
        border-radius: 7px !important;
        padding: 5px 11px !important;
        font-size: 11px !important;
        font-weight: 900 !important;
        color: #15264a !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] [aria-checked="true"] {
        background: linear-gradient(90deg, #2f9bff, #7b35ff) !important;
        color: #fff !important;
        box-shadow: 0 8px 16px rgba(80,95,255,0.22) !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stDownloadButton"] > button {
        width: auto !important;
        min-height: 34px !important;
        padding: 7px 13px !important;
        border-radius: 10px !important;
        border: 1px solid #d7e4f6 !important;
        background: #fff !important;
        color: #15264a !important;
        font-size: 11px !important;
        font-weight: 900 !important;
        box-shadow: 0 8px 18px rgba(42,70,120,0.06) !important;
    }
    [data-testid="stRadio"] label {
        display: none !important;
    }
    [data-testid="stRadio"] [role="radiogroup"] {
        background: #eef4ff !important;
        border: 1px solid #dce8f8 !important;
        border-radius: 9px !important;
        padding: 3px !important;
        gap: 0 !important;
    }
    [data-testid="stRadio"] [role="radio"] {
        border-radius: 7px !important;
        padding: 5px 11px !important;
        font-size: 11px !important;
        font-weight: 900 !important;
        color: #15264a !important;
    }
    [data-testid="stRadio"] [aria-checked="true"] {
        background: linear-gradient(90deg, #2f9bff, #7b35ff) !important;
        color: #fff !important;
        box-shadow: 0 8px 16px rgba(80,95,255,0.22) !important;
    }
    @media screen and (max-width: 1100px) {
        .forecast-ui-head {
            grid-template-columns: 1fr;
            justify-items: start;
        }
        .forecast-download {
            justify-self: start;
        }
    }
    .live-dashboard-grid {
        display: grid;
        grid-template-columns: minmax(270px, 0.82fr) minmax(660px, 2fr);
        gap: 20px;
        align-items: stretch;
        margin-bottom: 20px;
    }
    .dashboard-right-grid {
        display: grid;
        grid-template-columns: 0.9fr 1.1fr;
        gap: 20px;
        grid-template-rows: 122px 194px;
    }
    .dash-card {
        background:
            radial-gradient(circle at 94% 9%, rgba(247,251,255,0.96) 0 31px, transparent 32px),
            linear-gradient(180deg, rgba(255,255,255,0.96), rgba(253,254,255,0.9));
        border: 1px solid #dbe7f8;
        border-radius: 12px;
        box-shadow: 0 14px 34px rgba(42,70,120,0.06);
        position: relative;
        overflow: hidden;
        animation: fadeUp 0.8s cubic-bezier(.22,1,.36,1) both;
        transition: transform 0.3s, box-shadow 0.3s, border-color 0.3s;
    }
    .dash-card:hover {
        transform: translateY(-2px);
        border-color: rgba(110,150,230,0.34);
        box-shadow: 0 22px 48px rgba(42,70,120,0.12);
    }
    .dashboard-stat-card {
        padding: 22px 34px;
        min-height: 336px;
        display: flex;
        flex-direction: column;
    }
    .dashboard-top-card {
        padding: 24px 34px;
        min-height: 122px;
    }
    .dashboard-location-card {
        grid-column: 1 / -1;
        min-height: 194px;
        display: grid;
        grid-template-columns: minmax(250px, 0.82fr) minmax(320px, 1.55fr);
        padding: 28px 34px;
    }
    .dash-title {
        font-size: 14px;
        font-weight: 900;
        color: #15264a;
        margin-bottom: 10px;
    }
    .dash-label {
        font-size: 12px;
        font-weight: 800;
        color: #15264a;
        margin-top: 11px;
    }
    .dash-sub {
        font-size: 12px;
        font-weight: 700;
        color: #7e8ba5;
        line-height: 1.45;
    }
    .dash-aqi-number {
        font-size: 36px;
        line-height: 0.95;
        font-weight: 900;
        letter-spacing: -1px;
        margin: 7px 0 8px;
    }
    .dash-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
    }
    .dash-status-line {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        font-size: 12px;
        font-weight: 800;
        color: #3c4963;
    }
    .dash-status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        animation: livePulseSoft 2.1s ease-in-out infinite;
    }
    .dash-pollutant-value {
        color: #84bcff;
        font-weight: 900;
        margin-left: 8px;
    }
    .dash-risk-card {
        margin-top: auto;
        background: linear-gradient(135deg, rgba(240,248,255,0.94), rgba(243,232,255,0.92));
        border: 1px solid rgba(220,226,255,0.86);
        border-radius: 12px;
        padding: 16px 20px;
    }
    .dash-risk-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 12px;
    }
    .dash-risk-btn {
        color: #fff;
        font-size: 11px;
        font-weight: 900;
        padding: 7px 15px;
        border-radius: 9px;
        background: linear-gradient(90deg, #1da2ff, #8b3dff);
        box-shadow: 0 10px 22px rgba(88,96,255,0.28);
    }
    .dash-risk-percent {
        color: #ef1d38;
        font-size: 34px;
        font-weight: 900;
        letter-spacing: -1px;
        line-height: 1;
    }
    .dash-risk-desc {
        color: #22304a;
        font-size: 11px;
        line-height: 1.45;
        margin-top: 8px;
        max-width: 230px;
    }
    .dash-risk-track {
        height: 5px;
        width: 88px;
        margin-top: 10px;
        border-radius: 99px;
        background: #dce7fb;
        overflow: hidden;
    }
    .dash-risk-fill {
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, #19d2c4, #8b3dff);
    }
    .aqi-gauge {
        width: 78px;
        height: 78px;
        border-radius: 50%;
        background: conic-gradient(from -130deg, #ffd65a 0deg, #ff8c31 85deg, #9b4dff 180deg, transparent 181deg 360deg);
        position: relative;
        display: grid;
        place-items: center;
        transform: rotate(-8deg);
    }
    .aqi-gauge::before {
        content: '';
        width: 54px;
        height: 54px;
        border-radius: 50%;
        background: #fff;
        box-shadow: inset 0 0 0 1px #edf2fb;
    }
    .aqi-needle {
        position: absolute;
        width: 4px;
        height: 26px;
        border-radius: 99px;
        background: #2c8cff;
        transform-origin: 50% 90%;
        transform: rotate(var(--needle-angle));
        bottom: 27px;
        left: 37px;
        box-shadow: 0 4px 10px rgba(44,140,255,0.3);
    }
    .aqi-needle::after {
        content: '';
        position: absolute;
        width: 11px;
        height: 11px;
        border-radius: 50%;
        background: #ffcf48;
        left: 50%;
        bottom: -5px;
        transform: translateX(-50%);
    }
    .stream-live {
        color: #15d49b;
        font-size: 21px;
        font-weight: 900;
        margin-top: 18px;
        line-height: 1;
    }
    .sparkline {
        width: 156px;
        height: 64px;
        overflow: visible;
    }
    .sparkline polyline {
        fill: none;
        stroke: url(#sparkGradient);
        stroke-width: 4;
        stroke-linecap: round;
        stroke-linejoin: round;
        stroke-dasharray: 220;
        animation: sparkDraw 1.8s ease both;
    }
    .sparkline circle {
        fill: #16d68f;
        stroke: #fff;
        stroke-width: 2;
        filter: drop-shadow(0 4px 7px rgba(0,200,160,0.28));
    }
    .location-name {
        color: #ff2537;
        font-size: 20px;
        font-weight: 900;
        letter-spacing: -0.4px;
        margin: 14px 0 8px;
    }
    .location-name span {
        color: #263756;
        font-size: 17px;
        font-weight: 700;
    }
    .location-coords {
        color: #263756;
        font-size: 12px;
        font-weight: 800;
    }
    .mini-map {
        position: relative;
        min-height: 162px;
        border-radius: 10px;
        overflow: hidden;
        background:
            radial-gradient(circle at 72% 36%, rgba(136,91,255,0.14), transparent 12%),
            linear-gradient(115deg, rgba(255,255,255,0.2) 0 22%, transparent 22% 100%),
            #f7fbff;
    }
    .mini-map svg {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
    }
    .map-pin {
        position: absolute;
        right: 28%;
        top: 28%;
        width: 34px;
        height: 34px;
        border-radius: 50% 50% 50% 8px;
        background: linear-gradient(135deg, #a14dff, #6939ff);
        transform: rotate(-45deg);
        box-shadow: 0 12px 24px rgba(116,62,255,0.26);
        animation: pinBounce 2.5s ease-in-out infinite;
    }
    .map-pin::after {
        content: '';
        position: absolute;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #fff;
        left: 12px;
        top: 12px;
    }
    @keyframes sparkDraw {
        from { stroke-dashoffset: 220; opacity: 0.25; }
        to { stroke-dashoffset: 0; opacity: 1; }
    }
    @keyframes pinBounce {
        0%,100% { transform: rotate(-45deg) translate(0,0); }
        50% { transform: rotate(-45deg) translate(4px,-4px); }
    }

    /* GLASS PANEL */
    .glass-panel {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px; padding: 26px; margin-bottom: 18px;
        position: relative; overflow: hidden; transition: border-color 0.3s, transform 0.3s;
        animation: fadeUp 0.8s cubic-bezier(.22,1,.36,1) 0.25s both;
    }
    .glass-panel::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,200,255,0.4), transparent);
    }
    .glass-panel:hover { border-color: rgba(0,200,255,0.12); transform: translateY(-2px); }
    .panel-title {
        font-size: 14px; font-weight: 700; color: rgba(200,220,255,0.85);
        margin-bottom: 18px; letter-spacing: -0.2px; display: flex; align-items: center; gap: 8px;
    }
    .panel-title-icon {
        width: 28px; height: 28px; background: rgba(0,200,255,0.1);
        border: 1px solid rgba(0,200,255,0.2); border-radius: 8px;
        display: flex; align-items: center; justify-content: center; font-size: 13px;
    }

    /* ADVISORY */
    .advisory-wrap {
        border-radius: 22px; padding: 32px; position: relative; overflow: hidden;
        margin-top: 8px; border: 1px solid rgba(255,255,255,0.06);
        animation: fadeUp 0.8s cubic-bezier(.22,1,.36,1) 0.3s both;
    }
    .advisory-wrap::before {
        content: ''; position: absolute; inset: 0;
        background: radial-gradient(ellipse 50% 60% at 90% 50%, rgba(0,200,255,0.06) 0%, transparent 60%), rgba(255,255,255,0.025);
        pointer-events: none;
    }
    .advisory-tag {
        display: inline-flex; align-items: center; gap: 7px;
        padding: 5px 14px; border-radius: 100px;
        font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;
    }
    .advisory-title { font-size: 26px; font-weight: 800; color:black; letter-spacing: -0.5px; margin-bottom: 10px; }
    .advisory-body { font-size: 14px; color: rgba(180,200,255,0.5); line-height: 1.75; font-weight: 400; max-width: 600px; }

    /* MAP */
    .map-glass { border-radius: 20px; overflow: hidden; border: 1px solid rgba(0,200,255,0.12); box-shadow: 0 8px 40px rgba(0,0,0,0.4); }
    iframe { border-radius: 18px !important; border: none !important; }

    /* SIDEBAR — base styles */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.78), rgba(244,249,255,0.68)) !important;
        backdrop-filter: blur(24px) saturate(1.25) !important;
        -webkit-backdrop-filter: blur(24px) saturate(1.25) !important;
        border-right: 1px solid rgba(210,225,248,0.74) !important;
        box-shadow: 16px 0 44px rgba(36, 55, 99, 0.10) !important;
        transition: transform 0.35s cubic-bezier(.22,1,.36,1) !important;

    will-change: transform;  /* ✅ ADD THIS LINE */
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 22px 18px 22px !important;
        overflow-x: hidden !important;
    }

    /* ─── DESKTOP: always visible, never transformed ─── */
    @media screen and (min-width: 769px) {
        [data-testid="stSidebar"] {
            position: relative !important;
            transform: translateX(0) !important;
            width: 19rem !important;
            min-width: 280px !important;
            max-width: 320px !important;
            z-index: 9998 !important;
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
        }
        [data-testid="stSidebar"].sidebar-closed {
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            transform: translateX(-105%) !important;
            border-right: 0 !important;
            box-shadow: none !important;
            overflow: hidden !important;
        }
        [data-testid="stSidebar"].sidebar-closed > div:first-child {
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
        .mobile-menu-btn { /* Removed redundant z-index for desktop, as it's display:none */
         /* z-index: 999999 !important; */
        }
        .sidebar-overlay { display: none !important; }
    }
    .sidebar-logo-area {
        display: flex; align-items: center; gap: 10px;
        margin-bottom: 18px; padding: 0 4px 16px;
        border-bottom: 1px solid rgba(218,230,248,0.72);
        animation: sidebarRise 0.55s cubic-bezier(.22,1,.36,1) both;
    }
    .sidebar-icon {
        width: 36px; height: 36px; background: linear-gradient(135deg, #37a4ff, #7d35ff);
        border-radius: 12px; display: flex; align-items: center; justify-content: center;
        color: #fff; font-size: 19px; font-weight: 900;
        box-shadow: 0 10px 24px rgba(92, 98, 255, 0.28);
        animation: logoFloat 3.8s ease-in-out infinite;
    }
    .sidebar-brand-name {
        font-size: 16px; font-weight: 900; color: #13233d !important;
        letter-spacing: -0.4px; line-height: 1.05;
    }
    .sidebar-brand-sub {
        font-size: 11px; color: #6c7890; letter-spacing: 0.2px; font-weight: 600;
        margin-top: 2px;
    }

    [data-testid="stTextInput"] > div > div > input {
        background: #ffffff !important;
        border: 1px solid #d9e4f4 !important;
        border-radius: 12px !important; color: #17243d !important;
        caret-color: #2563ff !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 12px !important; font-weight: 600 !important;
        min-height: 34px !important;
        padding: 8px 13px 8px 34px !important; transition: all 0.25s !important;
        box-shadow: 0 8px 20px rgba(35, 62, 114, 0.05) !important;
    }
    [data-testid="stTextInput"] > div > div > input:focus {
        border-color: #78a8ff !important;
        box-shadow: 0 0 0 3px rgba(75,137,255,0.12), 0 10px 24px rgba(35, 62, 114, 0.08) !important;
        outline: none !important;
    }
    [data-testid="stTextInput"] > div > div > input::placeholder { color: #98a5bb !important; }
    [data-testid="stTextInput"] label {
        font-family: 'Poppins', sans-serif !important; color: #13233d !important;
        font-size: 9px !important; font-weight: 900 !important; letter-spacing: 1.4px !important;
        text-transform: uppercase !important;
    }
    [data-testid="stSidebar"] [data-testid="stTextInput"] {
        position: relative;
        animation: sidebarRise 0.55s cubic-bezier(.22,1,.36,1) 0.08s both;
    }
    [data-testid="stSidebar"] [data-testid="stTextInput"]::after {
        content: '\\1F50D';
        position: absolute;
        left: 12px;
        bottom: 9px;
        color: #345082;
        font-size: 17px;
        line-height: 1;
        pointer-events: none;
    }

    [data-testid="stButton"] > button {
        background: linear-gradient(90deg, #44a2ff 0%, #8237ff 100%) !important;
        color: #fff !important; border: none !important; border-radius: 14px !important;
        font-family: 'Poppins', sans-serif !important; font-size: 11px !important;
        font-weight: 800 !important; letter-spacing: 0.15px !important;
        min-height: 36px !important;
        padding: 9px 16px !important; width: 100% !important; margin-top: 6px !important;
        transition: transform 0.25s cubic-bezier(.22,1,.36,1), box-shadow 0.25s, filter 0.25s !important;
        box-shadow: 0 12px 24px rgba(98, 83, 255, 0.27) !important;
        animation: sidebarRise 0.55s cubic-bezier(.22,1,.36,1) 0.14s both;
    }
    [data-testid="stButton"] > button:hover {
        transform: translateY(-2px) !important;
        filter: saturate(1.08) brightness(1.03) !important;
        box-shadow: 0 16px 32px rgba(98, 83, 255, 0.34) !important;
    }
    [data-testid="stButton"] > button:active { transform: scale(0.97) !important; }

    .sidebar-section-title {
        display: flex; align-items: center; gap: 8px;
        color: #13233d; font-size: 9px; font-weight: 900;
        letter-spacing: 1.3px; text-transform: uppercase;
        margin: 22px 4px 9px;
        animation: sidebarRise 0.55s cubic-bezier(.22,1,.36,1) both;
    }
    .sidebar-section-title::after {
        content: ''; height: 1px; flex: 1; background: #e6edf8;
    }
    .sidebar-nav {
        display: grid; gap: 6px;
        margin: 6px 0 10px;
    }
    .sidebar-nav-item {
        display: grid;
        grid-template-columns: 28px 1fr;
        align-items: center;
        column-gap: 8px;
        min-height: 38px;
        padding: 0 14px;
        border-radius: 14px; color: #2d9cff;
        text-decoration: none !important;
        font-size: 12px; font-weight: 900;
        border: 1px solid transparent;
        transition: transform 0.22s cubic-bezier(.22,1,.36,1), background 0.22s, color 0.22s, box-shadow 0.22s;
        animation: sidebarRise 0.5s cubic-bezier(.22,1,.36,1) both;
    }
    .sidebar-nav-item:nth-child(1) { animation-delay: 0.04s; }
    .sidebar-nav-item:nth-child(2) { animation-delay: 0.08s; }
    .sidebar-nav-item:nth-child(3) { animation-delay: 0.12s; }
    .sidebar-nav-item:nth-child(4) { animation-delay: 0.16s; }
    .sidebar-nav-item:nth-child(5) { animation-delay: 0.20s; }
    .sidebar-nav-item:nth-child(6) { animation-delay: 0.24s; }
    .sidebar-nav-item:nth-child(7) { animation-delay: 0.28s; }
    .sidebar-nav-item.active,
    .sidebar-nav-item:hover {
        background: rgba(238,244,255,0.82);
        color: #3157ff;
        transform: translateX(2px);
        box-shadow: inset 0 0 0 1px rgba(86, 129, 255, 0.13);
    }
    .sidebar-nav-icon,
    .sidebar-info-icon {
        width: 17px; height: 17px; flex: 0 0 17px;
        color: currentColor;
        transition: transform 0.25s cubic-bezier(.22,1,.36,1), color 0.25s;
    }
    .sidebar-nav-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        font-size: 18px;
        font-weight: 900;
        line-height: 1;
        text-align: center;
    }
    .sidebar-nav-item:hover .sidebar-nav-icon,
    .sidebar-info-row:hover .sidebar-info-icon {
        transform: translateY(-1px) scale(1.12) rotate(-3deg);
    }
    .sidebar-info-list {
        display: grid; gap: 13px;
        margin: 8px 4px 18px;
        animation: sidebarRise 0.55s cubic-bezier(.22,1,.36,1) 0.18s both;
    }
    .sidebar-info-row {
        display: flex; align-items: center; gap: 10px;
        color: #24334d; font-size: 10.5px; font-weight: 700;
        transition: transform 0.22s, color 0.22s;
    }
    .sidebar-info-row:hover { transform: translateX(3px); color: #3149e8; }
    .sidebar-dot {
        width: 8px; height: 8px; border-radius: 50%; flex: 0 0 8px;
        box-shadow: 0 0 0 5px rgba(0, 196, 151, 0.08);
        animation: livePulseSoft 2.1s ease-in-out infinite;
    }
    .dot-green { background: #00c497; }
    .dot-blue { background: #2e87ff; box-shadow: 0 0 0 5px rgba(46, 135, 255, 0.08); animation-delay: 0.3s; }
    .dot-purple { background: #7b45ff; box-shadow: 0 0 0 5px rgba(123, 69, 255, 0.08); animation-delay: 0.6s; }
    .sidebar-status-card {
        margin-top: 24px;
        min-height: 170px;
        border: 1px solid #dde8f7;
        border-radius: 18px;
        background:
            radial-gradient(circle at 85% 12%, rgba(255,255,255,0.78) 0 11px, transparent 12px),
            linear-gradient(180deg, rgba(255,255,255,0.78) 0%, rgba(247,251,255,0.58) 100%);
        backdrop-filter: blur(16px) saturate(1.15);
        -webkit-backdrop-filter: blur(16px) saturate(1.15);
        box-shadow: 0 18px 40px rgba(35, 62, 114, 0.09);
        padding: 16px 12px 14px;
        text-align: center;
        overflow: hidden;
        position: relative;
        animation: sidebarRise 0.6s cubic-bezier(.22,1,.36,1) 0.28s both;
    }
    .sidebar-status-card::before {
        content: '';
        position: absolute; inset: 14px auto auto 50%;
        width: 84px; height: 84px; transform: translateX(-50%);
        border-radius: 50%;
        background: radial-gradient(circle, rgba(78, 160, 255, 0.16), transparent 68%);
        animation: haloPulse 3s ease-in-out infinite;
    }
    .satellite-orbit {
        position: relative; z-index: 1;
        width: 78px; height: 78px; margin: 2px auto 8px;
    }
    .satellite-orbit::before,
    .satellite-orbit::after {
        content: ''; position: absolute; inset: 14px;
        border: 2px solid rgba(87, 145, 255, 0.24);
        border-radius: 50%;
        transform: rotate(-24deg) scaleX(1.35);
    }
    .satellite-orbit::after {
        inset: 22px; transform: rotate(36deg) scaleX(1.6);
        border-color: rgba(123, 69, 255, 0.18);
    }
    .satellite-body {
        position: absolute; left: 28px; top: 24px;
        width: 24px; height: 24px; border-radius: 6px;
        background: linear-gradient(135deg, #85c7ff, #6a65ff);
        box-shadow: 0 8px 18px rgba(72, 111, 235, 0.24);
        animation: satelliteBob 2.6s ease-in-out infinite;
    }
    .satellite-body::before,
    .satellite-body::after {
        content: ''; position: absolute; top: 8px;
        width: 20px; height: 9px; border-radius: 3px;
        background: linear-gradient(135deg, #cfe8ff, #75a9ff);
    }
    .satellite-body::before { right: 24px; transform: rotate(-18deg); }
    .satellite-body::after { left: 24px; transform: rotate(18deg); }
    .status-title {
        position: relative; z-index: 1;
        color: #182944; font-size: 11px; font-weight: 900; line-height: 1.2;
        margin-top: 4px;
    }
    .status-subtitle {
        position: relative; z-index: 1;
        color: #40516d; font-size: 10px; font-weight: 700; margin-top: 8px;
    }
    .status-subtitle span {
        display: inline-block; width: 7px; height: 7px; border-radius: 50%;
        background: #00c497; margin-right: 6px;
        box-shadow: 0 0 0 5px rgba(0,196,151,0.10);
        vertical-align: middle;
    }

    /* Screenshot-matched sidebar polish */
    [data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 88% 18%, rgba(216,238,255,0.72), transparent 30%),
            linear-gradient(180deg, #ffffff 0%, #f7fbff 52%, #ffffff 100%) !important;
        border-right: 1px solid #e3edf8 !important;
        box-shadow: 18px 0 44px rgba(36,55,99,0.08) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 26px 18px 26px !important;
    }
    .sidebar-logo-area {
        border-bottom: 0 !important;
        margin-bottom: 28px !important;
        padding: 0 !important;
        gap: 12px !important;
    }
    .sidebar-icon {
        width: 44px !important;
        height: 44px !important;
        border-radius: 8px !important;
        font-size: 26px !important;
        background: linear-gradient(135deg, #18a8ff 0%, #8438ff 100%) !important;
        box-shadow: 0 14px 28px rgba(88,100,255,0.24) !important;
    }
    .sidebar-brand-name {
        font-size: 23px !important;
        color: #15264a !important;
        letter-spacing: -0.6px !important;
    }
    .sidebar-brand-sub {
        font-size: 12px !important;
        color: #15264a !important;
        font-weight: 600 !important;
        margin-top: 6px !important;
    }
    [data-testid="stSidebar"] [data-testid="stTextInput"] {
        margin-bottom: 8px !important;
    }
    [data-testid="stTextInput"] label {
        color: #15264a !important;
        font-size: 11px !important;
        letter-spacing: 1.6px !important;
        margin-bottom: 12px !important;
    }
    [data-testid="stTextInput"] label::after,
    .sidebar-section-title::after {
        background: #dce6f3 !important;
    }
    [data-testid="stTextInput"] > div > div > input {
        min-height: 42px !important;
        border-radius: 8px !important;
        border-color: #d8e4f3 !important;
        color: #15264a !important;
        font-size: 13px !important;
        padding-left: 43px !important;
        box-shadow: 0 10px 24px rgba(35,62,114,0.06) !important;
    }
    [data-testid="stSidebar"] [data-testid="stTextInput"]::after {
        left: 16px !important;
        bottom: 11px !important;
        color: #233d72 !important;
        font-size: 16px !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] > button {
        min-height: 43px !important;
        border-radius: 8px !important;
        margin-top: 8px !important;
        font-size: 12px !important;
        background: linear-gradient(90deg, #189fff 0%, #8536ff 100%) !important;
        box-shadow: 0 14px 26px rgba(87,92,255,0.23) !important;
    }
    .sidebar-section-title {
        color: #15264a !important;
        font-size: 11px !important;
        letter-spacing: 1.5px !important;
        margin: 28px 0 12px !important;
        gap: 14px !important;
    }
    .sidebar-nav {
        gap: 12px !important;
        margin: 0 0 24px !important;
    }
    .sidebar-nav-item {
        grid-template-columns: 27px 1fr !important;
        min-height: 46px !important;
        padding: 0 12px !important;
        border-radius: 10px !important;
        color: #1b2c50 !important;
        font-size: 14px !important;
        font-weight: 800 !important;
        column-gap: 12px !important;
    }
    .sidebar-nav-item.active,
    .sidebar-nav-item:hover {
        background: linear-gradient(90deg, rgba(231,241,255,0.95), rgba(239,244,255,0.86)) !important;
        color: #15264a !important;
        box-shadow: none !important;
        transform: none !important;
    }
    .sidebar-nav-icon {
        width: 25px !important;
        height: 25px !important;
        color: #1f3567 !important;
    }
    .sidebar-nav-icon svg,
    .sidebar-info-icon {
        width: 19px !important;
        height: 19px !important;
        stroke-width: 1.9 !important;
    }
    .sidebar-nav-item.active .sidebar-nav-icon {
        color: #455cff !important;
    }
    .sidebar-info-list {
        gap: 20px !important;
        margin: 10px 6px 28px !important;
    }
    .sidebar-info-row {
        color: #213453 !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        gap: 17px !important;
    }
    .sidebar-info-icon {
        color: #1f3567 !important;
        flex: 0 0 19px !important;
    }
    .sidebar-dot {
        width: 11px !important;
        height: 11px !important;
        flex-basis: 11px !important;
        box-shadow: 0 5px 12px rgba(29, 167, 255, 0.18) !important;
    }
    .dot-green { background: #18d29b !important; }
    .dot-blue { background: #238fff !important; }
    .dot-purple { background: #8737ff !important; }
    .sidebar-status-card {
        width: calc(100% - 16px) !important;
        min-height: 236px !important;
        margin: 88px auto 0 !important;
        border-radius: 18px !important;
        border: 1px solid #dbe7f6 !important;
        background:
            radial-gradient(circle at 86% 10%, transparent 0 18px, rgba(255,255,255,0.95) 19px 28px, transparent 29px),
            linear-gradient(180deg, rgba(255,255,255,0.9), rgba(249,252,255,0.86)) !important;
        box-shadow: 0 18px 42px rgba(35,62,114,0.10) !important;
        padding: 24px 16px 20px !important;
    }
    .sidebar-status-card::before {
        width: 112px !important;
        height: 112px !important;
        top: 30px !important;
        background: radial-gradient(circle, rgba(80,172,255,0.17), transparent 67%) !important;
    }
    .status-illustration {
        position: relative;
        z-index: 1;
        width: 108px;
        height: 92px;
        margin: 0 auto 12px;
        display: grid;
        place-items: center;
    }
    .status-illustration svg {
        width: 106px;
        height: 92px;
        overflow: visible;
        filter: drop-shadow(0 12px 18px rgba(72,111,235,0.16));
    }
    .satellite-orbit {
        display: none !important;
    }
    .status-title {
        color: #15264a !important;
        font-size: 16px !important;
        line-height: 1.25 !important;
        margin-top: 0 !important;
    }
    .status-subtitle {
        display: grid !important;
        grid-template-columns: 16px 1fr !important;
        column-gap: 12px !important;
        align-items: start !important;
        justify-content: center !important;
        width: fit-content !important;
        margin: 22px auto 0 !important;
        color: #8da0bd !important;
        font-size: 12px !important;
        line-height: 1.35 !important;
        text-align: left !important;
    }
    .status-subtitle span {
        width: 10px !important;
        height: 10px !important;
        margin: 4px 0 0 !important;
        background: #18d29b !important;
        box-shadow: 0 0 0 6px rgba(24,210,155,0.11) !important;
    }
    .status-subtitle strong {
        display: block;
        color: #15264a !important;
        font-size: 13px !important;
        margin-top: 3px;
    }
    .scroll-anchor {
        display: block;
        height: 1px;
        margin-top: -1px;
        scroll-margin-top: 28px;
    }
    @keyframes sidebarRise {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes logoFloat {
        0%,100% { transform: translateY(0); }
        50% { transform: translateY(-3px); }
    }
    @keyframes livePulseSoft {
        0%,100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(0.78); opacity: 0.65; }
    }
    @keyframes haloPulse {
        0%,100% { transform: translateX(-50%) scale(1); opacity: 0.85; }
        50% { transform: translateX(-50%) scale(1.16); opacity: 0.45; }
    }
    @keyframes satelliteBob {
        0%,100% { transform: translateY(0) rotate(-8deg); }
        50% { transform: translateY(-5px) rotate(5deg); }
    }

    [data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; border: 1px solid rgba(0,200,255,0.12) !important; }

    [data-testid="stInfo"] { background: rgba(0,200,255,0.06) !important; border: 1px solid rgba(0,200,255,0.2) !important; border-radius: 12px !important; color: #a0d8f0 !important; }
    [data-testid="stWarning"] { background: rgba(245,200,66,0.06) !important; border: 1px solid rgba(245,200,66,0.2) !important; border-radius: 12px !important; color: #f5e4a0 !important; }
    [data-testid="stError"] { background: rgba(255,77,109,0.06) !important; border: 1px solid rgba(255,77,109,0.2) !important; border-radius: 12px !important; color: #ffa0b0 !important; }

    [data-testid="stMetric"] { background: rgba(255,255,255,0.03) !important; border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 14px !important; padding: 18px 20px !important; }
    [data-testid="stMetricLabel"] p { font-family: 'Poppins', sans-serif !important; color: rgba(150,180,255,0.45) !important; font-size: 10px !important; font-weight: 700 !important; letter-spacing: 2px !important; text-transform: uppercase !important; }
    [data-testid="stMetricValue"] { font-family: 'Poppins', sans-serif !important; font-size: 34px !important; font-weight: 900 !important; color: #e8f0ff !important; }

    hr { border: none !important; height: 1px !important; background: linear-gradient(90deg, transparent, rgba(0,200,255,0.2), transparent) !important; margin: 32px 0 !important; }

    /* Streamlit typography overrides */
    h2, h3 { font-family: 'Poppins', sans-serif !important; font-weight: 700 !important; color: #e0eaff !important; letter-spacing: -0.3px !important; }

    /* Change Streamlit subheader text (e.g., “6-Hour Forecast”, “7-Day Trend”) */
    [data-testid="stSubheader"] div,
    [data-testid="stSubheader"] span,
    [data-testid="stSubheader"] { color: #98a6b2 !important; }

    /* Change Streamlit radio labels (e.g., “View Mode”, “📊 Chart”, “📋 Table”) */
    [data-testid="stRadio"] label,
    .stRadio label,
    [data-testid="stRadio"] span,
    .stRadio span { color: #98a6b2 !important; }



    @keyframes fadeUp { from { opacity: 0; transform: translateY(22px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes fadeDown { from { opacity: 0; transform: translateY(-12px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes livePulse { 0%,100% { opacity: 1; box-shadow: 0 0 8px #00c8ff; } 50% { opacity: .5; box-shadow: 0 0 16px #00c8ff; } }

    /* ─── RESPONSIVE BREAKPOINTS ─── */

    @media screen and (max-width: 1400px) {
        .block-container { padding: 0 1.5rem 3rem !important; }
        .navbar-title { font-size: 38px !important; }
        .hero-h1 { font-size: clamp(32px, 6vw, 52px) !important; }
        .kpi-row { grid-template-columns: repeat(2, 1fr) !important; gap: 12px !important; }
        .live-dashboard-grid { grid-template-columns: 1fr 2fr !important; gap: 18px !important; }
        .dashboard-right-grid { gap: 18px !important; }
    }

    @media screen and (max-width: 992px) {
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
    }

    @media screen and (max-width: 1024px) {
        [data-testid="stSidebar"] { width: 21rem !important; min-width: 240px !important; max-width: 320px !important; }
        .main-stats-card, .glass-panel { padding: 20px !important; }
        .navbar-title { font-size: 32px !important; }
        .stat-strip { gap: 2rem !important; flex-direction: column !important; align-items: center !important; }
        .live-dashboard-grid { grid-template-columns: 1fr !important; }
        .dashboard-right-grid { grid-template-columns: 1fr 1fr !important; }
    }

    /* ─── TABLET & MOBILE: show toggle button, slide sidebar offscreen ─── */
    @media screen and (max-width: 768px) {
        /* Show the floating menu button */
        .mobile-menu-btn { display: flex !important; }

        .stApp { background-color: white !important; }
        .block-container { padding: 0 1rem 2rem !important; max-width: 100% !important; }

        /* Push sidebar off-screen by default */
        [data-testid="stSidebar"] {
            position: fixed !important;
            top: 0 !important; left: 0 !important; bottom: 0 !important;
            width: 85vw !important;
            max-width: 350px !important;
            transform: translateX(-105%) !important;
            z-index: 9999 !important;
        }
        /* When JS adds .sidebar-open class, slide in */
        [data-testid="stSidebar"].sidebar-open {
            transform: translateX(0) !important;
        }
        [data-testid="stSidebar"].sidebar-closed {
            transform: translateX(-105%) !important;
        }

        .stream-table-card {
            overflow-x: auto !important;
            overflow-y: hidden !important;
            -webkit-overflow-scrolling: touch;
        }
        .stream-table-card::-webkit-scrollbar {
            height: 6px;
        }
        .stream-table {
            min-width: 760px;
            font-size: 12px;
        }
        .stream-table th,
        .stream-table td {
            padding: 11px 8px;
        }
        .stream-table th:first-child,
        .stream-table td:first-child {
            padding-left: 16px;
        }

        .navbar {
            flex-direction: column !important; gap: 1rem !important;
            padding: 1rem 0 !important; text-align: center !important;
        }
        .navbar-title { font-size: 28px !important; text-align: center !important; }
        .navbar-brand { justify-content: center !important; }
        .navbar-links { display: none !important; }
        .hero-h1 { font-size: clamp(28px, 8vw, 40px) !important; line-height: 1.2 !important; text-align: center !important; }
        .hero-sub { text-align: center !important; }
        .stat-strip { flex-direction: column !important; gap: 1rem !important; align-items: center !important; text-align: center !important; }
        .kpi-row { grid-template-columns: 1fr !important; gap: 12px !important; justify-items: center !important; }
        .dashboard-right-grid { grid-template-columns: 1fr !important; }
        .dashboard-location-card { grid-template-columns: 1fr !important; gap: 16px !important; }
        .dashboard-stat-card { min-height: auto !important; }
        .main-stats-card { padding: 1.5rem !important; text-align: center !important; }
        .card-title, .pollutant-info { text-align: center !important; }
        .aqi-big-number { font-size: 60px !important; text-align: center !important; }
        .sec-heading { text-align: center !important; justify-content: center !important; }
        .advisory-title { font-size: 20px !important; }
    }

    @media screen and (max-width: 480px) {
        .hero-sub { font-size: 14px !important; }
        [data-testid="stButton"] > button { padding: 14px 20px !important; font-size: 16px !important; }
        .aqi-big-number { font-size: 48px !important; }
    }

    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)
# =========================
# 📱 RESPONSIVE TOGGLE BUTTON + JS
# =========================    
inject_css()
st.markdown("""
<!-- Floating hamburger button — visible on every screen via CSS -->
<button class="mobile-menu-btn" id="mobileMenuBtn" aria-label="Toggle Menu" aria-expanded="false">&#9776;</button>

<!-- Dark overlay — tap to close sidebar -->
<div class="sidebar-overlay" id="sidebarOverlay"></div>
""", unsafe_allow_html=True)

components.html("""
<script>
(function () {
    const doc = window.parent.document;
    if (typeof window.parent.__airflowSidebarCleanup === 'function') {
        window.parent.__airflowSidebarCleanup();
    }
    const cleanupFns = [];
    window.parent.__airflowSidebarCleanup = function () {
        while (cleanupFns.length) {
            const cleanup = cleanupFns.pop();
            try { cleanup(); } catch (err) {}
        }
    };
    function bindManaged(target, eventName, handler) {
        target.addEventListener(eventName, handler);
        cleanupFns.push(function () {
            target.removeEventListener(eventName, handler);
        });
    }
    function setManagedInterval(handler, delay) {
        const intervalId = window.setInterval(handler, delay);
        cleanupFns.push(function () {
            window.clearInterval(intervalId);
        });
    }

    const sidebarStorageKey = 'airflow-sidebar-open';
    function readSavedSidebarState() {
        if (window.parent.innerWidth > 768) return true;
        try {
            return window.parent.localStorage.getItem(sidebarStorageKey) === 'true';
        } catch (err) {
            return false;
        }
    }
    function saveSidebarState() {
        try {
            window.parent.localStorage.setItem(sidebarStorageKey, sidebarOpen ? 'true' : 'false');
        } catch (err) {}
    }

    let sidebarOpen = readSavedSidebarState();
    let didBind = false;

    function getParts() {
        return {
            sidebar: doc.querySelector('[data-testid="stSidebar"]'),
            btn: doc.getElementById('mobileMenuBtn'),
            overlay: doc.getElementById('sidebarOverlay'),
            nativeToggleButton: doc.querySelector('[data-testid="stSidebarCollapseButton"]'),
            lastSyncTime: doc.getElementById('lastSyncTime')
        };
    }

    function closeSidebarOnMobile() {
        if (window.parent.innerWidth <= 768) {
            sidebarOpen = false;
            saveSidebarState();
            renderSidebarState();
        }
    }

    function getAnalyzeButton() {
        const buttons = Array.from(doc.querySelectorAll('[data-testid="stSidebar"] button'));
        return buttons.find(function (button) {
            return button.textContent.trim().toLowerCase() === 'analyse air quality';
        });
    }

    function submitCityAnalysis() {
        const cityInput = doc.querySelector('[data-testid="stSidebar"] input');
        const analyzeButton = getAnalyzeButton();
        if (!cityInput || !cityInput.value.trim() || !analyzeButton) return;
        closeSidebarOnMobile();
        window.setTimeout(function () {
            analyzeButton.click();
        }, 50);
    }

    function scrollToWorkspaceTarget(link) {
        const href = link.getAttribute('href') || '';
        const targetId = href.replace('#', '');
        const parts = getParts();
        const cityInput = doc.querySelector('[data-testid="stSidebar"] input');

        doc.querySelectorAll('.sidebar-nav-item').forEach(function (item) {
            item.classList.toggle('active', item === link);
        });

        if (targetId === 'settings') {
            if (cityInput) {
                cityInput.focus();
                cityInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            return;
        }

        let target = doc.getElementById(targetId);
        if (!target && targetId === 'alerts') {
            target = doc.getElementById('health-advisory');
        }
        if (!target && targetId === 'reports') {
            target = doc.getElementById('atmospheric-data-stream');
        }

        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        } else if (targetId === 'live-dashboard') {
            window.parent.scrollTo({ top: 0, behavior: 'smooth' });
        } else if (cityInput) {
            cityInput.focus();
            cityInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } else if (parts.lastSyncTime) {
            parts.lastSyncTime.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }

    function renderSidebarState() {
        const parts = getParts();
        if (!parts.sidebar || !parts.btn || !parts.overlay) return;

        const isMobile = window.parent.innerWidth <= 768;
        parts.sidebar.classList.toggle('sidebar-open', sidebarOpen);
        parts.sidebar.classList.toggle('sidebar-closed', !sidebarOpen);
        parts.overlay.classList.toggle('active', isMobile && sidebarOpen);
        parts.btn.classList.toggle('sidebar-is-open', sidebarOpen);
        parts.btn.innerHTML = sidebarOpen ? '&times;' : '&#9776;';
        parts.btn.setAttribute('aria-expanded', sidebarOpen ? 'true' : 'false');
    }

    function updateLastSyncTime() {
        const target = getParts().lastSyncTime;
        if (!target) return;
        target.textContent = new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: true
        });
    }

    function applyTheme(mode) {
        const isDark = mode === 'dark';
        doc.body.classList.toggle('airflow-dark', isDark);
        doc.documentElement.classList.toggle('airflow-dark', isDark);
        const toggle = doc.getElementById('themeToggleBtn');
        if (toggle) {
            toggle.classList.toggle('is-dark', isDark);
            toggle.setAttribute('aria-pressed', isDark ? 'true' : 'false');
            toggle.setAttribute('title', isDark ? 'Switch to light mode' : 'Switch to dark mode');
        }
        try {
            window.parent.localStorage.setItem('airflow-theme', mode);
        } catch (err) {}
    }

    function bindThemeToggle() {
        const toggle = doc.getElementById('themeToggleBtn');
        if (!toggle || toggle.dataset.bound === 'true') return;
        toggle.dataset.bound = 'true';
        const savedTheme = (function () {
            try { return window.parent.localStorage.getItem('airflow-theme') || 'light'; }
            catch (err) { return 'light'; }
        })();
        applyTheme(savedTheme);
        toggle.addEventListener('click', function () {
            applyTheme(doc.body.classList.contains('airflow-dark') ? 'light' : 'dark');
        });
    }

    function enhanceSidebarArtwork() {
        const iconMap = {
            'Dashboard': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><rect x="3.5" y="3.5" width="8" height="8" rx="1.7"/><rect x="12.5" y="12.5" width="8" height="8" rx="1.7"/><path d="M15.5 3.5h3a2 2 0 0 1 2 2v3"/><path d="M8.5 20.5h-3a2 2 0 0 1-2-2v-3"/><path d="M7.5 12.5v4"/><path d="M5.5 14.5h4"/></svg>',
            'Live AQI': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3.5 20 8v8l-8 4.5L4 16V8l8-4.5Z"/><path d="M12 8.3v3.8"/><path d="M10.2 10.2h3.6"/><path d="M12 15.5h.01"/></svg>',
            'Forecast': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M5.5 17.5h12.3a3.2 3.2 0 0 0 .4-6.38 5.1 5.1 0 0 0-9.84-1.4A3.8 3.8 0 0 0 5.5 17.5Z"/></svg>',
            'Analytics': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><path d="M8 16v-4"/><path d="M12 16V8"/><path d="M16 16v-7"/><path d="M7 18h10"/></svg>',
            'Alerts': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8a6 6 0 1 0-12 0c0 7-2 7-2 9h16c0-2-2-2-2-9"/><path d="M10 20a2 2 0 0 0 4 0"/></svg>',
            'Reports': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M7 3.5h7l3 3V20.5H7a2 2 0 0 1-2-2v-13a2 2 0 0 1 2-2Z"/><path d="M14 3.5v4h4"/><path d="M8.5 12h7"/><path d="M8.5 15.5h5"/></svg>',
            'Settings': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z"/><path d="M19.4 15a1.8 1.8 0 0 0 .36 1.98l.06.06a2.15 2.15 0 1 1-3.04 3.04l-.06-.06A1.8 1.8 0 0 0 14.74 19a1.8 1.8 0 0 0-1.1.34 1.8 1.8 0 0 0-.86 1.55V21a2.15 2.15 0 1 1-4.3 0v-.1a1.8 1.8 0 0 0-.86-1.55 1.8 1.8 0 0 0-2.06.28l-.06.06a2.15 2.15 0 1 1-3.04-3.04l.06-.06A1.8 1.8 0 0 0 3 14.6a1.8 1.8 0 0 0-.34-1.1A1.8 1.8 0 0 0 1.1 12H1a2.15 2.15 0 1 1 0-4.3h.1a1.8 1.8 0 0 0 1.55-.86 1.8 1.8 0 0 0-.28-2.06l-.06-.06A2.15 2.15 0 1 1 5.35 1.7l.06.06A1.8 1.8 0 0 0 7.4 2a1.8 1.8 0 0 0 1.1-.34A1.8 1.8 0 0 0 9.36.1V0h4.3v.1a1.8 1.8 0 0 0 .86 1.55 1.8 1.8 0 0 0 2.06-.28l.06-.06a2.15 2.15 0 1 1 3.04 3.04l-.06.06A1.8 1.8 0 0 0 19 6.4c.16.37.16.73.4 1.1A1.8 1.8 0 0 0 20.9 8H21a2.15 2.15 0 1 1 0 4.3h-.1a1.8 1.8 0 0 0-1.5 2.7Z" transform="translate(1.5 1.5) scale(.88)"/></svg>'
        };

        doc.querySelectorAll('.sidebar-nav-item').forEach(function (item) {
            const label = item.textContent.trim();
            const icon = item.querySelector('.sidebar-nav-icon');
            if (icon && iconMap[label] && !icon.dataset.enhanced) {
                icon.innerHTML = iconMap[label];
                icon.dataset.enhanced = 'true';
            }
        });

        const oldSatellite = doc.querySelector('.sidebar-status-card .satellite-orbit');
        if (oldSatellite && !doc.querySelector('.sidebar-status-card .status-illustration')) {
            oldSatellite.outerHTML = '<div class="status-illustration"><svg viewBox="0 0 120 102" fill="none" aria-hidden="true"><path d="M30 43c4-20 28-29 47-17 18 11 21 36 7 51-15 17-45 17-58-1-7-10-5-23 4-33Z" fill="#DDF1FF"/><path d="M66 66c13-4 21-16 18-29-18 1-34 12-39 30 6 2 14 2 21-1Z" fill="url(#dishGradient)"/><path d="M43 66c12 7 28 8 40 1" stroke="#7A8CF8" stroke-width="3" stroke-linecap="round"/><path d="M61 66 47 88" stroke="#6278DF" stroke-width="4" stroke-linecap="round"/><path d="M47 88h27" stroke="#6278DF" stroke-width="4" stroke-linecap="round"/><path d="M58 73 74 88" stroke="#6278DF" stroke-width="3" stroke-linecap="round"/><path d="M72 25c12 3 20 11 22 23" stroke="#25A7FF" stroke-width="4" stroke-linecap="round"/><path d="M75 34c6 2 10 6 11 12" stroke="#25A7FF" stroke-width="4" stroke-linecap="round"/><path d="M43 47 62 66" stroke="#8CA0FF" stroke-width="5" stroke-linecap="round"/><defs><linearGradient id="dishGradient" x1="44" y1="36" x2="83" y2="70" gradientUnits="userSpaceOnUse"><stop stop-color="#96B6FF"/><stop offset="1" stop-color="#5B71F0"/></linearGradient></defs></svg></div>';
        }
    }

    function bindSidebarControls() {
        const parts = getParts();
        if (!parts.sidebar || !parts.btn || !parts.overlay) {
            window.setTimeout(bindSidebarControls, 250);
            return;
        }

        if (parts.nativeToggleButton) {
            parts.nativeToggleButton.setAttribute('aria-hidden', 'true');
            parts.nativeToggleButton.setAttribute('tabindex', '-1');
        }

        if (!didBind) {
            bindManaged(parts.btn, 'click', function (e) {
                e.preventDefault();
                e.stopPropagation();
                sidebarOpen = !sidebarOpen;
                saveSidebarState();
                renderSidebarState();
            });

            bindManaged(parts.overlay, 'click', function () {
                sidebarOpen = false;
                saveSidebarState();
                renderSidebarState();
            });

            bindManaged(doc, 'click', function (e) {
                const analyzeButton = getAnalyzeButton();
                if (analyzeButton && e.target.closest('button') === analyzeButton) {
                    closeSidebarOnMobile();
                }
            });

            bindManaged(doc, 'keydown', function (e) {
                const sidebarInput = doc.querySelector('[data-testid="stSidebar"] input');
                if (e.key === 'Enter' && sidebarInput && e.target === sidebarInput) {
                    closeSidebarOnMobile();
                    window.setTimeout(submitCityAnalysis, 50);
                }
            });

            bindManaged(doc, 'click', function (e) {
                const link = e.target.closest('.sidebar-nav-item');
                if (!link) return;
                e.preventDefault();
                scrollToWorkspaceTarget(link);
                closeSidebarOnMobile();
            });

            bindManaged(window.parent, 'resize', function () {
                sidebarOpen = window.parent.innerWidth > 768 ? true : readSavedSidebarState();
                renderSidebarState();
            });

            didBind = true;
        }

        enhanceSidebarArtwork();
        bindThemeToggle();
        renderSidebarState();
        updateLastSyncTime();
    }

    try {
        applyTheme(window.parent.localStorage.getItem('airflow-theme') || 'light');
    } catch (err) {
        applyTheme('light');
    }
    bindSidebarControls();
    setManagedInterval(bindThemeToggle, 500);
    setManagedInterval(updateLastSyncTime, 1000);
})();
</script>
""", height=0, width=0)


# =========================
# 🧭 NAVBAR
# =========================
st.markdown("""
<div class="top-control-bar" aria-label="Application controls">
    <button class="top-icon-btn" type="button" title="Focus mode" aria-label="Focus mode">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 3.8 20.2 12 12 20.2 3.8 12 12 3.8Z"/>
            <path d="M12 8.5 15.5 12 12 15.5 8.5 12 12 8.5Z"/>
        </svg>
    </button>
    <button class="top-icon-btn top-theme-toggle" id="themeToggleBtn" type="button" aria-label="Switch theme" aria-pressed="false" title="Switch to dark mode">
        <svg class="sun-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="3.6"/>
            <path d="M12 2.8v2.1M12 19.1v2.1M4.1 4.1l1.5 1.5M18.4 18.4l1.5 1.5M2.8 12h2.1M19.1 12h2.1M4.1 19.9l1.5-1.5M18.4 5.6l1.5-1.5"/>
        </svg>
        <svg class="moon-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
            <path d="M20.2 14.2A7.6 7.6 0 0 1 9.8 3.8 8.4 8.4 0 1 0 20.2 14.2Z"/>
        </svg>
    </button>
    <div class="top-pill">AI</div>
    <div class="top-pill af">AF</div>
</div>
<div class="navbar">
    <div class="eyebrow">
        <div class="eyebrow-dot"></div>
        <span class="eyebrow-text">Live Air Monitoring System</span>
    </div>
    <div class="navbar-brand">
        <div class="navbar-logo">🍃</div>
        <div class="navbar-title">Air<span>Flow</span></div>
    </div>
    <div class="navbar-links">
        <a class="navbar-link" href="#">Air Quality</a>
        <a class="navbar-link" href="#">Forecasting</a>
        <a class="navbar-link" href="#">Map View</a>
        <a class="navbar-link" href="#">About</a>
        <span class="navbar-cta">Intelligence v2</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================
# 🪪 HERO
# =========================
st.markdown("""
<div class="hero">
    <div class="hero-h1">Air Quality<br><span class="accent">Intelligence</span></div>
    <p class="hero-sub">Real-time AQI tracking, pollutant breakdown, and ML-powered forecasts — built for precision environmental awareness.</p>
    <div class="stat-strip">
        <div class="stat-item"><div class="stat-number">500+</div><div class="stat-label">Cities Tracked</div></div>
        <div class="stat-divider"></div>
        <div class="stat-item"><div class="stat-number">6-Hr</div><div class="stat-label">Forecast Window</div></div>
        <div class="stat-divider"></div>
        <div class="stat-item"><div class="stat-number">7-Day</div><div class="stat-label">Trend Analysis</div></div>
        <div class="stat-divider"></div>
        <div class="stat-item"><div class="stat-number">Live</div><div class="stat-label">Data Stream</div></div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================
# 🎛️ SIDEBAR
# =========================
def request_city_analysis():
    st.session_state.run_from_city_enter = True


if "run_from_city_enter" not in st.session_state:
    st.session_state.run_from_city_enter = False


with st.sidebar:
    st.html("""
    <div class="sidebar-logo-area">
        <div class="sidebar-icon">A</div>
        <div>
            <div class="sidebar-brand-name">AirFlow</div>
            <div class="sidebar-brand-sub">Intelligence v2.0</div>
        </div>
    </div>
    """)

    user_city = st.text_input(
        "📍 CITY / REGION",
        placeholder="e.g. Delhi, London, Tokyo",
        on_change=request_city_analysis,
    )
    run = st.button("Analyse Air Quality")

    st.html("""
    <div class="sidebar-section-title">Workspace</div>
    <nav class="sidebar-nav" aria-label="Workspace navigation">
        <a class="sidebar-nav-item active" href="#live-dashboard">
            <span class="sidebar-nav-icon">▧</span>
            Dashboard
        </a>
        <a class="sidebar-nav-item" href="#atmospheric-data-stream">
            <span class="sidebar-nav-icon">◇</span>
            Live AQI
        </a>
        <a class="sidebar-nav-item" href="#forecast">
            <span class="sidebar-nav-icon">⌁</span>
            Forecast
        </a>
        <a class="sidebar-nav-item" href="#environmental-intelligence">
            <span class="sidebar-nav-icon">▥</span>
            Analytics
        </a>
        <a class="sidebar-nav-item" href="#health-advisory">
            <span class="sidebar-nav-icon">♧</span>
            Alerts
        </a>
        <a class="sidebar-nav-item" href="#reports">
            <span class="sidebar-nav-icon">▤</span>
            Reports
        </a>
        <a class="sidebar-nav-item" href="#settings">
            <span class="sidebar-nav-icon">⚙</span>
            Settings
        </a>
    </nav>

    <div class="sidebar-section-title">Data Sources</div>
    <div class="sidebar-info-list">
        <div class="sidebar-info-row"><span class="sidebar-dot dot-green"></span>WAQI - Live Feed</div>
        <div class="sidebar-info-row"><span class="sidebar-dot dot-blue"></span>ML Forecast Engine</div>
        <div class="sidebar-info-row"><span class="sidebar-dot dot-purple"></span>Geo Location API</div>
    </div>

    <div class="sidebar-section-title">Model Info</div>
    <div class="sidebar-info-list">
        <div class="sidebar-info-row">
            <svg class="sidebar-info-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 16l4-4 4 3 4-7 4 2"/></svg>
            Linear Regression
        </div>
        <div class="sidebar-info-row">
            <svg class="sidebar-info-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="8"/><path d="M12 8v5l3 2"/></svg>
            6-hour prediction horizon
        </div>
        <div class="sidebar-info-row">
            <svg class="sidebar-info-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19h16"/><path d="M6 16l4-5 4 3 4-8"/></svg>
            7-day rolling trend
        </div>
    </div>

    <div class="sidebar-status-card">
        <div class="satellite-orbit"><div class="satellite-body"></div></div>
        <div class="status-title">Live Monitoring<br>System Active</div>
        <div class="status-subtitle"><span></span>Last synced<br><strong id="lastSyncTime">--:--:--</strong></div>
    </div>
    """)

    st.markdown("""
    <div style="display:none;">
        <div style="font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:rgba(150,180,255,0.4);margin-bottom:12px;">Data Sources</div>
        <div style="font-size:12px;color:rgba(180,200,255,0.45);line-height:2.2;">
            <span style="display:inline-block;width:7px;height:7px;background:#00c8ff;border-radius:50%;box-shadow:0 0 6px #00c8ff;margin-right:8px;vertical-align:middle;"></span>WAQI — Live Feed<br>
            <span style="margin-left:15px;">ML Forecast Engine</span><br>
            <span style="margin-left:15px;">Geo Location API</span>
        </div>
    </div>
    <div style="display:none;">
        <div style="font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:rgba(150,130,255,0.4);margin-bottom:10px;">Model Info</div>
        <div style="font-size:12px;color:rgba(180,200,255,0.45);line-height:2.2;">
            Linear Regression<br>6-hour prediction horizon<br>7-day rolling trend
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================
# 🔗 SESSION STATE
# =========================
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False


# =========================
# 📤 FETCH DATA
# =========================
should_fetch_city = (run or st.session_state.run_from_city_enter) and user_city
st.session_state.run_from_city_enter = False

if should_fetch_city:
    try:
        corrected_city = correct_city(user_city)
        api_data = fetch_aqi(corrected_city)
        df = process_data(api_data)

        st.session_state.api_data = api_data
        st.session_state.df = df
        st.session_state.city = corrected_city
        st.session_state.data_loaded = True

        if corrected_city.lower() != user_city.lower():
            st.info(f"⟡ Showing results for: **{corrected_city}**")

    except Exception as e:
        st.error("Enter Correct City / Check Your Network Connection Status")


# =========================
# 📊 DISPLAY
# =========================
if st.session_state.data_loaded:

    api_data = st.session_state.api_data
    df = st.session_state.df
    city = st.session_state.city

    aqi_value = int(df["AQI"].iloc[0])
    status_label, status_color, status_emoji, status_bg, status_border = get_aqi_status(aqi_value)
    risk_pct = aqi_risk_percent(aqi_value)
    short_advice = health_advice(aqi_value)[:54] + "..."

    pollutant_cols = ["PM2.5", "PM10", "NO2", "O3"]
    dominant_pollutant = max(pollutant_cols, key=lambda col: float(df[col].iloc[0]) if col in df else 0)
    geo = api_data.get("city", {}).get("geo", [None, None])
    lat = geo[0] if len(geo) > 0 else None
    lon = geo[1] if len(geo) > 1 else None
    station_name = api_data.get("city", {}).get("name", city)
    location_suffix = station_name
    if isinstance(location_suffix, str) and location_suffix.lower().startswith(city.lower()):
        location_suffix = location_suffix[len(city):].lstrip(", ").strip()
    location_suffix_html = f", {location_suffix}" if location_suffix else ""

    try:
        lat_float = float(lat)
        lon_float = float(lon)
        lat_text = f"{abs(lat_float):.4f}&deg; {'N' if lat_float >= 0 else 'S'}"
        lon_text = f"{abs(lon_float):.4f}&deg; {'E' if lon_float >= 0 else 'W'}"
    except (TypeError, ValueError):
        lat_text = "Unavailable"
        lon_text = "Unavailable"

    spark_values = [float(df[col].iloc[0]) for col in pollutant_cols if col in df] + [float(aqi_value)]
    spark_min, spark_max = min(spark_values), max(spark_values)
    spark_range = spark_max - spark_min or 1
    spark_points_list = []
    for idx, value in enumerate(spark_values):
        x = 8 + idx * (126 / max(len(spark_values) - 1, 1))
        y = 50 - ((value - spark_min) / spark_range) * 38
        spark_points_list.append((x, y))
    spark_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in spark_points_list)
    spark_circles = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4"></circle>' for x, y in spark_points_list)
    needle_angle = min(max((aqi_value / 500) * 220 - 110, -110), 110)

    st.markdown('<div id="live-dashboard" class="scroll-anchor"></div><div class="dashboard-heading"><span class="dashboard-step">1</span> Live Dashboard</div>', unsafe_allow_html=True)

    # BUG FIX: repaired the broken unclosed div structure in this card
    st.html(f"""
    <div class="live-dashboard-grid">
        <div class="dash-card dashboard-stat-card">
            <div class="card-badge" style="background:{status_bg};border:1px solid {status_border};color:{status_color};">
                <span>{status_emoji}</span> {status_label}
            </div>
            <div class="dash-title">Main Statistics</div>
            <div class="dash-label">AQI (US)</div>
            <div class="dash-aqi-number" style="color:{status_color};">{aqi_value}</div>
            <div class="dash-label">Dominant Pollutant <span class="dash-pollutant-value">{dominant_pollutant}</span></div>
            <div class="dash-label">City</div>
            <div class="dash-sub" style="color:#6db2ff;font-weight:900;font-size:15px;">{city}</div>

            <div class="dash-risk-card">
                <div class="dash-risk-head">
                    <div class="dash-title" style="margin:0;font-size:14px;">Risk of Pollution</div>
                    <div class="dash-risk-btn">Details</div>
                </div>
                <div class="dash-label" style="margin-top:0;">RISK</div>
                <div class="dash-risk-percent">{risk_pct}%</div>
                <div class="dash-risk-desc">{short_advice}</div>
                <div class="dash-risk-track"><div class="dash-risk-fill" style="width:{risk_pct}%;"></div></div>
            </div>
        </div>

        <div class="dashboard-right-grid">
            <div class="dash-card dashboard-top-card">
                <div class="dash-row">
                    <div>
                        <div class="dash-title">AQI Index</div>
                        <div class="dash-aqi-number" style="color:{status_color};">{aqi_value}</div>
                        <div class="dash-status-line"><span class="dash-status-dot" style="background:{status_color};"></span>{status_label}</div>
                    </div>
                    <div class="aqi-gauge" style="--needle-angle:{needle_angle:.1f}deg;">
                        <div class="aqi-needle"></div>
                    </div>
                </div>
            </div>

            <div class="dash-card dashboard-top-card">
                <div class="dash-row">
                    <div>
                        <div class="dash-title">Live Data Stream</div>
                        <div class="stream-live">&bull; Live</div>
                        <div class="dash-sub" style="margin-top:10px;">Real-time feed active</div>
                    </div>
                    <svg class="sparkline" viewBox="0 0 150 64" aria-hidden="true">
                        <defs>
                            <linearGradient id="sparkGradient" x1="0" x2="1" y1="0" y2="0">
                                <stop offset="0%" stop-color="#23d0d1"/>
                                <stop offset="100%" stop-color="#20d986"/>
                            </linearGradient>
                        </defs>
                        <polyline points="{spark_points}"></polyline>
                        {spark_circles}
                    </svg>
                </div>
            </div>

            <div class="dash-card dashboard-location-card">
                <div>
                    <div class="dash-title">Location</div>
                    <div class="location-name">{city}<span>{location_suffix_html}</span></div>
                    <div class="location-coords">Lat {lat_text}, Lon {lon_text}</div>
                </div>
                <div class="mini-map">
                    <svg viewBox="0 0 520 210" aria-hidden="true">
                        <path d="M0 108 C70 92 112 116 166 94 S270 70 320 96 414 138 520 96" fill="none" stroke="#a9d4ff" stroke-width="8" opacity="0.7"/>
                        <path d="M335 -10 C350 38 352 86 335 136 S312 190 332 228" fill="none" stroke="#8bc9ff" stroke-width="10" opacity="0.45"/>
                        <g stroke="#dce8ff" stroke-width="2" opacity="0.9">
                            <path d="M22 26H492M12 62H512M18 98H500M8 134H510M30 170H490"/>
                            <path d="M70 0V210M128 18V210M186 0V190M244 20V210M302 0V210M360 24V210M418 0V184M476 18V210"/>
                            <path d="M55 190L160 35M145 205L252 18M250 198L390 30M360 200L505 55"/>
                        </g>
                    </svg>
                    <div class="map-pin"></div>
                </div>
            </div>
        </div>
    </div>
    """)
    short_advice = health_advice(aqi_value)[:54] + "…"

    legacy_dashboard_markup = '''

    left_col, right_col = st.columns([1, 2.2], gap="large")

    with left_col:
        # BUG FIX: repaired the broken unclosed div structure in this card
        st.markdown(f"""
        <div class="main-stats-card">
            <div class="card-badge" style="background:{status_bg};border:1px solid {status_border};color:{status_color};">
                <span>{status_emoji}</span> {status_label}
            </div>
            <div class="card-title">Main Statistics</div>
            <div style="font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:black;margin-bottom:2px;">AQI</div>
            <div class="aqi-big-number" style="color:{status_color};">{aqi_value}</div>
            <div class="pollutant-info" style="color:black;font-size:20px;">
                Dominant Pollutant &nbsp;<strong>PM2.5</strong> — <strong>{city}</strong>
            </div>
            <div class="risk-card">
                <div class="risk-card-header">
                    <div class="risk-card-title">Risk of Pollution</div>
                    <div class="risk-details-btn">Details</div>
                </div>
                <div style="font-size:20px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:black;margin-bottom:4px;">Risk</div>
                <div class="risk-percent">{risk_pct}%</div>
                <div class="risk-desc">{short_advice}</div>
                <div class="risk-bar-bg"><div class="risk-bar-fill" style="width:{risk_pct}%;"></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi-cell">
                <div class="kpi-tag" style="color:black;font-size:13px;">AQI Value</div>
                <div class="kpi-num" style="color:{status_color};">{aqi_value}</div>
                <div class="kpi-meta" style="color:black;">{status_emoji} {status_label}</div>
            </div>
            <div class="kpi-cell">
                <div class="kpi-tag" style="color:black;font-size:13px;">Location</div>
                <div class="kpi-num" style="font-size:20px;color:red;">{city}</div>
                <div class="kpi-meta" style="color:black;">📍 Selected Region</div>
            </div>
            <div class="kpi-cell">
                <div class="kpi-tag" style="color:black;font-size:20px;">Data Stream</div>
                <div class="kpi-num" style="color:#00c8ff;font-size:22px;">● Live</div>
                <div class="kpi-meta" style="color:black;font-size:12px;">Real-time feed active</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><div class="panel-title-icon">📊</div> Pollutant Breakdown</div>', unsafe_allow_html=True)
        fig = plot_pollutants(df)
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    '''

    st.markdown('<div id="pollutant-breakdown" class="scroll-anchor"></div><div class="numbered-section-heading"><span class="section-step">2</span> Pollutant Breakdown</div>', unsafe_allow_html=True)
    st.markdown('<div class="pollutant-chart-card">', unsafe_allow_html=True)
    fig_pollutants = plot_pollutants(df)
    if fig_pollutants is not None:
        st.pyplot(fig_pollutants, use_container_width=True)
    else:
        st.warning("Pollutant data unavailable")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div id="atmospheric-data-stream" class="scroll-anchor"></div><div id="reports" class="scroll-anchor"></div><div class="numbered-section-heading"><span class="section-step">3</span> Atmospheric Data Stream</div>', unsafe_allow_html=True)
    live_stream_df = build_live_stream_table(df, api_data)
    st.html(render_live_stream_table(live_stream_df, city))

    st.markdown("<hr>", unsafe_allow_html=True)


    # -------------------------
    # 🔮 FORECAST + 📈 TREND
    # -------------------------

    st.markdown('<div id="forecast" class="scroll-anchor"></div><div class="numbered-section-heading"><span class="section-step">4</span> Forecast &amp; Trend</div>', unsafe_allow_html=True)
    dummy = generate_dummy_data(df)
    model = train_model(dummy)

    try:
        predictions = predict_future(model, df)
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        predictions = None

    col1, col2 = st.columns(2)

    # =========================
    # 🔮 6-HOUR FORECAST
    # =========================
    with col1:
        if predictions is not None:
            components.html(
                render_forecast_card(
                    "6-Hour AQI Forecast",
                    predictions,
                    "Hour",
                    predictions.columns[1],
                    "#f5b52e",
                    "forecast.csv",
                    "six-hour-forecast-chart",
                ),
                height=350,
                scrolling=False,
            )
        else:
            st.warning("Forecast not available")
    # =========================
    # 📈 7-DAY TREND
    # =========================
    with col2:
        try:
            trend = generate_7day_trend(df)

            components.html(
                render_forecast_card(
                    "7-Day AQI Trend",
                    trend,
                    "Day",
                    trend.columns[1],
                    "#2f9bff",
                    "trend.csv",
                    "seven-day-trend-chart",
                ),
                height=350,
                scrolling=False,
            )

        except Exception as e:
            st.error(f"Trend Error: {e}")
    # =========================
    # 🗺️ MAP
    # =========================
    st.markdown('<div id="geospatial-view" class="scroll-anchor"></div><div class="sec-heading" style="color:red;">🗺️ Geospatial View</div>', unsafe_allow_html=True)

    geo = api_data.get("city", {}).get("geo", [None, None])
    lat = geo[0] if len(geo) > 0 else None
    lon = geo[1] if len(geo) > 1 else None

    st.markdown('<div class="map-glass">', unsafe_allow_html=True)

    if lat is not None and lon is not None:
        m = create_map(lat, lon, aqi_value, city)
        m.scrollWheelZoom = False
        m.doubleClickZoom = False
        m.touchZoom = False
        st_folium(m, width=None, height=460)
    else:
        st.warning("Geographic data unavailable for this location")

    st.markdown('</div>', unsafe_allow_html=True)
# ===========================
#  Health Advisory
# ===========================
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(f'<div id="environmental-intelligence" class="scroll-anchor"></div><div id="health-advisory" class="scroll-anchor"></div><div class="sec-heading" style="color:#00c8ff;">💡 Environmental Intelligence</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="advisory-wrap" style="background:rgba(255,255,255,0.025);">
        <div class="advisory-tag" style="background:{status_bg};border:1px solid {status_border};color:{status_color};">
            {status_emoji} {status_label}
        </div>
        <div class="advisory-title">Health Advisory</div>
        <p class="advisory-body" style="color:#00c8ff;">{health_advice(aqi_value)}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:52px'></div>", unsafe_allow_html=True)

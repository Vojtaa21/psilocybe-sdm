"""
PsiloSDM — Streamlit aplikace
Species Distribution Modeling pro Psilocybe spp.
"""
import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from model import PsilocybeSDM, SDMFeatures

# ── Konfigurace stránky ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="PsiloSDM",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f1f0f; }
[data-testid="stSidebar"] * { color: #c8e6c9 !important; }
[data-testid="stSidebar"] .stSlider label { color: #a5d6a7 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #a8e44a !important; }
[data-testid="stSidebar"] hr { border-color: #2d4a2d; }

.metric-card {
    background: #1a2e1a;
    border: 1px solid #2d4a2d;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.prob-big { font-size: 3rem; font-weight: 700; line-height: 1; }
.suit-label { font-size: 0.85rem; color: #888; margin-top: 4px; }

.edu-box {
    background: #f0f7f0;
    border-left: 3px solid #3B6D11;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 12px;
    font-size: 0.88rem;
    color: #333;
}
.violation-box {
    background: #fff8e1;
    border-left: 3px solid #f9a825;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ── Singleton model ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return PsilocybeSDM()

model = load_model()

# ── Sidebar — parametry ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🍄 PsiloSDM")
    st.markdown("*Species Distribution Modeling*")
    st.markdown("---")

    st.markdown("### Environmentální parametry")
    st.caption("Nastavte hodnoty pro vybraný bod nebo oblast.")

    bio01 = st.slider("🌡 Roční teplota (°C)", -5.0, 25.0, 9.5, 0.5,
                      help="BIO01 — průměrná roční teplota vzduchu ve 2 m")
    bio12 = st.slider("🌧 Roční srážky (mm)", 200, 2000, 780, 10,
                      help="BIO12 — součet ročních srážek")
    ph    = st.slider("🪨 pH půdy", 3.0, 9.0, 5.8, 0.1,
                      help="Reakce půdy — lysohlávky preferují mírně kyselé prostředí")
    elev  = st.slider("⛰ Nadmořská výška (m)", 100, 1500, 420, 10,
                      help="Výška nad mořem — ovlivňuje teplotu a vlhkost")
    ndvi  = st.slider("🌿 Vegetační index NDVI", 0.0, 1.0, 0.62, 0.01,
                      help="Normalized Difference Vegetation Index ze Sentinel-2")

    st.markdown("---")
    st.markdown("### Heatmapa")
    resolution = st.select_slider(
        "Rozlišení mřížky",
        options=[10, 20, 30, 40, 50],
        value=20,
        help="Počet buněk na stranu (vyšší = pomalejší ale přesnější)"
    )
    show_heatmap = st.checkbox("Zobrazit heatmapu výskytu", value=True)

    st.markdown("---")
    run_btn = st.button("▶ Spustit model", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### Datové zdroje")
    st.caption("ERA5 · ČHMÚ · GBIF · SoilGrids · Sentinel-2")
    st.caption("Model: Random Forest (demo mód)")

# ── Výpočet výsledku ─────────────────────────────────────────────────────────
features = SDMFeatures(
    bio01=bio01, bio04=750.0, bio12=float(bio12),
    bio15=25.0, ph=ph, elev=float(elev), ndvi=ndvi,
)
result = model.predict_point(features)

# Barva podle pravděpodobnosti
def prob_color(p):
    if p < 0.25: return "#ef5350"
    if p < 0.50: return "#ffa726"
    if p < 0.75: return "#66bb6a"
    return "#2e7d32"

color = prob_color(result.probability)

# ── Hlavní obsah ─────────────────────────────────────────────────────────────
st.markdown("## Analýza výskytu lysohlávek — ČR / SR")

col_map, col_result = st.columns([3, 2], gap="large")

# ── MAPA ────────────────────────────────────────────────────────────────────
with col_map:
    st.markdown("### Mapa pravděpodobnosti výskytu")

    # Základní mapa
    m = folium.Map(
        location=[49.5, 15.8],
        zoom_start=7,
        tiles="CartoDB positron",
    )

    # Heatmapa mřížky
    if show_heatmap:
        with st.spinner(f"Generuji heatmapu ({resolution}×{resolution})..."):
            grid = model.predict_grid(
                lat_range=(47.5, 51.2),
                lon_range=(12.0, 19.0),
                resolution=resolution,
            )
            lats = np.linspace(47.5, 51.2, resolution)
            lons = np.linspace(12.0, 19.0, resolution)
            dlat = (51.2 - 47.5) / resolution
            dlon = (19.0 - 12.0) / resolution

            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    prob = float(grid[i, j])
                    if prob < 0.10:
                        continue
                    hex_color = prob_color(prob)
                    folium.Rectangle(
                        bounds=[[lat, lon], [lat + dlat, lon + dlon]],
                        color=None,
                        fill=True,
                        fill_color=hex_color,
                        fill_opacity=prob * 0.65,
                        tooltip=f"P(výskyt) = {prob:.0%}",
                    ).add_to(m)

    # Legenda
    legend_html = """
    <div style="position:fixed;bottom:20px;left:20px;background:white;
                padding:10px 14px;border-radius:8px;border:1px solid #ccc;
                font-size:12px;z-index:1000">
        <b>Vhodnost prostředí</b><br>
        <span style="color:#2e7d32">●</span> Velmi vysoká (>75 %)<br>
        <span style="color:#66bb6a">●</span> Vysoká (50–75 %)<br>
        <span style="color:#ffa726">●</span> Střední (25–50 %)<br>
        <span style="color:#ef5350">●</span> Nízká (<25 %)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    map_data = st_folium(m, width="100%", height=480, returned_objects=["last_clicked"])

    # Kliknutý bod na mapě
    if map_data and map_data.get("last_clicked"):
        clicked = map_data["last_clicked"]
        st.info(f"📍 Kliknutý bod: {clicked['lat']:.4f}°N, {clicked['lng']:.4f}°E")

# ── VÝSLEDKY ─────────────────────────────────────────────────────────────────
with col_result:
    st.markdown("### Výsledek modelu")

    pct = int(result.probability * 100)
    st.markdown(f"""
    <div class="metric-card">
        <div class="prob-big" style="color:{color}">{pct} %</div>
        <div class="suit-label">pravděpodobnost výskytu</div>
        <div style="margin-top:8px;font-size:0.9rem;color:#aaa">
            Vhodnost: <strong style="color:{color}">{result.habitat_suitability}</strong>
            &nbsp;·&nbsp; Spolehlivost: <strong>{int(result.confidence*100)} %</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result.probability * 100,
        number={"suffix": " %", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0, 25],  "color": "#ffebee"},
                {"range": [25, 50], "color": "#fff8e1"},
                {"range": [50, 75], "color": "#f1f8e9"},
                {"range": [75, 100],"color": "#e8f5e9"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": result.probability * 100,
            },
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig_gauge.update_layout(
        height=200, margin=dict(t=20, b=0, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#333",
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Ekologická varování
    if result.constraint_violations:
        st.markdown("**⚠ Ekologická varování**")
        for v in result.constraint_violations:
            st.markdown(f'<div class="violation-box">⚠ {v}</div>', unsafe_allow_html=True)

    # Feature importance
    st.markdown("**Důležitost proměnných**")
    imp = result.feature_importance
    labels = {
        "bio01": "Teplota", "bio04": "Sezónnost t.",
        "bio12": "Srážky", "bio15": "Sezónnost sr.",
        "ph": "pH půdy", "elev": "Výška", "ndvi": "NDVI",
    }
    imp_df = pd.DataFrame({
        "Proměnná": [labels.get(k, k) for k in imp],
        "Důležitost": list(imp.values()),
    }).sort_values("Důležitost", ascending=True)

    fig_imp = px.bar(
        imp_df, x="Důležitost", y="Proměnná",
        orientation="h",
        color="Důležitost",
        color_continuous_scale=[[0, "#c8e6c9"], [1, "#2e7d32"]],
        text=imp_df["Důležitost"].apply(lambda x: f"{x:.0%}"),
    )
    fig_imp.update_layout(
        height=220, margin=dict(t=0, b=0, l=0, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False, coloraxis_showscale=False,
        xaxis_tickformat=".0%",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig_imp.update_traces(textposition="outside")
    st.plotly_chart(fig_imp, use_container_width=True)

# ── EDUKATIVNÍ SEKCE ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Jak model funguje")

edu1, edu2, edu3, edu4 = st.columns(4)

with edu1:
    st.markdown("""
    <div class="edu-box">
    <strong>🗃 Datové vstupy</strong><br>
    Model kombinuje bioklimatická data ERA5 (teplota, srážky), pH půdy ze SoilGrids
    a vegetační index NDVI ze Sentinel-2 satelitu.
    </div>
    """, unsafe_allow_html=True)

with edu2:
    st.markdown("""
    <div class="edu-box">
    <strong>🤖 Algoritmus</strong><br>
    Random Forest natrénovaný na přítomnostních záznamech z GBIF vs. pseudo-absencích
    (MaxEnt-like přístup). Výstup: kalibrovaná pravděpodobnost 0–1.
    </div>
    """, unsafe_allow_html=True)

with edu3:
    st.markdown("""
    <div class="edu-box">
    <strong>✅ Validace</strong><br>
    Model hodnotíme pomocí AUC-ROC na stratifikované 5-fold cross-validaci.
    Typická hodnota pro SDM: AUC 0.80–0.92.
    </div>
    """, unsafe_allow_html=True)

with edu4:
    st.markdown("""
    <div class="edu-box">
    <strong>🍄 Ekologie</strong><br>
    <em>Psilocybe semilanceata</em> preferuje vlhká louky,
    teploty 5–18 °C, pH 5–7 a nadmořské výšky do 900 m n. m.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("PsiloSDM · vzdělávací prototyp · data: ERA5, ČHMÚ, GBIF, SoilGrids, Sentinel-2")

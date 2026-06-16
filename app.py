"""
PsiloSDM v2 — Streamlit aplikace s reálnými daty
"""
import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px

from data_loader import (
    fetch_gbif_occurrences,
    build_presence_dataset,
    build_background_dataset,
    fetch_soil_properties,
    get_worldclim_value,
    fetch_elevation_batch,
    BBOX,
)
from model import PsilocybeSDM, FEATURE_LABELS

# ── Konfigurace ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PsiloSDM",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0d1f0d; }
[data-testid="stSidebar"] * { color: #c8e6c9 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #a8e44a !important; }
[data-testid="stSidebar"] hr { border-color: #1e3a1e; }
.stProgress > div > div { background: #2e7d32; }
.metric-big { font-size: 2.8rem; font-weight: 700; line-height: 1.1; }
.source-chip {
    display: inline-block; font-size: 11px; padding: 2px 8px;
    border-radius: 12px; margin: 2px; background: #e8f5e9; color: #2e7d32;
    border: 1px solid #a5d6a7;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = PsilocybeSDM()
if "trained" not in st.session_state:
    st.session_state.trained = False
if "auc" not in st.session_state:
    st.session_state.auc = None
if "clicked_point" not in st.session_state:
    st.session_state.clicked_point = None
if "heatmap_grid" not in st.session_state:
    st.session_state.heatmap_grid = None


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🍄 PsiloSDM")
    st.markdown("*AI analýza výskytu lysohlávek*")
    st.markdown("---")

    st.markdown("### Datové zdroje")
    st.markdown("""
    <span class="source-chip">GBIF</span>
    <span class="source-chip">SoilGrids</span>
    <span class="source-chip">WorldClim</span>
    <span class="source-chip">Open-Elevation</span>
    """, unsafe_allow_html=True)
    st.markdown("")

    # Trénink modelu
    if not st.session_state.trained:
        st.markdown("### Inicializace modelu")
        st.info("Model ještě nebyl natrénován na reálných datech.")
        if st.button("▶ Stáhnout data a natrénovat AI", use_container_width=True, type="primary"):
            with st.spinner("Stahuji výskytová data z GBIF..."):
                try:
                    presence_df = build_presence_dataset()
                    st.success(f"✓ Načteno {len(presence_df)} výskytů")
                except Exception as e:
                    st.error(f"Chyba GBIF: {e}")
                    presence_df = pd.DataFrame()

            with st.spinner("Generuji pseudo-absence..."):
                background_df = build_background_dataset(
                    n_points=max(500, len(presence_df) * 5)
                )

            with st.spinner("Trénuji Random Forest model..."):
                auc = st.session_state.model.train(presence_df, background_df)
                st.session_state.trained = True
                st.session_state.auc = auc
                st.success(f"✓ Model natrénován! AUC = {auc:.3f}")
                st.rerun()
    else:
        st.markdown("### Model")
        st.success(f"✓ Natrénován | AUC = {st.session_state.auc:.3f}")
        if st.button("↺ Přetrénovat", use_container_width=True):
            st.session_state.trained = False
            st.session_state.heatmap_grid = None
            st.rerun()

    st.markdown("---")
    st.markdown("### Heatmapa")
    resolution = st.select_slider(
        "Rozlišení", options=[15, 20, 25, 30], value=20,
        help="Vyšší = přesnější ale pomalejší"
    )
    if st.button("🗺 Vygenerovat heatmapu", use_container_width=True,
                 disabled=not st.session_state.trained):
        with st.spinner(f"Počítám {resolution}×{resolution} buněk..."):
            lats, lons, grid = st.session_state.model.predict_grid(
                lat_range=(BBOX["lat_min"], BBOX["lat_max"]),
                lon_range=(BBOX["lon_min"], BBOX["lon_max"]),
                resolution=resolution,
            )
            st.session_state.heatmap_grid = (lats, lons, grid)
        st.success("✓ Heatmapa připravena")

    st.markdown("---")
    st.markdown("### Ruční parametry")
    st.caption("Pro testování konkrétního bodu:")
    manual_mode = st.checkbox("Zadat parametry ručně")

    if manual_mode:
        m_temp  = st.slider("Teplota (°C)", -5.0, 25.0, 9.5, 0.5)
        m_rain  = st.slider("Srážky (mm)", 200, 2000, 780, 10)
        m_ph    = st.slider("pH půdy", 3.0, 9.0, 5.8, 0.1)
        m_elev  = st.slider("Výška (m)", 50, 1500, 420, 10)
        m_slope = st.slider("Sklon (°)", 0.0, 40.0, 8.0, 0.5)
        m_ndvi  = st.slider("NDVI", 0.0, 1.0, 0.60, 0.01)
        m_soc   = st.slider("Organická hmota (g/kg)", 5, 100, 25, 1)


# ── Hlavní obsah ─────────────────────────────────────────────────────────────
st.markdown("## 🍄 PsiloSDM — AI modelování výskytu lysohlávek")

if not st.session_state.trained:
    st.info("👈 Nejdřív natrénuj model v levém panelu — stáhne reálná data a připraví AI.")

    # Zobraz zatím co je k dispozici — GBIF záznamy
    with st.expander("Náhled GBIF dat (bez tréninku)", expanded=True):
        with st.spinner("Načítám GBIF záznamy..."):
            try:
                gbif_preview = fetch_gbif_occurrences(limit_per_species=50)
                st.write(f"Nalezeno **{len(gbif_preview)}** záznamů výskytu")
                if not gbif_preview.empty:
                    m = folium.Map(location=[49.5, 16.0], zoom_start=6,
                                   tiles="CartoDB positron")
                    for _, row in gbif_preview.iterrows():
                        folium.CircleMarker(
                            [row.lat, row.lon],
                            radius=5, color="#2e7d32",
                            fill=True, fill_opacity=0.7,
                            tooltip=f"{row.get('species','?')} ({row.get('year','')})",
                        ).add_to(m)
                    st_folium(m, width="100%", height=400)
            except Exception as e:
                st.warning(f"Nepodařilo se načíst GBIF data: {e}")
    st.stop()

# ── Po natrénování: hlavní layout ────────────────────────────────────────────
col_map, col_panel = st.columns([3, 2], gap="large")

with col_map:
    st.markdown("### Mapa pravděpodobnosti výskytu")

    # Sestav mapu
    m = folium.Map(
        location=[49.5, 16.0],
        zoom_start=7,
        tiles="CartoDB positron",
    )

    # GBIF výskytové body
    try:
        gbif_df = fetch_gbif_occurrences(limit_per_species=200)
        gbif_group = folium.FeatureGroup(name="GBIF výskyty", show=True)
        for _, row in gbif_df.iterrows():
            folium.CircleMarker(
                [row.lat, row.lon],
                radius=4,
                color="#ffffff",
                weight=1,
                fill=True,
                fill_color="#1b5e20",
                fill_opacity=0.85,
                tooltip=f"🍄 {row.get('species','Psilocybe')} ({row.get('year','')})",
            ).add_to(gbif_group)
        gbif_group.add_to(m)
    except Exception:
        pass

    # Heatmapa
    if st.session_state.heatmap_grid is not None:
        lats, lons, grid = st.session_state.heatmap_grid
        dlat = (lats[-1] - lats[0]) / len(lats)
        dlon = (lons[-1] - lons[0]) / len(lons)

        heat_group = folium.FeatureGroup(name="Heatmapa SDM", show=True)
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                prob = float(grid[i, j])
                if prob < 0.12:
                    continue
                if prob >= 0.75:   color = "#1b5e20"
                elif prob >= 0.50: color = "#388e3c"
                elif prob >= 0.25: color = "#f57f17"
                else:              color = "#c62828"

                folium.Rectangle(
                    bounds=[[lat, lon], [lat + dlat, lon + dlon]],
                    color=None, fill=True,
                    fill_color=color,
                    fill_opacity=min(0.75, prob * 0.9),
                    tooltip=f"P(výskyt) = {prob:.0%}",
                ).add_to(heat_group)
        heat_group.add_to(m)

    # Kliknutý bod
    if st.session_state.clicked_point:
        cp = st.session_state.clicked_point
        folium.Marker(
            [cp["lat"], cp["lon"]],
            icon=folium.Icon(color="red", icon="info-sign"),
            tooltip=f"Vybraný bod: {cp['lat']:.4f}°N, {cp['lon']:.4f}°E",
        ).add_to(m)

    folium.LayerControl().add_to(m)

    # Legenda
    legend = """
    <div style="position:fixed;bottom:20px;left:20px;background:white;
                padding:10px 14px;border-radius:8px;border:1px solid #ccc;
                font-size:12px;z-index:1000;box-shadow:0 2px 8px rgba(0,0,0,0.15)">
        <b>🍄 Vhodnost prostředí</b><br>
        <span style="color:#1b5e20">●</span> Velmi vysoká (&gt;75 %)<br>
        <span style="color:#388e3c">●</span> Vysoká (50–75 %)<br>
        <span style="color:#f57f17">●</span> Střední (25–50 %)<br>
        <span style="color:#c62828">●</span> Nízká (&lt;25 %)<br>
        <span style="color:#1b5e20">●</span> GBIF výskyty
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))

    map_data = st_folium(m, width="100%", height=500,
                         returned_objects=["last_clicked"])

    # Zpracuj klik na mapu
    if map_data and map_data.get("last_clicked"):
        clicked = map_data["last_clicked"]
        st.session_state.clicked_point = clicked


with col_panel:
    st.markdown("### Analýza bodu")

    # Určení bodu k analýze
    if manual_mode:
        features = {
            "bio01": m_temp, "bio04": 720.0,
            "bio12": float(m_rain), "bio15": 26.0,
            "elev": float(m_elev), "slope": m_slope,
            "ndvi": m_ndvi, "ph": m_ph,
            "soc": float(m_soc), "clay": 28.0,
            "sand": 35.0, "bulk_density": 1.2,
        }
        point_info = f"Ruční zadání"

    elif st.session_state.clicked_point:
        cp = st.session_state.clicked_point
        lat, lon = cp["lat"], cp["lon"]
        point_info = f"{lat:.4f}°N, {lon:.4f}°E"

        with st.spinner("Načítám data pro bod..."):
            climate = get_worldclim_value(lat, lon)
            soil = fetch_soil_properties(lat, lon)
            elev_list = fetch_elevation_batch([(lat, lon)])
            elev = elev_list[0] if elev_list else 300.0

        features = {
            **climate,
            "elev": elev,
            "slope": 8.0,  # aproximace bez DEM
            "ndvi": float(np.clip(
                0.5 + (climate["bio12"] - 600) / 3000, 0.2, 0.9
            )),
            **{k: soil.get(k, 0) for k in ["ph", "soc", "clay", "sand", "bulk_density"]},
        }
    else:
        st.info("👆 Klikni na mapu pro analýzu konkrétního bodu\nnebo zapni ruční zadání v levém panelu.")
        features = None
        point_info = None

    if features:
        result = st.session_state.model.predict_point(features)
        pct = int(result.probability * 100)

        st.markdown(f"📍 **{point_info}**")

        # Velké číslo
        st.markdown(
            f'<div class="metric-big" style="color:{result.suitability_color}">'
            f'{pct} %</div>'
            f'<div style="color:#666;font-size:0.9rem">pravděpodobnost výskytu · '
            f'vhodnost: <strong>{result.habitat_suitability}</strong></div>',
            unsafe_allow_html=True,
        )

        # Gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": " %", "font": {"size": 24}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": result.suitability_color, "thickness": 0.28},
                "steps": [
                    {"range": [0, 25],  "color": "#ffebee"},
                    {"range": [25, 50], "color": "#fff8e1"},
                    {"range": [50, 75], "color": "#f1f8e9"},
                    {"range": [75, 100],"color": "#e8f5e9"},
                ],
            },
        ))
        fig_g.update_layout(
            height=180, margin=dict(t=10, b=0, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_g, use_container_width=True)

        # AUC info
        if result.auc_score:
            st.caption(f"Model AUC (cross-validace): **{result.auc_score:.3f}**")

        # Varování
        if result.constraint_violations:
            with st.expander(f"⚠ {len(result.constraint_violations)} ekologických varování"):
                for v in result.constraint_violations:
                    st.warning(v)

        # Feature importance
        st.markdown("**Váha proměnných v modelu**")
        imp = result.feature_importance
        if imp:
            imp_df = pd.DataFrame({
                "Proměnná": [FEATURE_LABELS.get(k, k) for k in imp],
                "Důležitost": list(imp.values()),
            }).sort_values("Důležitost", ascending=True).tail(8)

            fig_i = px.bar(
                imp_df, x="Důležitost", y="Proměnná",
                orientation="h",
                color="Důležitost",
                color_continuous_scale=["#c8e6c9", "#1b5e20"],
                text=imp_df["Důležitost"].apply(lambda x: f"{x:.0%}"),
            )
            fig_i.update_layout(
                height=250, margin=dict(t=0, b=0, l=0, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False, coloraxis_showscale=False,
                xaxis_tickformat=".0%",
            )
            fig_i.update_traces(textposition="outside")
            st.plotly_chart(fig_i, use_container_width=True)

        # Detailní hodnoty pro bod
        with st.expander("Environmentální hodnoty bodu"):
            cols = st.columns(2)
            items = list(features.items())
            for i, (k, v) in enumerate(items):
                with cols[i % 2]:
                    label = FEATURE_LABELS.get(k, k)
                    st.metric(label, f"{v:.2f}")


# ── Spodní sekce — GBIF přehled ──────────────────────────────────────────────
st.markdown("---")
with st.expander("📊 Přehled GBIF výskytových dat", expanded=False):
    try:
        gbif_df = fetch_gbif_occurrences()
        c1, c2, c3 = st.columns(3)
        c1.metric("Celkem výskytů", len(gbif_df))
        c2.metric("Druhů", gbif_df["species"].nunique() if "species" in gbif_df.columns else "–")
        c3.metric("Zemí", gbif_df["country"].nunique() if "country" in gbif_df.columns else "–")

        if "year" in gbif_df.columns:
            year_counts = gbif_df.dropna(subset=["year"])["year"].astype(int).value_counts().sort_index()
            fig_y = px.bar(
                x=year_counts.index, y=year_counts.values,
                labels={"x": "Rok", "y": "Počet nálezů"},
                color_discrete_sequence=["#2e7d32"],
            )
            fig_y.update_layout(
                height=200, margin=dict(t=10, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_y, use_container_width=True)
    except Exception as e:
        st.warning(f"GBIF data nejsou dostupná: {e}")

st.caption("PsiloSDM v2 · data: GBIF · SoilGrids · WorldClim · Open-Elevation · model: Random Forest")

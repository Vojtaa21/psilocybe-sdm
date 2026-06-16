"""
PsiloSDM v2 — AI modelování výskytu lysohlávek
Kompletní Streamlit aplikace s reálnými daty
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

st.set_page_config(
    page_title="PsiloSDM · AI výskyt lysohlávek",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1a0a 0%, #0f2a0f 100%);
    border-right: 1px solid #1e3a1e;
}
[data-testid="stSidebar"] * { color: #c8e6c9 !important; }
[data-testid="stSidebar"] h1 { color: #a8e44a !important; font-size: 1.4rem !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #81c784 !important; font-size: 0.9rem !important;
    text-transform: uppercase; letter-spacing: 0.08em;
}
[data-testid="stSidebar"] hr { border-color: #1e3a1e !important; margin: 12px 0; }
.main .block-container { padding-top: 1.5rem; max-width: 1400px; }
h2 { font-size: 1.1rem !important; font-weight: 600 !important; color: #1b5e20 !important; }
.sdm-card {
    background: #ffffff; border: 1px solid #e8f5e9;
    border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.prob-display { font-size: 3.5rem; font-weight: 800; line-height: 1; letter-spacing: -0.02em; }
.prob-sublabel { font-size: 0.82rem; color: #666; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.06em; }
.source-badge {
    display: inline-block; font-size: 10px; font-weight: 600;
    padding: 3px 8px; border-radius: 20px; margin: 2px;
    background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7;
    text-transform: uppercase; letter-spacing: 0.04em;
}
.status-ok {
    display: inline-flex; align-items: center; gap: 6px;
    background: #e8f5e9; color: #1b5e20; border: 1px solid #a5d6a7;
    border-radius: 8px; padding: 6px 12px; font-size: 0.85rem; font-weight: 600;
}
.status-warn {
    background: #fff8e1; color: #f57f17; border: 1px solid #ffe082;
    border-radius: 8px; padding: 6px 12px; font-size: 0.85rem;
}
.eco-warning {
    background: #fff8e1; border-left: 3px solid #f9a825;
    border-radius: 0 8px 8px 0; padding: 8px 12px;
    margin-bottom: 6px; font-size: 0.82rem; color: #5d4037;
}
.edu-box {
    background: #f9fbe7; border-left: 3px solid #8bc34a;
    border-radius: 0 10px 10px 0; padding: 12px 16px;
    margin-bottom: 10px; font-size: 0.84rem; color: #33691e; line-height: 1.6;
}
.edu-box strong { display: block; font-size: 0.88rem; color: #1b5e20; margin-bottom: 4px; }
.mini-metric {
    background: #f9fbe7; border-radius: 8px; padding: 10px 12px;
    text-align: center; border: 1px solid #dcedc8;
}
.mini-metric-val { font-size: 1.2rem; font-weight: 700; color: #33691e; }
.mini-metric-label { font-size: 0.75rem; color: #689f38; margin-top: 2px; }
.sdm-divider { border: none; border-top: 1px solid #e8f5e9; margin: 20px 0; }
.stProgress > div > div { background-color: #2e7d32 !important; }
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {
        "model": PsilocybeSDM(),
        "trained": False,
        "auc": None,
        "clicked_point": None,
        "heatmap_grid": None,
        "gbif_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🍄 PsiloSDM")
    st.markdown("*AI analýza výskytu lysohlávek*")
    st.markdown("---")

    st.markdown("### Datové zdroje")
    st.markdown("""
    <span class="source-badge">GBIF</span>
    <span class="source-badge">SoilGrids</span>
    <span class="source-badge">WorldClim</span>
    <span class="source-badge">Open-Elevation</span>
    """, unsafe_allow_html=True)
    st.markdown("")

    st.markdown("### Model")
    if st.session_state.trained:
        auc = st.session_state.auc
        auc_color = "#1b5e20" if auc > 0.85 else "#f57f17"
        st.markdown(f"""
        <div class="status-ok">
            ✓ Natrénován &nbsp;·&nbsp;
            <span style="color:{auc_color}">AUC = {auc:.3f}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.session_state.gbif_count:
            st.caption(f"📍 {st.session_state.gbif_count} GBIF výskytů")
        if st.button("↺ Přetrénovat model", use_container_width=True):
            st.session_state.trained = False
            st.session_state.heatmap_grid = None
            st.rerun()
    else:
        st.markdown('<div class="status-warn">⏳ Čeká na trénink</div>', unsafe_allow_html=True)
        st.markdown("")
        if st.button("▶ Stáhnout data a natrénovat AI",
                     use_container_width=True, type="primary"):
            progress = st.progress(0, text="Stahuji GBIF data...")
            try:
                presence_df = build_presence_dataset()
                st.session_state.gbif_count = len(presence_df)
                progress.progress(40, text=f"Načteno {len(presence_df)} výskytů...")
            except Exception as e:
                st.error(f"Chyba: {e}")
                presence_df = pd.DataFrame()
            progress.progress(60, text="Generuji pseudo-absence...")
            background_df = build_background_dataset(n_points=max(500, len(presence_df) * 5))
            progress.progress(80, text="Trénuji Random Forest...")
            auc = st.session_state.model.train(presence_df, background_df)
            st.session_state.trained = True
            st.session_state.auc = auc
            progress.progress(100, text="Hotovo!")
            st.rerun()

    st.markdown("---")
    st.markdown("### Heatmapa")
    resolution = st.select_slider(
        "Rozlišení mřížky", options=[10, 15, 20, 25, 30], value=15,
        disabled=not st.session_state.trained,
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
    st.markdown("### Testovací bod")
    manual_mode = st.checkbox("Zadat parametry ručně")
    if manual_mode:
        m_temp  = st.slider("🌡 Teplota (°C)", -5.0, 25.0, 9.5, 0.5)
        m_rain  = st.slider("🌧 Srážky (mm/rok)", 200, 2000, 780, 10)
        m_ph    = st.slider("🪨 pH půdy", 3.0, 9.0, 5.8, 0.1)
        m_elev  = st.slider("⛰ Výška (m n. m.)", 50, 1500, 420, 10)
        m_slope = st.slider("📐 Sklon svahu (°)", 0.0, 40.0, 8.0, 0.5)
        m_ndvi  = st.slider("🌿 NDVI", 0.0, 1.0, 0.60, 0.01)
        m_soc   = st.slider("🌱 Organická hmota (g/kg)", 5, 100, 25, 1)


# ── Hlavní obsah ─────────────────────────────────────────────────────────────
col_title, col_stats = st.columns([2, 1])
with col_title:
    st.markdown("## 🍄 PsiloSDM — AI modelování výskytu lysohlávek")
    st.caption("Reálná data · Random Forest · ČR a Slovensko")
with col_stats:
    if st.session_state.trained:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="mini-metric"><div class="mini-metric-val">{st.session_state.gbif_count}</div><div class="mini-metric-label">GBIF výskytů</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="mini-metric"><div class="mini-metric-val">{st.session_state.auc:.2f}</div><div class="mini-metric-label">AUC skóre</div></div>', unsafe_allow_html=True)

st.markdown('<hr class="sdm-divider">', unsafe_allow_html=True)

if not st.session_state.trained:
    st.info("👈 Klikni na **'Stáhnout data a natrénovat AI'** v levém panelu pro spuštění.")
    with st.expander("🗺 Náhled GBIF dat — reálné nálezy lysohlávek", expanded=True):
        with st.spinner("Načítám záznamy z GBIF..."):
            try:
                preview_df = fetch_gbif_occurrences(limit_per_species=50)
                st.caption(f"Nalezeno **{len(preview_df)}** záznamů výskytu z GBIF")
                m_prev = folium.Map(location=[49.5, 16.0], zoom_start=6, tiles="CartoDB positron")
                for _, row in preview_df.iterrows():
                    folium.CircleMarker(
                        [row.lat, row.lon], radius=5, color="#fff", weight=1,
                        fill=True, fill_color="#2e7d32", fill_opacity=0.8,
                        tooltip=f"🍄 {row.get('species','Psilocybe')} · {row.get('year','')}",
                    ).add_to(m_prev)
                st_folium(m_prev, width="100%", height=420, key="preview_map")
            except Exception as e:
                st.warning(f"GBIF data nejsou dostupná: {e}")
    st.stop()

# ── Po natrénování ────────────────────────────────────────────────────────────
col_map, col_panel = st.columns([3, 2], gap="large")

with col_map:
    st.markdown("### Mapa pravděpodobnosti výskytu")

    m = folium.Map(location=[49.5, 16.5], zoom_start=7, tiles="CartoDB positron")

    try:
        gbif_df = fetch_gbif_occurrences(limit_per_species=200)
        gbif_layer = folium.FeatureGroup(name="🍄 GBIF výskyty", show=True)
        for _, row in gbif_df.iterrows():
            folium.CircleMarker(
                [row.lat, row.lon], radius=5, color="#fff", weight=1.5,
                fill=True, fill_color="#1b5e20", fill_opacity=0.85,
                tooltip=f"🍄 {row.get('species','Psilocybe')} · {row.get('year','')}",
            ).add_to(gbif_layer)
        gbif_layer.add_to(m)
    except Exception:
        pass

    if st.session_state.heatmap_grid is not None:
        lats_g, lons_g, grid = st.session_state.heatmap_grid
        dlat = (lats_g[-1] - lats_g[0]) / max(len(lats_g) - 1, 1)
        dlon = (lons_g[-1] - lons_g[0]) / max(len(lons_g) - 1, 1)
        heat_layer = folium.FeatureGroup(name="🌡 Heatmapa SDM", show=True)
        for i, lat in enumerate(lats_g):
            for j, lon in enumerate(lons_g):
                prob = float(grid[i, j])
                if prob < 0.12:
                    continue
                color = (
                    "#1b5e20" if prob >= 0.75 else
                    "#388e3c" if prob >= 0.50 else
                    "#f57f17" if prob >= 0.25 else "#c62828"
                )
                folium.Rectangle(
                    bounds=[[lat, lon], [lat + dlat, lon + dlon]],
                    color=None, fill=True, fill_color=color,
                    fill_opacity=min(0.72, prob * 0.85),
                    tooltip=f"P(výskyt) = {prob:.0%}",
                ).add_to(heat_layer)
        heat_layer.add_to(m)

    cp = st.session_state.clicked_point
    if cp:
        folium.Marker(
            [cp["lat"], cp["lon"]],
            icon=folium.Icon(color="red", icon="info-sign"),
            tooltip=f"📍 {cp['lat']:.4f}°N · {cp['lon']:.4f}°E",
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element("""
    <div style="position:fixed;bottom:24px;left:16px;background:white;padding:12px 16px;
                border-radius:10px;border:1px solid #ddd;font-size:12px;z-index:1000;
                box-shadow:0 2px 12px rgba(0,0,0,0.12)">
        <div style="font-weight:700;margin-bottom:6px;color:#1b5e20">Vhodnost prostředí</div>
        <div><span style="color:#1b5e20;font-size:16px">●</span> Velmi vysoká (&gt;75 %)</div>
        <div><span style="color:#388e3c;font-size:16px">●</span> Vysoká (50–75 %)</div>
        <div><span style="color:#f57f17;font-size:16px">●</span> Střední (25–50 %)</div>
        <div><span style="color:#c62828;font-size:16px">●</span> Nízká (&lt;25 %)</div>
        <div style="margin-top:6px;border-top:1px solid #eee;padding-top:6px">
        <span style="color:#1b5e20;font-size:16px">●</span> GBIF nálezy</div>
    </div>
    """))

    map_result = st_folium(m, width="100%", height=520, key="main_map",
                           returned_objects=["last_clicked"])

    # Opravené zpracování kliknutí
    if map_result:
        raw = map_result.get("last_clicked")
        if raw and isinstance(raw, dict):
            lat = raw.get("lat") or raw.get("latitude")
            lng = raw.get("lng") or raw.get("lon") or raw.get("longitude")
            if lat is not None and lng is not None:
                new_point = {"lat": float(lat), "lon": float(lng)}
                if new_point != st.session_state.clicked_point:
                    st.session_state.clicked_point = new_point
                    st.rerun()


with col_panel:
    features = None
    point_label = None

    if manual_mode:
        features = {
            "bio01": m_temp, "bio04": 720.0, "bio12": float(m_rain), "bio15": 26.0,
            "elev": float(m_elev), "slope": m_slope, "ndvi": m_ndvi, "ph": m_ph,
            "soc": float(m_soc), "clay": 28.0, "sand": 35.0, "bulk_density": 1.2,
        }
        point_label = "Ruční zadání parametrů"

    elif st.session_state.clicked_point:
        cp = st.session_state.clicked_point
        lat, lon = cp["lat"], cp["lon"]
        point_label = f"{lat:.4f}°N · {lon:.4f}°E"
        with st.spinner("Načítám data pro vybraný bod..."):
            climate = get_worldclim_value(lat, lon)
            soil = fetch_soil_properties(lat, lon)
            elev_vals = fetch_elevation_batch([(lat, lon)])
            elev = elev_vals[0] if elev_vals else 300.0
        features = {
            **climate,
            "elev": float(elev),
            "slope": 8.0,
            "ndvi": float(np.clip(0.5 + (climate["bio12"] - 600) / 3000, 0.2, 0.9)),
            "ph": soil.get("ph", 5.8),
            "soc": soil.get("soc", 20.0),
            "clay": soil.get("clay", 28.0),
            "sand": soil.get("sand", 35.0),
            "bulk_density": soil.get("bulk_density", 1.2),
        }

    if features:
        result = st.session_state.model.predict_point(features)
        pct = int(result.probability * 100)
        color = result.suitability_color

        st.markdown("### Výsledek analýzy")
        st.markdown(f"📍 `{point_label}`")

        st.markdown(f"""
        <div class="sdm-card">
            <div class="prob-display" style="color:{color}">{pct} %</div>
            <div class="prob-sublabel">pravděpodobnost výskytu</div>
            <div style="margin-top:10px;font-size:0.9rem">
                Vhodnost: <strong style="color:{color}">{result.habitat_suitability}</strong>
                &nbsp;·&nbsp; AUC: <strong>{result.confidence:.3f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=pct,
            number={"suffix": " %", "font": {"size": 26, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#ccc"},
                "bar": {"color": color, "thickness": 0.25}, "bgcolor": "white", "borderwidth": 0,
                "steps": [
                    {"range": [0, 25],  "color": "#ffebee"},
                    {"range": [25, 50], "color": "#fff8e1"},
                    {"range": [50, 75], "color": "#f1f8e9"},
                    {"range": [75, 100],"color": "#e8f5e9"},
                ],
                "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.85, "value": pct},
            },
        ))
        fig_g.update_layout(height=190, margin=dict(t=10, b=0, l=30, r=30),
                            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g, use_container_width=True)

        if result.constraint_violations:
            with st.expander(f"⚠ {len(result.constraint_violations)} ekologická varování"):
                for v in result.constraint_violations:
                    st.markdown(f'<div class="eco-warning">⚠ {v}</div>', unsafe_allow_html=True)

        if result.feature_importance:
            st.markdown("**Váha proměnných v modelu**")
            imp = result.feature_importance
            imp_df = pd.DataFrame({
                "Proměnná": [FEATURE_LABELS.get(k, k) for k in imp],
                "Důležitost": list(imp.values()),
            }).sort_values("Důležitost", ascending=True).tail(8)
            fig_i = px.bar(imp_df, x="Důležitost", y="Proměnná", orientation="h",
                           color="Důležitost", color_continuous_scale=["#c8e6c9", "#1b5e20"],
                           text=imp_df["Důležitost"].apply(lambda x: f"{x:.0%}"))
            fig_i.update_layout(height=240, margin=dict(t=0, b=0, l=0, r=40),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                showlegend=False, coloraxis_showscale=False,
                                xaxis={"tickformat": ".0%", "showgrid": False},
                                yaxis={"showgrid": False}, font={"size": 11})
            fig_i.update_traces(textposition="outside", marker_line_width=0)
            st.plotly_chart(fig_i, use_container_width=True)

        with st.expander("🔬 Environmentální hodnoty bodu"):
            cols = st.columns(2)
            items = [(k, v) for k, v in features.items() if k in FEATURE_LABELS]
            for i, (k, v) in enumerate(items):
                with cols[i % 2]:
                    st.metric(FEATURE_LABELS.get(k, k), f"{v:.2f}")
    else:
        st.markdown("### Analýza bodu")
        st.markdown("""
        <div class="sdm-card" style="text-align:center;padding:40px 24px;color:#999">
            <div style="font-size:2.5rem;margin-bottom:12px">🗺</div>
            <div style="font-weight:600;color:#555;margin-bottom:8px">Vyber bod na mapě</div>
            <div style="font-size:0.85rem">Klikni kamkoliv na mapu<br>nebo zapni ruční zadání v levém panelu</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="edu-box"><strong>🗃 Data</strong>
        Kombinujeme reálné nálezy z GBIF s půdními daty SoilGrids, klimatem WorldClim a nadmořskou výškou.</div>
        <div class="edu-box"><strong>🤖 AI model</strong>
        Random Forest natrénovaný na přítomnostních záznamech vs. pseudo-absencích. Výstup: pravděpodobnost 0–100 %.</div>
        <div class="edu-box"><strong>🍄 Ekologie</strong>
        Psilocybe semilanceata preferuje vlhké louky, 6–14 °C, pH 4.5–6.5, výška 100–800 m n. m.</div>
        """, unsafe_allow_html=True)

# ── Spodní sekce ─────────────────────────────────────────────────────────────
st.markdown('<hr class="sdm-divider">', unsafe_allow_html=True)
with st.expander("📊 Statistiky GBIF výskytových dat", expanded=False):
    try:
        gbif_all = fetch_gbif_occurrences()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Celkem výskytů", len(gbif_all))
        c2.metric("Druhů", gbif_all["species"].nunique() if "species" in gbif_all.columns else "–")
        c3.metric("Zemí", gbif_all["country"].nunique() if "country" in gbif_all.columns else "–")
        c4.metric("Rozsah let",
                  f"{int(gbif_all['year'].min())}–{int(gbif_all['year'].max())}"
                  if "year" in gbif_all.columns and not gbif_all["year"].isna().all() else "–")
        if "year" in gbif_all.columns:
            yr = gbif_all.dropna(subset=["year"])
            yr["year"] = yr["year"].astype(int)
            yr_counts = yr["year"].value_counts().sort_index()
            fig_yr = px.bar(x=yr_counts.index, y=yr_counts.values,
                            labels={"x": "Rok", "y": "Počet nálezů"},
                            color_discrete_sequence=["#2e7d32"])
            fig_yr.update_layout(height=200, margin=dict(t=10, b=0, l=0, r=0),
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 showlegend=False, xaxis={"showgrid": False},
                                 yaxis={"showgrid": True, "gridcolor": "#eee"})
            st.plotly_chart(fig_yr, use_container_width=True)
    except Exception as e:
        st.warning(f"Data nejsou dostupná: {e}")

st.caption("PsiloSDM v2 · data: GBIF · SoilGrids · WorldClim · Open-Elevation · model: Random Forest")

"""
Psy Space v4 — pokročilá vizualizace mapy s pravděpodobnostní vrstvou
Nové: barevná heatmapa regionu + dynamické polygony + vylepšený popup
"""

import streamlit as st
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
from datetime import date, timedelta

from data_sources import fetch_all

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURACE
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Psy Space",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #f9fbe7; color: #1a2e1a;
}
.block-container { padding-top: 1.2rem !important; max-width: 1280px; }
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #f1f8e9 0%, #e8f5e9 100%);
    border-right: 1px solid #0d3020;
}
[data-testid="stSidebar"] * { color: #2e4a2e !important; }
[data-testid="stSidebar"] h1 {
    color: #1b5e20 !important; font-size: 1.35rem !important;
    letter-spacing: 0.1em;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #e65100 !important; font-size: 0.78rem !important;
    text-transform: uppercase; letter-spacing: 0.14em;
}
[data-testid="stSidebar"] hr { border-color: #c8e6c9 !important; }
.psy-logo {
    font-size: 2.8rem; font-weight: 900; letter-spacing: 0.1em;
    background: linear-gradient(90deg, #2e7d32 0%, #558b2f 50%, #e65100 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.05;
}
.psy-species {
    font-size: 0.82rem; color: #2d6040; letter-spacing: 0.2em;
    text-transform: uppercase; margin-bottom: 1rem;
}
.prob-widget {
    background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e9 100%);
    border: 1px solid #a5d6a7; border-radius: 18px;
    padding: 28px 24px 22px; text-align: center;
    position: relative; overflow: hidden;
}
.prob-widget::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #2e7d32, transparent);
}
.prob-label {
    font-size: 0.72rem; color: #4a6741;
    text-transform: uppercase; letter-spacing: 0.18em; margin-bottom: 10px;
}
.prob-value { font-size: 5rem; font-weight: 900; line-height: 1; }
.prob-value.high   { color: #1b5e20; }
.prob-value.medium { color: #e65100; }
.prob-value.low    { color: #bf360c; }
.prob-value.none   { color: #9e9e9e; }
.prob-pct { font-size: 2rem; font-weight: 400; opacity: 0.7; }
.prob-status {
    display: inline-block; margin-top: 10px; padding: 4px 16px;
    border-radius: 20px; font-size: 0.8rem; font-weight: 700;
}
.prob-status.high   { background:#e8f5e9;color:#1b5e20;border:1px solid #a5d6a7; }
.prob-status.medium { background:#fff3e0;color:#e65100;border:1px solid #ffcc80; }
.prob-status.low    { background:#fbe9e7;color:#bf360c;border:1px solid #ffab91; }
.prob-status.none   { background:#f5f5f5;color:#757575;border:1px solid #e0e0e0; }
.prob-coords { margin-top:10px;font-size:0.75rem;color:#5a7a5a;font-family:monospace; }
.sec-widget {
    background:#ffffff;border:1px solid #c8e6c9;border-radius:14px;
    padding:16px 14px 12px;text-align:center;margin-bottom:10px;
    box-shadow:0 1px 4px rgba(0,0,0,0.06);
}
.sec-label { font-size:0.68rem;color:#5a7a5a;text-transform:uppercase;letter-spacing:0.15em;margin-bottom:6px; }
.sec-value { font-size:1.8rem;font-weight:800;color:#2e7d32;line-height:1; }
.sec-value.orange { color:#e65100; }
.sec-sub { font-size:0.72rem;color:#6a8a6a;margin-top:4px; }
.dc {
    background:#ffffff;border:1px solid #dcedc8;border-radius:9px;
    padding:9px 10px;text-align:center;
}
.dc-l { font-size:0.64rem;color:#6a8a6a;text-transform:uppercase;letter-spacing:0.08em; }
.dc-v { font-size:1rem;font-weight:700;color:#2e7d32;margin-top:2px; }
.sec-header {
    font-size:0.72rem;color:#e65100;text-transform:uppercase;
    letter-spacing:0.2em;border-bottom:1px solid #c8e6c9;
    padding-bottom:5px;margin:18px 0 10px;
}
.map-tip {
    background:#f1f8e9;border:1px solid #c8e6c9;border-radius:10px;
    padding:10px 14px;font-size:0.78rem;color:#4a6741;
    margin-bottom:10px;line-height:1.5;
}
.map-tip span { color:#2e7d32; font-weight:600; }
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:#f1f8e9; }
::-webkit-scrollbar-thumb { background:#a5d6a7;border-radius:2px; }

/* Skryj collapse/expand tlačítko sidebaru */
[data-testid="collapsedControl"] { display: none !important; }
button[kind="header"] { display: none !important; }
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}

/* Logo vylepšení */
.psy-logo-wrap {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 8px 0 4px;
    border-bottom: 2px solid #c8e6c9;
    margin-bottom: 6px;
}
.psy-logo-icon {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    border-radius: 14px;
    border: 1.5px solid #a5d6a7;
    display: flex; align-items: center; justify-content: center;
    font-size: 28px;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(46,125,50,0.15);
}
.psy-logo-text { line-height: 1.1; }
.psy-logo-name {
    font-size: 1.6rem; font-weight: 900;
    letter-spacing: 0.06em;
    background: linear-gradient(90deg, #1b5e20, #33691e, #e65100);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.psy-logo-sub {
    font-size: 0.7rem; color: #558b2f;
    letter-spacing: 0.14em; text-transform: uppercase;
    margin-top: 1px;
}
.psy-header-bar {
    display: flex; align-items: center; gap: 16px;
    padding: 12px 0 10px;
    border-bottom: 1.5px solid #c8e6c9;
    margin-bottom: 16px;
}
.psy-header-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    border-radius: 12px; border: 1px solid #a5d6a7;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; flex-shrink: 0;
}
.psy-header-title { line-height: 1.15; }
.psy-header-name {
    font-size: 1.9rem; font-weight: 900; letter-spacing: 0.04em;
    background: linear-gradient(90deg, #1b5e20 0%, #33691e 60%, #e65100 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.psy-header-species {
    font-size: 0.78rem; color: #558b2f;
    letter-spacing: 0.16em; text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KONSTANTY
# ══════════════════════════════════════════════════════════════════════════════
SPECIES = {
    "name": "Psilocybe semilanceata", "cz": "Lysohlávka kopinatá",
    "emoji": "🍄", "season": (8, 10),
}

# Regiony s [lat, lon, zoom, bbox]
REGIONS = {
    "Celá ČR + SR":            [49.5, 16.5, 7,  [47.5, 12.0, 51.2, 22.5]],
    "Šumava a Krušné hory":    [49.0, 13.5, 8,  [48.4, 12.0, 50.8, 14.8]],
    "Praha a okolí":           [50.0, 14.5, 9,  [49.8, 13.8, 50.3, 15.2]],
    "Krkonoše a Orlické hory": [50.6, 16.0, 9,  [50.2, 15.2, 51.0, 17.0]],
    "Jeseníky a Beskydy":      [49.8, 17.8, 9,  [49.3, 17.0, 50.2, 18.8]],
    "Bílé Karpaty":            [48.9, 17.5, 9,  [48.6, 17.0, 49.3, 18.5]],
    "Tatry a Orava (SK)":      [49.3, 19.8, 9,  [49.0, 19.0, 49.7, 21.0]],
    "Malé Karpaty (SK)":       [48.5, 17.3, 9,  [48.0, 16.8, 49.0, 18.0]],
}

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for k, v in {
    "clicked_lat":    None,
    "clicked_lon":    None,
    "click_prob":     None,
    "all_data":       None,
    "factor_scores":  None,
    "heatmap_region": None,   # název regionu pro který je heatmapa vygenerována
    "heatmap_points": None,   # seznam [lat, lon, weight] pro HeatMap vrstvu
    "heatmap_month":  None,   # měsíc při generování (sezóna mění výsledky)
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# PREDIKČNÍ MODEL
# ══════════════════════════════════════════════════════════════════════════════

def tri(v, ol, oh, al, ah):
    if v is None or v < al or v > ah: return 0.0
    if ol <= v <= oh: return 1.0
    return (v - al) / (ol - al) if v < ol else (ah - v) / (ah - oh)


def compute_probability(data: dict) -> tuple[float, dict]:
    """
    Predikční model v2 — váhy přepočítány podle ekologie Psilocybe semilanceata.
    Půda 30 % | Počasí 25 % | Land cover 20 % | NDVI 10 % | Klima 10 % | Terén 5 %
    """
    month = date.today().month
    in_season = SPECIES["season"][0] <= month <= SPECIES["season"][1]

    # ── TVRDÁ BLOKACE ────────────────────────────────────────────────────────
    land_cover = data.get("land_cover", "")
    BLOCKED = {"Zastavěná plocha", "Vodní plocha", "Sníh / led",
               "Holá půda", "Mangrovník"}
    empty = {k: 0.0 for k in ["pH půdy","Org. hmota","Textura půdy",
             "Teplota půdy","Vlhkost půdy","Srážky 7d","Land cover",
             "NDVI","Zastínění","Klima (bio01)","Klima (bio12)","Výška","Sklon"]}
    if land_cover in BLOCKED:
        return 0.0, empty
    if land_cover == "Orná půda":
        return min(0.08, data.get("gbif_density", 0) * 2), empty

    # ── PŮDA (30 %) ──────────────────────────────────────────────────────────
    f_ph      = tri(data["ph"], 4.5, 6.5, 3.5, 8.0)
    f_soc     = tri(data["soc"], 15, 60, 3, 100)
    clay      = data.get("clay", 28.0)
    sand      = data.get("sand", 35.0)
    silt      = max(0, 100 - clay - sand)
    f_texture = (tri(clay, 15, 45, 5, 65) * 0.5 +
                 tri(silt, 20, 50, 5, 70) * 0.3 +
                 tri(sand, 10, 40, 3, 70) * 0.2)
    soil_score = 0.12 * f_ph + 0.10 * f_soc + 0.08 * f_texture

    # ── POČASÍ (25 %) ────────────────────────────────────────────────────────
    f_temp_air  = tri(data["temp_now"], 6, 14, -2, 22)
    temp_soil   = data["temp_now"] * 0.85 - 1.5
    f_temp_soil = tri(temp_soil, 5, 12, 0, 18)
    f_soil_moist= tri(data.get("soil_moisture", 0.3), 0.25, 0.55, 0.08, 0.75)
    f_rain7     = tri(data["rain_7d"], 15, 55, 3, 120)
    weather_score = (0.05 * f_temp_air + 0.08 * f_temp_soil +
                     0.07 * f_soil_moist + 0.05 * f_rain7)

    # ── LAND COVER (20 %) ────────────────────────────────────────────────────
    f_landcover   = data.get("land_suitability", 0.5)
    landcover_score = 0.20 * f_landcover

    # ── NDVI + ZASTÍNĚNÍ (10 %) ──────────────────────────────────────────────
    ndvi    = data.get("ndvi", 0.55)
    f_ndvi  = tri(ndvi, 0.40, 0.80, 0.10, 0.95)
    f_canopy= tri(ndvi, 0.45, 0.72, 0.20, 0.90)
    ndvi_score = 0.06 * f_ndvi + 0.04 * f_canopy

    # ── KLIMA (10 %) ─────────────────────────────────────────────────────────
    f_bio01 = tri(data.get("bio01", 9.0), 6, 12, 2, 18)
    f_bio12 = tri(data.get("bio12", 700), 600, 1400, 300, 2200)
    climate_score = 0.05 * f_bio01 + 0.05 * f_bio12

    # ── TERÉN (5 %) ──────────────────────────────────────────────────────────
    f_elev  = tri(data["elev"], 100, 800, 30, 1400)
    slope   = min(data.get("slope", 8.0), 45.0)
    f_slope = tri(slope, 2, 20, 0, 40)
    f_north = (data.get("north_factor", 0.5) + 1) / 2
    terrain_score = 0.02 * f_elev + 0.015 * f_slope + 0.015 * f_north

    base = soil_score + weather_score + landcover_score + ndvi_score + climate_score + terrain_score

    # ── GBIF boost + penalizace ───────────────────────────────────────────────
    gbif_boost = min(0.10, data.get("gbif_count", 0) * 0.005)
    penalty = 1.0
    if data["temp_now"] > 24 or data["temp_now"] < -2: penalty *= 0.25
    if data.get("humidity", 70) < 40:                  penalty *= 0.45
    if data.get("wind", 0) > 15:                       penalty *= 0.80
    if data.get("water_deficit", 0) > 35:              penalty *= 0.65

    season_mult = 1.0 if in_season else 0.25
    prob = float(np.clip((base + gbif_boost) * season_mult * penalty, 0.0, 1.0))

    factors = {
        "pH půdy":       round(f_ph, 2),
        "Org. hmota":    round(f_soc, 2),
        "Textura půdy":  round(f_texture, 2),
        "Teplota půdy":  round(f_temp_soil, 2),
        "Vlhkost půdy":  round(f_soil_moist, 2),
        "Srážky 7d":     round(f_rain7, 2),
        "Land cover":    round(f_landcover, 2),
        "NDVI":          round(f_ndvi, 2),
        "Zastínění":     round(f_canopy, 2),
        "Klima (bio01)": round(f_bio01, 2),
        "Klima (bio12)": round(f_bio12, 2),
        "Výška":         round(f_elev, 2),
        "Sklon":         round(f_slope, 2),
    }
    return prob, factors


def prob_class(p):
    if p >= 0.65: return "high"
    if p >= 0.40: return "medium"
    if p >= 0.15: return "low"
    return "none"

def prob_label(p):
    if p >= 0.65: return "Velmi vhodné"
    if p >= 0.40: return "Podmíněně vhodné"
    if p >= 0.15: return "Málo vhodné"
    return "Nevhodné"


# ══════════════════════════════════════════════════════════════════════════════
# HEATMAPA — rychlý odhad pravděpodobnosti pro mřížku bodů
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gbif_for_region(bbox: list, region_name: str) -> list[dict]:
    """
    Stáhne GBIF záznamy výskytu lysohlávek pro celý region.
    Cachované na 1 hodinu — nepřenačítá se při každém kliknutí.
    Vrátí seznam {lat, lon, year} pro vykreslení na mapě.
    """
    import requests
    try:
        lat_min, lon_min, lat_max, lon_max = bbox
        resp = requests.get(
            "https://api.gbif.org/v1/occurrence/search",
            params={
                "scientificName":   "Psilocybe semilanceata",
                "decimalLatitude":  f"{lat_min},{lat_max}",
                "decimalLongitude": f"{lon_min},{lon_max}",
                "hasCoordinate":    True,
                "hasGeospatialIssue": False,
                "limit":            300,
            },
            timeout=10,
        )
        records = resp.json().get("results", [])
        return [
            {
                "lat":  r["decimalLatitude"],
                "lon":  r["decimalLongitude"],
                "year": r.get("year", ""),
                "species": r.get("species", "Psilocybe semilanceata"),
            }
            for r in records
            if r.get("decimalLatitude") and r.get("decimalLongitude")
        ]
    except Exception:
        return []


def compute_local_density(records: list[dict], lat: float, lon: float,
                          radius_deg: float = 0.5) -> int:
    """Spočítá počet GBIF nálezů v okolí bodu — pro velikost kruhu."""
    return sum(
        1 for r in records
        if abs(r["lat"] - lat) < radius_deg
        and abs(r["lon"] - lon) < radius_deg
    )


def prob_to_color_hex(p: float) -> str:
    """
    Mapuje pravděpodobnost 0–1 na hex barvu pro polygonové vrstvy.
    Barevná škála: tmavě modrá → zelená → žlutá → oranžová → červená
    """
    if p >= 0.75: return "#ff2200"   # velmi vysoká — červená
    if p >= 0.60: return "#ff6600"   # vysoká — oranžová
    if p >= 0.45: return "#ffaa00"   # střední-vysoká — amber
    if p >= 0.30: return "#ccdd00"   # střední — žlutozelená
    if p >= 0.15: return "#44cc44"   # nízká — zelená
    return "#006622"                  # velmi nízká — tmavě zelená


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="psy-logo-wrap">
        <div class="psy-logo-icon">🍄</div>
        <div class="psy-logo-text">
            <div class="psy-logo-name">PSY SPACE</div>
            <div class="psy-logo-sub">predikce výskytu hub</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Sledovaný druh")
    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid #c8e6c9;border-radius:10px;padding:12px 14px">
        <div style="font-size:1.3rem">{SPECIES['emoji']}</div>
        <div style="color:#1b5e20;font-weight:700;font-size:0.92rem">{SPECIES['cz']}</div>
        <div style="color:#5a7a5a;font-size:0.76rem;font-style:italic">{SPECIES['name']}</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    st.markdown("---")
    st.markdown("### Region")
    selected_region = st.selectbox(
        "Oblast", list(REGIONS.keys()), label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### Časový rámec")
    time_mode = st.radio(
        "Období", ["Aktuální týden", "Vlastní rozsah"],
        label_visibility="collapsed"
    )
    if time_mode == "Aktuální týden":
        d_from = date.today() - timedelta(days=date.today().weekday())
        d_to   = d_from + timedelta(days=6)
        st.caption(f"📅 {d_from.strftime('%d. %m.')} — {d_to.strftime('%d. %m. %Y')}")
        selected_month = date.today().month
    else:
        d_from = st.date_input("Od", value=date.today() - timedelta(days=7))
        d_to   = st.date_input("Do", value=date.today())
        selected_month = d_from.month

    st.markdown("---")

    # Ovládání mapy
    st.markdown("### Vizualizace mapy")
    show_clusters = st.checkbox("Zobrazit oblasti výskytu", value=True,
                                help="Zvýrazněné oblasti s historickými nálezy")
    show_gbif = st.checkbox("Zobrazit jednotlivé nálezy", value=True,
                            help="Každý historický GBIF záznam jako bod")

    st.markdown("---")
    st.markdown("### Datové zdroje")
    sources_map = {
        "weather": "Open-Meteo", "terrain": "Open-Elevation",
        "soil": "SoilGrids", "land_cover": "Copernicus",
        "ndvi": "MODIS NDVI", "gbif": "GBIF",
    }
    all_data_sources = (st.session_state.all_data or {}).get("_sources", {})
    for key, name in sources_map.items():
        status = all_data_sources.get(key, "⚪ čeká")
        cls = "src-ok" if "✅" in status else "src-warn"
        icon = status.split()[0] if status else "⚪"
        st.markdown(
            f'<span style="font-size:0.7rem;padding:2px 8px;border-radius:12px;'
            f'margin:2px;display:inline-block;'
            f'background:{"#e8f5e9" if "✅" in status else "#fff8e1"};'
            f'color:{"#2e7d32" if "✅" in status else "#e65100"};'
            f'border:1px solid {"#39ff1430" if "✅" in status else "#ff8c0030"}">'
            f'{icon} {name}</span>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# HLAVNÍ OBSAH
# ══════════════════════════════════════════════════════════════════════════════
region_data   = REGIONS[selected_region]
region_center = region_data[:3]   # [lat, lon, zoom]
region_bbox   = region_data[3]    # [lat_min, lon_min, lat_max, lon_max]

# Header
st.markdown(f"""
<div class="psy-header-bar">
    <div class="psy-header-icon">🍄</div>
    <div class="psy-header-title">
        <div class="psy-header-name">PSY SPACE</div>
        <div class="psy-header-species">
            {SPECIES['name']} &nbsp;·&nbsp; {selected_region}
        </div>
    </div>
    <div style="margin-left:auto;font-size:0.78rem;color:#5a7a5a;
                text-align:right;line-height:1.7">
        👆 Klikni na mapu<br>
        <span style="font-size:0.7rem;color:#8a9a8a">reálná data ze 6 zdrojů</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Widgety ──────────────────────────────────────────────────────────────────
col_prob, col_sec = st.columns([2, 1], gap="large")

with col_prob:
    p   = st.session_state.click_prob or 0.0
    cls = prob_class(p)
    lbl = prob_label(p) if p > 0 else "Klikni na mapu"
    coords = (
        f"{st.session_state.clicked_lat:.4f}°N · "
        f"{st.session_state.clicked_lon:.4f}°E"
        if st.session_state.clicked_lat else "čeká na výběr bodu…"
    )
    pct = int(p * 100)

    st.markdown(f"""
    <div class="prob-widget">
        <div class="prob-label">🍄 Pravděpodobnost výskytu</div>
        <div class="prob-value {cls}">{pct}<span class="prob-pct"> %</span></div>
        <span class="prob-status {cls}">{lbl}</span>
        <div class="prob-coords">{coords}</div>
    </div>""", unsafe_allow_html=True)

    if st.session_state.all_data:
        d = st.session_state.all_data
        st.markdown("""<div style="margin-top:10px;font-size:0.65rem;color:#2d6040;
        text-transform:uppercase;letter-spacing:0.12em;margin-bottom:6px">
        Vstupní data modelu</div>""", unsafe_allow_html=True)
        r1 = st.columns(4)
        r2 = st.columns(4)
        for col, (lbl2, val) in zip(r1, [
            ("🌡 Teplota", f"{d['temp_now']} °C"),
            ("💧 Vlhkost", f"{d['humidity']} %"),
            ("🌧 Srážky 7d", f"{d['rain_7d']} mm"),
            ("🪨 pH půdy", f"{d['ph']}"),
        ]):
            with col:
                st.markdown(
                    f'<div class="dc"><div class="dc-l">{lbl2}</div>'
                    f'<div class="dc-v">{val}</div></div>',
                    unsafe_allow_html=True,
                )
        for col, (lbl2, val) in zip(r2, [
            ("⛰ Výška", f"{int(d['elev'])} m"),
            ("📐 Sklon", f"{d.get('slope','–')}°"),
            ("🌿 NDVI", f"{d.get('ndvi','–')}"),
            ("🌍 Land cover", f"{d.get('land_cover','–')[:12]}"),
        ]):
            with col:
                st.markdown(
                    f'<div class="dc"><div class="dc-l">{lbl2}</div>'
                    f'<div class="dc-v">{val}</div></div>',
                    unsafe_allow_html=True,
                )

with col_sec:
    d          = st.session_state.all_data or {}
    gbif_count = d.get("gbif_count", 0)
    month_now  = date.today().month
    in_season  = SPECIES["season"][0] <= month_now <= SPECIES["season"][1]

    st.markdown(f"""
    <div class="sec-widget">
        <div class="sec-label">📍 GBIF nálezy v okolí</div>
        <div class="sec-value">{gbif_count}</div>
        <div class="sec-sub">záznamy do 30 km</div>
    </div>
    <div class="sec-widget">
        <div class="sec-label">📅 Hlavní sezóna</div>
        <div class="sec-value {'orange' if not in_season else ''}">
            {'✅ Probíhá' if in_season else '⏳ Mimo sezónu'}</div>
        <div class="sec-sub">optimum: srpen–říjen</div>
    </div>
    <div class="sec-widget">
        <div class="sec-label">🌍 Land cover</div>
        <div class="sec-value" style="font-size:0.85rem;color:#7dcc8a">
            {d.get('land_cover','–')}</div>
        <div class="sec-sub">vhodnost: {int(d.get('land_suitability',0)*100)} %</div>
    </div>""", unsafe_allow_html=True)

# ── Faktory modelu ────────────────────────────────────────────────────────────
if st.session_state.factor_scores:
    st.markdown(
        '<div class="sec-header">Váha faktorů v modelu</div>',
        unsafe_allow_html=True,
    )
    factors = st.session_state.factor_scores
    cols = st.columns(len(factors))
    for col, (name, score) in zip(cols, factors.items()):
        pct_f = int(score * 100)
        color = "#2e7d32" if score > 0.6 else "#f57f17" if score > 0.3 else "#c62828"
        with col:
            st.markdown(f"""
            <div class="dc" style="padding:8px 6px">
                <div class="dc-l">{name}</div>
                <div class="dc-v" style="color:{color};font-size:0.9rem">{pct_f} %</div>
                <div style="height:4px;border-radius:2px;margin-top:5px;
                    background:linear-gradient(90deg,{color} {pct_f}%,
                    #0d3020 {pct_f}%)"></div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INTERAKTIVNÍ MAPA S VRSTVOU PRAVDĚPODOBNOSTI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="sec-header">Interaktivní mapa — vrstva pravděpodobnosti výskytu</div>',
    unsafe_allow_html=True,
)
st.markdown("""
<div class="map-tip">
    <span>👆 Klikni na mapu</span> pro přesnou analýzu bodu (reálná data ze 6 API).
    Barevná vrstva zobrazuje <em>odhadnutou</em> vhodnost prostředí pro celý region
    — červená = nejvyšší pravděpodobnost, tmavě zelená = nejnižší.
</div>""", unsafe_allow_html=True)

# ── Načti GBIF záznamy pro region (cachované) ────────────────────────────────
# Cache se invaliduje při změně regionu — nepřenačítá se při kliknutí na mapu
gbif_records = fetch_gbif_for_region(
    bbox=region_bbox,
    region_name=selected_region,
)

# ── Sestav folium mapu ────────────────────────────────────────────────────────
m = folium.Map(
    location=[region_center[0], region_center[1]],
    zoom_start=region_center[2],
    tiles="CartoDB positron",      # světlý podklad — lépe viditelné vrstvy
    prefer_canvas=True,
)

# ── VRSTVA 1: Oblasti s vyšší pravděpodobností ───────────────────────────────
# Průhledné kruhy kolem míst s více nálezy = zvýrazněné oblasti výskytu.
# Velikost kruhu odpovídá hustotě historických nálezů v okolí.
if show_clusters and gbif_records:
    cluster_layer = folium.FeatureGroup(name="🟢 Oblasti výskytu", show=True)
    for rec in gbif_records:
        lat_g = rec["lat"]
        lon_g = rec["lon"]
        # Počet nálezů v okolí 0.5° → určuje velikost a opacity kruhu
        density = compute_local_density(gbif_records, lat_g, lon_g, 0.4)
        # Větší hustota = větší a výraznější kruh
        radius_m  = min(12000, 3000 + density * 1200)
        opacity   = min(0.45, 0.10 + density * 0.04)
        folium.Circle(
            location=[lat_g, lon_g],
            radius=radius_m,
            color="#1b5e20",
            fill=True,
            fill_color="#388e3c",
            fill_opacity=opacity,
            weight=0.5,
        ).add_to(cluster_layer)
    cluster_layer.add_to(m)

# ── VRSTVA 2: Jednotlivé GBIF nálezy ─────────────────────────────────────────
# Každý historický záznam jako malý zelený bod s tooltipem.
if show_gbif and gbif_records:
    gbif_layer = folium.FeatureGroup(name="📍 GBIF nálezy", show=True)
    for rec in gbif_records:
        folium.CircleMarker(
            location=[rec["lat"], rec["lon"]],
            radius=4,
            color="#ffffff",
            weight=1,
            fill=True,
            fill_color="#39ff14",
            fill_opacity=0.90,
            tooltip=f"🍄 {rec.get('species','Psilocybe')} · {rec.get('year','')}",
        ).add_to(gbif_layer)
    gbif_layer.add_to(m)

# ── VRSTVA 3: Marker kliknutého bodu s popup ─────────────────────────────────
# Po kliknutí se zobrazí přesná pravděpodobnost + všechna data v popup okně
if st.session_state.clicked_lat and st.session_state.click_prob is not None:
    p   = st.session_state.click_prob
    cls = prob_class(p)
    d   = st.session_state.all_data or {}

    hex_map = {
        "high":   "#1b5e20",
        "medium": "#f57f17",
        "low":    "#bf360c",
        "none":   "#616161",
    }
    col_map = {"high": "green", "medium": "orange", "low": "red", "none": "gray"}
    hx = hex_map[cls]

    # ── Popup HTML s kompletními daty bodu ───────────────────────────────────
    popup_html = f"""
    <div style="font-family:'Courier New',monospace;min-width:230px;
                max-width:260px;padding:8px;background:#fff;color:#1a1a1a;
                border-radius:8px;border:1px solid #e0e0e0">
        <div style="font-weight:700;font-size:1.3rem;color:{hx};
                    margin-bottom:6px;text-align:center">
            {int(p*100)} %
        </div>
        <div style="font-size:0.8rem;color:{hx};text-align:center;
                    margin-bottom:8px;letter-spacing:0.1em">
            {prob_label(p).upper()}
        </div>
        <hr style="border-color:#333;margin:6px 0">
        <table style="width:100%;font-size:0.75rem;border-collapse:collapse">
            <tr><td style="color:#555;padding:2px 0">🌡 Teplota</td>
                <td style="color:#1a1a1a;text-align:right">
                    <b>{d.get('temp_now','–')} °C</b></td></tr>
            <tr><td style="color:#666">💧 Vlhkost</td>
                <td style="color:#1a1a1a;text-align:right">
                    <b>{d.get('humidity','–')} %</b></td></tr>
            <tr><td style="color:#666">🌧 Srážky 7d</td>
                <td style="color:#1a1a1a;text-align:right">
                    <b>{d.get('rain_7d','–')} mm</b></td></tr>
            <tr><td style="color:#666">🪨 pH půdy</td>
                <td style="color:#1a1a1a;text-align:right">
                    <b>{d.get('ph','–')}</b></td></tr>
            <tr><td style="color:#666">🌱 Org. hmota</td>
                <td style="color:#1a1a1a;text-align:right">
                    <b>{d.get('soc','–')} g/kg</b></td></tr>
            <tr><td style="color:#666">⛰ Výška</td>
                <td style="color:#1a1a1a;text-align:right">
                    <b>{int(d.get('elev',0))} m n.m.</b></td></tr>
            <tr><td style="color:#666">📐 Sklon</td>
                <td style="color:#1a1a1a;text-align:right">
                    <b>{d.get('slope','–')}°</b></td></tr>
            <tr><td style="color:#666">🌿 NDVI</td>
                <td style="color:#1a1a1a;text-align:right">
                    <b>{d.get('ndvi','–')}</b></td></tr>
            <tr><td style="color:#666">🌍 Prostředí</td>
                <td style="color:#1a1a1a;text-align:right">
                    <b>{d.get('land_cover','–')}</b></td></tr>
            <tr><td style="color:#666">📍 GBIF</td>
                <td style="color:#1a1a1a;text-align:right">
                    <b>{d.get('gbif_count',0)} nálezů</b></td></tr>
        </table>
        <hr style="border-color:#333;margin:6px 0">
        <div style="font-size:0.68rem;color:#888;text-align:center">
            {st.session_state.clicked_lat:.5f}°N ·
            {st.session_state.clicked_lon:.5f}°E
        </div>
    </div>"""

    # Marker s ikonou
    folium.Marker(
        location=[st.session_state.clicked_lat, st.session_state.clicked_lon],
        icon=folium.Icon(color=col_map[cls], icon="leaf", prefix="fa"),
        popup=folium.Popup(popup_html, max_width=270),
        tooltip=f"🍄 {int(p*100)} % · {prob_label(p)} · klikni pro detail",
    ).add_to(m)

    # Dynamický kruh — velikost odpovídá pravděpodobnosti
    # 0 % → 200 m, 100 % → 1500 m
    circle_radius = int(200 + p * 1300)
    folium.Circle(
        location=[st.session_state.clicked_lat, st.session_state.clicked_lon],
        radius=circle_radius,
        color=hx,
        fill=True,
        fill_opacity=round(0.12 + p * 0.28, 2),
        weight=2,
        dash_array="6 4",   # přerušovaná čára pro elegantní vzhled
    ).add_to(m)

# ── Legenda mapy ──────────────────────────────────────────────────────────────
legend_html = """
<div style="position:fixed;bottom:24px;left:16px;
            background:rgba(255,255,255,0.95);padding:12px 16px;
            border-radius:12px;border:1px solid #c8e6c9;
            font-size:12px;z-index:1000;
            box-shadow:0 2px 10px rgba(0,0,0,0.12)">
    <div style="font-weight:700;margin-bottom:8px;color:#1b5e20;
                letter-spacing:0.05em;font-size:11px">
        🍄 HISTORICKÉ NÁLEZY
    </div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
        <div style="width:18px;height:18px;border-radius:50%;
                    background:#38823840;border:1px solid #388e3c;
                    flex-shrink:0"></div>
        <span style="color:#333">Oblast s více nálezy</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
        <div style="width:10px;height:10px;border-radius:50%;
                    background:#2e7d32;flex-shrink:0"></div>
        <span style="color:#333">Konkrétní GBIF nález</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px">
        <div style="width:10px;height:10px;border-radius:50%;
                    background:#e65100;flex-shrink:0"></div>
        <span style="color:#333">Tvůj vybraný bod</span>
    </div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Ovládání vrstev
folium.LayerControl(collapsed=False).add_to(m)

# ── Zobraz mapu a zachyť kliknutí ────────────────────────────────────────────
# st_folium vrátí last_clicked když uživatel klikne na mapu
# returned_objects říká které události chceme zachytit
map_data = st_folium(
    m,
    width="100%",
    height=540,
    key=f"psy_map_{selected_region}",  # reset mapy při změně regionu
    returned_objects=["last_clicked"],
)

# ── Zpracování kliknutí → stáhni reálná data → spusť model ──────────────────
# Kliknutí na mapu vrátí souřadnice v map_data["last_clicked"]
# Pro každý nový bod stáhneme data ze všech 6 API a spustíme compute_probability
if map_data:
    raw = map_data.get("last_clicked")
    if raw and isinstance(raw, dict):
        lat = raw.get("lat") or raw.get("latitude")
        lng = raw.get("lng") or raw.get("lon") or raw.get("longitude")
        if lat is not None and lng is not None:
            lat, lng = float(lat), float(lng)
            # Přepočítej pouze pokud uživatel klikl na jiný bod
            if (lat != st.session_state.clicked_lat or
                    lng != st.session_state.clicked_lon):
                with st.spinner("🌍 Stahuji reálná data ze 6 zdrojů…"):
                    all_data = fetch_all(lat, lng)
                prob, factors = compute_probability(all_data)
                # Ulož do session state → triggere rerun → mapa se překreslí
                st.session_state.clicked_lat   = lat
                st.session_state.clicked_lon   = lng
                st.session_state.click_prob    = prob
                st.session_state.all_data      = all_data
                st.session_state.factor_scores = factors
                st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:0.72rem;color:#6a8a6a;letter-spacing:0.1em">
    PSY SPACE v4 · Open-Meteo · SoilGrids · OpenTopography ·
    Copernicus ESA WorldCover · NASA MODIS · GBIF
</div>""", unsafe_allow_html=True)

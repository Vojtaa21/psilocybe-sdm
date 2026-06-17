"""
Psy Space v3 — maximální přesnost predikce výskytu Psilocybe semilanceata
Kombinuje: Open-Meteo + SoilGrids + Open-Elevation + Copernicus + MODIS + GBIF
"""

import streamlit as st
import numpy as np
import folium
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
    background-color: #03080f; color: #d0f0d0;
}
.block-container { padding-top: 1.2rem !important; max-width: 1280px; }
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #03080f 0%, #071a12 100%);
    border-right: 1px solid #0d3020;
}
[data-testid="stSidebar"] * { color: #7dcc8a !important; }
[data-testid="stSidebar"] h1 {
    color: #39ff14 !important; font-size: 1.35rem !important;
    letter-spacing: 0.1em; text-shadow: 0 0 16px #39ff1460;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #ff8c00 !important; font-size: 0.78rem !important;
    text-transform: uppercase; letter-spacing: 0.14em;
}
[data-testid="stSidebar"] hr { border-color: #0d3020 !important; }
.psy-logo {
    font-size: 2.8rem; font-weight: 900; letter-spacing: 0.1em;
    background: linear-gradient(90deg, #39ff14 0%, #a8e44a 50%, #ff8c00 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.05;
}
.psy-species {
    font-size: 0.82rem; color: #2d6040; letter-spacing: 0.2em;
    text-transform: uppercase; margin-bottom: 1rem;
}
.prob-widget {
    background: linear-gradient(135deg, #03120a 0%, #071f10 100%);
    border: 1px solid #0d4020; border-radius: 18px;
    padding: 28px 24px 22px; text-align: center;
    position: relative; overflow: hidden;
}
.prob-widget::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #39ff14, transparent);
}
.prob-label {
    font-size: 0.72rem; color: #2d6040;
    text-transform: uppercase; letter-spacing: 0.18em; margin-bottom: 10px;
}
.prob-value { font-size: 5rem; font-weight: 900; line-height: 1; }
.prob-value.high   { color: #39ff14; text-shadow: 0 0 30px #39ff1450; }
.prob-value.medium { color: #ffd700; text-shadow: 0 0 30px #ffd70040; }
.prob-value.low    { color: #ff8c00; text-shadow: 0 0 30px #ff8c0040; }
.prob-value.none   { color: #2d4030; }
.prob-pct { font-size: 2rem; font-weight: 400; opacity: 0.7; }
.prob-status {
    display: inline-block; margin-top: 10px; padding: 4px 16px;
    border-radius: 20px; font-size: 0.8rem; font-weight: 700;
}
.prob-status.high   { background:#0d3a10;color:#39ff14;border:1px solid #39ff1440; }
.prob-status.medium { background:#2a2000;color:#ffd700;border:1px solid #ffd70040; }
.prob-status.low    { background:#2a1000;color:#ff8c00;border:1px solid #ff8c0040; }
.prob-status.none   { background:#0d1a10;color:#2d6040;border:1px solid #0d3020; }
.prob-coords { margin-top:10px;font-size:0.75rem;color:#2d6040;font-family:monospace; }
.sec-widget {
    background:#03120a;border:1px solid #0d3020;border-radius:14px;
    padding:16px 14px 12px;text-align:center;margin-bottom:10px;
}
.sec-label { font-size:0.68rem;color:#2d6040;text-transform:uppercase;letter-spacing:0.15em;margin-bottom:6px; }
.sec-value { font-size:1.8rem;font-weight:800;color:#39ff14;line-height:1; }
.sec-value.orange { color:#ff8c00; }
.sec-sub { font-size:0.72rem;color:#2d6040;margin-top:4px; }
.data-row {
    display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-top:12px;
}
.data-row-3 { grid-template-columns:repeat(3,1fr); }
.dc {
    background:#03120a;border:1px solid #0d3020;border-radius:9px;
    padding:9px 10px;text-align:center;
}
.dc-l { font-size:0.64rem;color:#2d6040;text-transform:uppercase;letter-spacing:0.08em; }
.dc-v { font-size:1rem;font-weight:700;color:#7dcc8a;margin-top:2px; }
.dc-v.g { color:#39ff14; }
.dc-v.o { color:#ff8c00; }
.sec-header {
    font-size:0.72rem;color:#ff8c00;text-transform:uppercase;
    letter-spacing:0.2em;border-bottom:1px solid #0d3020;
    padding-bottom:5px;margin:18px 0 10px;
}
.map-tip {
    background:#03120a;border:1px solid #0d3020;border-radius:10px;
    padding:10px 14px;font-size:0.78rem;color:#2d6040;
    margin-bottom:10px;line-height:1.5;
}
.map-tip span { color:#39ff14; }
.src-row { display:flex;flex-wrap:wrap;gap:6px;margin:8px 0; }
.src-badge {
    font-size:0.66rem;padding:3px 9px;border-radius:20px;
    font-weight:600;letter-spacing:0.06em;text-transform:uppercase;
}
.src-ok   { background:#0d3010;color:#39ff14;border:1px solid #39ff1430; }
.src-warn { background:#1a1000;color:#ff8c00;border:1px solid #ff8c0030; }
.factor-bar {
    height:5px;border-radius:3px;margin-top:4px;
    background:linear-gradient(90deg, #39ff14, #0d3020);
}
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:#03080f; }
::-webkit-scrollbar-thumb { background:#0d3020;border-radius:2px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KONSTANTY
# ══════════════════════════════════════════════════════════════════════════════
SPECIES = {
    "name": "Psilocybe semilanceata", "cz": "Lysohlávka kopinatá",
    "emoji": "🍄", "season": (8, 10),
}
REGIONS = {
    "Celá ČR + SR":            [49.5, 16.5, 7],
    "Šumava a Krušné hory":    [49.0, 13.5, 8],
    "Praha a okolí":           [50.0, 14.5, 9],
    "Krkonoše a Orlické hory": [50.6, 16.0, 9],
    "Jeseníky a Beskydy":      [49.8, 17.8, 9],
    "Bílé Karpaty":            [48.9, 17.5, 9],
    "Tatry a Orava (SK)":      [49.3, 19.8, 9],
    "Malé Karpaty (SK)":       [48.5, 17.3, 9],
}

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for k, v in {
    "clicked_lat": None, "clicked_lon": None,
    "click_prob": None, "all_data": None,
    "factor_scores": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# PREDIKČNÍ MODEL — kombinuje všechna data
# ══════════════════════════════════════════════════════════════════════════════

def tri(v, ol, oh, al, ah):
    """Trojúhelníková skórovací funkce — 1.0 v optimu, 0 mimo rozsah."""
    if v is None or v < al or v > ah: return 0.0
    if ol <= v <= oh: return 1.0
    return (v - al) / (ol - al) if v < ol else (ah - v) / (ah - oh)


def compute_probability(data: dict) -> tuple[float, dict]:
    """
    Predikční model v2 — váhy přepočítány podle ekologie Psilocybe semilanceata.

    Nové váhování (inspirováno Stamets 1996 + GBIF SDM literatura):
      Půdní vlastnosti      30 % — pH a textura určují zda mycelium vůbec může růst
      Aktuální počasí       25 % — srážky + vlhkost = spouštěč plodnic
      Typ prostředí         20 % — louka vs. město vs. les (tvrdá blokace)
      Klimatická normála    10 % — dlouhodobý bioklimatický rámec
      NDVI / vegetace       10 % — zdraví a hustota vegetace
      Výška / sklon          5 % — modifikátor vlhkosti a teploty

    Nové faktory:
      - Vlhkost půdy z Open-Meteo (soil_moisture) — reálná, ne jen srážky
      - Teplota půdy odhadnutá z teploty vzduchu s korekcí (-2°C pod povrchem)
      - Canopy cover aproximovaný z NDVI (vysoké NDVI = hustý porost = stín)
    """
    month = date.today().month
    in_season = SPECIES["season"][0] <= month <= SPECIES["season"][1]

    # ── TVRDÁ BLOKACE — nevhodné land cover ─────────────────────────────────
    land_cover = data.get("land_cover", "")
    BLOCKED = {"Zastavěná plocha", "Vodní plocha", "Sníh / led",
               "Holá půda", "Mangrovník"}
    empty_factors = {k: 0.0 for k in [
        "Teplota","Teplota půdy","Vlhkost vzduchu","Vlhkost půdy",
        "Srážky 7d","Srážky 3d","pH půdy","Org. hmota",
        "Textura půdy","Land cover","NDVI","Zastínění",
        "Výška","Sklon/orientace","Klima","Sezóna",
    ]}
    if land_cover in BLOCKED:
        return 0.0, empty_factors
    if land_cover == "Orná půda":
        # Okraje polí — max 8 %
        return min(0.08, data.get("gbif_density", 0) * 2), empty_factors

    # ════════════════════════════════════════════════════════════════
    # SKUPINA A: PŮDNÍ VLASTNOSTI (30 %)
    # ════════════════════════════════════════════════════════════════

    # A1: pH půdy (12 %) — nejdůležitější půdní faktor
    # Optimum 4.5–6.5, absolutní hranice 3.5–8.0
    f_ph = tri(data["ph"], 4.5, 6.5, 3.5, 8.0)

    # A2: Organická hmota SOC (10 %) — mycelium potřebuje rozkládající se materiál
    f_soc = tri(data["soc"], 15, 60, 3, 100)

    # A3: Textura půdy (8 %) — jíl 15-45 % = dobrá retence vlhkosti
    # kombinace clay + sand pro optimální texturu
    clay = data.get("clay", 28.0)
    sand = data.get("sand", 35.0)
    silt = max(0, 100 - clay - sand)  # prach = zbytek
    f_texture = (
        tri(clay, 15, 45, 5, 65) * 0.5 +
        tri(silt, 20, 50, 5, 70) * 0.3 +
        tri(sand, 10, 40, 3, 70) * 0.2
    )

    soil_score = 0.12 * f_ph + 0.10 * f_soc + 0.08 * f_texture

    # ════════════════════════════════════════════════════════════════
    # SKUPINA B: AKTUÁLNÍ POČASÍ (25 %)
    # ════════════════════════════════════════════════════════════════

    # B1: Teplota vzduchu (5 %) — méně důležitá než teplota půdy
    f_temp_air = tri(data["temp_now"], 6, 14, -2, 22)

    # B2: Teplota půdy — odhad z teploty vzduchu (8 %)
    # Teplota půdy bývá o 2–4°C nižší než vzduch, s menší amplitudou
    temp_soil = data["temp_now"] * 0.85 - 1.5
    f_temp_soil = tri(temp_soil, 5, 12, 0, 18)

    # B3: Reálná vlhkost půdy z Open-Meteo (7 %)
    # soil_moisture_0_to_1cm je v m³/m³ (0–1)
    soil_moist = data.get("soil_moisture", 0.3)
    f_soil_moist = tri(soil_moist, 0.25, 0.55, 0.08, 0.75)

    # B4: Srážky 7 dní zpět (5 %) — spouštěč plodnic
    f_rain7 = tri(data["rain_7d"], 15, 55, 3, 120)

    weather_score = (
        0.05 * f_temp_air +
        0.08 * f_temp_soil +
        0.07 * f_soil_moist +
        0.05 * f_rain7
    )

    # ════════════════════════════════════════════════════════════════
    # SKUPINA C: TYP PROSTŘEDÍ — ESA WorldCover (20 %)
    # ════════════════════════════════════════════════════════════════
    f_landcover = data.get("land_suitability", 0.5)
    landcover_score = 0.20 * f_landcover

    # ════════════════════════════════════════════════════════════════
    # SKUPINA D: NDVI + ZASTÍNĚNÍ (10 %)
    # ════════════════════════════════════════════════════════════════

    ndvi = data.get("ndvi", 0.55)

    # D1: NDVI (6 %) — zdraví vegetace
    f_ndvi = tri(ndvi, 0.40, 0.80, 0.10, 0.95)

    # D2: Zastínění / canopy cover aproximace (4 %)
    # Vysoké NDVI (>0.7) = hustý porost = udržuje vlhkost
    # Optimum pro lysohlávky: středně hustá vegetace (louky se stromy)
    f_canopy = tri(ndvi, 0.45, 0.72, 0.20, 0.90)

    ndvi_score = 0.06 * f_ndvi + 0.04 * f_canopy

    # ════════════════════════════════════════════════════════════════
    # SKUPINA E: KLIMATICKÁ NORMÁLA WorldClim (10 %)
    # ════════════════════════════════════════════════════════════════

    # E1: Roční průměrná teplota (5 %)
    f_bio01 = tri(data.get("bio01", 9.0), 6, 12, 2, 18)

    # E2: Roční srážky (5 %)
    f_bio12 = tri(data.get("bio12", 700), 600, 1400, 300, 2200)

    climate_score = 0.05 * f_bio01 + 0.05 * f_bio12

    # ════════════════════════════════════════════════════════════════
    # SKUPINA F: VÝŠKA A TERÉN (5 %)
    # ════════════════════════════════════════════════════════════════

    # F1: Nadmořská výška (2 %)
    f_elev = tri(data["elev"], 100, 800, 30, 1400)

    # F2: Sklon svahu — oprava nereálných hodnot (1.5 %)
    slope = data.get("slope", 8.0)
    if slope > 45:
        slope = 8.0  # artefakt parsování — použij průměr
    f_slope = tri(slope, 2, 20, 0, 40)

    # F3: Severní orientace — vlhčí mikroklima (1.5 %)
    north = data.get("north_factor", 0.5)
    f_north = (north + 1) / 2

    terrain_score = 0.02 * f_elev + 0.015 * f_slope + 0.015 * f_north

    # ════════════════════════════════════════════════════════════════
    # SEZÓNA — multiplikátor (ne additivní váha)
    # ════════════════════════════════════════════════════════════════
    # Mimo sezónu se pravděpodobnost snižuje na 15 %, ne na 0
    season_mult = 1.0 if in_season else 0.15

    # ════════════════════════════════════════════════════════════════
    # CELKOVÉ SKÓRE
    # ════════════════════════════════════════════════════════════════
    base = (
        soil_score      +   # 30 %
        weather_score   +   # 25 %
        landcover_score +   # 20 %
        ndvi_score      +   # 10 %
        climate_score   +   # 10 %
        terrain_score       #  5 %
    )                       # = 100 %

    # GBIF boost — historické nálezy v okolí
    gbif_boost = min(0.10, data.get("gbif_count", 0) * 0.005)

    # Penalizace za extrémní podmínky
    penalty = 1.0
    if data["temp_now"] > 24 or data["temp_now"] < -2: penalty *= 0.25
    if data.get("humidity", 70) < 40:                  penalty *= 0.45
    if data.get("wind", 0) > 15:                       penalty *= 0.80
    if data.get("water_deficit", 0) > 35:              penalty *= 0.65

    prob = float(np.clip((base + gbif_boost) * season_mult * penalty, 0.0, 1.0))

    factors = {
        "pH půdy":        round(f_ph, 2),
        "Org. hmota":     round(f_soc, 2),
        "Textura půdy":   round(f_texture, 2),
        "Teplota půdy":   round(f_temp_soil, 2),
        "Vlhkost půdy":   round(f_soil_moist, 2),
        "Srážky 7d":      round(f_rain7, 2),
        "Land cover":     round(f_landcover, 2),
        "NDVI":           round(f_ndvi, 2),
        "Zastínění":      round(f_canopy, 2),
        "Klima (bio01)":  round(f_bio01, 2),
        "Klima (bio12)":  round(f_bio12, 2),
        "Výška":          round(f_elev, 2),
        "Sklon":          round(f_slope, 2),
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
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🍄 Psy Space")
    st.markdown("*predikce výskytu lysohlávek*")
    st.markdown("---")
    st.markdown("### Sledovaný druh")
    st.markdown(f"""
    <div style="background:#03120a;border:1px solid #0d3020;border-radius:10px;padding:12px 14px">
        <div style="font-size:1.3rem">{SPECIES['emoji']}</div>
        <div style="color:#39ff14;font-weight:700;font-size:0.92rem">{SPECIES['cz']}</div>
        <div style="color:#2d6040;font-size:0.76rem;font-style:italic">{SPECIES['name']}</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("---")
    st.markdown("### Region")
    selected_region = st.selectbox("Oblast", list(REGIONS.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Časový rámec")
    time_mode = st.radio("Období", ["Aktuální týden", "Vlastní rozsah"], label_visibility="collapsed")
    if time_mode == "Aktuální týden":
        d_from = date.today() - timedelta(days=date.today().weekday())
        d_to   = d_from + timedelta(days=6)
        st.caption(f"📅 {d_from.strftime('%d. %m.')} — {d_to.strftime('%d. %m. %Y')}")
    else:
        d_from = st.date_input("Od", value=date.today() - timedelta(days=7))
        d_to   = st.date_input("Do", value=date.today())
    st.markdown("---")
    st.markdown("### Datové zdroje")
    sources = (st.session_state.all_data or {}).get("_sources", {})
    src_list = [
        ("weather",    "Open-Meteo"),
        ("terrain",    "Open-Elevation"),
        ("soil",       "SoilGrids"),
        ("land_cover", "Copernicus"),
        ("ndvi",       "MODIS NDVI"),
        ("gbif",       "GBIF"),
    ]
    for key, name in src_list:
        status = sources.get(key, "⚪ čeká")
        cls = "src-ok" if "✅" in status else "src-warn" if "⚠" in status else "src-warn"
        icon = status.split()[0] if status else "⚪"
        st.markdown(
            f'<span class="{cls} src-badge">{icon} {name}</span>',
            unsafe_allow_html=True,
        )
    st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# HLAVNÍ OBSAH
# ══════════════════════════════════════════════════════════════════════════════
region_center = REGIONS[selected_region]

col_title, col_hint = st.columns([2, 1])
with col_title:
    st.markdown('<div class="psy-logo">PSY SPACE</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="psy-species">{SPECIES["emoji"]} {SPECIES["name"]} · {selected_region}</div>',
        unsafe_allow_html=True,
    )
with col_hint:
    st.markdown("""
    <div style="padding-top:16px;font-size:0.78rem;color:#2d6040;text-align:right;line-height:1.8">
        👆 Klikni na mapu pro analýzu<br>
        <span style="font-size:0.7rem">6 reálných datových zdrojů</span>
    </div>""", unsafe_allow_html=True)

# ── Hlavní widgety ────────────────────────────────────────────────────────────
col_prob, col_sec = st.columns([2, 1], gap="large")

with col_prob:
    p   = st.session_state.click_prob or 0.0
    cls = prob_class(p)
    lbl = prob_label(p) if p > 0 else "Klikni na mapu"
    coords = (
        f"{st.session_state.clicked_lat:.4f}°N · {st.session_state.clicked_lon:.4f}°E"
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

    # Data pod widgetem
    if st.session_state.all_data:
        d = st.session_state.all_data
        st.markdown("""
        <div style="margin-top:10px;font-size:0.65rem;color:#2d6040;
                    text-transform:uppercase;letter-spacing:0.12em;margin-bottom:6px">
            Vstupní data modelu
        </div>""", unsafe_allow_html=True)

        r1 = st.columns(4)
        r2 = st.columns(4)
        cells_r1 = [
            ("🌡 Teplota",     f"{d['temp_now']} °C"),
            ("💧 Vlhkost",     f"{d['humidity']} %"),
            ("🌧 Srážky 7d",   f"{d['rain_7d']} mm"),
            ("🌧 Srážky 3d",   f"{d['rain_3d']} mm"),
        ]
        cells_r2 = [
            ("🪨 pH půdy",     f"{d['ph']}"),
            ("⛰ Výška",       f"{int(d['elev'])} m"),
            ("📐 Sklon",       f"{d.get('slope', '–')}°"),
            ("🌿 NDVI",        f"{d.get('ndvi', '–')}"),
        ]
        for col, (lbl2, val) in zip(r1, cells_r1):
            with col:
                st.markdown(f'<div class="dc"><div class="dc-l">{lbl2}</div><div class="dc-v">{val}</div></div>', unsafe_allow_html=True)
        for col, (lbl2, val) in zip(r2, cells_r2):
            with col:
                st.markdown(f'<div class="dc"><div class="dc-l">{lbl2}</div><div class="dc-v">{val}</div></div>', unsafe_allow_html=True)

with col_sec:
    d = st.session_state.all_data or {}
    gbif_count = d.get("gbif_count", 0)
    month_now  = date.today().month
    in_season  = SPECIES["season"][0] <= month_now <= SPECIES["season"][1]

    st.markdown(f"""
    <div class="sec-widget">
        <div class="sec-label">📍 GBIF nálezy v okolí</div>
        <div class="sec-value">{gbif_count}</div>
        <div class="sec-sub">historické záznamy do 30 km</div>
    </div>
    <div class="sec-widget">
        <div class="sec-label">📅 Hlavní sezóna</div>
        <div class="sec-value {'orange' if not in_season else ''}">{'✅ Probíhá' if in_season else '⏳ Mimo sezónu'}</div>
        <div class="sec-sub">optimum: srpen–říjen</div>
    </div>
    <div class="sec-widget">
        <div class="sec-label">🌍 Land cover</div>
        <div class="sec-value" style="font-size:0.9rem;color:#7dcc8a">{d.get('land_cover', '–')}</div>
        <div class="sec-sub">vhodnost: {int(d.get('land_suitability', 0)*100)} %</div>
    </div>""", unsafe_allow_html=True)

# ── Faktory modelu ────────────────────────────────────────────────────────────
if st.session_state.factor_scores:
    st.markdown('<div class="sec-header">Váha faktorů v modelu</div>', unsafe_allow_html=True)
    factors = st.session_state.factor_scores
    cols = st.columns(len(factors))
    for col, (name, score) in zip(cols, factors.items()):
        pct_f = int(score * 100)
        color = "#39ff14" if score > 0.6 else "#ffd700" if score > 0.3 else "#ff8c00"
        with col:
            st.markdown(f"""
            <div class="dc" style="padding:8px 6px">
                <div class="dc-l">{name}</div>
                <div class="dc-v" style="color:{color};font-size:0.9rem">{pct_f} %</div>
                <div style="height:4px;border-radius:2px;margin-top:5px;
                            background:linear-gradient(90deg,{color} {pct_f}%,#0d3020 {pct_f}%)">
                </div>
            </div>""", unsafe_allow_html=True)

# ── Mapa ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-header">Interaktivní mapa — klikni pro predikci</div>', unsafe_allow_html=True)
st.markdown("""
<div class="map-tip">
    <span>👆 Klikni kamkoliv na mapu</span> — aplikace stáhne reálná data
    ze 6 zdrojů (počasí, srážky 14 dní, půda, terén, land cover, NDVI, GBIF)
    a okamžitě vypočítá pravděpodobnost výskytu lysohlávek.
</div>""", unsafe_allow_html=True)

m = folium.Map(
    location=[region_center[0], region_center[1]],
    zoom_start=region_center[2],
    tiles="CartoDB dark_matter",
    prefer_canvas=True,
)

if st.session_state.clicked_lat and st.session_state.click_prob is not None:
    p   = st.session_state.click_prob
    cls = prob_class(p)
    d   = st.session_state.all_data or {}
    hex_map = {"high":"#39ff14","medium":"#ffd700","low":"#ff8c00","none":"#444"}
    col_map = {"high":"green","medium":"orange","low":"red","none":"gray"}
    hx = hex_map[cls]

    popup_html = f"""
    <div style="font-family:monospace;min-width:210px;padding:6px">
        <div style="font-weight:700;font-size:1.2rem;color:{hx};margin-bottom:4px">
            {int(p*100)} % — {prob_label(p)}
        </div>
        <hr style="margin:5px 0;border-color:#eee">
        <div style="font-size:0.78rem;color:#333;line-height:1.9">
            🌡 {d.get('temp_now','–')} °C &nbsp;·&nbsp; 💧 {d.get('humidity','–')} %<br>
            🌧 Srážky 7d: <b>{d.get('rain_7d','–')} mm</b><br>
            🌧 Srážky 3d: <b>{d.get('rain_3d','–')} mm</b><br>
            🪨 pH: <b>{d.get('ph','–')}</b> &nbsp;·&nbsp; 🌱 SOC: {d.get('soc','–')} g/kg<br>
            ⛰ {int(d.get('elev',0))} m &nbsp;·&nbsp; 📐 sklon {d.get('slope','–')}°<br>
            🌍 {d.get('land_cover','–')}<br>
            🌿 NDVI: {d.get('ndvi','–')} &nbsp;·&nbsp; 📍 GBIF: {d.get('gbif_count',0)}
        </div>
        <hr style="margin:5px 0;border-color:#eee">
        <div style="font-size:0.68rem;color:#888">
            {st.session_state.clicked_lat:.5f}°N · {st.session_state.clicked_lon:.5f}°E
        </div>
    </div>"""

    folium.Marker(
        [st.session_state.clicked_lat, st.session_state.clicked_lon],
        icon=folium.Icon(color=col_map[cls], icon="leaf", prefix="fa"),
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"🍄 {int(p*100)} % · {prob_label(p)}",
    ).add_to(m)

    folium.Circle(
        [st.session_state.clicked_lat, st.session_state.clicked_lon],
        radius=int(200 + (st.session_state.click_prob or 0) * 1300), color=hx, fill=True, fill_opacity=0.15 + (st.session_state.click_prob or 0) * 0.25, weight=2,
    ).add_to(m)

map_data = st_folium(m, width="100%", height=500, key="psy_map",
                     returned_objects=["last_clicked"])

# ── Zpracování kliknutí ───────────────────────────────────────────────────────
if map_data:
    raw = map_data.get("last_clicked")
    if raw and isinstance(raw, dict):
        lat = raw.get("lat") or raw.get("latitude")
        lng = raw.get("lng") or raw.get("lon") or raw.get("longitude")
        if lat and lng:
            lat, lng = float(lat), float(lng)
            if lat != st.session_state.clicked_lat or lng != st.session_state.clicked_lon:
                with st.spinner("🌍 Stahuji data ze 6 zdrojů…"):
                    all_data = fetch_all(lat, lng)
                prob, factors = compute_probability(all_data)
                st.session_state.clicked_lat   = lat
                st.session_state.clicked_lon   = lng
                st.session_state.click_prob    = prob
                st.session_state.all_data      = all_data
                st.session_state.factor_scores = factors
                st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:0.72rem;color:#0d3020;letter-spacing:0.1em">
    PSY SPACE v3 · Open-Meteo · SoilGrids · Open-Elevation · Copernicus · MODIS · GBIF
</div>""", unsafe_allow_html=True)

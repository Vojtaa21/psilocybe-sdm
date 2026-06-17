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
    Hlavní predikční funkce — kombinuje všechna reálná data.

    Váhy jednotlivých faktorů jsou kalibrované na ekologii
    Psilocybe semilanceata (Stamets 1996, GBIF occurrence modeling).
    """
    month = date.today().month
    in_season = SPECIES["season"][0] <= month <= SPECIES["season"][1]

    # ── Faktor 1: Aktuální teplota (váha 0.18) ──────────────────────────────
    f_temp = tri(data["temp_now"], 6, 14, -2, 22)

    # ── Faktor 2: Vlhkost vzduchu (váha 0.14) ───────────────────────────────
    f_humidity = tri(data["humidity"], 75, 98, 45, 100)

    # ── Faktor 3: Srážky 7 dní zpět (váha 0.16) ─────────────────────────────
    # Lysohlávky reagují na srážky se zpožděním 5–10 dní!
    f_rain7 = tri(data["rain_7d"], 15, 55, 3, 120)

    # ── Faktor 4: Srážky 3 dny zpět (váha 0.08) ─────────────────────────────
    f_rain3 = tri(data["rain_3d"], 5, 25, 0, 60)

    # ── Faktor 5: Vlhkost půdy (váha 0.10) ──────────────────────────────────
    f_soil_moist = tri(data.get("soil_moisture", 0.3), 0.25, 0.6, 0.1, 0.8)

    # ── Faktor 6: pH půdy (váha 0.08) ───────────────────────────────────────
    f_ph = tri(data["ph"], 4.5, 6.5, 3.0, 8.0)

    # ── Faktor 7: Organická hmota půdy (váha 0.06) ──────────────────────────
    f_soc = tri(data["soc"], 15, 60, 3, 100)

    # ── Faktor 8: Nadmořská výška (váha 0.06) ───────────────────────────────
    f_elev = tri(data["elev"], 100, 800, 30, 1400)

    # ── Faktor 9: Sklon svahu (váha 0.04) ───────────────────────────────────
    # Mírný sklon = lepší odvodnění = méně záplav
    f_slope = tri(data.get("slope", 5), 2, 20, 0, 40)

    # ── Faktor 10: Severní orientace (váha 0.04) ─────────────────────────────
    # Sever = více vlhkosti a stínu
    north = data.get("north_factor", 0.5)
    f_north = (north + 1) / 2  # převod z [-1,1] na [0,1]

    # ── Faktor 11: Land cover (váha 0.10) ───────────────────────────────────
    f_landcover = data.get("land_suitability", 0.5)

    # ── Faktor 12: NDVI vegetační index (váha 0.04) ──────────────────────────
    f_ndvi = tri(data.get("ndvi", 0.55), 0.4, 0.8, 0.1, 0.95)

    # ── Faktor 13: Sezóna (váha 0.08) ───────────────────────────────────────
    f_season = 1.0 if in_season else 0.12

    # ── Váhovaný součet ──────────────────────────────────────────────────────
    base = (
        0.18 * f_temp       +
        0.14 * f_humidity   +
        0.16 * f_rain7      +
        0.08 * f_rain3      +
        0.10 * f_soil_moist +
        0.08 * f_ph         +
        0.06 * f_soc        +
        0.06 * f_elev       +
        0.04 * f_slope      +
        0.04 * f_north      +
        0.10 * f_landcover  +
        0.04 * f_ndvi       +
        0.08 * f_season
    )

    # ── GBIF boost (historická hustota nálezů) ───────────────────────────────
    gbif_boost = min(0.12, data.get("gbif_count", 0) * 0.006)

    # ── Penalizace za extrémní podmínky ──────────────────────────────────────
    penalty = 1.0
    if data["temp_now"] > 22 or data["temp_now"] < 0:  penalty *= 0.3
    if data["humidity"] < 45:                           penalty *= 0.5
    if data.get("wind", 0) > 15:                        penalty *= 0.8
    if data.get("water_deficit", 0) > 30:               penalty *= 0.7

    prob = float(np.clip((base + gbif_boost) * penalty, 0.0, 1.0))

    # Slovník skóre pro zobrazení
    factors = {
        "Teplota":       round(f_temp, 2),
        "Vlhkost":       round(f_humidity, 2),
        "Srážky 7d":     round(f_rain7, 2),
        "Srážky 3d":     round(f_rain3, 2),
        "Vlhkost půdy":  round(f_soil_moist, 2),
        "pH půdy":       round(f_ph, 2),
        "Org. hmota":    round(f_soc, 2),
        "Výška":         round(f_elev, 2),
        "Sklon":         round(f_slope, 2),
        "Orientace S":   round(f_north, 2),
        "Land cover":    round(f_landcover, 2),
        "NDVI":          round(f_ndvi, 2),
        "Sezóna":        round(f_season, 2),
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
        radius=8000, color=hx, fill=True, fill_opacity=0.07, weight=1,
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

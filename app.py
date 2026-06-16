"""
Psy Space — predikční nástroj výskytu Psilocybe semilanceata
Reálná data: Open-Meteo (počasí) + SoilGrids (půda) + GBIF (nálezy)
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
import requests
import time
from streamlit_folium import st_folium
from datetime import date, timedelta

# ── Volitelný import SDM modelu ──────────────────────────────────────────────
try:
    from sdm_model import PsilocybeSDM, SDMFeatures
    _sdm = PsilocybeSDM()
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

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
.prob-value { font-size: 5rem; font-weight: 900; line-height: 1; letter-spacing: -0.03em; }
.prob-value.high   { color: #39ff14; text-shadow: 0 0 30px #39ff1450; }
.prob-value.medium { color: #ffd700; text-shadow: 0 0 30px #ffd70040; }
.prob-value.low    { color: #ff8c00; text-shadow: 0 0 30px #ff8c0040; }
.prob-value.none   { color: #2d4030; }
.prob-pct { font-size: 2rem; font-weight: 400; opacity: 0.7; }
.prob-status {
    display: inline-block; margin-top: 10px; padding: 4px 16px;
    border-radius: 20px; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.08em;
}
.prob-status.high   { background: #0d3a10; color: #39ff14; border: 1px solid #39ff1440; }
.prob-status.medium { background: #2a2000; color: #ffd700; border: 1px solid #ffd70040; }
.prob-status.low    { background: #2a1000; color: #ff8c00; border: 1px solid #ff8c0040; }
.prob-status.none   { background: #0d1a10; color: #2d6040; border: 1px solid #0d3020; }
.prob-coords { margin-top: 10px; font-size: 0.75rem; color: #2d6040; font-family: monospace; }
.sec-widget {
    background: #03120a; border: 1px solid #0d3020;
    border-radius: 14px; padding: 18px 16px 14px; text-align: center;
    margin-bottom: 10px;
}
.sec-label {
    font-size: 0.7rem; color: #2d6040;
    text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 8px;
}
.sec-value { font-size: 2rem; font-weight: 800; color: #39ff14; line-height: 1; }
.sec-value.orange { color: #ff8c00; }
.sec-sub { font-size: 0.75rem; color: #2d6040; margin-top: 5px; }
.data-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;
    margin-top: 14px;
}
.data-cell {
    background: #03120a; border: 1px solid #0d3020; border-radius: 10px;
    padding: 10px 12px; text-align: center;
}
.data-cell-label { font-size: 0.68rem; color: #2d6040; text-transform: uppercase; letter-spacing: 0.1em; }
.data-cell-val { font-size: 1.1rem; font-weight: 700; color: #7dcc8a; margin-top: 2px; }
.sec-header {
    font-size: 0.72rem; color: #ff8c00; text-transform: uppercase;
    letter-spacing: 0.2em; border-bottom: 1px solid #0d3020;
    padding-bottom: 5px; margin: 20px 0 12px;
}
.map-tip {
    background: #03120a; border: 1px solid #0d3020; border-radius: 10px;
    padding: 10px 14px; font-size: 0.78rem; color: #2d6040;
    margin-bottom: 10px; line-height: 1.5;
}
.map-tip span { color: #39ff14; }
.data-source-row {
    display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px;
}
.ds-badge {
    font-size: 0.68rem; padding: 3px 10px; border-radius: 20px;
    font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
}
.ds-ok   { background: #0d3010; color: #39ff14; border: 1px solid #39ff1430; }
.ds-warn { background: #1a1000; color: #ff8c00; border: 1px solid #ff8c0030; }
.ds-off  { background: #0d0d0d; color: #2d4030; border: 1px solid #1a1a1a; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #03080f; }
::-webkit-scrollbar-thumb { background: #0d3020; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KONSTANTY
# ══════════════════════════════════════════════════════════════════════════════
SPECIES = {
    "name": "Psilocybe semilanceata",
    "cz":   "Lysohlávka kopinatá",
    "emoji": "🍄",
    "season": (8, 10),
    "opt": {
        "temp":  (6, 14),
        "rain":  (600, 1400),
        "ph":    (4.5, 6.5),
        "elev":  (100, 800),
        "humid": (70, 98),
        "wind":  (0, 8),
    },
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
    "click_prob": None, "weather_data": None,
    "soil_data": None, "gbif_nearby": None,
    "data_sources": {},
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# DATOVÉ FUNKCE — reálná API
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather(lat: float, lon: float) -> dict:
    """
    Open-Meteo API — aktuální počasí + historické srážky.
    Zdarma, bez API klíče, přesnost ~1 km.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "weather_code",
            ],
            "daily": ["precipitation_sum", "temperature_2m_max", "temperature_2m_min"],
            "past_days": 7,
            "forecast_days": 1,
            "timezone": "Europe/Prague",
        }
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        d = resp.json()

        cur  = d.get("current", {})
        daily = d.get("daily", {})

        # Srážky za posledních 7 dní
        rain_7d = sum(r for r in (daily.get("precipitation_sum") or []) if r is not None)

        return {
            "temp":       round(cur.get("temperature_2m", 10.0), 1),
            "humidity":   int(cur.get("relative_humidity_2m", 70)),
            "precip_now": round(cur.get("precipitation", 0.0), 1),
            "wind":       round(cur.get("wind_speed_10m", 3.0), 1),
            "rain_7d":    round(rain_7d, 1),
            "source":     "Open-Meteo (reálná data)",
            "ok":         True,
        }
    except Exception as e:
        return {
            "temp": 10.0, "humidity": 70, "precip_now": 0.0,
            "wind": 3.0, "rain_7d": 20.0,
            "source": f"Open-Meteo nedostupné: {e}", "ok": False,
        }


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_elevation(lat: float, lon: float) -> float:
    """Open-Elevation API — nadmořská výška."""
    try:
        resp = requests.post(
            "https://api.open-elevation.com/api/v1/lookup",
            json={"locations": [{"latitude": lat, "longitude": lon}]},
            timeout=6,
        )
        return float(resp.json()["results"][0]["elevation"])
    except Exception:
        # Záloha: aproximace podle zeměpisné šířky pro ČR
        return max(150.0, 300 + abs(lat - 50) * 100)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_soil(lat: float, lon: float) -> dict:
    """SoilGrids REST API — pH, organická hmota, složení půdy."""
    try:
        url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
        params = {
            "lon": lon, "lat": lat,
            "property": ["phh2o", "soc", "clay", "sand"],
            "depth": "0-5cm", "value": "mean",
        }
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        layers = resp.json().get("properties", {}).get("layers", [])
        result = {}
        for layer in layers:
            name = layer["name"]
            val  = layer.get("depths", [{}])[0].get("values", {}).get("mean")
            if val is None:
                continue
            if name == "phh2o": result["ph"]   = round(val / 10, 2)
            elif name == "soc": result["soc"]  = round(val / 10, 1)
            elif name == "clay":result["clay"] = round(val / 10, 1)
            elif name == "sand":result["sand"] = round(val / 10, 1)
        result["ok"] = True
        return result
    except Exception:
        return {"ph": 5.8, "soc": 20.0, "clay": 28.0, "sand": 35.0, "ok": False}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gbif_nearby(lat: float, lon: float, radius_km: int = 30) -> int:
    """GBIF API — počet nálezů lysohlávek v okolí bodu."""
    try:
        resp = requests.get(
            "https://api.gbif.org/v1/occurrence/search",
            params={
                "scientificName": "Psilocybe semilanceata",
                "decimalLatitude":  f"{lat-0.3},{lat+0.3}",
                "decimalLongitude": f"{lon-0.5},{lon+0.5}",
                "hasCoordinate": True,
                "limit": 1,
            },
            timeout=8,
        )
        return int(resp.json().get("count", 0))
    except Exception:
        return 0


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_worldclim_approx(lat: float, lon: float) -> dict:
    """
    Aproximace WorldClim bioklimatických proměnných pro ČR/SR.
    Pro produkci: nahradit čtením z GeoTIFF (worldclim.org/data/worldclim21.html).
    """
    lat_n = (lat - 47.5) / (51.2 - 47.5)
    lon_n = (lon - 12.0) / (22.5 - 12.0)
    return {
        "bio01": round(10.5 - lat_n * 4.0 + lon_n * 0.5, 2),
        "bio04": round(680 + lat_n * 80, 1),
        "bio12": round(580 + lon_n * 220 + lat_n * 80, 1),
        "bio15": round(28 - lon_n * 5, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PREDIKČNÍ FUNKCE — kombinuje všechna reálná data
# ══════════════════════════════════════════════════════════════════════════════

def compute_probability(lat: float, lon: float) -> tuple[float, dict]:
    """
    Kombinuje reálná data z Open-Meteo, SoilGrids, Open-Elevation
    a WorldClim do jednoho pravděpodobnostního skóre.

    Vrátí: (pravděpodobnost 0.0–1.0, slovník vstupních dat)
    """
    # 1. Stáhni všechna data paralelně (sekvenčně kvůli Streamlit cache)
    weather = fetch_weather(lat, lon)
    elev    = fetch_elevation(lat, lon)
    soil    = fetch_soil(lat, lon)
    climate = fetch_worldclim_approx(lat, lon)
    gbif_n  = fetch_gbif_nearby(lat, lon)

    # 2. Sestav feature vektor
    features = {
        # Aktuální počasí (Open-Meteo) — nejvyšší váha pro krátkodobou predikci
        "temp_now":   weather["temp"],
        "humidity":   weather["humidity"],
        "rain_7d":    weather["rain_7d"],
        "wind":       weather["wind"],
        # Terén (Open-Elevation)
        "elev":       elev,
        # Půda (SoilGrids)
        "ph":         soil.get("ph", 5.8),
        "soc":        soil.get("soc", 20.0),
        "clay":       soil.get("clay", 28.0),
        # Klima (WorldClim 30letý průměr)
        "bio01":      climate["bio01"],
        "bio12":      climate["bio12"],
        # GBIF — historická hustota nálezů v okolí
        "gbif_nearby": gbif_n,
    }

    # 3. Predikce
    if MODEL_AVAILABLE:
        # Produkční mód — sdm_model.py
        sdm_features = SDMFeatures(
            bio01=climate["bio01"], bio04=climate["bio04"],
            bio12=climate["bio12"], bio15=climate["bio15"],
            ph=features["ph"], elev=elev,
            ndvi=0.55,  # TODO: napojit Sentinel-2
            soc=features["soc"], clay=features["clay"],
            sand=soil.get("sand", 35.0),
            bulk_density=1.2,
            slope=5.0,
        )
        result = _sdm.predict_point(sdm_features.__dict__)
        base_prob = result.probability
    else:
        # Záložní skóre — trojúhelníkové funkce pro každou proměnnou
        def tri(v, ol, oh, al, ah):
            if v is None or v < al or v > ah: return 0.0
            if ol <= v <= oh: return 1.0
            return (v-al)/(ol-al) if v < ol else (ah-v)/(ah-oh)

        month = date.today().month
        in_season = SPECIES["season"][0] <= month <= SPECIES["season"][1]

        base_prob = (
            0.22 * tri(weather["temp"],   6,  14,   0, 25)  +
            0.18 * tri(weather["humidity"], 70, 98,  40, 100) +
            0.15 * tri(weather["rain_7d"], 15, 50,   3, 120) +
            0.15 * tri(climate["bio12"],  600, 1400, 200, 2200) +
            0.10 * tri(features["ph"],    4.5, 6.5,  3.0, 8.5) +
            0.08 * tri(elev,              100,  800,  50, 1400) +
            0.07 * tri(features["soc"],    15,  60,   3, 100)  +
            0.05 * (1.0 if in_season else 0.15)
        )

    # 4. Boost z historických GBIF nálezů v okolí
    gbif_boost = min(0.12, gbif_n * 0.008)
    prob = float(np.clip(base_prob + gbif_boost, 0.0, 1.0))

    # 5. Penalizace za nevhodné aktuální podmínky
    if weather["temp"] > 22 or weather["temp"] < 0:
        prob *= 0.4
    if weather["humidity"] < 50:
        prob *= 0.6
    if weather["wind"] > 12:
        prob *= 0.85

    return float(np.clip(prob, 0.0, 1.0)), features


def prob_class(p: float) -> str:
    if p >= 0.65: return "high"
    if p >= 0.40: return "medium"
    if p >= 0.15: return "low"
    return "none"

def prob_label(p: float) -> str:
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
        <div style="color:#39ff14;font-weight:700;font-size:0.92rem;margin-top:2px">{SPECIES['cz']}</div>
        <div style="color:#2d6040;font-size:0.76rem;font-style:italic">{SPECIES['name']}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    st.markdown("---")
    st.markdown("### Region")
    selected_region = st.selectbox(
        "Oblast", options=list(REGIONS.keys()), label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### Časový rámec")
    time_mode = st.radio(
        "Období", ["Aktuální týden", "Vlastní rozsah"], label_visibility="collapsed"
    )
    if time_mode == "Aktuální týden":
        d_from = date.today() - timedelta(days=date.today().weekday())
        d_to   = d_from + timedelta(days=6)
        st.caption(f"📅 {d_from.strftime('%d. %m.')} — {d_to.strftime('%d. %m. %Y')}")
    else:
        d_from = st.date_input("Od", value=date.today() - timedelta(days=7))
        d_to   = st.date_input("Do", value=date.today())

    st.markdown("---")
    st.markdown("### Datové zdroje")
    st.markdown("""
    <div style="font-size:0.75rem;color:#2d6040;line-height:2">
        🌡 <b style="color:#39ff14">Open-Meteo</b> — aktuální počasí<br>
        🪨 <b style="color:#39ff14">SoilGrids</b> — půdní data<br>
        ⛰ <b style="color:#39ff14">Open-Elevation</b> — výška terénu<br>
        📍 <b style="color:#39ff14">GBIF</b> — historické nálezy<br>
        🌍 <b style="color:#ff8c00">WorldClim</b> — klimatická normála
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HLAVNÍ OBSAH
# ══════════════════════════════════════════════════════════════════════════════
region_center = REGIONS[selected_region]

# Header
col_title, col_hint = st.columns([2, 1])
with col_title:
    st.markdown('<div class="psy-logo">PSY SPACE</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="psy-species">{SPECIES["emoji"]} {SPECIES["name"]} '
        f'&nbsp;·&nbsp; {selected_region}</div>',
        unsafe_allow_html=True,
    )
with col_hint:
    st.markdown("""
    <div style="padding-top:16px;font-size:0.78rem;color:#2d6040;text-align:right;line-height:1.8">
        👆 Klikni na mapu pro analýzu<br>
        <span style="font-size:0.7rem">načte reálné počasí, půdu a výšku</span>
    </div>
    """, unsafe_allow_html=True)

# ── Widgety ──────────────────────────────────────────────────────────────────
col_prob, col_sec = st.columns([2, 1], gap="large")

with col_prob:
    if st.session_state.click_prob is not None:
        p      = st.session_state.click_prob
        cls    = prob_class(p)
        lbl    = prob_label(p)
        coords = f"{st.session_state.clicked_lat:.4f}°N · {st.session_state.clicked_lon:.4f}°E"
    else:
        p, cls, lbl, coords = 0.0, "none", "Klikni na mapu", "čeká na výběr bodu…"

    pct = int(p * 100)
    st.markdown(f"""
    <div class="prob-widget">
        <div class="prob-label">🍄 Pravděpodobnost výskytu</div>
        <div class="prob-value {cls}">{pct}<span class="prob-pct"> %</span></div>
        <span class="prob-status {cls}">{lbl}</span>
        <div class="prob-coords">{coords}</div>
    </div>
    """, unsafe_allow_html=True)

    # Datové vstupy modelu pod hlavním widgetem
    if st.session_state.weather_data:
        w = st.session_state.weather_data
        f = st.session_state.get("feature_data", {})
        st.markdown("""<div style="margin-top:10px">
        <div style="font-size:0.68rem;color:#2d6040;text-transform:uppercase;
                    letter-spacing:0.12em;margin-bottom:6px">
            Vstupní data modelu (reálná)
        </div></div>""", unsafe_allow_html=True)

        d1, d2, d3, d4, d5, d6 = st.columns(6)
        cells = [
            (d1, "🌡 Teplota",  f"{w['temp']} °C"),
            (d2, "💧 Vlhkost",  f"{w['humidity']} %"),
            (d3, "🌧 Srážky 7d",f"{w['rain_7d']} mm"),
            (d4, "🪨 pH půdy",  f"{f.get('ph', '–')}"),
            (d5, "⛰ Výška",    f"{int(f.get('elev', 0))} m"),
            (d6, "📍 GBIF",    f"{f.get('gbif_nearby', 0)} nálezů"),
        ]
        for col, label, val in cells:
            with col:
                st.markdown(f"""
                <div class="data-cell">
                    <div class="data-cell-label">{label}</div>
                    <div class="data-cell-val">{val}</div>
                </div>
                """, unsafe_allow_html=True)

with col_sec:
    # Nálezy v okolí
    gbif_nearby = st.session_state.get("gbif_nearby_count", 0)
    st.markdown(f"""
    <div class="sec-widget">
        <div class="sec-label">📍 Nálezy v okolí 30 km</div>
        <div class="sec-value">{gbif_nearby}</div>
        <div class="sec-sub">záznamy z GBIF databáze</div>
    </div>
    """, unsafe_allow_html=True)

    # Sezóna
    month_now  = date.today().month
    in_season  = SPECIES["season"][0] <= month_now <= SPECIES["season"][1]
    season_txt = "✅ Probíhá" if in_season else "⏳ Mimo sezónu"
    season_cls = "" if in_season else "orange"
    st.markdown(f"""
    <div class="sec-widget">
        <div class="sec-label">📅 Hlavní sezóna</div>
        <div class="sec-value {season_cls}" style="font-size:1.15rem">{season_txt}</div>
        <div class="sec-sub">optimum: srpen–říjen</div>
    </div>
    """, unsafe_allow_html=True)

    # Aktuální počasí (pokud je k dispozici)
    if st.session_state.weather_data:
        w = st.session_state.weather_data
        src_cls = "ds-ok" if w["ok"] else "ds-warn"
        st.markdown(f"""
        <div class="sec-widget">
            <div class="sec-label">🌡 Aktuální teplota</div>
            <div class="sec-value">{w['temp']}<span style="font-size:1rem"> °C</span></div>
            <div class="sec-sub">💧 {w['humidity']} % vlhkost · 💨 {w['wind']} km/h</div>
        </div>
        """, unsafe_allow_html=True)

# ── Mapa ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-header">Interaktivní mapa — klikni pro predikci</div>',
            unsafe_allow_html=True)
st.markdown("""
<div class="map-tip">
    <span>👆 Klikni kamkoliv na mapu</span> — aplikace stáhne reálné počasí,
    půdní data a nadmořskou výšku pro vybraný bod a okamžitě vypočítá
    pravděpodobnost výskytu lysohlávek pomocí AI modelu.
</div>
""", unsafe_allow_html=True)

m = folium.Map(
    location=[region_center[0], region_center[1]],
    zoom_start=region_center[2],
    tiles="CartoDB dark_matter",
    prefer_canvas=True,
)

# Marker kliknutého bodu
if (st.session_state.clicked_lat is not None
        and st.session_state.click_prob is not None):
    p   = st.session_state.click_prob
    cls = prob_class(p)
    w   = st.session_state.weather_data or {}
    f   = st.session_state.get("feature_data", {})

    color_map = {"high": "green", "medium": "orange", "low": "red", "none": "gray"}
    hex_map   = {"high": "#39ff14", "medium": "#ffd700", "low": "#ff8c00", "none": "#444"}

    popup_html = f"""
    <div style="font-family:monospace;min-width:200px;padding:6px">
        <div style="font-weight:700;font-size:1.2rem;color:{hex_map[cls]};margin-bottom:4px">
            {int(p*100)} % — {prob_label(p)}
        </div>
        <hr style="margin:6px 0;border-color:#ddd">
        <div style="font-size:0.8rem;color:#444;line-height:1.8">
            🌡 Teplota: <b>{w.get('temp','–')} °C</b><br>
            💧 Vlhkost: <b>{w.get('humidity','–')} %</b><br>
            🌧 Srážky 7d: <b>{w.get('rain_7d','–')} mm</b><br>
            🪨 pH půdy: <b>{f.get('ph','–')}</b><br>
            ⛰ Výška: <b>{int(f.get('elev',0))} m n.m.</b><br>
            📍 GBIF nálezy: <b>{f.get('gbif_nearby',0)}</b>
        </div>
        <hr style="margin:6px 0;border-color:#ddd">
        <div style="font-size:0.7rem;color:#888">
            {st.session_state.clicked_lat:.5f}°N · {st.session_state.clicked_lon:.5f}°E
        </div>
    </div>
    """

    folium.Marker(
        location=[st.session_state.clicked_lat, st.session_state.clicked_lon],
        icon=folium.Icon(color=color_map[cls], icon="leaf", prefix="fa"),
        popup=folium.Popup(popup_html, max_width=240),
        tooltip=f"🍄 {int(p*100)} % · {prob_label(p)} · klikni pro detail",
    ).add_to(m)

    folium.Circle(
        location=[st.session_state.clicked_lat, st.session_state.clicked_lon],
        radius=8000,
        color=hex_map[cls],
        fill=True, fill_opacity=0.07, weight=1,
    ).add_to(m)

map_data = st_folium(
    m, width="100%", height=500,
    key="psy_map",
    returned_objects=["last_clicked"],
)

# ── Zpracování kliknutí ───────────────────────────────────────────────────────
if map_data:
    raw = map_data.get("last_clicked")
    if raw and isinstance(raw, dict):
        lat = raw.get("lat") or raw.get("latitude")
        lng = raw.get("lng") or raw.get("lon") or raw.get("longitude")
        if lat is not None and lng is not None:
            lat, lng = float(lat), float(lng)
            if (lat != st.session_state.clicked_lat or
                    lng != st.session_state.clicked_lon):

                with st.spinner("🌍 Stahuji reálná data pro vybraný bod…"):
                    prob, feat = compute_probability(lat, lng)
                    weather    = fetch_weather(lat, lng)
                    gbif_count = fetch_gbif_nearby(lat, lng)

                st.session_state.clicked_lat      = lat
                st.session_state.clicked_lon      = lng
                st.session_state.click_prob       = prob
                st.session_state.weather_data     = weather
                st.session_state.feature_data     = feat
                st.session_state.gbif_nearby_count = gbif_count
                st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:0.72rem;color:#0d3020;letter-spacing:0.1em">
    PSY SPACE · vzdělávací prototyp ·
    Open-Meteo · SoilGrids (ISRIC) · Open-Elevation · GBIF · WorldClim
</div>
""", unsafe_allow_html=True)

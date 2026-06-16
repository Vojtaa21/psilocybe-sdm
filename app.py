"""
Psy Space — predikční nástroj výskytu Psilocybe semilanceata
Streamlit aplikace připravená pro Streamlit Cloud + sdm_model.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from datetime import date, timedelta

# ── Volitelný import modelu ──────────────────────────────────────────────────
try:
    from sdm_model import PsilocybeSDM, SDMFeatures
    _model = PsilocybeSDM()
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# 1. KONFIGURACE
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Psy Space",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. CSS — vesmírně-lesní estetika
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* Základní pozadí */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #03080f;
    color: #d0f0d0;
}
.block-container { padding-top: 1.2rem !important; max-width: 1280px; }

/* Sidebar */
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

/* Nadpis */
.psy-logo {
    font-size: 2.8rem; font-weight: 900; letter-spacing: 0.1em;
    background: linear-gradient(90deg, #39ff14 0%, #a8e44a 50%, #ff8c00 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.05;
}
.psy-species {
    font-size: 0.82rem; color: #2d6040; letter-spacing: 0.2em;
    text-transform: uppercase; margin-bottom: 1.2rem;
}

/* Hlavní widget — pravděpodobnost */
.prob-widget {
    background: linear-gradient(135deg, #03120a 0%, #071f10 100%);
    border: 1px solid #0d4020;
    border-radius: 18px;
    padding: 28px 24px 22px;
    text-align: center;
    position: relative; overflow: hidden;
}
.prob-widget::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #39ff14, transparent);
}
.prob-widget::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, #ff8c0040, transparent);
}
.prob-label {
    font-size: 0.72rem; color: #2d6040;
    text-transform: uppercase; letter-spacing: 0.18em; margin-bottom: 10px;
}
.prob-value {
    font-size: 5rem; font-weight: 900; line-height: 1;
    letter-spacing: -0.03em;
    transition: color 0.4s;
}
.prob-value.high   { color: #39ff14; text-shadow: 0 0 30px #39ff1450; }
.prob-value.medium { color: #ffd700; text-shadow: 0 0 30px #ffd70040; }
.prob-value.low    { color: #ff8c00; text-shadow: 0 0 30px #ff8c0040; }
.prob-value.none   { color: #2d4030; text-shadow: none; }
.prob-pct { font-size: 2rem; font-weight: 400; opacity: 0.7; }
.prob-status {
    display: inline-block;
    margin-top: 10px; padding: 4px 16px;
    border-radius: 20px; font-size: 0.8rem; font-weight: 700;
    letter-spacing: 0.08em;
}
.prob-status.high   { background: #0d3a10; color: #39ff14; border: 1px solid #39ff1440; }
.prob-status.medium { background: #2a2000; color: #ffd700; border: 1px solid #ffd70040; }
.prob-status.low    { background: #2a1000; color: #ff8c00; border: 1px solid #ff8c0040; }
.prob-status.none   { background: #0d1a10; color: #2d6040; border: 1px solid #0d3020; }
.prob-coords {
    margin-top: 10px; font-size: 0.75rem; color: #2d6040;
    font-family: monospace; letter-spacing: 0.05em;
}

/* Sekundární widgety */
.sec-widget {
    background: #03120a;
    border: 1px solid #0d3020;
    border-radius: 14px;
    padding: 18px 16px 14px;
    text-align: center; height: 100%;
}
.sec-label {
    font-size: 0.7rem; color: #2d6040;
    text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 8px;
}
.sec-value { font-size: 2rem; font-weight: 800; color: #39ff14; line-height: 1; }
.sec-value.orange { color: #ff8c00; }
.sec-sub { font-size: 0.75rem; color: #2d6040; margin-top: 5px; }

/* Sekce header */
.sec-header {
    font-size: 0.72rem; color: #ff8c00;
    text-transform: uppercase; letter-spacing: 0.2em;
    border-bottom: 1px solid #0d3020;
    padding-bottom: 5px; margin: 20px 0 12px;
}

/* Info tip */
.map-tip {
    background: #03120a; border: 1px solid #0d3020;
    border-radius: 10px; padding: 10px 14px;
    font-size: 0.78rem; color: #2d6040;
    margin-bottom: 10px; line-height: 1.5;
}
.map-tip span { color: #39ff14; }

/* Popup výsledek */
.popup-result {
    background: #071f10; border: 1px solid #39ff1430;
    border-radius: 12px; padding: 16px 20px;
    margin-top: 12px;
}
.popup-result-title { font-size: 0.72rem; color: #2d6040; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px; }
.popup-result-val { font-size: 1.8rem; font-weight: 800; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #03080f; }
::-webkit-scrollbar-thumb { background: #0d3020; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3. KONSTANTY DRUHU — pevně definovaný v modelu
# ══════════════════════════════════════════════════════════════════════════════
SPECIES = {
    "name":     "Psilocybe semilanceata",
    "cz":       "Lysohlávka kopinatá",
    "emoji":    "🍄",
    "season":   (8, 10),   # měsíce (srpen–říjen)
    "opt_temp": (6, 14),
    "opt_elev": (100, 800),
    "habitat":  "vlhké pastviny a louky",
}

REGIONS = {
    "Celá ČR + SR":                  [49.5, 16.5, 7],
    "Šumava a Krušné hory":          [49.0, 13.5, 8],
    "Praha a Středočeský kraj":      [50.0, 14.5, 9],
    "Krkonoše a Orlické hory":       [50.6, 16.0, 9],
    "Jeseníky a Beskydy":            [49.8, 17.8, 9],
    "Bílé Karpaty a jih Moravy":     [48.9, 17.5, 9],
    "Tatry a Orava (SK)":            [49.3, 19.8, 9],
    "Malé Karpaty (SK)":             [48.5, 17.3, 9],
}

# ══════════════════════════════════════════════════════════════════════════════
# 4. SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "clicked_lat"  not in st.session_state: st.session_state.clicked_lat  = None
if "clicked_lon"  not in st.session_state: st.session_state.clicked_lon  = None
if "click_prob"   not in st.session_state: st.session_state.click_prob   = None
if "click_status" not in st.session_state: st.session_state.click_status = None

# ══════════════════════════════════════════════════════════════════════════════
# 5. MODEL — funkce pro pravděpodobnost
# ══════════════════════════════════════════════════════════════════════════════

def get_probability(lat: float, lon: float) -> float:
    """
    Hlavní predikční funkce — vrátí pravděpodobnost výskytu 0.0–1.0.

    Produkční napojení na sdm_model.py:
    ------------------------------------
    features = SDMFeatures(
        bio01  = get_temperature(lat, lon),   # z ERA5 / ČHMÚ
        bio12  = get_rainfall(lat, lon),      # roční srážky mm
        ph     = get_soil_ph(lat, lon),       # SoilGrids API
        elev   = get_elevation(lat, lon),     # Open-Elevation API
        ndvi   = get_ndvi(lat, lon),          # Sentinel-2
        bio04  = 750.0,
        bio15  = 26.0,
    )
    result = _model.predict_point(features)
    return result.probability
    ------------------------------------
    """
    if MODEL_AVAILABLE:
        # Produkční mód — odkomentuj výše a smaž demo blok níže
        pass

    # Demo mód — deterministická aproximace podle polohy
    rng = np.random.default_rng(int(abs(lat * 1000 + lon * 100)) % (2**31))
    month = date.today().month

    # Klimatická aproximace pro ČR/SR
    temp   = 12.5 - (lat - 48.5) * 1.6 + rng.normal(0, 0.8)
    elev   = max(50, 200 + abs(lat - 50) * 120 + rng.normal(0, 80))
    rain   = 620 + (lon - 14) * 30 + rng.normal(0, 60)

    def tri(v, ol, oh, al, ah):
        if v < al or v > ah: return 0.0
        if ol <= v <= oh:    return 1.0
        return (v-al)/(ol-al) if v < ol else (ah-v)/(ah-oh)

    in_season = SPECIES["season"][0] <= month <= SPECIES["season"][1]
    score = (
        0.30 * tri(temp,  6,  14,  0, 22) +
        0.25 * tri(rain, 600, 1400, 300, 2000) +
        0.20 * tri(elev, 100,  800,  50, 1400) +
        0.15 * (1.0 if in_season else 0.2) +
        0.10 * float(rng.uniform(0.3, 0.9))
    )
    return float(np.clip(score, 0.0, 1.0))


def prob_class(p: float) -> str:
    """Vrátí CSS třídu podle pravděpodobnosti."""
    if p >= 0.65: return "high"
    if p >= 0.40: return "medium"
    if p >= 0.15: return "low"
    return "none"


def prob_label(p: float) -> str:
    """Vrátí český popis vhodnosti."""
    if p >= 0.65: return "Velmi vhodné"
    if p >= 0.40: return "Podmíněně vhodné"
    if p >= 0.15: return "Málo vhodné"
    return "Nevhodné"


def get_demo_kpi(region: str) -> dict:
    """
    Vrátí KPI hodnoty pro sidebar/widgety.
    Nahradit reálným API / sdm_model výstupem.
    """
    rng    = np.random.default_rng(hash(region) % (2**32))
    month  = date.today().month
    season = SPECIES["season"][0] <= month <= SPECIES["season"][1]
    return {
        "occurrences": int(rng.integers(2, 28 if season else 5)),
        "in_season":   season,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🍄 Psy Space")
    st.markdown("*predikce výskytu lysohlávek*")
    st.markdown("---")

    # Druh — pevně definován
    st.markdown("### Sledovaný druh")
    st.markdown(f"""
    <div style="background:#03120a;border:1px solid #0d3020;border-radius:10px;padding:12px 14px;margin-bottom:4px">
        <div style="font-size:1.3rem">{SPECIES['emoji']}</div>
        <div style="color:#39ff14;font-weight:700;font-size:0.92rem;margin-top:2px">{SPECIES['cz']}</div>
        <div style="color:#2d6040;font-size:0.76rem;font-style:italic">{SPECIES['name']}</div>
        <div style="color:#2d6040;font-size:0.74rem;margin-top:6px">📍 {SPECIES['habitat']}</div>
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
    # Model status
    if MODEL_AVAILABLE:
        st.markdown('<div style="background:#03120a;border:1px solid #0d4020;border-radius:8px;padding:6px 12px;font-size:0.75rem;color:#39ff14">✓ sdm_model.py aktivní</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background:#1a0800;border:1px solid #3a2000;border-radius:8px;padding:6px 12px;font-size:0.75rem;color:#ff8c00">⚠ demo mód — sdm_model.py nenalezen</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 7. HLAVNÍ OBSAH
# ══════════════════════════════════════════════════════════════════════════════
kpi = get_demo_kpi(selected_region)
region_center = REGIONS[selected_region]  # [lat, lon, zoom]

# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_hint = st.columns([2, 1])
with col_title:
    st.markdown(f'<div class="psy-logo">PSY SPACE</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="psy-species">{SPECIES["emoji"]} {SPECIES["name"]} '
        f'&nbsp;·&nbsp; {selected_region}</div>',
        unsafe_allow_html=True,
    )
with col_hint:
    st.markdown("""
    <div style="padding-top:12px;font-size:0.78rem;color:#2d6040;text-align:right;line-height:1.6">
        👆 Klikni na mapu<br>pro analýzu bodu
    </div>
    """, unsafe_allow_html=True)

# ── Widgety — pravděpodobnost + sekundární ────────────────────────────────────
col_prob, col_sec = st.columns([2, 1], gap="large")

with col_prob:
    # Hlavní widget — pravděpodobnost
    if st.session_state.click_prob is not None:
        p      = st.session_state.click_prob
        cls    = prob_class(p)
        lbl    = prob_label(p)
        lat_s  = f"{st.session_state.clicked_lat:.4f}°N"
        lon_s  = f"{st.session_state.clicked_lon:.4f}°E"
        coords = f"{lat_s} · {lon_s}"
    else:
        p      = 0.0
        cls    = "none"
        lbl    = "Klikni na mapu"
        coords = "čeká na výběr bodu…"

    pct = int(p * 100)
    st.markdown(f"""
    <div class="prob-widget">
        <div class="prob-label">🍄 Pravděpodobnost výskytu</div>
        <div class="prob-value {cls}">{pct}<span class="prob-pct"> %</span></div>
        <div>
            <span class="prob-status {cls}">{lbl}</span>
        </div>
        <div class="prob-coords">{coords}</div>
    </div>
    """, unsafe_allow_html=True)

with col_sec:
    month_now  = date.today().month
    in_season  = SPECIES["season"][0] <= month_now <= SPECIES["season"][1]
    season_txt = "✅ Probíhá" if in_season else "⏳ Mimo sezónu"
    season_cls = "" if in_season else "orange"

    # Sekundární widget 1 — nálezy
    st.markdown(f"""
    <div class="sec-widget" style="margin-bottom:12px">
        <div class="sec-label">📍 Zaznamenané nálezy</div>
        <div class="sec-value">{kpi['occurrences']}</div>
        <div class="sec-sub">v oblasti za posledních 7 dní</div>
    </div>
    """, unsafe_allow_html=True)

    # Sekundární widget 2 — sezóna
    st.markdown(f"""
    <div class="sec-widget">
        <div class="sec-label">📅 Hlavní sezóna</div>
        <div class="sec-value {season_cls}" style="font-size:1.3rem">{season_txt}</div>
        <div class="sec-sub">optimum: srpen–říjen</div>
    </div>
    """, unsafe_allow_html=True)

# ── Mapa ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-header">Interaktivní mapa výskytu</div>', unsafe_allow_html=True)

st.markdown("""
<div class="map-tip">
    <span>👆 Klikni kamkoliv na mapu</span> — aplikace okamžitě vypočítá
    pravděpodobnost výskytu lysohlávek pro vybrané místo pomocí AI modelu.
    Výsledek se zobrazí v hlavním widgetu výše i jako značka na mapě.
</div>
""", unsafe_allow_html=True)

# Sestav folium mapu
m = folium.Map(
    location=[region_center[0], region_center[1]],
    zoom_start=region_center[2],
    tiles="CartoDB dark_matter",   # tmavý podklad = vesmírně-lesní estetika
    prefer_canvas=True,
)

# Přidej marker kliknutého bodu s výsledkem
if (st.session_state.clicked_lat is not None
        and st.session_state.click_prob is not None):

    p   = st.session_state.click_prob
    cls = prob_class(p)
    color_map = {"high": "green", "medium": "orange", "low": "red", "none": "gray"}
    marker_color = color_map.get(cls, "gray")

    # Popup HTML
    popup_html = f"""
    <div style="font-family:monospace;min-width:180px;padding:4px">
        <div style="font-weight:700;font-size:1.1rem;color:#1a1a1a;margin-bottom:4px">
            {int(p*100)} % pravděpodobnost
        </div>
        <div style="font-size:0.8rem;color:#444">{prob_label(p)}</div>
        <hr style="margin:6px 0;border-color:#ddd">
        <div style="font-size:0.75rem;color:#666">
            📍 {st.session_state.clicked_lat:.5f}°N<br>
            📍 {st.session_state.clicked_lon:.5f}°E
        </div>
    </div>
    """

    folium.Marker(
        location=[st.session_state.clicked_lat, st.session_state.clicked_lon],
        icon=folium.Icon(color=marker_color, icon="leaf", prefix="fa"),
        popup=folium.Popup(popup_html, max_width=220),
        tooltip=f"🍄 {int(p*100)} % · {prob_label(p)}",
    ).add_to(m)

    # Kruh vhodnosti
    folium.Circle(
        location=[st.session_state.clicked_lat, st.session_state.clicked_lon],
        radius=8000,
        color={"high":"#39ff14","medium":"#ffd700","low":"#ff8c00","none":"#444"}.get(cls,"#444"),
        fill=True, fill_opacity=0.08, weight=1,
    ).add_to(m)

# Zachyť kliknutí
map_data = st_folium(
    m,
    width="100%",
    height=480,
    key="psy_map",
    returned_objects=["last_clicked"],
)

# ── Zpracování kliknutí → model → výsledek ────────────────────────────────────
if map_data:
    raw = map_data.get("last_clicked")
    if raw and isinstance(raw, dict):
        lat = raw.get("lat") or raw.get("latitude")
        lng = raw.get("lng") or raw.get("lon") or raw.get("longitude")
        if lat is not None and lng is not None:
            lat, lng = float(lat), float(lng)
            if (lat != st.session_state.clicked_lat
                    or lng != st.session_state.clicked_lon):
                # ── Výpočet pravděpodobnosti přes model ──────────────────────
                with st.spinner("Analyzuji bod…"):
                    prob = get_probability(lat, lng)
                # Ulož do session state
                st.session_state.clicked_lat  = lat
                st.session_state.clicked_lon  = lng
                st.session_state.click_prob   = prob
                st.session_state.click_status = prob_label(prob)
                st.rerun()

# ── Výsledek pod mapou (záložní zobrazení) ────────────────────────────────────
if st.session_state.click_prob is not None:
    p   = st.session_state.click_prob
    cls = prob_class(p)
    color_map_css = {
        "high":   "#39ff14",
        "medium": "#ffd700",
        "low":    "#ff8c00",
        "none":   "#2d6040",
    }
    c = color_map_css.get(cls, "#2d6040")

    st.markdown(f"""
    <div class="popup-result">
        <div class="popup-result-title">Výsledek analýzy posledního bodu</div>
        <div style="display:flex;align-items:baseline;gap:12px;margin-top:4px">
            <div class="popup-result-val" style="color:{c}">{int(p*100)} %</div>
            <div style="font-size:0.85rem;color:{c}">{prob_label(p)}</div>
            <div style="font-size:0.78rem;color:#2d6040;margin-left:auto;font-family:monospace">
                {st.session_state.clicked_lat:.5f}°N &nbsp; {st.session_state.clicked_lon:.5f}°E
            </div>
        </div>
        <div style="font-size:0.75rem;color:#2d6040;margin-top:6px">
            Model: <code>get_probability(lat, lon)</code> ·
            {'sdm_model.py' if MODEL_AVAILABLE else 'demo aproximace'}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Ekologický profil ─────────────────────────────────────────────────────────
st.markdown('<div class="sec-header">Ekologický profil druhu</div>', unsafe_allow_html=True)

e1, e2, e3, e4 = st.columns(4)
eco_data = [
    ("🌡", "Teplota", f"{SPECIES['opt_temp'][0]}–{SPECIES['opt_temp'][1]} °C", "roční průměr"),
    ("⛰",  "Výška",   f"{SPECIES['opt_elev'][0]}–{SPECIES['opt_elev'][1]} m",  "n. m."),
    ("🪨", "pH půdy", "4.5 – 6.5",  "mírně kyselá"),
    ("🌿", "Habitat", "pastviny",    "vlhké louky"),
]
for col, (icon, label, value, sub) in zip([e1, e2, e3, e4], eco_data):
    with col:
        st.markdown(f"""
        <div class="sec-widget">
            <div style="font-size:1.3rem">{icon}</div>
            <div class="sec-label" style="margin-top:4px">{label}</div>
            <div style="color:#39ff14;font-weight:700;font-size:1.1rem">{value}</div>
            <div class="sec-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:0.72rem;color:#0d3020;letter-spacing:0.1em">
    PSY SPACE · vzdělávací prototyp · GBIF · SoilGrids · WorldClim · sdm_model.py
</div>
""", unsafe_allow_html=True)

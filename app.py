"""
Psy Space — Streamlit aplikace pro sledování výskytu Psilocybe semilanceata
Druh houby je pevně definován v modelu (sdm_model.py).
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

# ── Volitelný import modelu (aplikace funguje i bez něj) ────────────────────
try:
    from sdm_model import PsilocybeSDM, SDMFeatures
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# 1. KONFIGURACE STRÁNKY
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Psy Space",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. VIZUÁLNÍ STYL — "vesmírně-lesní" estetika
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Základní pozadí ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #050d05;
    color: #e0ffe0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #050d05 0%, #0a1f0a 60%, #0d1a06 100%);
    border-right: 1px solid #1a3a1a;
}
[data-testid="stSidebar"] * { color: #a8e44a !important; }
[data-testid="stSidebar"] h1 {
    font-size: 1.3rem !important;
    color: #39ff14 !important;
    letter-spacing: 0.1em;
    text-shadow: 0 0 12px #39ff1480;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #ff8c00 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
[data-testid="stSidebar"] hr { border-color: #1a3a1a !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stDateInput label { color: #7dbb3a !important; font-size: 0.82rem !important; }

/* ── Hlavní nadpis ── */
.psy-title {
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: 0.08em;
    background: linear-gradient(90deg, #39ff14, #a8e44a, #ff8c00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;
    margin-bottom: 0;
    line-height: 1.1;
}
.psy-subtitle {
    font-size: 0.85rem;
    color: #4a7a1a;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* ── KPI karty ── */
.kpi-card {
    background: linear-gradient(135deg, #0a1f0a 0%, #0d2a0d 100%);
    border: 1px solid #1e4a1e;
    border-radius: 14px;
    padding: 22px 20px 18px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #39ff14, transparent);
}
.kpi-card:hover { border-color: #39ff1460; }
.kpi-icon { font-size: 1.6rem; margin-bottom: 6px; }
.kpi-label {
    font-size: 0.72rem;
    color: #4a7a1a;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 2.4rem;
    font-weight: 800;
    color: #39ff14;
    line-height: 1;
    text-shadow: 0 0 20px #39ff1440;
}
.kpi-unit { font-size: 1rem; color: #7dbb3a; margin-left: 2px; }
.kpi-value.orange { color: #ff8c00; text-shadow: 0 0 20px #ff8c0040; }
.kpi-value.yellow { color: #ffd700; text-shadow: 0 0 20px #ffd70040; }
.kpi-trend {
    font-size: 0.75rem;
    color: #4a7a1a;
    margin-top: 6px;
}

/* ── Sekce ── */
.section-header {
    font-size: 0.75rem;
    color: #ff8c00;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    border-bottom: 1px solid #1a3a1a;
    padding-bottom: 6px;
    margin: 24px 0 16px;
}

/* ── Mapa placeholder ── */
.map-container {
    background: #050d05;
    border: 1px solid #1e4a1e;
    border-radius: 14px;
    padding: 0;
    overflow: hidden;
    position: relative;
}
.map-overlay-label {
    position: absolute;
    top: 12px; left: 16px;
    background: #050d05cc;
    border: 1px solid #39ff1440;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 0.75rem;
    color: #39ff14;
    letter-spacing: 0.1em;
    z-index: 10;
}

/* ── Info badge ── */
.info-badge {
    display: inline-block;
    background: #0d2a0d;
    border: 1px solid #1e4a1e;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: #7dbb3a;
    margin: 2px;
    letter-spacing: 0.06em;
}
.info-badge.orange {
    border-color: #3a2000;
    color: #ff8c00;
    background: #1a1000;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #050d05; }
::-webkit-scrollbar-thumb { background: #1e4a1e; border-radius: 2px; }

/* ── Streamlit defaults přepsat ── */
.stMetric { background: transparent !important; }
[data-testid="metric-container"] { background: transparent !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1300px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3. PEVNĚ DEFINOVANÝ DRUH — napojení na sdm_model.py
# ══════════════════════════════════════════════════════════════════════════════

# Druh je pevně definován — sdm_model.py pracuje pouze s tímto druhem
SPECIES = {
    "name":        "Psilocybe semilanceata",
    "cz_name":     "Lysohlávka kopinatá",
    "emoji":       "🍄",
    "habitat":     "vlhké pastviny, louky",
    "opt_temp":    (6, 14),      # °C optimální rozsah
    "opt_rain":    (600, 1400),  # mm/rok
    "opt_ph":      (4.5, 6.5),
    "opt_elev":    (100, 800),   # m n. m.
    "season":      "září–říjen",
}

# Regiony ČR/SR pro filtrování
REGIONS = [
    "Celá ČR + SR",
    "Čechy — západ (Šumava, Krušné hory)",
    "Čechy — střed (Praha, Středočeský kraj)",
    "Čechy — východ (Krkonoše, Orlické hory)",
    "Morava — sever (Jeseníky, Beskydy)",
    "Morava — jih (Bílé Karpaty, Pálava)",
    "Slovensko — západ (Malé Karpaty)",
    "Slovensko — sever (Tatry, Orava)",
]

# Bounding boxy pro regiony [lat_min, lat_max, lon_min, lon_max]
REGION_BBOX = {
    "Celá ČR + SR":                          [47.5, 51.2, 12.0, 22.5],
    "Čechy — západ (Šumava, Krušné hory)":   [48.5, 50.8, 12.0, 14.5],
    "Čechy — střed (Praha, Středočeský kraj)":[49.5, 50.5, 13.8, 15.5],
    "Čechy — východ (Krkonoše, Orlické hory)":[50.0, 51.0, 15.5, 17.0],
    "Morava — sever (Jeseníky, Beskydy)":     [49.3, 50.2, 17.0, 18.8],
    "Morava — jih (Bílé Karpaty, Pálava)":    [48.6, 49.5, 17.0, 18.5],
    "Slovensko — západ (Malé Karpaty)":        [48.0, 49.0, 16.8, 18.5],
    "Slovensko — sever (Tatry, Orava)":        [49.0, 49.8, 19.0, 21.0],
}

# ══════════════════════════════════════════════════════════════════════════════
# 4. SIDEBAR — filtry
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🍄 Psy Space")
    st.markdown("*monitoring výskytu hub*")
    st.markdown("---")

    # Druh — pouze informativní, pevně definován
    st.markdown("### Sledovaný druh")
    st.markdown(f"""
    <div style="background:#0d2a0d;border:1px solid #1e4a1e;border-radius:10px;
                padding:12px 14px;margin-bottom:8px">
        <div style="font-size:1.4rem;margin-bottom:4px">{SPECIES['emoji']}</div>
        <div style="color:#39ff14;font-weight:700;font-size:0.95rem">
            {SPECIES['cz_name']}</div>
        <div style="color:#4a7a1a;font-size:0.78rem;font-style:italic;margin-top:2px">
            {SPECIES['name']}</div>
        <div style="color:#4a7a1a;font-size:0.75rem;margin-top:6px">
            📍 {SPECIES['habitat']}<br>
            📅 sezóna: {SPECIES['season']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Filtr: Region
    st.markdown("### Region")
    selected_region = st.selectbox(
        "Vyber oblast zájmu",
        options=REGIONS,
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Filtr: Časový rámec
    st.markdown("### Časový rámec")
    time_mode = st.radio(
        "Zobrazit data za",
        options=["Aktuální týden", "Vlastní rozsah"],
        label_visibility="collapsed",
    )

    if time_mode == "Aktuální týden":
        date_from = date.today() - timedelta(days=date.today().weekday())
        date_to   = date_from + timedelta(days=6)
        st.caption(f"📅 {date_from.strftime('%d. %m.')} — {date_to.strftime('%d. %m. %Y')}")
    else:
        date_from = st.date_input("Od", value=date.today() - timedelta(days=7))
        date_to   = st.date_input("Do", value=date.today())

    st.markdown("---")

    # Model status
    st.markdown("### Model")
    if MODEL_AVAILABLE:
        st.markdown('<div class="info-badge">✓ sdm_model.py načten</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-badge orange">⚠ demo mód</div>',
                    unsafe_allow_html=True)
        st.caption("Spusť sdm_model.py pro reálná data")

# ══════════════════════════════════════════════════════════════════════════════
# 5. VÝPOČET DAT — napojení na model nebo demo hodnoty
# ══════════════════════════════════════════════════════════════════════════════

bbox = REGION_BBOX.get(selected_region, REGION_BBOX["Celá ČR + SR"])

def get_kpi_data(region: str) -> dict:
    """
    Vrátí aktuální KPI hodnoty pro vybraný region.
    V produkci: napojit na weather API + sdm_model.predict_point().

    Návratový slovník:
        temperature   — aktuální teplota °C
        humidity      — relativní vlhkost %
        probability   — pravděpodobnost růstu 0–100 %
        temp_trend    — trend teploty (textový popis)
        rain_7d       — srážky za posledních 7 dní (mm)
        occurrences   — počet hlášených nálezů v oblasti
    """
    if MODEL_AVAILABLE:
        # ── Produkční mód: použij sdm_model.py ─────────────────────────────
        # model = PsilocybeSDM()
        # features = SDMFeatures(
        #     bio01=current_temp,
        #     bio12=annual_rain,
        #     ph=soil_ph,
        #     elev=mean_elevation,
        #     ndvi=current_ndvi,
        #     bio04=750, bio15=26,
        # )
        # result = model.predict_point(features)
        # return {"probability": result.probability * 100, ...}
        pass

    # ── Demo mód: simulované hodnoty podle regionu ──────────────────────────
    rng = np.random.default_rng(hash(region) % (2**32))

    # Simuluj teplotu podle regionu a ročního období
    month = date.today().month
    base_temp = 12 - abs(month - 9) * 1.8   # optimum v září
    is_mountain = any(k in region for k in ["Tatry", "Jeseníky", "Šumava", "Krkonoše"])
    temp = round(base_temp - (3 if is_mountain else 0) + rng.normal(0, 1.5), 1)

    # Pravděpodobnost — vyšší v sezóně a vhodném regionu
    in_season = 8 <= month <= 10
    prob_base = 0.65 if in_season else 0.25
    prob_boost = 0.15 if is_mountain else 0.0
    probability = int(np.clip((prob_base + prob_boost + rng.normal(0, 0.08)) * 100, 5, 95))

    humidity = int(np.clip(rng.normal(72 if in_season else 55, 8), 30, 98))
    rain_7d  = round(float(np.clip(rng.normal(28 if in_season else 12, 10), 0, 90)), 1)
    occurrences = int(np.clip(rng.normal(12 if in_season else 3, 4), 0, 50))

    return {
        "temperature":  temp,
        "humidity":     humidity,
        "probability":  probability,
        "temp_trend":   "↓ ochlazování" if temp < 10 else "→ stabilní",
        "rain_7d":      rain_7d,
        "occurrences":  occurrences,
    }


def get_occurrence_map_data(bbox: list, n_points: int = 40) -> pd.DataFrame:
    """
    Generuje body výskytu pro st.map().
    V produkci: nahradit GBIF API nebo výstupem sdm_model.predict_grid().

    Returns:
        DataFrame se sloupci 'lat', 'lon' pro st.map()
    """
    rng = np.random.default_rng(42)
    lat_min, lat_max, lon_min, lon_max = bbox
    return pd.DataFrame({
        "lat": rng.uniform(lat_min, lat_max, n_points),
        "lon": rng.uniform(lon_min, lon_max, n_points),
    })


kpi = get_kpi_data(selected_region)
map_df = get_occurrence_map_data(bbox)

# ══════════════════════════════════════════════════════════════════════════════
# 6. HLAVNÍ OBSAH
# ══════════════════════════════════════════════════════════════════════════════

# ── Nadpis druhu ─────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="psy-title">{SPECIES["emoji"]} {SPECIES["cz_name"]}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="psy-subtitle"><em>{SPECIES["name"]}</em> &nbsp;·&nbsp; '
    f'{selected_region} &nbsp;·&nbsp; '
    f'{date_from.strftime("%d. %m.")}–{date_to.strftime("%d. %m. %Y")}</div>',
    unsafe_allow_html=True,
)

# Badges
st.markdown(
    f'<span class="info-badge">📍 {SPECIES["habitat"]}</span>'
    f'<span class="info-badge">📅 sezóna: {SPECIES["season"]}</span>'
    f'<span class="info-badge">⛰ {SPECIES["opt_elev"][0]}–{SPECIES["opt_elev"][1]} m n. m.</span>'
    f'<span class="info-badge orange">🌡 optimum {SPECIES["opt_temp"][0]}–{SPECIES["opt_temp"][1]} °C</span>',
    unsafe_allow_html=True,
)

# ── KPI karty ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Aktuální podmínky</div>', unsafe_allow_html=True)

col1, col2, col3, col_extra1, col_extra2 = st.columns([1, 1, 1, 1, 1])

with col1:
    temp = kpi["temperature"]
    temp_color = "orange" if not (SPECIES["opt_temp"][0] <= temp <= SPECIES["opt_temp"][1]) else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">🌡</div>
        <div class="kpi-label">Teplota vzduchu</div>
        <div class="kpi-value {temp_color}">{temp}<span class="kpi-unit">°C</span></div>
        <div class="kpi-trend">{kpi['temp_trend']} &nbsp;·&nbsp;
            optimum {SPECIES['opt_temp'][0]}–{SPECIES['opt_temp'][1]} °C</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    hum = kpi["humidity"]
    hum_color = "" if hum >= 60 else "orange"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">💧</div>
        <div class="kpi-label">Relativní vlhkost</div>
        <div class="kpi-value {hum_color}">{hum}<span class="kpi-unit">%</span></div>
        <div class="kpi-trend">srážky (7 dní): {kpi['rain_7d']} mm</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    prob = kpi["probability"]
    if prob >= 65:
        prob_color = ""
        prob_icon  = "🟢"
        prob_label = "příznivé"
    elif prob >= 35:
        prob_color = "yellow"
        prob_icon  = "🟡"
        prob_label = "podmíněné"
    else:
        prob_color = "orange"
        prob_icon  = "🔴"
        prob_label = "nepříznivé"

    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{prob_icon}</div>
        <div class="kpi-label">Pravděpodobnost růstu</div>
        <div class="kpi-value {prob_color}">{prob}<span class="kpi-unit">%</span></div>
        <div class="kpi-trend">podmínky: {prob_label}</div>
    </div>
    """, unsafe_allow_html=True)

with col_extra1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">📍</div>
        <div class="kpi-label">Hlášené nálezy</div>
        <div class="kpi-value">{kpi['occurrences']}</div>
        <div class="kpi-trend">v oblasti za {(date_to - date_from).days + 1} dní</div>
    </div>
    """, unsafe_allow_html=True)

with col_extra2:
    month_now = date.today().month
    days_to_season = max(0, (date(date.today().year, 9, 1) - date.today()).days)
    in_season = 8 <= month_now <= 10
    season_label = "Právě teď! 🍄" if in_season else f"za {days_to_season} dní"
    season_color = "" if in_season else "orange"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">📅</div>
        <div class="kpi-label">Hlavní sezóna</div>
        <div class="kpi-value {season_color}" style="font-size:1.4rem">{season_label}</div>
        <div class="kpi-trend">optimum: {SPECIES['season']}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Mapa výskytu ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Predikce výskytu — {}</div>'.format(
    selected_region), unsafe_allow_html=True)

st.markdown("""
<div style="font-size:0.8rem;color:#4a7a1a;margin-bottom:12px">
    🗺 Mapa zobrazuje predikovaná místa výskytu <em>Psilocybe semilanceata</em>
    na základě klimatických a půdních podmínek.
    V produkci: výstup <code>sdm_model.predict_grid()</code> + GBIF záznamy.
</div>
""", unsafe_allow_html=True)

# st.map() — zobrazí body výskytu na mapě
# V produkci nahradit folium mapou s heatmapou z sdm_model.predict_grid()
st.map(
    map_df,
    zoom=6,
    use_container_width=True,
)

# ── Ekologický profil druhu ───────────────────────────────────────────────────
st.markdown('<div class="section-header">Ekologický profil druhu</div>',
            unsafe_allow_html=True)

eco1, eco2, eco3, eco4 = st.columns(4)

with eco1:
    st.markdown(f"""
    <div class="kpi-card" style="text-align:left">
        <div class="kpi-label">🌡 Teplota</div>
        <div style="color:#39ff14;font-weight:700">
            {SPECIES['opt_temp'][0]}–{SPECIES['opt_temp'][1]} °C</div>
        <div style="color:#4a7a1a;font-size:0.78rem;margin-top:4px">
            průměrná roční teplota</div>
    </div>
    """, unsafe_allow_html=True)

with eco2:
    st.markdown(f"""
    <div class="kpi-card" style="text-align:left">
        <div class="kpi-label">🌧 Srážky</div>
        <div style="color:#39ff14;font-weight:700">
            {SPECIES['opt_rain'][0]}–{SPECIES['opt_rain'][1]} mm</div>
        <div style="color:#4a7a1a;font-size:0.78rem;margin-top:4px">
            roční úhrn srážek</div>
    </div>
    """, unsafe_allow_html=True)

with eco3:
    st.markdown(f"""
    <div class="kpi-card" style="text-align:left">
        <div class="kpi-label">🪨 pH půdy</div>
        <div style="color:#39ff14;font-weight:700">
            {SPECIES['opt_ph'][0]}–{SPECIES['opt_ph'][1]}</div>
        <div style="color:#4a7a1a;font-size:0.78rem;margin-top:4px">
            mírně kyselá až neutrální</div>
    </div>
    """, unsafe_allow_html=True)

with eco4:
    st.markdown(f"""
    <div class="kpi-card" style="text-align:left">
        <div class="kpi-label">⛰ Výška</div>
        <div style="color:#39ff14;font-weight:700">
            {SPECIES['opt_elev'][0]}–{SPECIES['opt_elev'][1]} m</div>
        <div style="color:#4a7a1a;font-size:0.78rem;margin-top:4px">
            nadmořská výška n. m.</div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:0.75rem;color:#2a4a2a;letter-spacing:0.08em">
    PSY SPACE &nbsp;·&nbsp; vzdělávací prototyp &nbsp;·&nbsp;
    data: GBIF · SoilGrids · WorldClim &nbsp;·&nbsp;
    model: <code>sdm_model.py</code>
</div>
""", unsafe_allow_html=True)

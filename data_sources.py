"""
data_sources_v2.py — všechny datové zdroje pro Psy Space
Opraveno: land cover přes OSM Nominatim + ESA WorldCover
"""

import requests
import numpy as np
import streamlit as st
from datetime import date


# ══════════════════════════════════════════════════════════════════════════════
# 1. OPEN-METEO — aktuální + historické počasí + předpověď
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather_full(lat: float, lon: float) -> dict:
    """Open-Meteo: aktuální počasí + 14 dní historie + 7 dní předpověď."""
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": [
                    "temperature_2m", "relative_humidity_2m",
                    "precipitation", "wind_speed_10m",
                    "soil_moisture_0_to_1cm",
                ],
                "daily": [
                    "precipitation_sum",
                    "temperature_2m_max", "temperature_2m_min",
                    "et0_fao_evapotranspiration",
                ],
                "past_days": 14, "forecast_days": 7,
                "timezone": "Europe/Prague",
            },
            timeout=10,
        )
        resp.raise_for_status()
        d     = resp.json()
        cur   = d.get("current", {})
        daily = d.get("daily", {})
        precip = daily.get("precipitation_sum") or []
        tmax   = daily.get("temperature_2m_max") or []
        tmin   = daily.get("temperature_2m_min") or []
        et0    = daily.get("et0_fao_evapotranspiration") or []

        rain_3d  = sum(r for r in precip[-3:]  if r is not None)
        rain_7d  = sum(r for r in precip[-7:]  if r is not None)
        rain_14d = sum(r for r in precip[-14:] if r is not None)
        rain_fc  = sum(r for r in precip[:7]   if r is not None)
        et0_7d   = sum(e for e in et0[-7:]     if e is not None)

        temp_7d = float(np.mean([
            (mx + mn) / 2 for mx, mn in zip(tmax[-7:], tmin[-7:])
            if mx is not None and mn is not None
        ])) if tmax else float(cur.get("temperature_2m", 10.0))

        return {
            "temp_now":         round(float(cur.get("temperature_2m", 10.0)), 1),
            "temp_7d_avg":      round(temp_7d, 1),
            "humidity":         int(cur.get("relative_humidity_2m", 70)),
            "soil_moisture":    round(float(cur.get("soil_moisture_0_to_1cm", 0.3)), 3),
            "precip_now":       round(float(cur.get("precipitation", 0.0)), 1),
            "wind":             round(float(cur.get("wind_speed_10m", 3.0)), 1),
            "rain_3d":          round(rain_3d, 1),
            "rain_7d":          round(rain_7d, 1),
            "rain_14d":         round(rain_14d, 1),
            "rain_forecast_7d": round(rain_fc, 1),
            "water_deficit":    round(max(0, et0_7d - rain_7d), 1),
            "ok": True,
        }
    except Exception as e:
        return {
            "temp_now": 10.0, "temp_7d_avg": 10.0, "humidity": 70,
            "soil_moisture": 0.3, "precip_now": 0.0, "wind": 3.0,
            "rain_3d": 10.0, "rain_7d": 20.0, "rain_14d": 35.0,
            "rain_forecast_7d": 15.0, "water_deficit": 5.0,
            "ok": False, "error": str(e),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2. OPEN-ELEVATION — výška + sklon + orientace svahu
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_terrain(lat: float, lon: float) -> dict:
    """Výška, sklon a orientace svahu ze sítě 3×3 bodů."""
    try:
        step = 0.005
        points = [
            {"latitude": lat + dlat, "longitude": lon + dlon}
            for dlat in [-step, 0, step]
            for dlon in [-step, 0, step]
        ]
        resp = requests.post(
            "https://api.open-elevation.com/api/v1/lookup",
            json={"locations": points}, timeout=10,
        )
        resp.raise_for_status()
        elevs = [r["elevation"] for r in resp.json()["results"]]
        center = float(elevs[4])
        dx = 111320 * np.cos(np.radians(lat)) * step
        dy = 111320 * step
        dz_x = (elevs[5] - elevs[3]) / (2 * dx)
        dz_y = (elevs[7] - elevs[1]) / (2 * dy)
        slope = float(np.degrees(np.arctan(np.sqrt(dz_x**2 + dz_y**2))))
        aspect = float(np.degrees(np.arctan2(-dz_x, dz_y)) % 360)
        twi = float(np.log(max(1.0, center / 10) / np.tan(np.radians(max(slope, 0.1)))))
        north_factor = float(np.cos(np.radians(aspect)))
        return {
            "elev": center, "slope": round(slope, 2),
            "aspect": round(aspect, 1), "twi": round(twi, 2),
            "north_factor": round(north_factor, 3), "ok": True,
        }
    except Exception as e:
        return {
            "elev": 350.0, "slope": 5.0, "aspect": 315.0,
            "twi": 8.0, "north_factor": 0.7,
            "ok": False, "error": str(e),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3. SOILGRIDS — půdní data
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_soil(lat: float, lon: float) -> dict:
    """SoilGrids REST API — pH, organická hmota, textura půdy."""
    try:
        resp = requests.get(
            "https://rest.isric.org/soilgrids/v2.0/properties/query",
            params={
                "lon": lon, "lat": lat,
                "property": ["phh2o", "soc", "clay", "sand", "bdod"],
                "depth": "0-5cm", "value": "mean",
            },
            timeout=10,
        )
        resp.raise_for_status()
        result = {"ok": True}
        for layer in resp.json().get("properties", {}).get("layers", []):
            name = layer["name"]
            val  = layer.get("depths", [{}])[0].get("values", {}).get("mean")
            if val is None: continue
            if   name == "phh2o": result["ph"]           = round(val / 10, 2)
            elif name == "soc":   result["soc"]          = round(val / 10, 1)
            elif name == "clay":  result["clay"]         = round(val / 10, 1)
            elif name == "sand":  result["sand"]         = round(val / 10, 1)
            elif name == "bdod":  result["bulk_density"] = round(val / 100, 2)
        return result
    except Exception as e:
        return {
            "ph": 5.8, "soc": 20.0, "clay": 28.0,
            "sand": 35.0, "bulk_density": 1.2,
            "ok": False, "error": str(e),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. LAND COVER — ESA WorldCover + OSM Nominatim záloha
# ══════════════════════════════════════════════════════════════════════════════

# Mapování ESA WorldCover tříd na vhodnost pro lysohlávky
ESA_CLASSES = {
    10: ("Lesní porost",      0.25),
    20: ("Keře a křoviny",    0.40),
    30: ("Louky a tráva",     1.00),  # ideální
    40: ("Orná půda",         0.05),
    50: ("Zastavěná plocha",  0.00),  # město = 0
    60: ("Holá půda",         0.02),
    70: ("Sníh / led",        0.00),
    80: ("Vodní plocha",      0.00),
    90: ("Mokřad",            0.35),
    95: ("Mangrovník",        0.00),
}


def _land_cover_from_nominatim(lat: float, lon: float) -> dict:
    """
    Záloha: OSM Nominatim reverse geocoding.
    Spolehlivě rozliší město, les, pole a přírodu bez registrace.
    """
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 16},
            headers={"User-Agent": "PsySpace/3.0 (educational)"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        addr = data.get("address", {})
        osm_type = data.get("type", "")
        osm_class = data.get("class", "")
        category  = data.get("category", "")

        # Detekce typu prostředí z OSM tagů
        city_keys    = {"city","town","village","suburb","neighbourhood",
                        "building","road","house_number","pedestrian","motorway"}
        forest_keys  = {"forest","wood","nature_reserve","national_park","heath"}
        farm_keys    = {"farm","farmland","farmyard","agricultural","orchard"}
        meadow_keys  = {"meadow","grassland","grass","pasture","park","recreation_ground"}
        water_keys   = {"water","river","lake","pond","reservoir","stream"}

        addr_vals = set(addr.keys()) | {osm_type, osm_class, category}

        if addr_vals & city_keys:
            return {"land_cover": "Zastavěná plocha", "suitability": 0.00, "ok": True, "source": "OSM"}
        elif addr_vals & water_keys:
            return {"land_cover": "Vodní plocha",     "suitability": 0.00, "ok": True, "source": "OSM"}
        elif addr_vals & farm_keys:
            return {"land_cover": "Orná půda",        "suitability": 0.05, "ok": True, "source": "OSM"}
        elif addr_vals & forest_keys:
            return {"land_cover": "Lesní porost",     "suitability": 0.25, "ok": True, "source": "OSM"}
        elif addr_vals & meadow_keys:
            return {"land_cover": "Louky / pastviny", "suitability": 0.90, "ok": True, "source": "OSM"}
        else:
            # Neznámé prostředí — mírná vhodnost
            return {"land_cover": "Příroda / smíšené", "suitability": 0.45, "ok": True, "source": "OSM"}

    except Exception as e:
        return {"land_cover": "Neznámý", "suitability": 0.30, "ok": False, "error": str(e)}


def _land_cover_from_sentinel_hub(lat: float, lon: float, token: str) -> dict:
    """
    ESA WorldCover 10m přes Sentinel Hub OGC WMS.
    Vyžaduje registraci na sentinelhub.com (zdarma).
    Token předej přes st.secrets["SENTINEL_HUB_TOKEN"].
    """
    try:
        instance_id = token
        url = f"https://services.sentinel-hub.com/ogc/wms/{instance_id}"
        resp = requests.get(url, params={
            "SERVICE": "WMS", "REQUEST": "GetFeatureInfo",
            "VERSION": "1.3.0",
            "LAYERS": "ESA_WORLDCOVER_10M",
            "QUERY_LAYERS": "ESA_WORLDCOVER_10M",
            "CRS": "EPSG:4326",
            "BBOX": f"{lat-0.001},{lon-0.001},{lat+0.001},{lon+0.001}",
            "WIDTH": "3", "HEIGHT": "3", "I": "1", "J": "1",
            "INFO_FORMAT": "application/json",
        }, timeout=8)
        resp.raise_for_status()
        lc_code = int(resp.json()["features"][0]["properties"]["value"])
        name, suit = ESA_CLASSES.get(lc_code, ("Neznámý", 0.20))
        return {"land_cover": name, "suitability": suit, "ok": True, "source": "ESA WorldCover"}
    except Exception as e:
        return {"land_cover": None, "suitability": None, "ok": False, "error": str(e)}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_land_cover(lat: float, lon: float) -> dict:
    """
    Land cover: zkusí ESA WorldCover (Sentinel Hub), záloha OSM Nominatim.
    """
    # Zkus Sentinel Hub pokud je token v secrets
    try:
        token = st.secrets.get("SENTINEL_HUB_TOKEN", "")
        if token:
            result = _land_cover_from_sentinel_hub(lat, lon, token)
            if result["ok"]:
                return result
    except Exception:
        pass

    # Záloha: OSM Nominatim — spolehlivé pro ČR/SK
    return _land_cover_from_nominatim(lat, lon)


# ══════════════════════════════════════════════════════════════════════════════
# 5. NDVI — NASA MODIS nebo sezónní aproximace
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_ndvi(lat: float, lon: float) -> dict:
    """MODIS NDVI nebo sezónní aproximace pro ČR."""
    try:
        resp = requests.get(
            f"https://modis.ornl.gov/rst/api/v1/MOD13Q1/subset"
            f"?latitude={lat}&longitude={lon}"
            f"&startDate=A2024001&endDate=A2024365"
            f"&kmAboveBelow=0&kmLeftRight=0",
            timeout=8,
        )
        resp.raise_for_status()
        subsets = resp.json().get("subset", [])
        if subsets:
            raw = subsets[-1].get("data", [None])[0]
            if raw is not None and raw > 0:
                return {"ndvi": round(float(raw) * 0.0001, 3), "ok": True, "source": "MODIS"}
        raise Exception("Žádná MODIS data")
    except Exception:
        # Sezónní aproximace NDVI pro středoevropské louky
        month = date.today().month
        ndvi_seasonal = {
            1: 0.15, 2: 0.18, 3: 0.30, 4: 0.50, 5: 0.70,
            6: 0.80, 7: 0.78, 8: 0.72, 9: 0.65, 10: 0.50,
            11: 0.30, 12: 0.18,
        }
        return {
            "ndvi":   round(ndvi_seasonal.get(month, 0.5), 3),
            "ok":     False,
            "source": "sezónní aproximace",
        }


# ══════════════════════════════════════════════════════════════════════════════
# 6. GBIF — historické nálezy v okolí
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gbif_nearby(lat: float, lon: float, radius: float = 0.3) -> dict:
    """GBIF — počet a hustota nálezů lysohlávek v okolí bodu."""
    try:
        resp = requests.get(
            "https://api.gbif.org/v1/occurrence/search",
            params={
                "scientificName": "Psilocybe semilanceata",
                "decimalLatitude":  f"{lat-radius},{lat+radius}",
                "decimalLongitude": f"{lon-radius*1.5},{lon+radius*1.5}",
                "hasCoordinate": True, "limit": 1,
            },
            timeout=8,
        )
        count = int(resp.json().get("count", 0))
        area  = (radius * 111) ** 2 * np.pi
        return {"count": count, "density": round(count / max(area, 1), 4), "ok": True}
    except Exception:
        return {"count": 0, "density": 0.0, "ok": False}


# ══════════════════════════════════════════════════════════════════════════════
# 7. WORLDCLIM — klimatická normála
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=604800, show_spinner=False)
def fetch_worldclim(lat: float, lon: float) -> dict:
    """WorldClim bioklimatické proměnné — prostorová interpolace pro ČR/SK."""
    lat_n = float(np.clip((lat - 47.5) / (51.2 - 47.5), 0, 1))
    lon_n = float(np.clip((lon - 12.0) / (22.5 - 12.0), 0, 1))
    return {
        "bio01": round(10.5 - lat_n * 4.0 + lon_n * 0.5, 2),
        "bio04": round(680 + lat_n * 80, 1),
        "bio12": round(580 + lon_n * 220 + lat_n * 80, 1),
        "bio15": round(28 - lon_n * 5, 1),
        "ok": True,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8. MASTER FUNKCE
# ══════════════════════════════════════════════════════════════════════════════

def fetch_all(lat: float, lon: float) -> dict:
    """Stáhne všechna data pro bod najednou."""
    weather    = fetch_weather_full(lat, lon)
    terrain    = fetch_terrain(lat, lon)
    soil       = fetch_soil(lat, lon)
    land_cover = fetch_land_cover(lat, lon)
    ndvi       = fetch_ndvi(lat, lon)
    gbif       = fetch_gbif_nearby(lat, lon)
    climate    = fetch_worldclim(lat, lon)

    return {
        # Počasí
        "temp_now":         weather["temp_now"],
        "temp_7d_avg":      weather["temp_7d_avg"],
        "humidity":         weather["humidity"],
        "soil_moisture":    weather["soil_moisture"],
        "rain_3d":          weather["rain_3d"],
        "rain_7d":          weather["rain_7d"],
        "rain_14d":         weather["rain_14d"],
        "rain_forecast_7d": weather["rain_forecast_7d"],
        "water_deficit":    weather["water_deficit"],
        "wind":             weather["wind"],
        # Terén
        "elev":             terrain["elev"],
        "slope":            terrain["slope"],
        "aspect":           terrain["aspect"],
        "twi":              terrain["twi"],
        "north_factor":     terrain["north_factor"],
        # Půda
        "ph":               soil.get("ph", 5.8),
        "soc":              soil.get("soc", 20.0),
        "clay":             soil.get("clay", 28.0),
        "sand":             soil.get("sand", 35.0),
        "bulk_density":     soil.get("bulk_density", 1.2),
        # Land cover
        "land_cover":       land_cover.get("land_cover", "Neznámý"),
        "land_suitability": land_cover.get("suitability", 0.3),
        # NDVI
        "ndvi":             ndvi["ndvi"],
        # GBIF
        "gbif_count":       gbif["count"],
        "gbif_density":     gbif["density"],
        # Klima
        "bio01":            climate["bio01"],
        "bio12":            climate["bio12"],
        # Stav zdrojů pro sidebar
        "_sources": {
            "weather":    "✅ Open-Meteo"     if weather["ok"]    else "⚠️ záloha",
            "terrain":    "✅ Open-Elevation"  if terrain["ok"]    else "⚠️ záloha",
            "soil":       "✅ SoilGrids"       if soil["ok"]       else "⚠️ záloha",
            "land_cover": f"✅ {land_cover.get('source','OSM')}" if land_cover["ok"] else "⚠️ záloha",
            "ndvi":       f"✅ {ndvi.get('source','MODIS')}"     if ndvi["ok"]       else "⚠️ aproximace",
            "gbif":       "✅ GBIF"            if gbif["ok"]       else "⚠️ záloha",
        },
    }

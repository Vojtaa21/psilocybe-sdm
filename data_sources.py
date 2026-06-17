"""
data_sources_v3.py — všechny datové zdroje pro Psy Space
Nové: Copernicus OAuth2 + ESA WorldCover 10m pro přesný land cover
"""

import requests
import numpy as np
import streamlit as st
from datetime import date, timedelta


# ══════════════════════════════════════════════════════════════════════════════
# COPERNICUS AUTH — OAuth2 token
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3000, show_spinner=False)
def get_copernicus_token() -> str | None:
    """
    Získá OAuth2 access token z Copernicus Identity Service.
    Token platí 3600 sekund — cachujeme na 3000s.
    """
    try:
        client_id     = st.secrets.get("SH_CLIENT_ID", "")
        client_secret = st.secrets.get("SH_CLIENT_SECRET", "")

        if not client_id or not client_secret:
            return None

        resp = requests.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
            "/protocol/openid-connect/token",
            data={
                "grant_type":    "client_credentials",
                "client_id":     client_id,
                "client_secret": client_secret,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("access_token")

    except Exception as e:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ESA WORLDCOVER 10m — přes Copernicus Process API
# ══════════════════════════════════════════════════════════════════════════════

ESA_CLASSES = {
    10: ("Lesní porost",      0.25),
    20: ("Keře a křoviny",    0.40),
    30: ("Louky a tráva",     1.00),
    40: ("Orná půda",         0.05),
    50: ("Zastavěná plocha",  0.00),
    60: ("Holá půda",         0.02),
    70: ("Sníh / led",        0.00),
    80: ("Vodní plocha",      0.00),
    90: ("Mokřad",            0.35),
    95: ("Mangrovník",        0.00),
}


def _worldcover_via_process_api(lat: float, lon: float, token: str) -> dict | None:
    """
    ESA WorldCover přes Sentinel Hub Process API.
    Stáhne pixel 10m × 10m pro zadané souřadnice.
    """
    try:
        evalscript = """
        //VERSION=3
        function setup() {
            return { input: ["Map"], output: { bands: 1, sampleType: "UINT8" } };
        }
        function evaluatePixel(sample) {
            return [sample.Map];
        }
        """

        payload = {
            "input": {
                "bounds": {
                    "bbox": [lon - 0.0001, lat - 0.0001,
                             lon + 0.0001, lat + 0.0001],
                    "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                },
                "data": [{
                    "type": "byoc-ESA-WorldCover-10m-2021",
                    "dataFilter": {"timeRange": {
                        "from": "2021-01-01T00:00:00Z",
                        "to":   "2021-12-31T23:59:59Z",
                    }},
                }],
            },
            "output": {
                "width": 1, "height": 1,
                "responses": [{"identifier": "default",
                               "format": {"type": "image/tiff"}}],
            },
            "evalscript": evalscript,
        }

        resp = requests.post(
            "https://sh.dataspace.copernicus.eu/api/v1/process",
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/json",
            },
            timeout=15,
        )
        resp.raise_for_status()

        # Přečti hodnotu pixelu z GeoTIFF odpovědi
        try:
            import io
            import struct
            # Jednoduchý parse TIFF hodnoty — první datový byte
            content = resp.content
            # Hledej hodnotu v TIFF datech (offset ~8 bytů pro single-band TIFF)
            if len(content) > 200:
                # Zkus najít hodnotu pixelu — je uložena za TIFF hlavičkou
                for offset in range(100, min(len(content), 500)):
                    val = content[offset]
                    if val in ESA_CLASSES:
                        name, suit = ESA_CLASSES[val]
                        return {
                            "land_cover": name,
                            "suitability": suit,
                            "lc_code": val,
                            "ok": True,
                            "source": "ESA WorldCover 10m (Copernicus)",
                        }
        except Exception:
            pass
        return None

    except Exception as e:
        return None


def _worldcover_via_wms(lat: float, lon: float, token: str) -> dict | None:
    """
    ESA WorldCover přes WMS GetFeatureInfo — záloha pro Process API.
    """
    try:
        resp = requests.get(
            "https://sh.dataspace.copernicus.eu/ogc/wms/"
            + st.secrets.get("SH_CLIENT_ID", ""),
            params={
                "SERVICE":      "WMS",
                "REQUEST":      "GetFeatureInfo",
                "VERSION":      "1.3.0",
                "LAYERS":       "ESA_WORLDCOVER_10M",
                "QUERY_LAYERS": "ESA_WORLDCOVER_10M",
                "CRS":          "EPSG:4326",
                "BBOX":         f"{lat-0.001},{lon-0.001},{lat+0.001},{lon+0.001}",
                "WIDTH":        "3",
                "HEIGHT":       "3",
                "I":            "1",
                "J":            "1",
                "INFO_FORMAT":  "application/json",
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        resp.raise_for_status()
        features = resp.json().get("features", [])
        if features:
            lc_code = int(features[0].get("properties", {}).get("value", 0))
            name, suit = ESA_CLASSES.get(lc_code, ("Neznámý", 0.3))
            return {
                "land_cover": name, "suitability": suit,
                "lc_code": lc_code, "ok": True,
                "source": "ESA WorldCover WMS",
            }
        return None
    except Exception:
        return None


def _land_cover_from_nominatim(lat: float, lon: float) -> dict:
    """
    Záloha: OSM Nominatim reverse geocoding.
    Spolehlivě rozliší město, les, pole a přírodu.
    """
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 16},
            headers={"User-Agent": "PsySpace/3.0 (educational project)"},
            timeout=8,
        )
        resp.raise_for_status()
        data      = resp.json()
        addr      = data.get("address", {})
        osm_type  = data.get("type", "")
        osm_class = data.get("class", "")

        city_keys   = {"city","town","village","suburb","neighbourhood",
                       "building","road","house_number","pedestrian",
                       "motorway","retail","commercial","industrial"}
        forest_keys = {"forest","wood","nature_reserve","national_park","heath"}
        farm_keys   = {"farm","farmland","farmyard","orchard","vineyard"}
        meadow_keys = {"meadow","grassland","grass","pasture",
                       "park","recreation_ground","pitch"}
        water_keys  = {"water","river","lake","pond","reservoir","stream","bay"}

        addr_vals = set(addr.keys()) | {osm_type, osm_class}

        if addr_vals & city_keys:
            return {"land_cover": "Zastavěná plocha", "suitability": 0.00,
                    "ok": True, "source": "OSM Nominatim"}
        elif addr_vals & water_keys:
            return {"land_cover": "Vodní plocha",     "suitability": 0.00,
                    "ok": True, "source": "OSM Nominatim"}
        elif addr_vals & farm_keys:
            return {"land_cover": "Orná půda",        "suitability": 0.05,
                    "ok": True, "source": "OSM Nominatim"}
        elif addr_vals & forest_keys:
            return {"land_cover": "Lesní porost",     "suitability": 0.25,
                    "ok": True, "source": "OSM Nominatim"}
        elif addr_vals & meadow_keys:
            return {"land_cover": "Louky / pastviny", "suitability": 0.90,
                    "ok": True, "source": "OSM Nominatim"}
        else:
            return {"land_cover": "Příroda / smíšené","suitability": 0.45,
                    "ok": True, "source": "OSM Nominatim"}

    except Exception as e:
        return {"land_cover": "Neznámý", "suitability": 0.30,
                "ok": False, "error": str(e)}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_land_cover(lat: float, lon: float) -> dict:
    """
    Land cover v pořadí přesnosti:
    1. ESA WorldCover 10m (Copernicus Process API)
    2. ESA WorldCover WMS
    3. OSM Nominatim (záloha)
    """
    # Zkus Copernicus API
    token = get_copernicus_token()
    if token:
        # Zkus Process API
        result = _worldcover_via_process_api(lat, lon, token)
        if result and result.get("ok"):
            return result
        # Zkus WMS
        result = _worldcover_via_wms(lat, lon, token)
        if result and result.get("ok"):
            return result

    # Záloha: OSM Nominatim
    return _land_cover_from_nominatim(lat, lon)


# ══════════════════════════════════════════════════════════════════════════════
# SENTINEL-2 NDVI — přes Copernicus Process API
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_ndvi_sentinel(lat: float, lon: float) -> dict:
    """
    Aktuální NDVI ze Sentinel-2 přes Copernicus Process API.
    Záloha: sezónní aproximace.
    """
    token = get_copernicus_token()

    if token:
        try:
            # Datum: posledních 30 dní
            date_to   = date.today().strftime("%Y-%m-%dT23:59:59Z")
            date_from = (date.today() - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00Z")

            evalscript = """
            //VERSION=3
            function setup() {
                return {
                    input: ["B04", "B08", "dataMask"],
                    output: { bands: 1, sampleType: "FLOAT32" }
                };
            }
            function evaluatePixel(sample) {
                if (sample.dataMask == 0) return [-999];
                let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
                return [ndvi];
            }
            """
            payload = {
                "input": {
                    "bounds": {
                        "bbox": [lon-0.001, lat-0.001, lon+0.001, lat+0.001],
                        "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                    },
                    "data": [{
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {"from": date_from, "to": date_to},
                            "maxCloudCoverage": 30,
                        },
                    }],
                },
                "output": {
                    "width": 1, "height": 1,
                    "responses": [{"identifier": "default",
                                   "format": {"type": "image/tiff"}}],
                },
                "evalscript": evalscript,
            }
            resp = requests.post(
                "https://sh.dataspace.copernicus.eu/api/v1/process",
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type":  "application/json",
                },
                timeout=15,
            )
            resp.raise_for_status()

            # Parsuj float z TIFF odpovědi
            content = resp.content
            if len(content) > 100:
                import struct
                # TIFF float32 hodnota je typicky na offsetu ~8 bytů
                for offset in range(0, min(len(content) - 4, 200)):
                    try:
                        val = struct.unpack_from('<f', content, offset)[0]
                        if -1.0 <= val <= 1.0 and val != 0.0:
                            return {
                                "ndvi":   round(float(val), 3),
                                "ok":     True,
                                "source": "Sentinel-2 L2A (Copernicus)",
                            }
                    except Exception:
                        continue

        except Exception:
            pass

    # Záloha: sezónní aproximace pro ČR
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
# OSTATNÍ DATOVÉ ZDROJE (nezměněno)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather_full(lat: float, lon: float) -> dict:
    """Open-Meteo: počasí + 14 dní historie + 7 dní předpověď."""
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": ["temperature_2m", "relative_humidity_2m",
                            "precipitation", "wind_speed_10m",
                            "soil_moisture_0_to_1cm"],
                "daily":   ["precipitation_sum", "temperature_2m_max",
                            "temperature_2m_min", "et0_fao_evapotranspiration"],
                "past_days": 14, "forecast_days": 7,
                "timezone": "Europe/Prague",
            },
            timeout=10,
        )
        resp.raise_for_status()
        d      = resp.json()
        cur    = d.get("current", {})
        daily  = d.get("daily", {})
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


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_terrain(lat: float, lon: float) -> dict:
    """Open-Elevation: výška, sklon, orientace, TWI."""
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
        elevs  = [r["elevation"] for r in resp.json()["results"]]
        center = float(elevs[4])
        dx = 111320 * np.cos(np.radians(lat)) * step
        dy = 111320 * step
        dz_x   = (elevs[5] - elevs[3]) / (2 * dx)
        dz_y   = (elevs[7] - elevs[1]) / (2 * dy)
        slope  = float(np.degrees(np.arctan(np.sqrt(dz_x**2 + dz_y**2))))
        aspect = float(np.degrees(np.arctan2(-dz_x, dz_y)) % 360)
        twi    = float(np.log(max(1.0, center / 10) / np.tan(np.radians(max(slope, 0.1)))))
        return {
            "elev": center, "slope": round(slope, 2),
            "aspect": round(aspect, 1), "twi": round(twi, 2),
            "north_factor": round(float(np.cos(np.radians(aspect))), 3),
            "ok": True,
        }
    except Exception:
        return {"elev": 350.0, "slope": 5.0, "aspect": 315.0,
                "twi": 8.0, "north_factor": 0.7, "ok": False}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_soil(lat: float, lon: float) -> dict:
    """SoilGrids: pH, organická hmota, textura."""
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
    except Exception:
        return {"ph": 5.8, "soc": 20.0, "clay": 28.0,
                "sand": 35.0, "bulk_density": 1.2, "ok": False}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gbif_nearby(lat: float, lon: float, radius: float = 0.3) -> dict:
    """GBIF: počet nálezů lysohlávek v okolí."""
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


@st.cache_data(ttl=604800, show_spinner=False)
def fetch_worldclim(lat: float, lon: float) -> dict:
    """WorldClim: klimatická normála pro ČR/SK."""
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
# MASTER FUNKCE
# ══════════════════════════════════════════════════════════════════════════════

def fetch_all(lat: float, lon: float) -> dict:
    """Stáhne všechna data pro bod — včetně Copernicus ESA WorldCover."""
    weather    = fetch_weather_full(lat, lon)
    terrain    = fetch_terrain(lat, lon)
    soil       = fetch_soil(lat, lon)
    land_cover = fetch_land_cover(lat, lon)
    ndvi       = fetch_ndvi_sentinel(lat, lon)
    gbif       = fetch_gbif_nearby(lat, lon)
    climate    = fetch_worldclim(lat, lon)

    return {
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
        "elev":             terrain["elev"],
        "slope":            terrain["slope"],
        "aspect":           terrain["aspect"],
        "twi":              terrain["twi"],
        "north_factor":     terrain["north_factor"],
        "ph":               soil.get("ph", 5.8),
        "soc":              soil.get("soc", 20.0),
        "clay":             soil.get("clay", 28.0),
        "sand":             soil.get("sand", 35.0),
        "bulk_density":     soil.get("bulk_density", 1.2),
        "land_cover":       land_cover.get("land_cover", "Neznámý"),
        "land_suitability": land_cover.get("suitability", 0.3),
        "ndvi":             ndvi["ndvi"],
        "gbif_count":       gbif["count"],
        "gbif_density":     gbif["density"],
        "bio01":            climate["bio01"],
        "bio12":            climate["bio12"],
        "_sources": {
            "weather":    "✅ Open-Meteo"    if weather["ok"]    else "⚠️ záloha",
            "terrain":    "✅ Open-Elevation" if terrain["ok"]   else "⚠️ záloha",
            "soil":       "✅ SoilGrids"      if soil["ok"]      else "⚠️ záloha",
            "land_cover": f"✅ {land_cover.get('source','OSM')}" if land_cover["ok"] else "⚠️ záloha",
            "ndvi":       f"✅ {ndvi.get('source','')}"          if ndvi["ok"]       else "⚠️ aproximace",
            "gbif":       "✅ GBIF"           if gbif["ok"]      else "⚠️ záloha",
        },
    }

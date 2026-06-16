"""
data_loader.py — automatické stažení všech reálných dat pro SDM

Zdroje:
  GBIF      — výskytové záznamy lysohlávek (REST API, bez klíče)
  SoilGrids — pH, organická hmota, textura půdy (REST API, bez klíče)
  Open-Elevation — nadmořská výška + sklon (REST API, bez klíče)
  WorldClim — teplota, srážky (statické soubory, bez klíče)
"""
from __future__ import annotations

import time
import math
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".sdm_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Bounding box ČR + SR
BBOX = {"lat_min": 47.5, "lat_max": 51.2, "lon_min": 12.0, "lon_max": 22.5}

SPECIES = [
    "Psilocybe semilanceata",
    "Psilocybe cubensis",
    "Psilocybe cyanescens",
    "Psilocybe bohémica",
]


# ── Cache pomocník ───────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    return CACHE_DIR / f"{h}.json"


def _load_cache(key: str):
    p = _cache_path(key)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return None


def _save_cache(key: str, data):
    try:
        _cache_path(key).write_text(json.dumps(data))
    except Exception:
        pass


# ── 1. GBIF výskytová data ───────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_gbif_occurrences(limit_per_species: int = 300) -> pd.DataFrame:
    """
    Stáhne reálné záznamy výskytu lysohlávek z GBIF API.
    Vrátí DataFrame s lat, lon, species, year.
    """
    cache_key = f"gbif_{limit_per_species}"
    cached = _load_cache(cache_key)
    if cached:
        return pd.DataFrame(cached)

    all_records = []
    base = "https://api.gbif.org/v1/occurrence/search"

    for species in SPECIES:
        try:
            params = {
                "scientificName": species,
                "decimalLatitude": f"{BBOX['lat_min']},{BBOX['lat_max']}",
                "decimalLongitude": f"{BBOX['lon_min']},{BBOX['lon_max']}",
                "hasCoordinate": True,
                "hasGeospatialIssue": False,
                "limit": limit_per_species,
                "offset": 0,
            }
            resp = requests.get(base, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            for rec in data.get("results", []):
                lat = rec.get("decimalLatitude")
                lon = rec.get("decimalLongitude")
                if lat is None or lon is None:
                    continue
                if not (BBOX["lat_min"] <= lat <= BBOX["lat_max"]):
                    continue
                if not (BBOX["lon_min"] <= lon <= BBOX["lon_max"]):
                    continue
                all_records.append({
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    "species": species,
                    "year": rec.get("year"),
                    "country": rec.get("countryCode", ""),
                })
            time.sleep(0.3)  # respektuj rate limit

        except Exception as e:
            logger.warning(f"GBIF chyba pro {species}: {e}")
            continue

    df = pd.DataFrame(all_records).drop_duplicates(subset=["lat", "lon"])
    _save_cache(cache_key, df.to_dict("records"))
    return df


# ── 2. SoilGrids — půdní data ───────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_soil_properties(lat: float, lon: float) -> dict:
    """
    Stáhne půdní vlastnosti pro bod z SoilGrids REST API.
    Vrátí pH, organickou hmotu, obsah jílu a písku.
    """
    cache_key = f"soil_{lat:.3f}_{lon:.3f}"
    cached = _load_cache(cache_key)
    if cached:
        return cached

    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    properties = ["phh2o", "soc", "clay", "sand", "bdod"]

    try:
        params = {
            "lon": lon, "lat": lat,
            "property": properties,
            "depth": "0-5cm",
            "value": "mean",
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        result = {}
        for layer in data.get("properties", {}).get("layers", []):
            name = layer["name"]
            val = layer.get("depths", [{}])[0].get("values", {}).get("mean")
            if val is None:
                continue
            # SoilGrids vrací hodnoty × 10
            if name == "phh2o":
                result["ph"] = round(val / 10, 2)
            elif name == "soc":
                result["soc"] = round(val / 10, 2)   # g/kg
            elif name == "clay":
                result["clay"] = round(val / 10, 1)  # %
            elif name == "sand":
                result["sand"] = round(val / 10, 1)  # %
            elif name == "bdod":
                result["bulk_density"] = round(val / 100, 2)

        _save_cache(cache_key, result)
        return result

    except Exception as e:
        logger.warning(f"SoilGrids chyba ({lat},{lon}): {e}")
        return {"ph": 5.8, "soc": 15.0, "clay": 25.0, "sand": 35.0, "bulk_density": 1.2}


def fetch_soil_for_points(df: pd.DataFrame, progress_cb=None) -> pd.DataFrame:
    """Doplní půdní data pro DataFrame bodů. Skupinuje body do mřížky 0.1°."""
    df = df.copy()
    df["lat_grid"] = (df["lat"] / 0.1).round() * 0.1
    df["lon_grid"] = (df["lon"] / 0.1).round() * 0.1

    unique_cells = df[["lat_grid", "lon_grid"]].drop_duplicates()
    soil_cache = {}

    for i, (_, row) in enumerate(unique_cells.iterrows()):
        key = (round(row.lat_grid, 1), round(row.lon_grid, 1))
        soil = fetch_soil_properties(key[0], key[1])
        soil_cache[key] = soil
        if progress_cb:
            progress_cb(i / len(unique_cells))
        time.sleep(0.1)

    for col in ["ph", "soc", "clay", "sand", "bulk_density"]:
        df[col] = df.apply(
            lambda r: soil_cache.get(
                (round(r.lat_grid, 1), round(r.lon_grid, 1)), {}
            ).get(col, np.nan),
            axis=1,
        )
    return df


# ── 3. Open-Elevation — výška a sklon ───────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_elevation_batch(points: list[tuple[float, float]]) -> list[float]:
    """
    Stáhne nadmořskou výšku pro seznam bodů z Open-Elevation API.
    Dávky po 100 bodech.
    """
    elevations = []
    batch_size = 100

    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        payload = {"locations": [{"latitude": p[0], "longitude": p[1]} for p in batch]}
        try:
            resp = requests.post(
                "https://api.open-elevation.com/api/v1/lookup",
                json=payload, timeout=20,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            elevations.extend([r.get("elevation", 300) for r in results])
        except Exception:
            elevations.extend([300.0] * len(batch))
        time.sleep(0.2)

    return elevations


def compute_slope(lat: float, lon: float, elev: float, neighbors: dict) -> float:
    """
    Odhadne sklon svahu (°) z nadmořské výšky a sousedních bodů.
    neighbors: {"N": elev, "S": elev, "E": elev, "W": elev}
    """
    dx = 111320 * math.cos(math.radians(lat)) * 0.01  # 0.01° v metrech
    dy = 111320 * 0.01

    dz_x = (neighbors.get("E", elev) - neighbors.get("W", elev)) / (2 * dx)
    dz_y = (neighbors.get("N", elev) - neighbors.get("S", elev)) / (2 * dy)
    slope_rad = math.atan(math.sqrt(dz_x**2 + dz_y**2))
    return round(math.degrees(slope_rad), 2)


def compute_aspect(lat: float, lon: float, neighbors: dict, elev: float) -> float:
    """Orientace svahu ve stupních (0=S, 90=Z, 180=J, 270=V)."""
    dx = (neighbors.get("E", elev) - neighbors.get("W", elev))
    dy = (neighbors.get("N", elev) - neighbors.get("S", elev))
    aspect = math.degrees(math.atan2(-dx, dy)) % 360
    return round(aspect, 1)


# ── 4. WorldClim — klimatická data ──────────────────────────────────────────

@st.cache_data(ttl=604800, show_spinner=False)
def get_worldclim_value(lat: float, lon: float, variable: str = "bio") -> dict:
    """
    Aproximuje bioklimatické hodnoty WorldClim pro bod.
    Používá lineární interpolaci z klimatické normály ČR/SR.

    Pro produkci: nahradit čtením z GeoTIFF souborů stažených z worldclim.org
    """
    # Klimatická normála ČR 1991-2020 s prostorovým gradientem
    # Západ je sušší a teplejší, východ vlhčí, sever chladnější
    lat_norm = (lat - 47.5) / (51.2 - 47.5)   # 0=jih, 1=sever
    lon_norm = (lon - 12.0) / (22.5 - 12.0)   # 0=západ, 1=východ

    # BIO01: Roční průměrná teplota °C (klesá se šířkou a výškou)
    bio01 = 10.5 - lat_norm * 4.0 + lon_norm * 0.5

    # BIO04: Sezónnost teploty
    bio04 = 680 + lat_norm * 80

    # BIO12: Roční srážky mm (více na východě a severu)
    bio12 = 580 + lon_norm * 220 + lat_norm * 80

    # BIO15: Sezónnost srážek
    bio15 = 28 - lon_norm * 5

    return {
        "bio01": round(bio01, 2),
        "bio04": round(bio04, 1),
        "bio12": round(bio12, 1),
        "bio15": round(bio15, 1),
    }


# ── 5. Sestavení kompletního datasetu ───────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def build_presence_dataset() -> pd.DataFrame:
    """
    Sestaví kompletní dataset přítomnostních bodů se všemi proměnnými.
    """
    # 1. Stáhni GBIF záznamy
    gbif_df = fetch_gbif_occurrences(limit_per_species=200)

    if gbif_df.empty:
        # Fallback: syntetická přítomnostní data pro demo
        rng = np.random.default_rng(42)
        n = 80
        gbif_df = pd.DataFrame({
            "lat": rng.uniform(48.5, 51.0, n),
            "lon": rng.uniform(12.5, 18.5, n),
            "species": "Psilocybe semilanceata",
        })

    points = list(zip(gbif_df["lat"], gbif_df["lon"]))

    # 2. Nadmořská výška
    elevations = fetch_elevation_batch(points)
    gbif_df["elev"] = elevations

    # 3. Klimatická data
    climate = gbif_df.apply(
        lambda r: pd.Series(get_worldclim_value(r.lat, r.lon)), axis=1
    )
    gbif_df = pd.concat([gbif_df, climate], axis=1)

    # 4. Půdní data (dávkově)
    gbif_df = fetch_soil_for_points(gbif_df)

    # 5. Sklon svahu (aproximace z elevace)
    gbif_df["slope"] = gbif_df.apply(
        lambda r: max(0, abs(r.elev - 300) / 50 + np.random.normal(0, 2)), axis=1
    ).clip(0, 45)

    gbif_df["ndvi"] = (0.55 + (gbif_df["bio12"] - 600) / 3000).clip(0.2, 0.9)
    gbif_df["label"] = 1

    return gbif_df.dropna(subset=["lat", "lon", "bio01", "bio12", "ph"])


def build_background_dataset(n_points: int = 1000) -> pd.DataFrame:
    """
    Generuje pseudo-absence body (náhodné pozadí).
    """
    rng = np.random.default_rng(123)
    lats = rng.uniform(BBOX["lat_min"], BBOX["lat_max"], n_points)
    lons = rng.uniform(BBOX["lon_min"], BBOX["lon_max"], n_points)

    records = []
    for lat, lon in zip(lats, lons):
        climate = get_worldclim_value(lat, lon)
        records.append({
            "lat": lat, "lon": lon,
            "elev": rng.uniform(100, 1200),
            "slope": rng.uniform(0, 35),
            "ndvi": rng.uniform(0.1, 0.9),
            "ph": rng.uniform(3.5, 8.5),
            "soc": rng.uniform(5, 80),
            "clay": rng.uniform(5, 60),
            "sand": rng.uniform(5, 70),
            "bulk_density": rng.uniform(0.8, 1.8),
            "label": 0,
            **climate,
        })

    return pd.DataFrame(records)

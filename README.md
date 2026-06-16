# 🍄 PsiloSDM

Vzdělávací aplikace pro Species Distribution Modeling lysohlávek (*Psilocybe* spp.)
postavená na Streamlit + Folium + Plotly.

## Živá ukázka

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

## Funkce

- Interaktivní mapa ČR/SR s heatmapou pravděpodobnosti výskytu
- Nastavitelné environmentální parametry (teplota, srážky, pH, výška, NDVI)
- Gauge vizualizace výsledku + importance proměnných
- Edukativní sekce o SDM metodologii

## Spuštění lokálně

```bash
git clone https://github.com/TVŮJ-USERNAME/psilocybe-sdm
cd psilocybe-sdm
pip install -r requirements.txt
streamlit run app.py
```

## Nasazení na Streamlit Community Cloud

1. Nahraj repozitář na GitHub
2. Jdi na [share.streamlit.io](https://share.streamlit.io)
3. Klikni **New app** → vyber repozitář → main file: `app.py`
4. Deploy → hotovo za ~2 minuty

## Struktura

```
├── app.py              # hlavní Streamlit aplikace
├── model.py            # SDM model (Random Forest demo mód)
├── requirements.txt    # závislosti pro Streamlit Cloud
└── .streamlit/
    └── config.toml     # téma a konfigurace serveru
```

## Datové zdroje (pro produkční verzi)

| Zdroj | Data | Formát |
|---|---|---|
| Copernicus ERA5 | Teplota, srážky | NetCDF |
| ČHMÚ | Klimatické normy | GeoTIFF / WCS |
| GBIF | Výskytové záznamy | CSV / API |
| SoilGrids | pH půdy | GeoTIFF |
| Sentinel-2 | NDVI | GeoTIFF |

## Roadmapa (Render verze)

- [ ] FastAPI backend s PostGIS
- [ ] Napojení na reálná ERA5 data přes `cdsapi`
- [ ] GBIF occurrence records pro ČR/SR
- [ ] Natrénovaný Random Forest model
- [ ] Tile server pro výkonnou heatmapu

## Licence

MIT — volně použitelné pro vzdělávací účely.

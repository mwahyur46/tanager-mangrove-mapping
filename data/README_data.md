# Data Access

All Tanager-1 scenes are available via Planet Open STAC (surface reflectance, HDF5 format).
Atmospheric correction applied by Planet — no further radiometric processing required.

## Tanager-1 Scene IDs

| Site | Scene ID | Date | Notes |
|---|---|---|---|
| Sangatta, Kutai Timur (ID) | `20250302_030003_92_4001` | 2 Mar 2025 | Training + GEDI fusion |
| Gujarat, India | `20250311_061550_53_4001` | 11 Mar 2025 | Gulf of Kutch, Avicennia marina |
| El Salvador | `20250223_165546_32_4001` | 23 Feb 2025 | Bahia de Jiquilisco, Ramsar site |
| Belize | `20250824_171857_84_4001` | 24 Aug 2025 | Caribbean barrier reef mangrove |
| Ho Chi Minh, Vietnam | `20250407_035527_47_4001` | 7 Apr 2025 | Confirm coverage of Can Gio Reserve |

## Download Instructions

```bash
# Install Planet SDK
pip install planet

# Authenticate
planet auth init

# Download a scene (replace SCENE_ID)
planet data asset-get \
  --item-type PSScene \
  --asset-type ortho_analytic_sr \
  SCENE_ID
```

Full documentation: https://developers.planet.com/docs/

## Reference Datasets

| Dataset | Source | Notes |
|---|---|---|
| Global Mangrove Watch v3 (GMW v3) | https://www.globalmangrovewatch.org | Validation reference |
| GEDI L4A v2 | https://lpdaac.usgs.gov/products/gedi02_av002/ | AGB footprints, Sangatta |
| AOI polygons | `data/aoi/` | GeoJSON, one file per site |
| GMW v3 subsets | `data/gmw_v3/` | GeoJSON, clipped to AOI per site |

## Local Data Folders (gitignored)

| Folder | Contents |
|---|---|
| `data/raw/` | HDF5 originals downloaded from Planet STAC |
| `data/processed/` | GeoTIFF converted from HDF5 (relevant bands only) |

# Data Access

All Tanager-1 scenes are available via Planet Open STAC (surface reflectance, HDF5 format).
Atmospheric correction applied by Planet - no further radiometric processing required.

## Tanager-1 Scene IDs

| Site | Scene ID | Date | Role |
|---|---|---|---|
| Sangatta, Kutai Timur (ID) | `20250302_030003_92_4001` | 2 Mar 2025 | Training anchor |
| Gujarat, India | `20250311_061550_53_4001` | 11 Mar 2025 | Transfer site 1 - Gulf of Kutch, Avicennia marina |
| El Salvador | `20250223_165546_32_4001` | 23 Feb 2025 | Transfer site 2 - Bahia de Jiquilisco, Ramsar site |
| Belize (strip 1) | `20250824_171853_67_4001` | 24 Aug 2025 | Transfer site 3a - Caribbean barrier reef mangrove |
| Belize (strip 2) | `20250824_171857_84_4001` | 24 Aug 2025 | Transfer site 3b - adjacent strip |
| Australia | `20250608_014315_58_4001` | 8 Jun 2025 | Transfer site 4 |

Ho Chi Minh City (`20250407_035527_47_4001`) was assessed and dropped - insufficient mangrove coverage.

## Download Instructions

Scenes are distributed via the Planet Tanager Open Data Competition STAC catalog as HDF5 surface reflectance products (`ortho_sr_hdf5`). Download through the competition portal or Planet Explorer.

## Reference Datasets

| Dataset | Source | Notes |
|---|---|---|
| Global Mangrove Watch v3 (GMW v3) | https://www.globalmangrovewatch.org | Validation reference (independent - never used in training) |
| AOI polygons | `data/aoi/` | GeoJSON + shapefile per site |
| GMW v3 subsets | `data/gmw_v3/` | GeoJSON + shapefile, clipped to AOI per site |

## Local Data Folders (gitignored)

| Folder | Contents |
|---|---|
| `data/raw/` | HDF5 originals downloaded from Planet STAC |
| `data/processed/` | GeoTIFF converted from HDF5 (relevant bands only) |

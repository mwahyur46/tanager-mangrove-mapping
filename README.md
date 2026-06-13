# tanager-mangrove-mapping

**Transferable Mangrove Extent and Biomass Mapping from Tanager-1 Hyperspectral Imagery Using Adaptive Spectral Thresholds**

Submitted to the [Planet Tanager Open Data Competition](https://www.planet.com/) (August 2026.)

---

## Overview

This repository implements a transferable framework for mangrove extent and biomass mapping using Tanager-1 hyperspectral imagery (426 bands, 380-2500 nm, 30 m resolution).

**Core innovation:** per-scene adaptive spectral threshold calibration via histogram bimodal detection (with Otsu fallback) — eliminating the need for fixed thresholds from external studies and enabling transferability across geographic regions without retraining. Open water is masked via MNDWI before thresholding to keep bimodal peaks clean.

**Pipeline:**

```
Tanager-1 HDF5 (data/raw/)
    └── Step 0: inspect_hdf5() → HDF5 → GeoTIFF (5 bands only) → data/processed/
    └── Step 1: Spectral Indices (NDMI, MNDWI, MVI, SAVI, EMI)
    └── Step 2: Water mask (MNDWI) + Adaptive Threshold per scene  ← core innovation
                bimodal valley detection; Otsu fallback if unimodal
    └── Step 3: Pseudo-labels (MVI ∧ NDMI) → RF + XGBoost (class-balanced)
    └── Step 4: GEDI L4A Fusion — wall-to-wall AGB + carbon map (Sangatta)
    └── Step 5: Transferability — Steps 1–3 applied to 4 new sites, no retraining
                per-scene thresholds saved to outputs/results/thresholds_{scene_id}.json
```

## Study Sites

| Site | Scene ID | Role |
|---|---|---|
| Sangatta, Kutai Timur (Indonesia) | `20250302_030003_92_4001` | Training + GEDI fusion |
| Gujarat, India | `20250311_061550_53_4001` | Transferability |
| El Salvador | `20250223_165546_32_4001` | Transferability |
| Belize | `20250824_171857_84_4001` | Transferability |
| Ho Chi Minh, Vietnam | `20250407_035527_47_4001` | Transferability (TBC) |

## Repository Structure

```
tanager-mangrove-mapping/
├── notebooks/
│   ├── 01_preprocessing.ipynb       # HDF5 inspection, band extraction, water mask, adaptive threshold
│   ├── 02_classification.ipynb      # Pseudo-labels, RF + XGBoost (class-balanced), extent map
│   ├── 03_gedi_fusion.ipynb         # GEDI L4A join, AGB regression, carbon map (Sangatta)
│   └── 04_transferability.ipynb     # run_all_transfer_sites() → 4 sites, no retraining
├── src/
│   ├── preprocessing.py             # HDF5 I/O, GeoTIFF conversion, indices, water mask, adaptive threshold, save_raster
│   ├── classification.py            # Pseudo-labels, RF + XGBoost, evaluation, compare_models
│   ├── gedi_utils.py                # GEDI loading, spatial join (vectorised), AGB regression, carbon conversion
│   └── transferability.py          # run_transfer_scene(), run_all_transfer_sites(), TRANSFER_SITES registry
├── data/
│   ├── README_data.md               # Download instructions (no large files in repo)
│   ├── raw/                         # HDF5 originals from Planet STAC (gitignored)
│   ├── processed/                   # GeoTIFF band files + output rasters (gitignored)
│   ├── aoi/                         # AOI polygons per site (.geojson)
│   └── gmw_v3/                      # GMW v3 validation subsets (.geojson)
├── outputs/
│   ├── figures/                     # Maps + plots (150 dpi PNG)
│   ├── models/                      # Trained RF + XGBoost + AGB regressor (.joblib)
│   └── results/                     # Accuracy tables (.csv), thresholds per scene (.json)
├── docs/
│   └── technical_memo.pdf           # Methods write-up (W11–W12)
└── mangrove_tanager_final.ipynb     # Competition submission notebook
```

## Setup

```bash
conda env create -f environment.yml
conda activate tanager-mangrove
```

Data download instructions: see `data/README_data.md`.

## Running the Pipeline

Run notebooks in order. Each notebook saves outputs consumed by the next.

| Notebook | Key input | Key output |
|---|---|---|
| `01_preprocessing` | `data/raw/{scene_id}.h5` | `data/processed/*_nm.tif`, `outputs/results/thresholds_*.json` |
| `02_classification` | processed GeoTIFFs + thresholds JSON | `outputs/models/rf_*.joblib`, `data/processed/extent_mangrove_*.tif` |
| `03_gedi_fusion` | extent map + `data/raw/gedi_l4a_sangatta.geojson` | `data/processed/agb_map_*.tif`, `carbon_map_*.tif` |
| `04_transferability` | RF model + processed GeoTIFFs (transfer sites) | `outputs/results/thresholds_*.json` per site, `transferability_summary.csv` |

**Before running notebook 01:** call `inspect_hdf5()` on your Tanager `.h5` file and verify the path constants in `src/preprocessing.py` (`REFLECTANCE_PATH`, `WAVELENGTH_PATH`, `LON_PATH`, `LAT_PATH`) match your file's internal structure.

## License

MIT License. See `LICENSE`.

## References

- Bunting, P., Rosenqvist, A., Lucas, R. M., Rebelo, L., Hilarides, L., Thomas, N., Hardy, A., Itoh, T., Shimada, M., & Finlayson, C. M. (2018). The Global Mangrove Watch—A new 2010 Global Baseline of Mangrove Extent. Remote Sensing, 10(10), 1669. https://doi.org/10.3390/rs10101669
- Hu, T., Zhang, Y., Su, Y., Zheng, Y., Lin, G., & Guo, Q. (2020). Mapping the global mangrove forest aboveground biomass using multisource remote sensing data. Remote Sensing, 12(10), 1690. https://doi.org/10.3390/rs12101690
- Lassalle, G., Ferreira, M. P., La Rosa, L. E. C., Scafutto, R. D. M., & De Souza Filho, C. R. (2022). Advances in multi- and hyperspectral remote sensing of mangrove species: A synthesis and study case on airborne and multisource spaceborne imagery. ISPRS Journal of Photogrammetry and Remote Sensing, 195, 298–312. https://doi.org/10.1016/j.isprsjprs.2022.12.003
- Munawaroh, M., Wijaya, M. S., Winarso, G., Rudiastuti, A., Rahmila, Y. I., Suardana, A. a. P., Anggraini, N., & Rahmadi. (2025). Assessment of spatial misclassification in mangrove vs. Non-Mangrove Remote sensing classifications. ICARES, 1–7. https://doi.org/10.1109/icares67579.2025.11371528
- Nie, X., Xue, Z., & Li, X. (2026). Label-free mangrove mapping from temporally consistent PlanetScope imagery with interpretable deep unfolding network. ISPRS Journal of Photogrammetry and Remote Sensing, 235, 19–37. https://doi.org/10.1016/j.isprsjprs.2026.02.035
- Rahmila, Y. I., Prasetyo, L. B., Kusmana, C., Suyadi, Basyuni, M., Slamet, B., Pranoto, B., Yulianti, M., Yeny, I., Halwany, W., Rahmania, R., Januar, H. I., Adji, A. S., & Munawaroh. (2026). Mangrove Ecosystem Health Index (MEHI): a new method to evaluate mangrove ecosystem health at landscape scale using spatial metrics, canopy density, and potential disturbance based on hexagonal grid. Forest Science and Technology, 1–20. https://doi.org/10.1080/21580103.2026.2616443
- Shendryk, Y. (2022). Fusing GEDI with earth observation data for large area aboveground biomass mapping. International Journal of Applied Earth Observation and Geoinformation, 115, 103108. https://doi.org/10.1016/j.jag.2022.103108
- Zhang, R., & Fan, J. (2025). Classification and Carbon-Stock estimation of mangroves in Dongzhaigang based on Multi-Source remote sensing data using Google Earth Engine. Remote Sensing, 17(6), 964. https://doi.org/10.3390/rs17060964

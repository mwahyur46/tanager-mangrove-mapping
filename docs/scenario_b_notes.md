# Scenario B - Multispectral-Equivalent Comparison

## Positioning

Scenario B is developed for **journal publication (Objective 3) only**.
It is not included in the Planet Tanager Open Data Competition submission.

Rationale:

- The competition evaluates capabilities enabled by Tanager. Downgrading
  Tanager to Sentinel-2 bandpasses does not showcase the sensor.
- Adding Scenario B to competition deliverables would fragment focus from
  the core innovation (adaptive threshold + zero-shot transferability).
- Objective 3 (quantifying REIP and EMI contribution) is a scientific
  question suited to peer-reviewed venues, not a demonstration project.

## Research Question

To what extent do hyperspectral-specific features (REIP, EMI) improve
mangrove classification accuracy and cross-site transferability compared
to a multispectral-equivalent feature stack derived from the same Tanager
scene?

## Design

| Aspect          | Scenario A (competition + journal) | Scenario B (journal only) |
| --------------- | ---------------------------------- | ------------------------- |
| Sensor input    | Tanager 426-band VSWIR             | Tanager resampled to S2   |
| Features        | NDMI, MVI, MNDWI, SAVI, REIP, EMI  | NDMI, MVI, MNDWI, SAVI    |
| Feature count   | 6                                  | 4                         |
| Bandpass source | Discrete wavelengths (literature)  | S2 SRF convolution        |

Both scenarios share:

- Pseudo-label generation (adaptive spectral threshold)
- Model (XGBoost)
- Training scene (Sangatta)
- Transfer sites (Gujarat, El Salvador, Belize, Australia)
- Validation (GMW v3)

Only the feature stack differs.

## Methodology

1. Load Tanager HDF5 (426 bands) from `data/raw/`.
2. Convolve with Sentinel-2 SRF (ESA Handbook, COPE-GSEG-EOPG-TN-15-0007)
   to produce B3, B4, B8, B11 equivalent reflectance.
3. Compute NDMI, MVI, MNDWI, SAVI from resampled bands.
4. Apply adaptive threshold to generate Scenario B pseudo-labels.
5. Train XGBoost on Scenario B Sangatta pseudo-labels.
6. Apply to 4 transfer sites without retraining.
7. Compare Kappa A vs Kappa B per site.

## Expected Outcome

If REIP and EMI contribute meaningfully, Kappa A > Kappa B systematically
across sites. If contribution is negligible, Kappa A approximates Kappa B,
supporting a null hypothesis that S2-equivalent features suffice for
mangrove mapping in this pipeline.

## Data Requirements

- Tanager HDF5 raw files in `data/raw/`.
- Sentinel-2 SRF file in `data/sentinel2_srf/` (see README there).

## Output

- `data/processed_s2eq/{site_key}_s2eq_bands.tif` (4-band stack)
- `data/processed_s2eq/{site_key}_s2eq_indices.tif` (4-index stack)

Site key convention follows Scenario A
(e.g. `sangatta_20250302_030003_92_4001`).

## Limitation Note (for Methods / Limitations section)

Spectral indices are computed using wavelength positions anchored to the
original definitions in the primary literature (Scenario A) or bandpasses
from the Sentinel-2 mission (Scenario B). Data-driven or scene-adaptive
band selection is not evaluated in this study. This design choice
preserves the semantic equivalence of each index across sites and
sensors, and ensures that observed differences in transfer performance
can be attributed to ecological and threshold-related factors rather than
feature-definition variability.

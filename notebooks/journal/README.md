# Journal Notebooks

Experimental notebooks for ongoing development beyond the competition submission.
These are not part of the Planet Tanager Open Data Competition 2026 (Scenario A) pipeline.

---

## Purpose

The journal tracks **Scenario B: multispectral-equivalent ablation** — a study asking whether
the same mangrove extent mapping approach works when the full 426-band Tanager spectrum is
collapsed to a simulated 4-band multispectral stack (Blue, Green, Red, NIR), equivalent to
what a sensor like Sentinel-2 would provide.

The motivation is to isolate how much of the pipeline's performance comes from hyperspectral
features (REIP, EMI, NDRE) versus indices computable from conventional multispectral imagery
(NDVI, MNDWI, NDMI, SAVI, MVI). This frames Tanager's hyperspectral advantage more concretely.

---

## Notebooks

| Notebook | Corresponds to | Description |
|---|---|---|
| `01b_multispectral_equivalent.ipynb` | NB01 | Spectral resampling via Sentinel-2A SRF convolution, band extraction, indices for the 4-band stack |
| `02b_classification.ipynb` | NB02 | XGBoost training on multispectral-equivalent features, GMW v3 evaluation |
| `03b_transferability.ipynb` | NB03 | Zero-shot transfer using the multispectral-equivalent model |

---

## Additional Dependencies

These notebooks require files not needed by the competition pipeline:

| Dependency | Location | Purpose |
|---|---|---|
| Sentinel-2A SRF files | `data/sentinel2_srf/` | Per-band spectral response functions for convolution |
| `src/spectral_resampling.py` | `src/` | SRF convolution of the full 426-band cube to simulated S2A bands |

The full 426-band HDF5 files are required (same as NB01) since convolution operates on the
complete spectrum before any band reduction.

---

## Status

Active development. Notebooks may be incomplete or produce intermediate results.
Do not treat outputs here as final results.

---

## Relationship to Competition Notebooks

Scenario B shares the same site registry, HDF5 loading, and GMW v3 evaluation logic as
the competition pipeline. What differs is the feature set: REIP and indices that require
wavelengths outside the simulated multispectral range (e.g. EMI at 2200 nm, NDRE at 720 nm)
are excluded. This makes the comparison controlled at the feature level.

# Sentinel-2 Spectral Response Function (SRF)

## Source

ESA Sentinel-2 MSI Spectral Response Function.
Reference document: COPE-GSEG-EOPG-TN-15-0007.

Download from ESA Sentinel Online:
https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/document-library

Or Sentiwiki (direct file link, subject to change):
https://sentiwiki.copernicus.eu/web/s2-documents

Expected filename pattern:
`COPE-GSEG-EOPG-TN-15-0007 - Sentinel-2 Spectral Responses Functions ...xlsx`

## Format

XLSX with columns: `SR_WL` (wavelength nm), followed by band columns
`S2A_SR_AV_B1`, `S2A_SR_AV_B2`, ..., `S2A_SR_AV_B12` (Sentinel-2A) and
equivalent `S2B_SR_AV_Bx` columns (Sentinel-2B).

Each band column contains relative response (0 to 1) at each 1 nm step,
covering approximately 400 to 2400 nm.

## Bands Used in Scenario B

Only 4 bands are needed for the Scenario B feature stack
(NDMI, MVI, MNDWI, SAVI):

| Band | Center (nm) | Purpose                     |
| ---- | ----------- | --------------------------- |
| B3   | 560         | Green (MNDWI, MVI)          |
| B4   | 665         | Red (SAVI)                  |
| B8   | 842         | NIR (NDMI, MVI, SAVI)       |
| B11  | 1610        | SWIR1 (NDMI, MNDWI, MVI)    |

## Platform Choice

Default: Sentinel-2A (S2A). S2A and S2B SRFs are near-identical.
Use S2A columns unless specifically comparing platforms.

## After Download

Place the XLSX file in this directory. The path is consumed by
`src/spectral_resampling.load_sentinel2_srf()`.

Suggested filename after download:
`s2_srf_v5.0.xlsx` (or keep ESA original filename)

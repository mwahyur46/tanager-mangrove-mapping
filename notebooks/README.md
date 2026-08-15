# Notebooks

Competition pipeline notebooks for mangrove extent mapping from Planet Tanager-1 hyperspectral imagery.
Run in order — each stage depends on outputs from the previous one.

---

## Pipeline

```
01_preprocessing  -->  02_classification  -->  03_transferability
```

| Notebook | Stage | Produces |
|---|---|---|
| `01_preprocessing.ipynb` | Band extraction, spectral indices, adaptive thresholds, coastal candidate mask | 6-band GeoTIFF, REIP raster, threshold JSON |
| `02_classification.ipynb` | Pseudo-label generation, XGBoost training, GMW v3 evaluation on training site | Trained models, extent map, accuracy CSV |
| `03_transferability.ipynb` | Zero-shot transfer to 5 scenes across 4 sites | Per-site thresholds, accuracy CSVs, transferability summary |

---

## Execution Environment

Primary environment is **Google Colab** — notebooks have Colab paths hardcoded in their setup cells.

For local runs, in each notebook's setup cell:
- Comment out the Colab `ROOT` path
- Uncomment `ROOT = Path('..').resolve()`

A standalone local equivalent of NB03 is available at the repo root:

```bash
python run_03.py
```

---

## Sites

| Key | Scene ID | Role |
|---|---|---|
| `sangatta` | `20250302_030003_92_4001` | Training anchor (Kalimantan, Indonesia) |
| `gujarat` | `20250311_061550_53_4001` | Transfer site 1 (India) |
| `elsalvador` | `20250223_165546_32_4001` | Transfer site 2 (Central America) |
| `belize` | `20250824_171853_67_4001` | Transfer site 3a (Central America) |
| `belize2` | `20250824_171857_84_4001` | Transfer site 3b, adjacent strip |
| `australia` | `20250608_014315_58_4001` | Transfer site 4 (Queensland) |

---

## Notes

- GMW v3 is used for validation only — never as training input or threshold input.
- Primary classifier is XGBoost Tuned. Random Forest is a comparison baseline.
- `MVI` always uses Otsu thresholding (`force_otsu_indices=['MVI']`), never bimodal valley detection.
- The `journal/` subfolder contains experimental notebooks that are not part of this competition submission.

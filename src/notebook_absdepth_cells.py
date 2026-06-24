# =============================================================================
# NOTEBOOK INTEGRATION CELLS -- Absorption Depth Feature (continuum removal)
# =============================================================================
# Place src/continuum.py in the src/ folder first.
# =============================================================================


# ================================ NOTEBOOK 01 =================================
# Goal: compute the 1640 nm absorption-depth feature from the full HDF5 spectrum
# and save it as a GeoTIFF for use in 02_classification.ipynb.
#
# IMPORTANT: absorption depth needs the FULL 426-band spectrum, so it must be
# computed from the HDF5 (load_hdf5 output), NOT from the 5-band GeoTIFF.


# ----------------------------------------------------------------------------
# MARKDOWN CELL (new section, e.g. "## 6. Diagnostic Feature: Absorption Depth")
# ----------------------------------------------------------------------------
"""
## 6. Diagnostic Feature: Absorption Depth (1640 nm)

Continuum-removed band depth at the 1640 nm SWIR water-absorption feature.
This exploits Tanager's full 426-band spectrum to derive a physically-meaningful
feature that broadband multispectral sensors (Sentinel-2, Landsat) cannot
reproduce, since it requires many narrow contiguous bands around the absorption
trough.

Band depth: D = 1 - (R_band / R_continuum), where the continuum is linearly
interpolated between two shoulder wavelengths (van der Meer, 2004;
Clark & Roush, 1984).

This feature feeds the RF/XGBoost classifier only. The adaptive threshold and
pseudo-labels remain based on the 5 spectral indices, so the core methodology
is unchanged.

References:
- Clark, R.N. & Roush, T.L. (1984). doi:10.1029/JB089iB07p06329
- van der Meer, F. (2004). doi:10.1016/j.jag.2003.09.001
"""


# ----------------------------------------------------------------------------
# CODE CELL: import
# ----------------------------------------------------------------------------
"""
import importlib
import src.continuum as _cont
importlib.reload(_cont)
from src.continuum import absorption_depth_1640
"""


# ----------------------------------------------------------------------------
# CODE CELL: compute + save (single-scene, e.g. Sangatta)
# Requires `hdf5_data` = load_hdf5(...) for the current scene to be in memory.
# If only the GeoTIFF is loaded, re-run load_hdf5() for this scene first.
# ----------------------------------------------------------------------------
"""
# ============================================================
# Compute 1640 nm absorption depth from the full HDF5 spectrum
# ============================================================
hdf5_data = load_hdf5(h5_path(SITE))          # full 426-band spectrum
abs_depth = absorption_depth_1640(hdf5_data)

# Save as single-band GeoTIFF (same grid as the 5-band bands.tif)
import rasterio
absdepth_path = DATA_PROC / f'absdepth1640_{SITE}_{SCENE_ID}.tif'
with rasterio.open(
    absdepth_path, 'w',
    driver='GTiff',
    height=abs_depth.shape[0], width=abs_depth.shape[1],
    count=1, dtype='float32',
    crs=hdf5_data['crs'], transform=hdf5_data['transform'],
    compress='lzw', nodata=np.nan,
) as dst:
    dst.write(abs_depth, 1)
print(f'Absorption depth saved : {absdepth_path.name}')
"""


# ----------------------------------------------------------------------------
# OPTIONAL CODE CELL: spectral signature plot (strong storytelling figure)
# Shows WHY hyperspectral matters: mangrove vs non-mangrove absorption shape.
# ----------------------------------------------------------------------------
"""
# ============================================================
# Spectral signature comparison around 1640 nm
# Mean reflectance spectrum: pseudo-label mangrove vs non-mangrove
# (Run AFTER pseudo-labels exist; uses candidate zone for context)
# ============================================================
wl   = hdf5_data['wavelengths']
refl = hdf5_data['reflectance']

# Use a wavelength window around the feature
win = (wl >= 1400) & (wl <= 1900)

# Quick mangrove proxy from indices (MVI+NDMI threshold) within candidate zone
mvi  = indices['MVI']; ndmi = indices['NDMI']
man_mask = (mvi > thresholds['MVI']) & (ndmi > thresholds['NDMI']) & candidate_mask
non_mask = (~man_mask) & candidate_mask

man_spec = np.nanmean(refl[:, man_mask], axis=1)
non_spec = np.nanmean(refl[:, non_mask], axis=1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(wl[win], man_spec[win], color='#2ca02c', label='Mangrove (proxy)')
ax.plot(wl[win], non_spec[win], color='#8c564b', label='Non-mangrove')
ax.axvline(1640, color='red', ls='--', lw=1, label='1640 nm feature')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Surface reflectance')
ax.set_title(f'Spectral Signature around 1640 nm -- {SITE}')
ax.legend()
plt.tight_layout()
plt.savefig(ROOT / 'outputs' / 'figures' / f'spectral_sig_{SITE}.png',
            dpi=150, bbox_inches='tight')
plt.show()
"""


# ----------------------------------------------------------------------------
# FOR THE MULTI-SITE LOOP: add inside the loop, after candidate_mask step
# ----------------------------------------------------------------------------
"""
    # Diagnostic feature: 1640 nm absorption depth (from full HDF5 spectrum)
    abs_depth = absorption_depth_1640(data_hdf5)   # data_hdf5 = load_hdf5(...) in loop
    absdepth_path = DATA_PROC / f'absdepth1640_{site}_{scene_id}.tif'
    with rasterio.open(
        absdepth_path, 'w', driver='GTiff',
        height=abs_depth.shape[0], width=abs_depth.shape[1],
        count=1, dtype='float32',
        crs=data_hdf5['crs'], transform=data_hdf5['transform'],
        compress='lzw', nodata=np.nan,
    ) as dst:
        dst.write(abs_depth, 1)
"""


# ================================ NOTEBOOK 02 =================================
# Goal: load the absorption-depth GeoTIFF and pass it as an extra feature to
# both build_feature_matrix() and predict_extent().


# ----------------------------------------------------------------------------
# CODE CELL: load absorption depth (after loading indices + candidate mask)
# ----------------------------------------------------------------------------
"""
# ============================================================
# Load 1640 nm absorption-depth feature (from 01_preprocessing)
# ============================================================
absdepth_path = DATA_PROC / f'absdepth1640_{SITE}_{SCENE_ID}.tif'
with rasterio.open(absdepth_path) as src:
    abs_depth = src.read(1).astype(np.float32)

extra_features = {'AbsDepth1640': abs_depth}
print(f'Absorption depth loaded: '
      f'{np.isfinite(abs_depth).sum():,} valid px')
"""


# ----------------------------------------------------------------------------
# REPLACE the build_feature_matrix cell (section 3) with:
# ----------------------------------------------------------------------------
"""
X, y, feature_names = build_feature_matrix(
    indices, labels, extra_features=extra_features
)
X_train, X_test, y_train, y_test = split_data(X, y)
"""


# ----------------------------------------------------------------------------
# REPLACE the predict_extent cell (section 6) with:
# ----------------------------------------------------------------------------
"""
# ============================================================
# Predict full scene using RF (primary model)
# Same feature stack as training: 5 indices + absorption depth
# ============================================================
h, w        = list(indices.values())[0].shape
extent_map  = predict_extent(rf_model, indices,
                              original_shape=(h, w),
                              candidate_mask=candidate_mask,
                              extra_features=extra_features)

extent_path = DATA_PROC / f'extent_mangrove_{SITE}_{SCENE_ID}.tif'
with rasterio.open(
    extent_path, 'w',
    driver='GTiff', height=h, width=w,
    count=1, dtype='int8',
    crs=data['crs'], transform=data['transform'],
    compress='lzw'
) as dst:
    dst.write(extent_map, 1)
print(f'Extent map saved : {extent_path}')
"""

"""
Sentinel-2 equivalent resampling of Tanager-1 hyperspectral imagery.

This module implements Scenario B of the transferability study.
Tanager 426-band VSWIR reflectance is convolved with Sentinel-2 MSI
spectral response functions (SRF) to produce a multispectral-equivalent
raster stack for comparison against the hyperspectral-optimized
Scenario A.

Purpose: Isolate the contribution of Tanager's high spectral resolution
features (REIP, EMI) to mangrove classification accuracy and
transferability, by comparing model performance on a feature stack
derived from the same scene but downsampled to Sentinel-2 bandpasses.

Scope: Scenario B is developed for journal publication (Objective 3)
only. It is not included in the Planet Tanager Open Data Competition
submission.

Authors     : Muhammad Wahyu Ramadhan, Athar Abdurrahman B., Diniyarti
Competition : Planet Tanager Open Data Competition 2026
Topic       : Transferable Mangrove Extent Mapping Using
              Adaptive Spectral Thresholds
Date        : July 2026
"""

import numpy as np
import pandas as pd
import rasterio
from pathlib import Path


# ============================================================================
# Sentinel-2 bands used in Scenario B feature stack
# (NDVI, MNDWI, NDMI, CMRI, NDRE, SAVI, MVI, EMI)
# Center wavelengths (nm) from ESA Sentinel-2 User Handbook.
# SRF source: COPE-GSEG-EOPG-TN-15-0007 v4.0 (2024).
# ============================================================================
S2_TARGET_BANDS = {
    'green'   : {'code': 'B3',  'center': 560,  'used_in': ['MNDWI', 'MVI']},
    'red'     : {'code': 'B4',  'center': 665,  'used_in': ['NDVI', 'CMRI', 'SAVI']},
    'rededge' : {'code': 'B5',  'center': 705,  'used_in': ['NDRE']},
    'nir'     : {'code': 'B8',  'center': 842,  'used_in': ['NDVI', 'NDMI', 'CMRI', 'NDRE', 'SAVI', 'MVI', 'EMI']},
    'swir1'   : {'code': 'B11', 'center': 1610, 'used_in': ['MNDWI', 'NDMI', 'CMRI', 'MVI']},
    'swir2'   : {'code': 'B12', 'center': 2190, 'used_in': ['EMI']},
}

# Mapping: band name -> SRF column name in ESA XLSX
_SRF_COL = {
    'S2A': {
        'green'   : 'S2A_SR_AV_B3',
        'red'     : 'S2A_SR_AV_B4',
        'rededge' : 'S2A_SR_AV_B5',
        'nir'     : 'S2A_SR_AV_B8',
        'swir1'   : 'S2A_SR_AV_B11',
        'swir2'   : 'S2A_SR_AV_B12',
    },
    'S2B': {
        'green'   : 'S2B_SR_AV_B3',
        'red'     : 'S2B_SR_AV_B4',
        'rededge' : 'S2B_SR_AV_B5',
        'nir'     : 'S2B_SR_AV_B8',
        'swir1'   : 'S2B_SR_AV_B11',
        'swir2'   : 'S2B_SR_AV_B12',
    },
}


# ============================================================================
# 1. Load Sentinel-2 SRF
# ============================================================================

def load_sentinel2_srf(srf_path, platform='S2A'):
    """
    Load Sentinel-2 spectral response function (SRF) from ESA XLSX file.

    Reads sheet 'Spectral Responses (S2A)' or 'Spectral Responses (S2B)'.
    Only loads the 6 bands needed for Scenario B: B3, B4, B5, B8, B11, B12.

    Parameters
    ----------
    srf_path : str or Path
        Path to ESA SRF XLSX file (COPE-GSEG-EOPG-TN-15-0007 v4.0).
    platform : {'S2A', 'S2B'}
        Sentinel-2 platform. Default 'S2A'.

    Returns
    -------
    dict
        Keys are band names ('green', 'red', 'rededge', 'nir', 'swir1', 'swir2').
        Values are tuples of (wavelengths_nm, weights) as float32 ndarrays.
        Only rows where SRF > 0 are retained.
    """
    sheet = f'Spectral Responses ({platform})'
    cols  = _SRF_COL[platform]

    df = pd.read_excel(srf_path, sheet_name=sheet)

    srf_dict = {}
    for band_name, col in cols.items():
        wl     = df['SR_WL'].values.astype(np.float32)
        w      = df[col].values.astype(np.float32)
        active = w > 0.0
        srf_dict[band_name] = (wl[active], w[active])

        print(f'  SRF {platform} {band_name:<6}: '
              f'{wl[active].min():.0f}-{wl[active].max():.0f} nm, '
              f'peak={wl[w.argmax()]:.0f} nm, '
              f'{active.sum()} points')

    return srf_dict


# ============================================================================
# 2. Resample Tanager -> S2-equivalent bands
# ============================================================================

def resample_hyperspectral_to_s2(tanager_stack, tanager_wavelengths,
                                  srf_dict):
    """
    Convolve Tanager 426-band reflectance with Sentinel-2 SRF.

    For each S2 band, computes:
        R_S2 = sum(R_Tanager * SRF(lambda)) / sum(SRF(lambda))

    where SRF weights are linearly interpolated from the ESA SRF table
    onto the Tanager wavelength positions within each bandpass.

    Parameters
    ----------
    tanager_stack : ndarray, shape (n_bands, height, width)
        Tanager surface reflectance cube.
    tanager_wavelengths : ndarray, shape (n_bands,)
        Center wavelength (nm) of each Tanager band.
    srf_dict : dict
        Output of load_sentinel2_srf().

    Returns
    -------
    dict
        Keys: 'green', 'red', 'rededge', 'nir', 'swir1', 'swir2'.
        Values: 2D float32 ndarrays of resampled reflectance.
    """
    n_bands, h, w = tanager_stack.shape
    resampled     = {}

    for band_name, (srf_wl, srf_w) in srf_dict.items():
        # Interpolate SRF weights onto Tanager wavelength positions
        srf_interp = np.interp(
            tanager_wavelengths, srf_wl, srf_w,
            left=0.0, right=0.0
        ).astype(np.float32)

        # Only use Tanager bands that fall within the SRF active range
        active = srf_interp > 0.0
        n_active = int(active.sum())

        if n_active == 0:
            raise ValueError(
                f'No Tanager bands overlap S2 {band_name} bandpass '
                f'({srf_wl.min():.0f}-{srf_wl.max():.0f} nm). '
                f'Check tanager_wavelengths covers this range.'
            )

        # Weighted average: shape (n_active,) broadcast over (H, W)
        w_active    = srf_interp[active]                   # (n_active,)
        stack_slice = tanager_stack[active].astype(np.float32)  # (n_active, H, W)

        r_s2 = (
            np.einsum('b,bhw->hw', w_active, stack_slice) /
            w_active.sum()
        )

        resampled[band_name] = r_s2
        print(f'  Resampled {band_name:<6}: '
              f'{n_active} Tanager bands used, '
              f'mean R = {np.nanmean(r_s2):.4f}')

    return resampled


# ============================================================================
# 3. Compute Scenario B spectral indices
# ============================================================================

def compute_scenario_b_indices(resampled_bands, savi_L=0.5):
    """
    Compute Scenario B feature stack from S2-equivalent bands.

    Features (8 total):
        NDVI  = (NIR - RED) / (NIR + RED)
        MNDWI = (GREEN - SWIR1) / (GREEN + SWIR1)
        NDMI  = (NIR - SWIR1) / (NIR + SWIR1)
        CMRI  = NDVI - MNDWI
        NDRE  = (NIR - REDEDGE) / (NIR + REDEDGE)
        SAVI  = ((NIR - RED) / (NIR + RED + L)) * (1 + L)
        MVI   = (NIR - GREEN) / (SWIR1 - GREEN)
        EMI   = (NIR - SWIR2) / (NIR + SWIR2)

    Parameters
    ----------
    resampled_bands : dict
        Output of resample_hyperspectral_to_s2(). Must contain
        'green', 'red', 'rededge', 'nir', 'swir1', 'swir2'.
    savi_L : float
        Soil brightness correction factor (Huete, 1988). Default 0.5.

    Returns
    -------
    dict
        Keys: 'NDVI', 'MNDWI', 'NDMI', 'CMRI', 'NDRE', 'SAVI', 'MVI', 'EMI'. 
        Values are 2D float32 ndarrays.
    """
    green   = resampled_bands['green'].astype(np.float32)
    red     = resampled_bands['red'].astype(np.float32)
    rededge = resampled_bands['rededge'].astype(np.float32)
    nir     = resampled_bands['nir'].astype(np.float32)
    swir1   = resampled_bands['swir1'].astype(np.float32)
    swir2   = resampled_bands['swir2'].astype(np.float32)

    eps = 1e-10   # avoid division by zero
    MVI_CLIP = (-1, 20)

    ndvi  = np.clip((nir - red) / (nir + red + eps), -1, 1)
    mndwi = np.clip((green - swir1) / (green + swir1 + eps), -1, 1)
    ndmi  = np.clip((nir - swir1) / (nir + swir1 + eps), -1, 1)
    cmri  = ndvi - mndwi
    ndre  = np.clip((nir - rededge) / (nir + rededge + eps), -1, 1)
    savi  = np.clip(((nir - red) / (nir + red + savi_L + eps)) * (1 + savi_L), -1, 1)
    mvi   = np.clip((nir - green) / (swir1 - green + eps), MVI_CLIP[0], MVI_CLIP[1])
    emi   = np.clip((nir - swir2) / (nir + swir2 + eps), -1, 1)

    indices = {
        'NDVI'  : ndvi.astype(np.float32),
        'MNDWI' : mndwi.astype(np.float32),
        'NDMI'  : ndmi.astype(np.float32),
        'CMRI'  : cmri.astype(np.float32),
        'NDRE'  : ndre.astype(np.float32),
        'SAVI'  : savi.astype(np.float32),
        'MVI'   : mvi.astype(np.float32),
        'EMI'   : emi.astype(np.float32),
    }

    for name, arr in indices.items():
        print(f'  {name:<6}: min={np.nanmin(arr):.4f}, '
              f'max={np.nanmax(arr):.4f}, '
              f'mean={np.nanmean(arr):.4f}')

    return indices


# ============================================================================
# 4. Write Scenario B outputs to disk
# ============================================================================

def write_scenario_b_outputs(indices_dict, reference_tif, out_dir, site_key):
    """
    Write Scenario B index stack as multi-band GeoTIFF.

    Inherits CRS and transform from the Scenario A reference GeoTIFF
    to ensure pixel-perfect alignment between scenarios.

    Output filename: {out_dir}/{site_key}_s2eq_indices.tif
    Band order     : NDVI, MNDWI, NDMI, CMRI, NDRE, SAVI, MVI, EMI

    Parameters
    ----------
    indices_dict : dict
        Output of compute_scenario_b_indices().
    reference_tif : str or Path
        Existing Scenario A GeoTIFF (e.g. sangatta_..._bands.tif).
        Used only to inherit CRS, transform, and shape.
    out_dir : str or Path
        Output directory (data/processed_s2eq/).
    site_key : str
        Site identifier (e.g. 'sangatta_20250302_030003_92_4001').
    """
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{site_key}_s2eq_indices.tif'

    band_order = ['NDVI', 'MNDWI', 'NDMI', 'CMRI', 'NDRE', 'SAVI', 'MVI', 'EMI']

    with rasterio.open(reference_tif) as ref:
        profile = ref.profile.copy()

    # Remove tiling keys inherited from reference_tif that only apply to
    # TILED=YES profiles; passing BLOCKXSIZE/BLOCKYSIZE without TILED=YES
    # triggers a rasterio CPLE_IllegalArg warning.
    for key in ('blockxsize', 'blockysize', 'tiled'):
        profile.pop(key, None)

    profile.update(
        count    = len(band_order),
        dtype    = 'float32',
        compress = 'lzw',
        nodata   = np.nan,
    )

    with rasterio.open(out_path, 'w', **profile) as dst:
        for i, name in enumerate(band_order, start=1):
            dst.write(indices_dict[name], i)
            dst.set_band_description(i, name)

    print(f'  Scenario B output : {out_path.name}')
    print(f'  Bands (in order)  : {band_order}')
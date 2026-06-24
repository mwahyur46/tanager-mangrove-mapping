# =============================================================================
# preprocessing.py PATCHES
# =============================================================================
# This file contains 3 functions to update preprocessing.py:
#
#   1. NEW       : compute_coastal_candidate_mask()
#   2. MODIFIED  : detect_bimodal_threshold()    (add valid_mask parameter)
#   3. MODIFIED  : apply_adaptive_threshold()    (add candidate_mask parameter)
#
# Apply by:
#   - Adding function 1 to preprocessing.py (e.g. before detect_bimodal_threshold)
#   - Replacing existing detect_bimodal_threshold() with the version below
#   - Replacing existing apply_adaptive_threshold() with the version below
#
# New import required at top of preprocessing.py:
#   from scipy.ndimage import distance_transform_edt
# =============================================================================

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu


# =============================================================================
# 1. NEW: Adaptive coastal candidate mask
# =============================================================================

def compute_coastal_candidate_mask(data: dict,
                                    indices: dict,
                                    return_diagnostics: bool = False):
    """
    Compute adaptive coastal candidate mask for mangrove detection.
    All thresholds derived per-scene via Otsu (no fixed parameters).

    Pipeline:
      1. Water mask      : MNDWI > Otsu(MNDWI)
      2. Vegetation mask : SAVI  > Otsu(SAVI)
      3. Distance transform from water (Euclidean, pixels)
      4. Buffer threshold: Otsu(distance | vegetated pixels)
      5. Candidate       : vegetation AND distance<buffer AND NOT water

    Rationale: Otsu on distance-from-water within vegetated pixels separates
    "coastal vegetation (mangrove candidate)" from "inland vegetation
    (rainforest, plantation, revegetated mining)". Buffer width is therefore
    adaptive to scene geography (wide in Belize, narrow in Sangatta).

    Parameters
    ----------
    data    : dict from load_geotiff_bands()
    indices : dict from compute_all_indices()
    return_diagnostics : if True, return (mask, dict_of_intermediates)

    Returns
    -------
    candidate_mask : 2D bool array (True = coastal vegetated pixel)
    diagnostics    : dict (optional) with water_t, veg_t, buffer_t_px,
                     buffer_t_m, pixel_size_m, water_mask, veg_mask, dist_px
    """
    mndwi = indices['MNDWI']
    savi  = indices['SAVI']

    # 1. Water mask (Otsu on MNDWI)
    mndwi_flat = mndwi[np.isfinite(mndwi)]
    if len(mndwi_flat) == 0:
        raise ValueError("MNDWI has no finite pixels")
    water_t    = float(threshold_otsu(mndwi_flat))
    water_mask = (mndwi > water_t) & np.isfinite(mndwi)

    # 2. Vegetation mask (Otsu on SAVI)
    savi_flat = savi[np.isfinite(savi)]
    if len(savi_flat) == 0:
        raise ValueError("SAVI has no finite pixels")
    veg_t    = float(threshold_otsu(savi_flat))
    veg_mask = (savi > veg_t) & np.isfinite(savi)

    # 3. Distance from water (pixels, Euclidean)
    dist_px = distance_transform_edt(~water_mask)

    # 4. Adaptive buffer (Otsu on distance within vegetated pixels)
    dist_veg = dist_px[veg_mask & np.isfinite(dist_px)]
    if len(dist_veg) == 0:
        raise ValueError("No vegetated pixels — check SAVI threshold or data")
    buffer_t_px = float(threshold_otsu(dist_veg))

    # 5. Candidate
    candidate = veg_mask & (dist_px < buffer_t_px) & (~water_mask)

    # Convert buffer to meters (approx, assumes square pixels)
    try:
        pixel_size_m = float(abs(data['transform'].a))
    except (KeyError, AttributeError):
        pixel_size_m = 1.0
    buffer_t_m = buffer_t_px * pixel_size_m

    print(f"  Coastal candidate mask:")
    print(f"  water_t  (MNDWI)    : {water_t:.4f}")
    print(f"  veg_t    (SAVI)     : {veg_t:.4f}")
    print(f"  buffer_t (distance) : {buffer_t_px:.1f} px (~{buffer_t_m:.0f} m)")
    print(f"  candidate pixels    : {int(candidate.sum()):,} ({100*candidate.mean():.1f}%)")

    if return_diagnostics:
        diag = {
            'water_t'      : water_t,
            'veg_t'        : veg_t,
            'buffer_t_px'  : buffer_t_px,
            'buffer_t_m'   : buffer_t_m,
            'pixel_size_m' : pixel_size_m,
            'water_mask'   : water_mask,
            'veg_mask'     : veg_mask,
            'dist_px'      : dist_px,
        }
        return candidate, diag
    return candidate


# =============================================================================
# 2. MODIFIED: detect_bimodal_threshold (added valid_mask parameter)
# =============================================================================

def detect_bimodal_threshold(index_array: np.ndarray,
                              n_bins: int = 256,
                              min_peak_distance: int = 20,
                              water_mask: np.ndarray = None,
                              valid_mask: np.ndarray = None,
                              force_otsu: bool = False) -> tuple:
    """
    Detect threshold between two modes in a histogram (bimodal distribution).
    Falls back to Otsu if fewer than 2 peaks are found.

    Parameters
    ----------
    index_array       : 2D np.ndarray of spectral index values
    n_bins            : number of histogram bins
    min_peak_distance : minimum bin separation between peaks
    water_mask        : optional 2D bool array; water pixels excluded
                        (ignored if valid_mask is provided)
    valid_mask        : optional 2D bool array; if provided, histogram is
                        computed ONLY from pixels within valid_mask
                        (overrides water_mask). Use this to constrain
                        threshold calibration to a coastal candidate zone.
    force_otsu        : if True, skip valley-peak detection entirely and
                        use Otsu directly. Use for right-skewed indices
                        (e.g. MVI) where a small secondary peak in the
                        long tail causes valley detection to pick a
                        threshold far into the tail (too strict).

    Returns
    -------
    (float, str) : (threshold value, method used: 'bimodal', 'otsu', or 'empty')
    """
    arr = index_array

    if valid_mask is not None:
        # Constrain histogram to candidate zone
        finite_in_mask = valid_mask & np.isfinite(arr)
        flat = arr[finite_in_mask].ravel()
    else:
        arr_work = arr.copy()
        if water_mask is not None:
            arr_work[water_mask] = np.nan
        flat = arr_work[np.isfinite(arr_work)].ravel()

    if len(flat) == 0:
        return float('nan'), 'empty'

    if force_otsu:
        return float(threshold_otsu(flat)), 'otsu_forced'

    counts, bin_edges = np.histogram(flat, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    peaks, _ = find_peaks(counts, distance=min_peak_distance)

    if len(peaks) >= 2:
        top2 = peaks[np.argsort(counts[peaks])[-2:]]
        top2 = np.sort(top2)
        valley_idx = np.argmin(counts[top2[0]:top2[1]]) + top2[0]
        return float(bin_centers[valley_idx]), 'bimodal'

    otsu_t = float(threshold_otsu(flat))
    return otsu_t, 'otsu'


# =============================================================================
# 3. MODIFIED: apply_adaptive_threshold (added candidate_mask parameter)
# =============================================================================

def apply_adaptive_threshold(indices: dict, scene_id: str,
                              water_mask: np.ndarray = None,
                              candidate_mask: np.ndarray = None,
                              force_otsu_indices: list = None) -> dict:
    """
    Apply per-scene adaptive threshold to each index.
    Uses bimodal valley detection; falls back to Otsu if scene is unimodal.

    Parameters
    ----------
    indices            : dict from compute_all_indices()
    scene_id           : str identifier for logging
    water_mask         : optional 2D bool array; water excluded
                         (ignored if candidate_mask is provided)
    candidate_mask     : optional 2D bool array; if provided, thresholds are
                         derived ONLY from pixels within this zone
                         (recommended: coastal candidate mask)
    force_otsu_indices : optional list of index names (e.g. ['MVI']) that
                         should skip valley-peak detection and use Otsu
                         directly. Use for right-skewed indices where a
                         small secondary peak in the long tail causes
                         valley detection to pick an overly strict threshold.

    Returns
    -------
    dict of thresholds per index name (NaN if detection failed entirely)
    """
    domain = "candidate-zone" if candidate_mask is not None else "scene"
    force_otsu_indices = force_otsu_indices or []
    print(f"\n  Adaptive thresholds (domain={domain}) -- scene: {scene_id}")
    thresholds = {}
    for name, arr in indices.items():
        t, method = detect_bimodal_threshold(
            arr,
            water_mask=water_mask,
            valid_mask=candidate_mask,
            force_otsu=(name in force_otsu_indices),
        )
        thresholds[name] = t
        print(f"  {name:<8}: threshold = {t:.4f}  [{method}]")
    return thresholds

# =============================================================================
# continuum.py
# Hyperspectral diagnostic features via continuum removal (CR)
# =============================================================================
#
# Purpose
# -------
# Derive physically-meaningful diagnostic features from Tanager's full 426-band
# spectrum that CANNOT be reproduced from broadband multispectral sensors
# (Sentinel-2, Landsat). These features are computed from many narrow contiguous
# bands around a target absorption feature, then reduced to a single per-pixel
# value (absorption depth) used as an additional input to RF/XGBoost.
#
# This complements the 5 spectral indices: indices drive the adaptive threshold
# and pseudo-labels (unchanged), while absorption depth adds spectral-shape
# information that distinguishes mangrove leaf water content from other coastal
# vegetation.
#
# Method
# ------
# Continuum removal isolates an absorption feature by dividing the reflectance
# spectrum by its convex-hull (or straight-line) continuum. Band depth at the
# absorption minimum quantifies feature strength:
#
#     D = 1 - (R_b / R_c)
#
# where R_b is reflectance at the absorption band and R_c is the interpolated
# continuum reflectance at the same wavelength (van der Meer, 2004).
#
# References
# ----------
# Clark, R.N. & Roush, T.L. (1984). Reflectance Spectroscopy: Quantitative
#   Analysis Techniques for Remote Sensing Applications. Journal of Geophysical
#   Research, 89(B7), 6329-6340. doi:10.1029/JB089iB07p06329
#   (seminal definition of continuum and band depth)
#
# van der Meer, F. (2004). Analysis of Spectral Absorption Features in
#   Hyperspectral Imagery. International Journal of Applied Earth Observation
#   and Geoinformation, 5(1), 55-68. doi:10.1016/j.jag.2003.09.001
#   (band-depth formula D = 1 - Rb/Rc applied to discrete-band imagery, with
#    linear interpolation of the continuum across shoulder bands)
#
# Target absorption feature for mangrove
# --------------------------------------
# 1640 nm SWIR water-absorption region. Mangroves grow in saline conditions and
# their leaf water content and internal structure produce a characteristic
# water-absorption response. The 1640 nm band already anchors NDMI/MNDWI/MVI,
# so the diagnostic feature is built around a wavelength already central to the
# pipeline (Gao, 1996, NDWI; Xu, 2006, MNDWI).
# =============================================================================

import numpy as np


# =============================================================================
# 1. Per-pixel absorption depth at a single feature
# =============================================================================

def absorption_depth(data: dict,
                     left_nm: float,
                     center_nm: float,
                     right_nm: float,
                     tolerance_nm: float = 15.0) -> np.ndarray:
    """
    Compute continuum-removed band depth at a target absorption feature.

    The continuum is a straight line between the two shoulder wavelengths
    (left_nm, right_nm). Band depth at center_nm is:

        D = 1 - (R_center / R_continuum_at_center)

    Higher D = deeper absorption = stronger feature.

    Parameters
    ----------
    data         : dict from load_hdf5() with keys 'reflectance' (B,H,W) and
                   'wavelengths' (B,)
    left_nm       : left shoulder wavelength (continuum start), e.g. 1500
    center_nm     : absorption minimum wavelength, e.g. 1640
    right_nm      : right shoulder wavelength (continuum end), e.g. 1750
    tolerance_nm  : max allowed deviation when matching nearest band

    Returns
    -------
    2D np.ndarray (H, W) of band depth values, NaN where reflectance invalid

    Notes
    -----
    Continuum line is interpolated linearly between shoulders, following
    van der Meer (2004) for discrete-band (non-continuous) imagery.
    """
    wl  = data['wavelengths']
    refl = data['reflectance']

    def _nearest_band(target):
        idx = int(np.argmin(np.abs(wl - target)))
        if np.abs(wl[idx] - target) > tolerance_nm:
            raise ValueError(
                f"No band within {tolerance_nm} nm of {target} nm "
                f"(nearest = {wl[idx]:.1f} nm)"
            )
        return idx

    i_left   = _nearest_band(left_nm)
    i_center = _nearest_band(center_nm)
    i_right  = _nearest_band(right_nm)

    wl_left,   wl_center,  wl_right  = wl[i_left], wl[i_center], wl[i_right]
    R_left     = refl[i_left]
    R_center   = refl[i_center]
    R_right    = refl[i_right]

    # Linear continuum interpolated at center wavelength
    # R_c = R_left + (R_right - R_left) * (wl_center - wl_left)/(wl_right - wl_left)
    frac  = (wl_center - wl_left) / (wl_right - wl_left)
    R_cont = R_left + (R_right - R_left) * frac

    # Band depth; guard against divide-by-zero / invalid continuum
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = 1.0 - (R_center / R_cont)

    # Invalid where any input is non-finite, continuum <= 0, or continuum
    # too small (near-zero Rc causes extreme depth values in cloud/shadow
    # pixels where SWIR reflectance collapses; threshold 0.05 is conservative
    # for surface-reflectance data -- Gomez et al. (2008) note CR degrades
    # at low-SNR conditions typical of airborne/satellite SWIR acquisition)
    invalid = (
        ~(np.isfinite(R_center) & np.isfinite(R_cont))
        | (R_cont < 0.05)
    )
    depth[invalid] = np.nan

    # Clip to physically plausible range [-1, 1]
    # D > 1 : R_center < 0 (sensor noise/artefact)
    # D < -1: R_center >> R_cont (specular reflection / saturated pixel)
    depth = np.clip(depth, -1.0, 1.0)
    # Re-apply NaN after clip (clip does not preserve NaN)
    depth[invalid] = np.nan

    n_valid = int(np.isfinite(depth).sum())
    print(f"  Absorption depth @ {wl_center:.0f} nm "
          f"(shoulders {wl_left:.0f}/{wl_right:.0f}):")
    print(f"  valid px       : {n_valid:,}")
    print(f"  depth range    : {np.nanmin(depth):.4f} to {np.nanmax(depth):.4f}")
    print(f"  depth mean     : {np.nanmean(depth):.4f}")

    return depth.astype(np.float32)


# =============================================================================
# 2. Red-edge Inflection Point (REIP)
# =============================================================================

# Default red-edge window for mangrove mapping.
# Based on Sangatta spectral diagnostics: mangrove sigmoid rise from ~660nm
# to ~770nm, with maximum slope (REIP) around 710-730nm. Non-mangrove
# shows a much flatter response in the same region, giving strong separability.
REIP_WINDOW = {'left_nm': 670.0, 'right_nm': 760.0}


def compute_reip(data: dict,
                 left_nm: float = 670.0,
                 right_nm: float = 760.0,
                 tolerance_nm: float = 10.0) -> np.ndarray:
    """
    Compute Red-edge Inflection Point (REIP) per pixel.

    REIP is the wavelength at which the first derivative of the reflectance
    spectrum reaches its maximum within the red-edge region [left_nm, right_nm].
    It is computed by finding the peak of the first-order finite difference
    across all bands in the window.

    For mangrove, the sigmoid rise from red absorption (~660nm) to NIR plateau
    (~760nm) produces a sharp, well-defined REIP around 710-730nm. Non-vegetated
    and non-mangrove surfaces show a flatter red-edge, with REIP either absent
    or at different wavelengths. This makes REIP a physically-grounded
    separability feature that requires many narrow contiguous bands and cannot
    be reliably reproduced from broadband multispectral sensors
    (Sentinel-2, Landsat).

    Parameters
    ----------
    data        : dict from load_hdf5() with 'reflectance' (B,H,W) and
                  'wavelengths' (B,)
    left_nm     : start of red-edge window (nm), default 670
    right_nm    : end of red-edge window (nm), default 760
    tolerance_nm: tolerance for nearest-band matching

    Returns
    -------
    2D np.ndarray (H, W) of REIP values in nm. NaN where spectrum is invalid
    or no clear inflection point found.

    Notes
    -----
    Method: maximum first derivative (finite difference) across window bands,
    following standard remote sensing practice.
    Output is the wavelength (nm) of peak derivative, not the derivative value.
    """
    wl   = data['wavelengths']
    refl = data['reflectance']
    H, W = refl.shape[1], refl.shape[2]

    def _nearest_band(target):
        idx = int(np.argmin(np.abs(wl - target)))
        if np.abs(wl[idx] - target) > tolerance_nm:
            raise ValueError(
                f"No band within {tolerance_nm} nm of {target} nm "
                f"(nearest = {wl[idx]:.1f} nm)"
            )
        return idx

    i_left  = _nearest_band(left_nm)
    i_right = _nearest_band(right_nm)

    # Extract bands in window: shape (n_bands_window, H, W)
    bands   = refl[i_left:i_right + 1].astype(np.float32)
    wl_win  = wl[i_left:i_right + 1]
    n_bands = bands.shape[0]

    if n_bands < 3:
        raise ValueError(
            f"Only {n_bands} bands in [{left_nm}, {right_nm}] nm window -- "
            "too few for derivative computation"
        )

    # First derivative: finite difference between adjacent bands
    # Shape: (n_bands-1, H, W)
    dR    = np.diff(bands, axis=0)
    dwl   = np.diff(wl_win)                   # wavelength step per interval (nm)
    deriv = dR / dwl[:, None, None]            # normalize by wavelength step

    # Wavelength at midpoint of each derivative interval
    wl_mid = (wl_win[:-1] + wl_win[1:]) / 2.0  # shape (n_bands-1,)

    # REIP = wavelength of peak derivative per pixel
    peak_idx  = np.argmax(deriv, axis=0)       # shape (H, W)
    reip_map  = wl_mid[peak_idx]               # shape (H, W)

    # Mask invalid pixels: any NaN/inf in the window -> NaN REIP
    invalid = ~np.all(np.isfinite(bands), axis=0)
    reip_map[invalid] = np.nan

    # Also mask pixels where peak derivative is non-positive (no real red-edge)
    peak_deriv_vals = deriv[peak_idx,
                            np.arange(H)[:, None],
                            np.arange(W)[None, :]]
    reip_map[peak_deriv_vals <= 0] = np.nan

    n_valid = int(np.isfinite(reip_map).sum())
    print(f"  REIP (red-edge inflection point):")
    print(f"  window         : {wl_win[0]:.0f} - {wl_win[-1]:.0f} nm "
          f"({n_bands} bands)")
    print(f"  valid px       : {n_valid:,}")
    print(f"  REIP range     : {np.nanmin(reip_map):.1f} to "
          f"{np.nanmax(reip_map):.1f} nm")
    print(f"  REIP mean      : {np.nanmean(reip_map):.1f} nm")

    return reip_map.astype(np.float32)


# =============================================================================
# 3. Convenience: mangrove water-absorption feature at 1640 nm
# =============================================================================

# Default shoulders for the 1640 nm SWIR water-absorption feature.
# Shoulders chosen at reflectance highs flanking the water absorption trough,
# avoiding the deeper 1400 nm and 1900 nm atmospheric water bands.
WATER_1640 = {'left_nm': 1500.0, 'center_nm': 1640.0, 'right_nm': 1780.0}


def absorption_depth_1640(data: dict, tolerance_nm: float = 15.0) -> np.ndarray:
    """
    Mangrove leaf-water absorption depth at 1640 nm (SWIR1).
    Wrapper around absorption_depth() with preset shoulders (WATER_1640).

    See module docstring for physical rationale and references.
    """
    return absorption_depth(
        data,
        left_nm=WATER_1640['left_nm'],
        center_nm=WATER_1640['center_nm'],
        right_nm=WATER_1640['right_nm'],
        tolerance_nm=tolerance_nm,
    )

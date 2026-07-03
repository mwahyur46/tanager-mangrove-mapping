# =============================================================================
# continuum.py
# Hyperspectral diagnostic feature: Red-edge Inflection Point (REIP)
# =============================================================================
#
# Purpose
# -------
# Derive a physically-meaningful diagnostic feature from Tanager's full 426-band
# spectrum that CANNOT be reproduced from broadband multispectral sensors
# (Sentinel-2, Landsat). REIP is computed from many narrow contiguous bands
# across the red-edge region, then reduced to a single per-pixel value (the
# wavelength of maximum reflectance slope) used as an additional input to
# RF/XGBoost.
#
# This complements the 5 spectral indices: indices drive the adaptive threshold
# and pseudo-labels (unchanged), while REIP adds spectral-shape information that
# distinguishes mangrove canopy structure from other coastal vegetation.
#
# Note (history)
# --------------
# A continuum-removal absorption-depth feature at 1640 nm was previously
# evaluated here but removed: the Sangatta SWIR spectrum rises monotonically
# across 1500-1780 nm, so no genuine absorption trough exists (continuum values
# collapse toward zero, producing degenerate depth values). Its contribution to
# classification accuracy was negligible (delta Kappa within noise). REIP is
# retained as the sole hyperspectral hook.
# =============================================================================

import numpy as np


# =============================================================================
# Red-edge Inflection Point (REIP)
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

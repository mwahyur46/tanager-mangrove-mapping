# =============================================================================
# preprocessing.py
# HDF5 loader, HDF5->GeoTIFF conversion, band extractor,
# spectral indices, adaptive threshold
# =============================================================================

import re
import h5py
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_bounds
from rasterio.transform import Affine
from rasterio.crs import CRS
from pathlib import Path
from scipy.signal import find_peaks
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu


# =============================================================================
# 1. HDF5 I/O
# =============================================================================

# =============================================================================
# Target wavelengths (nm) — selected from Tanager 426-band VSWIR spectrum
# based on the original definitions of each spectral index.
#
# References (band positions in nm):
#   Green 560  : Xu (2006) MNDWI definition           doi:10.1080/01431160600589179
#                Baloloy et al. (2020) MVI            doi:10.1016/j.isprsjprs.2020.06.001
#   Red   660  : Huete (1988) SAVI                    doi:10.1016/0034-4257(88)90106-X
#   NIR   860  : Gao (1996) NDWI/NDMI                 doi:10.1016/S0034-4257(96)00067-3
#                Baloloy et al. (2020) MVI            doi:10.1016/j.isprsjprs.2020.06.001
#                Huete (1988) SAVI                    doi:10.1016/0034-4257(88)90106-X
#                Rahmila et al. (2026) EMI            doi:10.1080/21580103.2026.2616443
#   SWIR1 1640 : Xu (2006) MNDWI                      doi:10.1080/01431160600589179
#                Baloloy et al. (2020) MVI            doi:10.1016/j.isprsjprs.2020.06.001
#                Gao (1996) NDMI                      doi:10.1016/S0034-4257(96)00067-3
#   SWIR2 2200 : Rahmila et al. (2026) EMI            doi:10.1080/21580103.2026.2616443
# =============================================================================
RELEVANT_WAVELENGTHS = {
    'green' : 560,    # MNDWI, MVI
    'red'   : 660,    # SAVI
    'nir'   : 860,    # NDMI, MVI, SAVI, EMI
    'swir1' : 1640,   # NDMI, MNDWI, MVI
    'swir2' : 2200,   # EMI
}


def inspect_hdf5(filepath: str) -> None:
    """
    Print the full HDF5 tree (groups, datasets, shapes, dtypes, key attributes).
    Run this first on a new Tanager file to confirm the internal structure
    before implementing load_hdf5().
    """
    def _print_tree(name, obj):
        indent = "  " * name.count("/")
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{name}  shape={obj.shape}  dtype={obj.dtype}")
        else:
            print(f"{indent}{name}/")

    with h5py.File(filepath, 'r') as f:
        print(f"File: {filepath}")
        print(f"Root keys: {list(f.keys())}")
        f.visititems(_print_tree)
        # Print top-level attributes (often contain CRS / geotransform info)
        if f.attrs:
            print("\nRoot attributes:")
            for k, v in f.attrs.items():
                print(f"  {k}: {v}")


def _parse_ul(meta: str) -> tuple:
    """Extract UpperLeftPointMtrs (x, y) from StructMetadata string."""
    match = re.search(r'UpperLeftPointMtrs=\(([0-9.\-]+),([0-9.\-]+)\)', meta)
    if not match:
        raise KeyError("UpperLeftPointMtrs not found in StructMetadata.")
    return float(match.group(1)), float(match.group(2))


def _parse_pixel(meta: str) -> float:
    """Derive pixel size (metres) from UL/LR corners + XDim in StructMetadata."""
    ul   = re.search(r'UpperLeftPointMtrs=\(([0-9.\-]+),([0-9.\-]+)\)', meta)
    lr   = re.search(r'LowerRightMtrs=\(([0-9.\-]+),([0-9.\-]+)\)', meta)
    xdim = re.search(r'XDim=(\d+)', meta)
    if not (ul and lr and xdim):
        raise KeyError("Cannot parse pixel size from StructMetadata.")
    ul_x = float(ul.group(1))
    lr_x = float(lr.group(1))
    return (lr_x - ul_x) / int(xdim.group(1))


def load_hdf5(filepath: str) -> dict:
    """
    Load Tanager-1 HDF5 scene and return metadata + reflectance array.

    Parameters
    ----------
    filepath : str
        Path to Tanager .h5 file.

    Returns
    -------
    dict with keys:
        'reflectance' : np.ndarray, shape (bands, rows, cols), float32, scaled [0, 1]
        'wavelengths' : np.ndarray, shape (bands,), float32 — reconstructed from sensor spec
        'crs'         : rasterio.crs.CRS — read from epsg_code attribute
        'transform'   : rasterio.Affine — derived from StructMetadata UL corner + pixel size

    Notes
    -----
    Tanager HDF5 structure confirmed via inspect_hdf5():
        Reflectance : HDFEOS/GRIDS/HYP/Data Fields/surface_reflectance  (426, 850, 810)
        Wavelengths : not stored in file — reconstructed as np.linspace(380, 2500, 426)
        CRS         : HDFEOS/GRIDS/HYP attrs['epsg_code']  (e.g. 32650 for Sangatta)
        Geotransform: parsed from HDFEOS INFORMATION/StructMetadata.0 (UTM, metres)
    """
    REFLECTANCE_PATH = 'HDFEOS/GRIDS/HYP/Data Fields/surface_reflectance'

    with h5py.File(filepath, 'r') as f:
        reflectance = f[REFLECTANCE_PATH][:]                        # (426, rows, cols)

        # Wavelength not stored in file — reconstruct from Tanager sensor spec
        wavelengths = np.linspace(380, 2500, reflectance.shape[0]).astype(np.float32)

        # CRS — read epsg_code dynamically (varies per scene/UTM zone)
        epsg = int(f['HDFEOS/GRIDS/HYP'].attrs['epsg_code'])
        crs  = CRS.from_epsg(epsg)

        # Geotransform — parse UL corner + pixel size from StructMetadata
        meta      = f['HDFEOS INFORMATION/StructMetadata.0'][()].decode()
        ul        = _parse_ul(meta)
        pixel     = _parse_pixel(meta)
        transform = Affine(pixel, 0, ul[0], 0, -pixel, ul[1])

    reflectance = reflectance.astype(np.float32)
    # Mask nodata pixels — Tanager uses -9999 as nodata sentinel
    reflectance[reflectance <= -9999] = np.nan
    # No scaling needed — Tanager SR already in [0, 1] range

    print(f"  HDF5 loaded     : {Path(filepath).name}")
    print(f"  Shape (B,H,W)   : {reflectance.shape}")
    print(f"  CRS             : EPSG:{epsg}")
    print(f"  Pixel size      : {pixel:.1f} m")
    print(f"  Wavelength range: {wavelengths.min():.0f}-{wavelengths.max():.0f} nm")

    return {
        'reflectance' : reflectance,
        'wavelengths' : wavelengths,
        'crs'         : crs,
        'transform'   : transform,
    }


def extract_band(data: dict, wavelength_nm: float, tolerance_nm: float = 5.0) -> np.ndarray:
    """
    Extract the band closest to a target wavelength.

    Parameters
    ----------
    data          : dict from load_hdf5()
    wavelength_nm : target wavelength in nm
    tolerance_nm  : maximum allowed deviation

    Returns
    -------
    2D np.ndarray of reflectance values
    """
    wavelengths = data['wavelengths']
    idx = np.argmin(np.abs(wavelengths - wavelength_nm))
    if np.abs(wavelengths[idx] - wavelength_nm) > tolerance_nm:
        raise ValueError(f"No band within {tolerance_nm} nm of {wavelength_nm} nm")
    return data['reflectance'][idx]


# =============================================================================
# 2. HDF5 -> GeoTIFF Conversion
# =============================================================================

def hdf5_to_geotiff(hdf5_path: str, output_dir: str,
                    scene_id: str, site: str = None,
                    wavelengths_nm: dict = None) -> str:
    """
    Convert relevant bands from Tanager HDF5 to a single multiband GeoTIFF.
    All bands stacked into one file — 1 file per scene instead of 1 per band.

    Parameters
    ----------
    hdf5_path      : path to input .h5 file (in data/raw/)
    output_dir     : output directory (data/processed/)
    scene_id       : scene identifier, e.g. '20250302_030003_92_4001'
    site           : site name prefix, e.g. 'sangatta'
    wavelengths_nm : dict of {band_name: wavelength_nm} to export.
                     Defaults to RELEVANT_WAVELENGTHS.

    Returns
    -------
    str : path to output multiband GeoTIFF
    """
    if wavelengths_nm is None:
        wavelengths_nm = RELEVANT_WAVELENGTHS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data       = load_hdf5(hdf5_path)
    band_names = list(wavelengths_nm.keys())
    n_bands    = len(band_names)
    prefix     = f"{site}_{scene_id}" if site else scene_id
    out_path   = output_dir / f"{prefix}_bands.tif"

    # Extract all bands first to get shape
    arrays = [extract_band(data, wl) for wl in wavelengths_nm.values()]

    with rasterio.open(
        out_path, 'w',
        driver='GTiff',
        height=arrays[0].shape[0],
        width=arrays[0].shape[1],
        count=n_bands,
        dtype=arrays[0].dtype,
        crs=data['crs'],
        transform=data['transform'],
        compress='lzw'
    ) as dst:
        for i, (arr, name, wl) in enumerate(zip(arrays, band_names, wavelengths_nm.values()), start=1):
            dst.write(arr, i)
            dst.update_tags(i, name=name, wavelength_nm=wl)   # band metadata
            print(f"  Band {i}/{n_bands}  {name:<8} ({wl} nm)")

    print(f"  Output        : {out_path.name}")
    return str(out_path)


def load_geotiff_bands(processed_dir: str, scene_id: str, site: str = None) -> dict:
    """
    Load multiband GeoTIFF (output of hdf5_to_geotiff) into a dict of arrays.
    Use this instead of load_hdf5() once conversion is done.

    Parameters
    ----------
    processed_dir : path to data/processed/
    scene_id      : scene identifier string (e.g. '20250302_030003_92_4001')
    site          : site name prefix (e.g. 'sangatta')

    Returns
    -------
    dict: {'reflectance_bands': {band_name: 2D array}, 'crs': ..., 'transform': ...}
    """
    processed_dir = Path(processed_dir)
    prefix        = f"{site}_{scene_id}" if site else scene_id
    tif_path      = processed_dir / f"{prefix}_bands.tif"

    if not tif_path.exists():
        raise FileNotFoundError(f"Multiband GeoTIFF not found: {tif_path}")

    bands = {}
    meta  = {}
    with rasterio.open(tif_path) as src:
        meta['crs']       = src.crs
        meta['transform'] = src.transform
        for i in range(1, src.count + 1):
            # Read band name from tag; fall back to 'band_{i}' if missing
            tag       = src.tags(i)
            band_name = tag.get('name', f'band_{i}')
            bands[band_name] = src.read(i).astype(np.float32)

    print(f"  Bands loaded  : {list(bands.keys())}")
    print(f"  CRS           : {meta['crs']}")
    return {'reflectance_bands': bands, **meta}


def save_raster(array: np.ndarray,
                output_path: str,
                reference_data: dict,
                nodata: float = None,
                dtype: str = None) -> str:
    """
    Write a 2D numpy array to a GeoTIFF using CRS + transform from a reference dict.

    Parameters
    ----------
    array          : 2D np.ndarray to write (e.g. extent map, AGB map)
    output_path    : full path for output .tif
    reference_data : dict with 'crs' and 'transform' keys (from load_geotiff_bands())
    nodata         : value to mark as nodata (default: -1 for int, NaN for float)
    dtype          : rasterio dtype string; inferred from array if None

    Returns
    -------
    str : output_path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dtype is None:
        dtype = rasterio.dtypes.get_minimum_dtype(int(array.max())) if np.issubdtype(array.dtype, np.integer) \
                else 'float32'
    if nodata is None:
        nodata = -1 if np.issubdtype(array.dtype, np.integer) else float('nan')

    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=dtype,
        crs=reference_data['crs'],
        transform=reference_data['transform'],
        nodata=nodata,
        compress='lzw',
    ) as dst:
        dst.write(array.astype(dtype), 1)

    print(f"  Raster saved   : {output_path.name}")
    return str(output_path)


# =============================================================================
# 3. Spectral Indices
# =============================================================================

def _get_band(data: dict, band_name: str) -> np.ndarray:
    """Helper: get band array from load_geotiff_bands() output."""
    bands = data['reflectance_bands']
    if band_name not in bands:
        raise KeyError(f"Band '{band_name}' not found. Available: {list(bands.keys())}")
    return bands[band_name].astype(np.float32)


def compute_ndmi(data: dict) -> np.ndarray:
    """NDMI = (NIR - SWIR1) / (NIR + SWIR1)"""
    nir, swir1 = _get_band(data, 'nir'), _get_band(data, 'swir1')
    return np.clip((nir - swir1) / (nir + swir1 + 1e-10), -1, 1)


def compute_mndwi(data: dict) -> np.ndarray:
    """MNDWI = (Green - SWIR1) / (Green + SWIR1)"""
    green, swir1 = _get_band(data, 'green'), _get_band(data, 'swir1')
    return np.clip((green - swir1) / (green + swir1 + 1e-10), -1, 1)


def compute_mvi(data: dict) -> np.ndarray:
    """MVI = (NIR - Green) / (SWIR1 - Green)"""
    nir   = _get_band(data, 'nir')
    green = _get_band(data, 'green')
    swir1 = _get_band(data, 'swir1')
    return np.clip((nir - green) / (swir1 - green + 1e-10), -1, 20)


def compute_savi(data: dict, L: float = 0.5) -> np.ndarray:
    """SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)"""
    nir = _get_band(data, 'nir')
    red = _get_band(data, 'red')
    return np.clip(((nir - red) / (nir + red + L + 1e-10)) * (1 + L), -1, 1)


def compute_emi(data: dict) -> np.ndarray:
    """
    EMI = (NIR - SWIR2) / (NIR + SWIR2)
    Reference: Rahmila et al. (2026) doi:10.1080/21580103.2026.2616443
    """
    nir   = _get_band(data, 'nir')
    swir2 = _get_band(data, 'swir2')
    return np.clip((nir - swir2) / (nir + swir2 + 1e-10), -1, 1)


def compute_all_indices(data: dict) -> dict:
    """Compute all 5 indices and return as dict of 2D arrays."""
    return {
        'NDMI'  : compute_ndmi(data),
        'MNDWI' : compute_mndwi(data),
        'MVI'   : compute_mvi(data),
        'SAVI'  : compute_savi(data),
        'EMI'   : compute_emi(data),
    }


# =============================================================================
# 4. Adaptive Threshold (Core Innovation)
# =============================================================================

def compute_water_mask(data: dict, mndwi_threshold: float = 0.0) -> np.ndarray:
    """
    Generate a boolean mask of open water pixels using MNDWI.
    Masking water before bimodal detection makes mangrove/non-mangrove
    peaks cleaner by removing a third mode (open water).

    Parameters
    ----------
    data             : dict from load_geotiff_bands()
    mndwi_threshold  : pixels with MNDWI > threshold are flagged as water

    Returns
    -------
    2D bool np.ndarray — True = water pixel (exclude from thresholding)
    """
    mndwi = compute_mndwi(data)
    return mndwi > mndwi_threshold


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

# =============================================================================
# 5. Adaptive coastal candidate mask
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
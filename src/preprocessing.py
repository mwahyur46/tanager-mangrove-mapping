# =============================================================================
# preprocessing.py
# HDF5 loader, HDF5->GeoTIFF conversion, band extractor,
# spectral indices, adaptive threshold
# =============================================================================

import h5py
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu


# =============================================================================
# 1. HDF5 I/O
# =============================================================================

# Target wavelengths (nm) needed for 5 indices — only these bands are exported
RELEVANT_WAVELENGTHS = {
    'green' : 560,    # MNDWI, MVI
    'red'   : 665,    # SAVI
    'nir'   : 860,    # NDMI, MNDWI, MVI, SAVI, EMI
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


def load_hdf5(filepath: str) -> dict:
    """
    Load Tanager HDF5 scene and return metadata + reflectance array.

    Parameters
    ----------
    filepath : str
        Path to Tanager .h5 file.

    Returns
    -------
    dict with keys: 'reflectance' (np.ndarray, shape [bands, rows, cols]),
                    'wavelengths' (np.ndarray, shape [bands]),
                    'crs' (rasterio.crs.CRS),
                    'transform' (rasterio.Affine)

    Notes
    -----
    Path constants below follow the EMIT L2A / Tanager-1 HDF5 convention.
    Run inspect_hdf5() on your file first to verify — adjust the path
    strings if your file uses different group names.
    """
    # ---- Adjust these path strings to match actual HDF5 structure ----
    REFLECTANCE_PATH = 'reflectance/reflectance'   # (bands, rows, cols) float32
    WAVELENGTH_PATH  = 'sensor_band_parameters/wavelengths'
    # Geolocation: look for either a geolocation group or root attributes
    LON_PATH         = 'location/lon'              # (rows, cols)
    LAT_PATH         = 'location/lat'              # (rows, cols)
    # ------------------------------------------------------------------

    with h5py.File(filepath, 'r') as f:
        reflectance = f[REFLECTANCE_PATH][:]        # (bands, rows, cols)
        wavelengths = f[WAVELENGTH_PATH][:]

        # Build affine transform from corner coordinates
        if LON_PATH in f and LAT_PATH in f:
            lons = f[LON_PATH][:]
            lats = f[LAT_PATH][:]
            west, east = float(lons.min()), float(lons.max())
            south, north = float(lats.min()), float(lats.max())
            _, rows, cols = reflectance.shape
            transform = from_bounds(west, south, east, north, cols, rows)
        else:
            # Fall back to root attributes (some Tanager variants store geotransform here)
            gt = f.attrs.get('geotransform', None)
            if gt is not None:
                from rasterio.transform import Affine
                transform = Affine(*gt[:6])
            else:
                raise KeyError(
                    "Cannot find geolocation data. Run inspect_hdf5() "
                    "and update LON_PATH / LAT_PATH accordingly."
                )

        crs_wkt = f.attrs.get('coordinate_system_string', 'EPSG:4326')
        crs = CRS.from_user_input(crs_wkt)

    reflectance = reflectance.astype(np.float32)
    # Scale to [0, 1] reflectance if stored as integers (common: scale factor 0.0001)
    if reflectance.max() > 10:
        reflectance = reflectance * 0.0001

    print(f"  HDF5 loaded    : {Path(filepath).name}")
    print(f"  Shape (B,H,W)  : {reflectance.shape}")
    print(f"  Wavelength range: {wavelengths.min():.0f}–{wavelengths.max():.0f} nm")

    return {
        'reflectance' : reflectance,
        'wavelengths' : wavelengths.astype(np.float32),
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
                    scene_id: str,
                    wavelengths_nm: dict = None) -> dict:
    """
    Convert relevant bands from Tanager HDF5 to a multi-band GeoTIFF.
    Only exports bands needed for spectral indices (5-8 bands, not all 426).

    Analogous to unpacking only the tools you need from a 426-slot toolbox.

    Parameters
    ----------
    hdf5_path    : path to input .h5 file (in data/raw/)
    output_dir   : output directory (data/processed/)
    scene_id     : used for output filename, e.g. '20250302_030003_92_4001'
    wavelengths_nm : dict of {band_name: wavelength_nm} to export.
                     Defaults to RELEVANT_WAVELENGTHS.

    Returns
    -------
    dict: {band_name: output_tif_path}
    """
    if wavelengths_nm is None:
        wavelengths_nm = RELEVANT_WAVELENGTHS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_hdf5(hdf5_path)

    output_paths = {}
    for band_name, wl in wavelengths_nm.items():
        band_array = extract_band(data, wl)
        out_path = output_dir / f"{scene_id}_{band_name}_{wl}nm.tif"

        with rasterio.open(
            out_path, 'w',
            driver='GTiff',
            height=band_array.shape[0],
            width=band_array.shape[1],
            count=1,
            dtype=band_array.dtype,
            crs=data['crs'],
            transform=data['transform'],
            compress='lzw'          # lossless compression — keeps file size manageable
        ) as dst:
            dst.write(band_array, 1)

        output_paths[band_name] = str(out_path)
        print(f"  {band_name:<8} ({wl} nm) -> {out_path.name}")

    return output_paths


def load_geotiff_bands(processed_dir: str, scene_id: str) -> dict:
    """
    Load previously converted GeoTIFF bands back into a dict of arrays.
    Use this instead of load_hdf5() once conversion is done.

    Parameters
    ----------
    processed_dir : path to data/processed/
    scene_id      : scene identifier string

    Returns
    -------
    dict: {'reflectance': {band_name: 2D array}, 'crs': ..., 'transform': ...}
    """
    processed_dir = Path(processed_dir)
    # Match only band files: {scene_id}_{band_name}_{wavelength}nm.tif
    # Excludes derived outputs like {scene_id}_mangrove_extent.tif
    tif_files = [
        p for p in processed_dir.glob(f"{scene_id}_*.tif")
        if p.stem.endswith('nm')
    ]

    if not tif_files:
        raise FileNotFoundError(f"No band GeoTIFFs found for scene {scene_id} in {processed_dir}")

    bands = {}
    meta = {}
    for tif in sorted(tif_files):
        # parse band name from filename: sceneID_bandname_WAVELENGTHnm.tif
        parts = tif.stem.replace(scene_id + '_', '').split('_')
        band_name = parts[0]
        with rasterio.open(tif) as src:
            bands[band_name] = src.read(1).astype(np.float32)
            if not meta:
                meta['crs']       = src.crs
                meta['transform'] = src.transform

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
    return (nir - swir1) / (nir + swir1 + 1e-10)


def compute_mndwi(data: dict) -> np.ndarray:
    """MNDWI = (Green - SWIR1) / (Green + SWIR1)"""
    green, swir1 = _get_band(data, 'green'), _get_band(data, 'swir1')
    return (green - swir1) / (green + swir1 + 1e-10)


def compute_mvi(data: dict) -> np.ndarray:
    """MVI = (NIR - Green) / (SWIR1 - Green)"""
    nir   = _get_band(data, 'nir')
    green = _get_band(data, 'green')
    swir1 = _get_band(data, 'swir1')
    return (nir - green) / (swir1 - green + 1e-10)


def compute_savi(data: dict, L: float = 0.5) -> np.ndarray:
    """SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)"""
    nir = _get_band(data, 'nir')
    red = _get_band(data, 'red')
    return ((nir - red) / (nir + red + L + 1e-10)) * (1 + L)


def compute_emi(data: dict) -> np.ndarray:
    """
    EMI = (NIR - SWIR2) / (NIR + SWIR2)
    Reference: Rahmila et al. (2026) doi:10.1080/21580103.2026.2616443
    """
    nir   = _get_band(data, 'nir')
    swir2 = _get_band(data, 'swir2')
    return (nir - swir2) / (nir + swir2 + 1e-10)


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
                              water_mask: np.ndarray = None) -> tuple:
    """
    Detect threshold between two modes in a histogram (bimodal distribution).
    Falls back to Otsu thresholding when fewer than 2 peaks are found.

    Parameters
    ----------
    index_array       : 2D np.ndarray of spectral index values
    n_bins            : number of histogram bins
    min_peak_distance : minimum bin separation between peaks
    water_mask        : optional 2D bool array — water pixels excluded before detection

    Returns
    -------
    (float, str) : (threshold value, method used — 'bimodal' or 'otsu')
    """
    arr = index_array.copy()
    if water_mask is not None:
        arr[water_mask] = np.nan

    flat = arr[np.isfinite(arr)].ravel()
    counts, bin_edges = np.histogram(flat, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    peaks, _ = find_peaks(counts, distance=min_peak_distance)

    if len(peaks) >= 2:
        top2 = peaks[np.argsort(counts[peaks])[-2:]]
        top2 = np.sort(top2)
        valley_idx = np.argmin(counts[top2[0]:top2[1]]) + top2[0]
        return float(bin_centers[valley_idx]), 'bimodal'

    # Fallback: Otsu thresholding (minimises intra-class variance)
    otsu_t = float(threshold_otsu(flat))
    return otsu_t, 'otsu'


def apply_adaptive_threshold(indices: dict, scene_id: str,
                              water_mask: np.ndarray = None) -> dict:
    """
    Apply per-scene adaptive threshold to each index.
    Uses bimodal valley detection; falls back to Otsu if scene is unimodal.

    Parameters
    ----------
    indices    : dict from compute_all_indices()
    scene_id   : str identifier for logging
    water_mask : optional 2D bool array from compute_water_mask()

    Returns
    -------
    dict of thresholds per index name (None if detection failed entirely)
    """
    print(f"\n  Adaptive thresholds — scene: {scene_id}")
    thresholds = {}
    for name, arr in indices.items():
        t, method = detect_bimodal_threshold(arr, water_mask=water_mask)
        thresholds[name] = t
        print(f"  {name:<8}: threshold = {t:.4f}  [{method}]")
    return thresholds

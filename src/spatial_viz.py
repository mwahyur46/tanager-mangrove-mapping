"""
Spatial Visualization Helpers

On-the-fly raster reprojection to EPSG:4326, RGB composite generation
with percentile contrast stretching, and standardized geographic map-axes
formatting - shared across all visualization cells in the competition and
journal notebooks to ensure a consistent look across all map panels.
"""

import os

# ---------------------------------------------------------------------------
# Fix PROJ database version mismatch: force GDAL/rasterio to use the same
# proj.db that pyproj ships with, preventing the "DATABASE.LAYOUT.VERSION.MINOR
# = 2 whereas >= 6 is expected" error that occurs when another PROJ installation
# (e.g. QGIS, OSGeo4W, user pip env) pollutes the search path.
# ---------------------------------------------------------------------------
try:
    import pyproj
    _proj_data = pyproj.datadir.get_data_dir()
    os.environ.setdefault('PROJ_LIB',  _proj_data)
    os.environ.setdefault('PROJ_DATA', _proj_data)
except Exception:
    pass

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.ticker as mticker

DST_CRS = 'EPSG:4326'


def reproject_raster_to_4326(src, band=1):
    """
    Reprojects a single-band raster to EPSG:4326 on-the-fly.

    Parameters
    ----------
    src  : open rasterio dataset
    band : int
        Band index to read (1-based, default 1).

    Returns
    -------
    tuple of (img_data, extent)
        img_data is a float32 2-D array; extent is
        [left, right, bottom, top] in geographic degrees.
    """
    from rasterio.crs import CRS as RasterioCRS
    import pyproj
    with rasterio.Env():
        # Bypass rasterio's PROJ DB lookup entirely by using pyproj
        _pyproj_4326 = pyproj.CRS.from_epsg(4326)
        dst_crs_obj = RasterioCRS.from_wkt(_pyproj_4326.to_wkt())
        
    if src.crs and src.crs.to_string() != DST_CRS and src.crs != dst_crs_obj:
        with rasterio.Env():
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs_obj, src.width, src.height, *src.bounds
            )
            img_data = np.empty((height, width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, band),
                destination=img_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs_obj,
                resampling=Resampling.nearest,
            )
        left   = transform.c
        right  = transform.c + transform.a * width
        top    = transform.f
        bottom = transform.f + transform.e * height
    else:
        img_data = src.read(band).astype(np.float32)
        left, bottom, right, top = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top

    return img_data, [left, right, bottom, top]


def reproject_array_to_4326(arr, crs, transform):
    """
    Reprojects an in-memory single-band numpy array to EPSG:4326 on-the-fly.
    Useful when the array is already in memory (not yet written to a GeoTIFF).

    Parameters
    ----------
    arr       : np.ndarray, shape (H, W)
    crs       : rasterio CRS or anything accepted by CRS.from_user_input()
        Source coordinate reference system.
    transform : affine.Affine
        Affine transform of the source array.

    Returns
    -------
    tuple of (img_data, extent)
        img_data is a float32 2-D array; extent is
        [left, right, bottom, top] in geographic degrees.
    """
    from rasterio.crs import CRS as RasterioCRS
    import pyproj

    with rasterio.Env():
        src_crs = crs if hasattr(crs, 'to_epsg') else RasterioCRS.from_user_input(crs)
        # Bypass rasterio's PROJ DB lookup entirely by using pyproj
        _pyproj_4326 = pyproj.CRS.from_epsg(4326)
        dst_crs_obj = RasterioCRS.from_wkt(_pyproj_4326.to_wkt())

    h, w    = arr.shape
    src_arr = arr.astype(np.float32)

    left   = transform.c
    top    = transform.f
    right  = transform.c + transform.a * w
    bottom = transform.f + transform.e * h   # e is negative for north-up

    if src_crs.to_string() == DST_CRS or src_crs == dst_crs_obj:
        return src_arr, [left, right, bottom, top]

    with rasterio.Env():
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs_obj, w, h,
            left=left, bottom=bottom, right=right, top=top,
        )
        dst_arr = np.empty((dst_height, dst_width), dtype=np.float32)
        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs_obj,
            resampling=Resampling.nearest,
        )
    dst_left   = dst_transform.c
    dst_right  = dst_transform.c + dst_transform.a * dst_width
    dst_top    = dst_transform.f
    dst_bottom = dst_transform.f + dst_transform.e * dst_height

    return dst_arr, [dst_left, dst_right, dst_bottom, dst_top]


def reproject_rgb_composite_to_4326(src, bands):
    """
    Extracts a 3-band composite from a raster dataset, applies a 2-98%
    contrast stretch on valid pixels, reprojects on-the-fly to EPSG:4326,
    and returns an RGBA array plus extent.

    Parameters
    ----------
    src   : open rasterio dataset
    bands : list of 3 band identifiers - integers (1-based) or strings
        such as 'B8', 'B11', 'B4'.

    Returns
    -------
    tuple of (rgba_img, extent)
        rgba_img is (H, W, 4) float32 in [0, 1]; alpha=0 for nodata pixels.
        extent is [left, right, bottom, top] in geographic degrees.
    """
    import re

    # 1. Resolve band indices
    resolved_indices = []
    descriptions = [d.lower().strip() if d else '' for d in src.descriptions]

    for identifier in bands:
        idx_found = None
        if isinstance(identifier, int) and 1 <= identifier <= src.count:
            idx_found = identifier
        else:
            id_str = str(identifier).lower().strip()
            for i, desc in enumerate(descriptions, start=1):
                if id_str == desc or desc.endswith(id_str) or id_str in desc:
                    idx_found = i
                    break
            if idx_found is None:
                digits = re.findall(r'\d+', id_str)
                if digits:
                    val = int(digits[0])
                    for i, desc in enumerate(descriptions, start=1):
                        desc_digits = re.findall(r'\d+', desc)
                        if desc_digits and int(desc_digits[-1]) == val:
                            idx_found = i
                            break
                    if idx_found is None and 1 <= val <= src.count:
                        idx_found = val
        if idx_found is None:
            print(f'[WARNING] Could not resolve band identifier "{identifier}", falling back to band 1.')
            idx_found = 1
        resolved_indices.append(idx_found)

    # 2. Reproject or read each of the 3 bands
    rgb_layers = []

    if src.crs and src.crs.to_string() != DST_CRS:
        transform, width, height = calculate_default_transform(
            src.crs, DST_CRS, src.width, src.height, *src.bounds
        )
        for b_idx in resolved_indices:
            img_data = np.empty((height, width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, b_idx),
                destination=img_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=DST_CRS,
                resampling=Resampling.nearest,
            )
            rgb_layers.append(img_data)
        left   = transform.c
        right  = transform.c + transform.a * width
        top    = transform.f
        bottom = transform.f + transform.e * height
        extent = [left, right, bottom, top]
    else:
        for b_idx in resolved_indices:
            rgb_layers.append(src.read(b_idx).astype(np.float32))
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    # 3. 2-98 percentile contrast stretch per channel, with nodata masking
    nodata_val = src.nodata if src.nodata is not None else -9999
    h, w       = rgb_layers[0].shape
    rgba       = np.zeros((h, w, 4), dtype=np.float32)

    stacked           = np.stack(rgb_layers, axis=-1)
    pixel_is_nodata   = np.all(stacked == 0, axis=-1) | np.all(stacked == nodata_val, axis=-1)
    valid_mask        = (~np.any(np.isnan(stacked), axis=-1)
                         & ~np.any(np.isinf(stacked), axis=-1)
                         & ~pixel_is_nodata)

    for i, layer in enumerate(rgb_layers):
        if np.any(valid_mask):
            valid_pixels = layer[valid_mask]
            p2, p98 = np.percentile(valid_pixels, [2, 98])
            norm = np.clip((layer - p2) / (p98 - p2 + 1e-10), 0.0, 1.0) if p98 > p2 \
                   else np.clip(layer, 0.0, 1.0)
        else:
            norm = np.clip(layer, 0.0, 1.0)
        rgba[:, :, i] = norm * valid_mask

    rgba[:, :, 3] = valid_mask.astype(np.float32)
    return rgba, extent


def format_map_axes(ax, fontsize=11, spine_color='black', spine_width=2):
    """
    Applies standard geographic formatting to a map axes:
    longitude/latitude labels, tick locators, degree formatters,
    rotated Y-axis labels, and optional spine styling.

    Parameters
    ----------
    ax          : matplotlib.axes.Axes
    fontsize    : int
        Font size for axis labels and tick labels (default 11).
    spine_color : str or None
        Color for all four spines. Pass None to skip spine styling.
    spine_width : float
        Line width for spines (default 2).

    Returns
    -------
    None
    """
    ax.set_xlabel('Longitude', fontsize=fontsize)
    ax.set_ylabel('Latitude', fontsize=fontsize)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f°'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f°'))

    ax.yaxis.set_tick_params(rotation=90)

    if spine_color is not None:
        for spine in ax.spines.values():
            spine.set_edgecolor(spine_color)
            spine.set_linewidth(spine_width)

"""
Raster I/O, Spectral Index Computation & Point Extraction

Reads satellite rasters, computes spectral indices (NDVI, NDWI, MNDWI,
NDMI, CMRI, MVI, NDRE, SAVI, EVI) for Sentinel-2 and PlanetScope, writes
enriched *_indices.tif files, and extracts pixel values at the 60 field
sample locations from Rijal & Saintilan (2026).
"""

import os
import numpy as np
import pandas as pd
import rasterio


def process_sensor_raster(gdf, in_path, fallback_bands, sensor_type):
    """
    Reads raster, computes indices (saving a new TIF), extracts points,
    and generates feature summary.
    """
    if not os.path.exists(in_path):
        print(f"[WARNING] Raster not found: {in_path}")
        return pd.DataFrame(index=gdf.index)

    base_name = os.path.splitext(os.path.basename(in_path))[0]
    dir_name = os.path.dirname(in_path)

    out_tif = os.path.join(dir_name, f"{base_name}_indices.tif")
    out_summary = os.path.join(dir_name, f"{base_name}_indices_summary.csv")

    print(f"\nProcessing {sensor_type} Raster...")

    with rasterio.open(in_path) as src:
        meta = src.meta.copy()
        data = src.read().astype(np.float32)
        raster_bands = list(src.descriptions)

        # Resolve band names
        if not any(raster_bands):
            raster_bands = fallback_bands[:src.count]
        else:
            raster_bands = [b if b is not None else f"Band_{i+1}"
                            for i, b in enumerate(raster_bands)]

    eps = 1e-8

    # --- Sentinel-2 Indices ---
    if sensor_type == 's2':
        # Indices based on standard: B3(1), B4(2), B5(3), B8(6), B8A(7), B11(8)
        b3, b4, b5, b8, b8a, b11 = data[1], data[2], data[3], data[6], data[7], data[8]

        ndvi  = (b8 - b4)  / (b8 + b4 + eps)
        ndwi  = (b3 - b8)  / (b3 + b8 + eps)
        mndwi = (b3 - b11) / (b3 + b11 + eps)
        ndmi  = (b8 - b11) / (b8 + b11 + eps)
        cmri  = ndvi - ndwi
        ndre  = (b8a - b5) / (b8a + b5 + eps)
        savi  = 1.5 * (b8 - b4) / (b8 + b4 + 0.5)

        indices   = np.stack([ndvi, mndwi, ndmi, cmri, ndre, savi])
        idx_names = ['s2_NDVI', 's2_MNDWI', 's2_NDMI', 's2_CMRI', 's2_NDRE', 's2_SAVI']

    # --- PlanetScope Indices ---
    elif sensor_type == 'ps':
        # Indices based on PS 8-band standard: Red(5), RedEdge(6), NIR(7)
        red, rededge, nir = data[5], data[6], data[7]

        ndvi = (nir - red) / (nir + red + eps)
        savi = 1.5 * (nir - red) / (nir + red + 0.5)
        ndre = (nir - rededge) / (nir + rededge + eps)

        indices   = np.stack([ndvi, savi, ndre])
        idx_names = ['ps_NDVI', 'ps_SAVI', 'ps_NDRE']

    # Combine and save new TIF
    new_data = np.vstack([data, indices])
    new_band_names = raster_bands + idx_names

    meta.update(count=new_data.shape[0], dtype='float32')
    with rasterio.open(out_tif, 'w', **meta) as dst:
        dst.write(new_data)
        dst.descriptions = tuple(new_band_names)

    print(f"  [+] Saved raster with indices to: {out_tif}")

    # Extract points from the new TIF
    with rasterio.open(out_tif) as src:
        gdf_proj = gdf.to_crs(src.crs)
        coords = [(geom.x, geom.y) for geom in gdf_proj.geometry]
        sampled_values = list(src.sample(coords))
        df_extracted = pd.DataFrame(sampled_values, columns=new_band_names, index=gdf.index)

    # Generate and export Feature Summary
    summary_df = pd.DataFrame({
        'Feature': df_extracted.columns,
        'Min':  df_extracted.min(),
        'Max':  df_extracted.max(),
        'Mean': df_extracted.mean()
    })
    summary_df.to_csv(out_summary, index=False)
    print(f"  [+] Saved feature summary to: {out_summary}")
    print(f"\n  [{sensor_type.upper()}] Feature Summary (Console):")
    print(summary_df.to_string(index=False))

    return df_extracted


def extract_only(gdf, in_path, fallback_bands, sensor_type):
    """
    Extracts pixel values directly without creating indices (Used for Embeddings).
    """
    print(f"\nProcessing {sensor_type} Raster...")
    if not os.path.exists(in_path):
        print(f"[WARNING] Raster not found: {in_path}")
        return pd.DataFrame(index=gdf.index)

    with rasterio.open(in_path) as src:
        raster_bands = list(src.descriptions)
        if not any(raster_bands):
            raster_bands = fallback_bands[:src.count]
        else:
            raster_bands = [b if b is not None else f"Band_{i+1}"
                            for i, b in enumerate(raster_bands)]

        gdf_proj = gdf.to_crs(src.crs)
        coords = [(geom.x, geom.y) for geom in gdf_proj.geometry]
        sampled_values = list(src.sample(coords))
        df_extracted = pd.DataFrame(sampled_values, columns=raster_bands, index=gdf.index)

    print(f"  [+] Extracted {len(df_extracted.columns)} features.")
    return df_extracted

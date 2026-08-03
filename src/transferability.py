# =============================================================================
# transferability.py
# Apply the trained pipeline to new scenes without retraining.
# Each new scene gets its own per-scene adaptive thresholds and candidate mask
# -- that is the core transferability mechanism (no fixed thresholds, no new
# labels needed). The XGBoost model trained on Sangatta is applied as-is.
# =============================================================================

import json
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
from rasterio.warp import reproject, Resampling

from src.preprocessing import (
    load_geotiff_bands,
    compute_all_indices,
    compute_coastal_candidate_mask,
    apply_adaptive_threshold,
)
from src.classification import predict_extent, load_model
from src.evaluation import rasterize_gmw, evaluate_against_gmw


# =============================================================================
# Site registry
# =============================================================================

# Training site: Sangatta (20250302_030003_92_4001)
# Transfer sites: Gujarat, El Salvador, Belize, Australia
# Ho Chi Minh dropped (insufficient mangrove coverage -- confirmed W8)

TRANSFER_SITES = {
    'gujarat'    : '20250311_061550_53_4001',
    'elsalvador' : '20250223_165546_32_4001',
    'belize'     : '20250824_171853_67_4001',
    'belize2'    : '20250824_171857_84_4001',
    'australia'  : '20250608_014315_58_4001',
}

SITE_LABELS = {
    'gujarat'    : 'Gujarat, India',
    'elsalvador' : 'El Salvador',
    'belize'     : 'Belize 1',
    'belize2'    : 'Belize 2',
    'australia'  : 'Australia',
}



# =============================================================================
# Helper: resample HDF5-grid raster to GeoTIFF/indices grid
# =============================================================================

def _resample_to_grid(src_array, src_transform, src_crs,
                      dst_shape, dst_transform, dst_crs):
    """Resample 2D float32 array from source grid to destination grid."""
    dst = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=src_array,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


# =============================================================================
# Single-scene transfer
# =============================================================================

def run_transfer_scene(site: str,
                       scene_id: str,
                       processed_dir: str,
                       model_path: str,
                       results_dir: str,
                       gmw_dir: str = None) -> dict:
    """
    Apply trained RF classifier to a single new scene using per-scene
    adaptive thresholds and coastal candidate mask. No retraining.

    Pipeline per transfer site:
      1. Load 8-band GeoTIFF (from 01_preprocessing)
      2. Compute spectral indices
      3. Coastal candidate mask (adaptive Otsu -- same as training)
      4. Per-scene adaptive threshold (MVI force_otsu)
      5. Predict extent with pre-trained XGBoost model
      6. Save extent GeoTIFF
      7. (optional) Evaluate against GMW v3

    Parameters
    ----------
    site          : short site key (e.g. 'gujarat')
    scene_id      : Tanager scene identifier string
    processed_dir : path to data/processed/
    model_path    : path to saved .joblib RF model from 02_classification
    results_dir   : path to outputs/results/ for threshold JSON + metrics
    gmw_dir       : path to data/gmw_v3/ (optional; skip eval if None)

    Returns
    -------
    dict with scene metadata, mangrove area, thresholds, and metrics (if GMW)
    """
    site_label = SITE_LABELS.get(site, site)
    print(f"\n{'='*60}")
    print(f"  Transfer site  : {site_label}  ({scene_id})")
    print(f"{'='*60}")

    processed_dir = Path(processed_dir)
    results_dir   = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load bands
    data    = load_geotiff_bands(str(processed_dir), scene_id, site=site)
    indices = compute_all_indices(data)

    # 2. Coastal candidate mask (adaptive Otsu per scene)
    candidate_mask = compute_coastal_candidate_mask(data, indices)

    # 3. Per-scene adaptive threshold (MVI uses Otsu directly)
    thresholds = apply_adaptive_threshold(
        indices, scene_id=scene_id,
        candidate_mask=candidate_mask,
        force_otsu_indices=['MVI'],
    )

    # Save thresholds
    thresh_path = results_dir / f'thresholds_{site}_{scene_id}.json'
    with open(thresh_path, 'w') as f:
        json.dump(
            {k: float(v) for k, v in thresholds.items()
             if v is not None and np.isfinite(v)}, f, indent=2
        )

    # 4. Load pre-trained model
    model = load_model(model_path)

    # 5. Predict extent
    h, w       = list(indices.values())[0].shape
    extent_map = predict_extent(
        model, indices,
        original_shape=(h, w),
        candidate_mask=candidate_mask,
        extra_features=None,
    )

    # 7. Save extent GeoTIFF
    extent_path = processed_dir / f'extent_mangrove_{site}_{scene_id}.tif'
    with rasterio.open(
        extent_path, 'w',
        driver='GTiff', height=h, width=w,
        count=1, dtype='int8',
        crs=data['crs'], transform=data['transform'],
        compress='lzw', nodata=-1,
    ) as dst:
        dst.write(extent_map, 1)

    n_mangrove = int(np.sum(extent_map == 1))
    area_ha    = n_mangrove * 0.09   # 30m pixel = 0.09 ha
    print(f"  Extent saved   : {extent_path.name}")
    print(f"  Mangrove area  : {area_ha:,.1f} ha  ({n_mangrove:,} px)")

    result = {
        'site'          : site,
        'site_label'    : site_label,
        'scene_id'      : scene_id,
        'n_mangrove_px' : n_mangrove,
        'area_ha'       : area_ha,
        'thresholds'    : thresholds,
        'extent_path'   : str(extent_path),
        'has_reip'      : False,
    }

    # 8. Evaluate against GMW v3 (optional)
    if gmw_dir is not None:
        gmw_path = Path(gmw_dir) / f'gmw_{site}_{scene_id}.geojson'
        if gmw_path.exists():
            gmw_raster = rasterize_gmw(
                str(gmw_path),
                reference_shape=(h, w),
                transform=data['transform'],
                crs=data['crs'],
            )
            metrics = evaluate_against_gmw(
                extent_map, gmw_raster,
                eval_mask=candidate_mask,
                model_name=f'RF ({site_label})',
            )
            metrics['metrics_table'].to_csv(
                results_dir / f'gmw_eval_{site}_{scene_id}.csv', index=False
            )
            result['metrics'] = {
                'kappa'     : metrics['kappa'],
                'precision' : metrics['precision'],
                'recall'    : metrics['recall'],
                'f1'        : metrics['f1_mangrove'],
                'iou'       : metrics['IoU'],
            }
        else:
            print(f"  GMW v3 not found : {gmw_path.name} -- skipping eval")
            result['metrics'] = None

    return result


# =============================================================================
# Multi-site transfer loop
# =============================================================================

def run_all_transfer_sites(processed_dir: str,
                            model_path: str,
                            results_dir: str,
                            gmw_dir: str = None,
                            sites: dict = None) -> pd.DataFrame:
    """
    Run transfer pipeline across all (or specified) sites.

    Parameters
    ----------
    processed_dir : path to data/processed/
    model_path    : path to saved .joblib RF model
    results_dir   : path to outputs/results/
    gmw_dir       : path to data/gmw_v3/ (optional)
    sites         : dict {site_key: scene_id}; defaults to TRANSFER_SITES

    Returns
    -------
    pd.DataFrame: one row per site with area, thresholds, and metrics
    """
    if sites is None:
        sites = TRANSFER_SITES

    results = []
    failed  = []

    for site, scene_id in sites.items():
        try:
            r = run_transfer_scene(
                site=site,
                scene_id=scene_id,
                processed_dir=processed_dir,
                model_path=model_path,
                results_dir=results_dir,
                gmw_dir=gmw_dir,
            )
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR -- {site} ({scene_id}): {e}")
            failed.append({'site': site, 'scene_id': scene_id, 'error': str(e)})

    # Build summary DataFrame
    rows = []
    for r in results:
        row = {
            'site'          : r['site_label'],
            'scene_id'      : r['scene_id'],
            'area_ha'       : r['area_ha'],
            'n_mangrove_px' : r['n_mangrove_px'],
            'has_reip'      : r['has_reip'],
        }
        if r.get('metrics'):
            row.update(r['metrics'])
        rows.append(row)

    df = pd.DataFrame(rows)

    print(f"\n{'='*60}")
    print("  Transfer summary:")
    cols = ['site', 'area_ha', 'n_mangrove_px']
    if not df.empty:
        if 'kappa' in df.columns:
            cols += ['kappa', 'f1', 'recall']
        print(df[cols].to_string(index=False))
    else:
        print("  [No successful transfers]")

    if failed:
        print(f"\n  Failed ({len(failed)}):")
        for f in failed:
            print(f"    {f['site']}: {f['error']}")

    return df

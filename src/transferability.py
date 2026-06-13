# =============================================================================
# transferability.py
# Apply the trained pipeline to new scenes without retraining.
# Each new scene gets its own per-scene adaptive thresholds — that is the
# core transferability mechanism (no fixed thresholds, no new labels needed).
# =============================================================================

import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.preprocessing import (
    load_geotiff_bands,
    compute_all_indices,
    compute_water_mask,
    apply_adaptive_threshold,
    save_raster,
)
from src.classification import predict_extent, load_model


# =============================================================================
# Site registry
# =============================================================================

# Each entry: scene_id -> human label used in reports
TRANSFER_SITES = {
    '20250311_061550_53_4001': 'Gujarat, India',
    '20250223_165546_32_4001': 'El Salvador',
    '20250824_171857_84_4001': 'Belize',
    '20250407_035527_47_4001': 'Ho Chi Minh, Vietnam',
}


# =============================================================================
# Single-scene transfer
# =============================================================================

def run_transfer_scene(scene_id: str,
                        processed_dir: str,
                        model_path: str,
                        output_dir: str,
                        results_dir: str = None) -> dict:
    """
    Apply trained classifier to a single new scene using adaptive thresholds.
    No retraining — only new per-scene thresholds are computed.

    Parameters
    ----------
    scene_id      : Tanager scene identifier string
    processed_dir : path to data/processed/ (must contain scene GeoTIFFs)
    model_path    : path to saved .joblib RF model from notebook 02
    output_dir    : where to write the output extent GeoTIFF

    Returns
    -------
    dict with keys: scene_id, site_name, n_mangrove_px, threshold_method,
                    thresholds, output_tif
    """
    site_name = TRANSFER_SITES.get(scene_id, scene_id)
    print(f"\n{'='*60}")
    print(f"  Transfer site  : {site_name}  ({scene_id})")

    # 1. Load bands
    data = load_geotiff_bands(processed_dir, scene_id)

    # 2. Spectral indices
    indices = compute_all_indices(data)

    # 3. Water mask — exclude open water before adaptive thresholding
    water_mask = compute_water_mask(data)

    # 4. Per-scene adaptive thresholds (core transferability step)
    thresholds = apply_adaptive_threshold(indices, scene_id,
                                          water_mask=water_mask)

    # 5. Save thresholds to outputs/results/ for reproducibility and debugging
    out_results = Path(results_dir) if results_dir else Path(output_dir).parent / "outputs" / "results"
    out_results.mkdir(parents=True, exist_ok=True)
    thresh_path = out_results / f"thresholds_{scene_id}.json"
    with open(thresh_path, "w") as f:
        json.dump({k: v for k, v in thresholds.items() if v is not None}, f, indent=2)
    print(f"  Thresholds saved: {thresh_path.name}")

    # 6. Load pre-trained model
    model = load_model(model_path)

    # 7. Predict extent
    h, w = list(indices.values())[0].shape
    extent_map = predict_extent(model, indices, original_shape=(h, w))

    # 8. Save output raster
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_tif = output_dir / f"{scene_id}_mangrove_extent.tif"
    save_raster(extent_map, str(out_tif), reference_data=data, nodata=-1)

    n_mangrove = int(np.sum(extent_map == 1))
    area_ha    = n_mangrove * 0.09   # 30 m pixel = 0.09 ha

    print(f"  Mangrove area  : {area_ha:,.1f} ha  ({n_mangrove:,} px)")

    return {
        'scene_id'        : scene_id,
        'site_name'       : site_name,
        'n_mangrove_px'   : n_mangrove,
        'area_ha'         : area_ha,
        'thresholds'      : thresholds,
        'output_tif'      : str(out_tif),
    }


# =============================================================================
# Multi-site transfer loop
# =============================================================================

def run_all_transfer_sites(processed_dir: str,
                            model_path: str,
                            output_dir: str,
                            results_dir: str = None,
                            scene_ids: list = None) -> pd.DataFrame:
    """
    Run transfer pipeline across all (or specified) sites and return a summary table.

    Parameters
    ----------
    processed_dir : path to data/processed/
    model_path    : path to saved .joblib RF model
    output_dir    : base output directory; per-scene subdirs are created
    scene_ids     : list of scene IDs to process; defaults to all TRANSFER_SITES

    Returns
    -------
    pd.DataFrame: one row per site with area and threshold summary
    """
    if scene_ids is None:
        scene_ids = list(TRANSFER_SITES.keys())

    results = []
    failed  = []

    for sid in scene_ids:
        try:
            r = run_transfer_scene(
                scene_id=sid,
                processed_dir=processed_dir,
                model_path=model_path,
                output_dir=output_dir,
                results_dir=results_dir,
            )
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR — {sid}: {e}")
            failed.append({'scene_id': sid, 'error': str(e)})

    df = pd.DataFrame(results)

    print(f"\n{'='*60}")
    print("  Transfer summary:")
    if not df.empty:
        print(df[['site_name', 'area_ha', 'n_mangrove_px']].to_string(index=False))
    if failed:
        print(f"\n  Failed sites ({len(failed)}):")
        for f in failed:
            print(f"    {f['scene_id']}: {f['error']}")

    return df

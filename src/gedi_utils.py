# =============================================================================
# gedi_utils.py
# GEDI L4A footprint loading, spatial join, AGB regression, carbon conversion
# =============================================================================

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# IPCC 2006 tropical forest AGB-to-carbon ratio (tC / t dry biomass).
# For mangroves specifically, IPCC Wetlands Supplement (2013) recommends
# the same 0.451 but note this excludes belowground biomass carbon.
AGB_TO_CARBON_FACTOR = 0.451


# =============================================================================
# 1. Load GEDI Footprints
# =============================================================================

def load_gedi_footprints(geojson_path: str,
                          min_sensitivity: float = 0.95) -> gpd.GeoDataFrame:
    """
    Load GEDI L4A footprints exported from GEE by Athar.
    Applies quality filters on load.

    Expected columns: agbd, agbd_se, l4_quality_flag, sensitivity, geometry

    Parameters
    ----------
    geojson_path    : path to data/raw/gedi_l4a_sangatta.geojson
    min_sensitivity : minimum lidar sensitivity (default 0.95)

    Returns
    -------
    Filtered GeoDataFrame
    """
    gdf = gpd.read_file(geojson_path)

    before = len(gdf)

    # Quality filters
    gdf = gdf[gdf['l4_quality_flag'] == 1].copy()
    gdf = gdf[gdf['sensitivity'] >= min_sensitivity].copy()
    gdf = gdf[gdf['agbd'] > 0].copy()           # remove negative/zero AGB
    gdf = gdf.dropna(subset=['agbd']).copy()

    print(f"  GEDI footprints:")
    print(f"  before filter  : {before:,}")
    print(f"  after filter   : {len(gdf):,}")

    return gdf.reset_index(drop=True)


# =============================================================================
# 2. Spatial Join: GEDI footprints <-> Tanager pixels
# =============================================================================

def spatial_join_gedi_tanager(gedi_gdf: gpd.GeoDataFrame,
                               indices: dict,
                               tif_path: str) -> pd.DataFrame:
    """
    Extract Tanager spectral index values at each GEDI footprint location.
    Joins by pixel coordinate lookup — no resampling needed (point-in-pixel).

    Parameters
    ----------
    gedi_gdf : GeoDataFrame from load_gedi_footprints()
    indices  : dict of 2D index arrays from compute_all_indices()
    tif_path : path to any processed GeoTIFF (for CRS + transform reference)

    Returns
    -------
    pd.DataFrame with columns: agbd, agbd_se + one column per index
    """
    with rasterio.open(tif_path) as src:
        transform = src.transform
        crs       = src.crs

    # Reproject GEDI to match Tanager CRS if needed
    if gedi_gdf.crs != crs:
        gedi_gdf = gedi_gdf.to_crs(crs)

    feature_names = list(indices.keys())
    h, w = list(indices.values())[0].shape

    # Vectorised coordinate → pixel conversion (avoids row-by-row Python loop)
    xs = gedi_gdf.geometry.x.values
    ys = gedi_gdf.geometry.y.values
    rows_px, cols_px = rowcol(transform, xs, ys)
    rows_px = np.array(rows_px)
    cols_px = np.array(cols_px)

    # Keep only footprints that fall within the raster extent
    in_bounds = (rows_px >= 0) & (rows_px < h) & (cols_px >= 0) & (cols_px < w)
    gedi_sub  = gedi_gdf[in_bounds].copy().reset_index(drop=True)
    rows_px   = rows_px[in_bounds]
    cols_px   = cols_px[in_bounds]

    df = pd.DataFrame({
        'agbd'   : gedi_sub['agbd'].values,
        'agbd_se': gedi_sub['agbd_se'].values,
    })
    for fname in feature_names:
        df[fname] = indices[fname][rows_px, cols_px].astype(np.float32)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  Joined footprints: {len(df):,} valid samples")
    return df


# =============================================================================
# 3. AGB Regression
# =============================================================================

def train_agb_regressor(df: pd.DataFrame,
                         feature_names: list,
                         n_estimators: int = 200,
                         random_state: int = 42) -> tuple:
    """
    Train Random Forest regressor to predict AGB from spectral indices.
    Weighted by inverse of agbd_se (higher uncertainty = lower weight).

    Parameters
    ----------
    df            : DataFrame from spatial_join_gedi_tanager()
    feature_names : list of index column names to use as features
    n_estimators  : int
        Number of trees (default 200).
    random_state  : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    (fitted RandomForestRegressor, metrics dict)
    """
    X = df[feature_names].values
    y = df['agbd'].values
    w = 1.0 / (df['agbd_se'].values + 1e-6)    # inverse SE as sample weight

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y, w, test_size=0.2, random_state=random_state
    )

    reg = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        oob_score=True
    )
    reg.fit(X_train, y_train, sample_weight=w_train)

    y_pred  = reg.predict(X_test)
    r2      = r2_score(y_test, y_pred)
    mae     = mean_absolute_error(y_test, y_pred)

    print(f"  AGB regressor:")
    print(f"  OOB R2         : {reg.oob_score_:.4f}")
    print(f"  Test R2        : {r2:.4f}")
    print(f"  Test MAE       : {mae:.2f} Mg/ha")

    metrics = {'oob_r2': reg.oob_score_, 'test_r2': r2, 'test_mae': mae}
    return reg, metrics


# =============================================================================
# 4. Wall-to-Wall AGB Prediction
# =============================================================================

def predict_wall_to_wall_agb(regressor: RandomForestRegressor,
                              indices: dict,
                              mangrove_mask: np.ndarray) -> np.ndarray:
    """
    Generate wall-to-wall AGB map within mangrove extent only.

    Parameters
    ----------
    regressor     : fitted RandomForestRegressor
    indices       : dict of 2D index arrays
    mangrove_mask : 2D boolean array (True = mangrove pixel from classification)

    Returns
    -------
    2D np.ndarray of AGB (Mg/ha), NaN outside mangrove extent
    """
    feature_names = list(indices.keys())
    h, w          = mangrove_mask.shape

    agb_map = np.full((h, w), np.nan, dtype=np.float32)

    # Predict only within mangrove extent
    X_mang = np.stack(
        [indices[f][mangrove_mask] for f in feature_names], axis=1
    ).astype(np.float32)

    valid = np.all(np.isfinite(X_mang), axis=1)
    preds = np.full(X_mang.shape[0], np.nan)
    preds[valid] = regressor.predict(X_mang[valid])

    agb_map[mangrove_mask] = preds

    total_px  = int(np.sum(mangrove_mask))
    valid_px  = int(np.sum(valid))
    mean_agb  = float(np.nanmean(agb_map))
    print(f"  AGB map:")
    print(f"  mangrove px    : {total_px:,}")
    print(f"  predicted px   : {valid_px:,}")
    print(f"  mean AGB       : {mean_agb:.2f} Mg/ha")

    return agb_map


# =============================================================================
# 5. Carbon Conversion
# =============================================================================

def agb_to_carbon(agb_map: np.ndarray,
                  conversion_factor: float = AGB_TO_CARBON_FACTOR) -> np.ndarray:
    """
    Convert AGB (Mg/ha) to carbon stock (MgC/ha).

    Default factor: 0.451 (IPCC 2006 — tropical forest).
    NaN pixels preserved.

    Parameters
    ----------
    agb_map           : 2D np.ndarray from predict_wall_to_wall_agb()
    conversion_factor : AGB-to-carbon ratio

    Returns
    -------
    2D np.ndarray of carbon stock (MgC/ha)
    """
    carbon_map = agb_map * conversion_factor
    print(f"  Carbon map:")
    print(f"  mean carbon    : {float(np.nanmean(carbon_map)):.2f} MgC/ha")
    print(f"  conversion     : {conversion_factor} (IPCC 2006)")
    return carbon_map


# =============================================================================
# 6. Save / Load
# =============================================================================

def save_regressor(model, filepath: str):
    """
    Save AGB regressor to .joblib.

    Parameters
    ----------
    model    : fitted RandomForestRegressor from train_agb_regressor()
    filepath : str
        Destination path (e.g. outputs/models/agb_regressor_sangatta.joblib).
    """
    joblib.dump(model, filepath)
    print(f"  Regressor saved : {filepath}")


def load_regressor(filepath: str):
    """
    Load AGB regressor from .joblib.

    Parameters
    ----------
    filepath : str
        Path to saved .joblib file.

    Returns
    -------
    Fitted RandomForestRegressor.
    """
    return joblib.load(filepath)

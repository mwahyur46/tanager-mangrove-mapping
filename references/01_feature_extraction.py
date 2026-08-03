"""
Feature Extraction & Train-Test Dataset Creation (Multi-Sensor)

Extracts per-sensor feature matrices from the 60 field sample plots collected
by Rijal & Saintilan (2026) at Bama Beach, Baluran National Park, and produces
independent train (70%) / test (30%) splits for each sensor × target combination.

The 60 plots were originally designed for 10 m Sentinel-2 imagery. That same
sampling framework is intentionally kept unchanged across all three sensors —
including 3 m PlanetScope — to isolate the effect of the dataset rather than
optimising each sensor's sampling individually.

Spectral indices (NDVI, NDWI, MNDWI, NDMI, CMRI, MVI, NDRE, SAVI, EVI) are
computed for Sentinel-2 and PlanetScope and written to new *_indices.tif files.
Google Satellite Embeddings (64 latent bands) are extracted directly.

Prediction targets (all sourced from the original field survey):
  - Above-Ground Carbon  (agc_mg_c_ha)      — regression
  - Tree height          (tree_height_m)     — regression
  - Canopy cover         (canopy_cover_pct)  — regression
  - Dominant species     (species_id)        — classification (4 classes)
"""

import os
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
import warnings

# Ignore division by zero warnings in raster math
warnings.filterwarnings('ignore')

from src.config import (
    SAMPLES_PATH, S2_PATH, EMBED_PATH, PS_PATH,
    S2_BANDS, EMBED_BANDS, PS_BANDS,
    CLASSIFICATION_TARGET, REGRESSION_TARGET, TARGET_COLS,
    TRAIN_TEST_DIR, RANDOM_STATE
)
from src.raster import process_sensor_raster, extract_only

# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    print("====================================================")
    print("Starting Feature Extraction & Splitting by Sensor...")
    print("====================================================")
    os.makedirs(TRAIN_TEST_DIR, exist_ok=True)

    # 1. Load samples
    if not os.path.exists(SAMPLES_PATH):
        raise FileNotFoundError(f"Sample points file not found: {SAMPLES_PATH}")

    gdf_samples = gpd.read_file(SAMPLES_PATH)
    valid_targets = [col for col in TARGET_COLS if col in gdf_samples.columns]
    df_base = pd.DataFrame(gdf_samples[valid_targets])
    print(f"Loaded {len(gdf_samples)} sample points from {SAMPLES_PATH}.")

    # 2 & 3 & 4. Process Rasters
    df_s2    = process_sensor_raster(gdf_samples, S2_PATH, S2_BANDS, 's2')
    df_ps    = process_sensor_raster(gdf_samples, PS_PATH, PS_BANDS, 'ps')
    df_embed = extract_only(gdf_samples, EMBED_PATH, EMBED_BANDS, 'embed')

    datasets = {'s2': df_s2, 'ps': df_ps, 'embed': df_embed}

    print("\n====================================================")
    print("Generating Train-Test Datasets")
    print("====================================================")

    reg_targets = REGRESSION_TARGET if isinstance(REGRESSION_TARGET, list) else [REGRESSION_TARGET]

    for prefix, df_feat in datasets.items():
        if df_feat.empty or df_feat.isna().all().all():
            continue

        print(f"\n>>> SENSOR: {prefix.upper()} <<<")

        # --- CLASSIFICATION ---
        if CLASSIFICATION_TARGET in df_base.columns:
            class_cols = [CLASSIFICATION_TARGET]
            if 'dominant_species' in df_base.columns:
                class_cols.append('dominant_species')

            df_class = pd.concat([df_base[class_cols], df_feat], axis=1).dropna(
                subset=[CLASSIFICATION_TARGET])

            try:
                df_c_train, df_c_test = train_test_split(
                    df_class, test_size=0.30, random_state=RANDOM_STATE,
                    stratify=df_class[CLASSIFICATION_TARGET]
                )
            except ValueError:
                df_c_train, df_c_test = train_test_split(
                    df_class, test_size=0.30, random_state=RANDOM_STATE)

            c_train_path   = os.path.join(TRAIN_TEST_DIR, f'{prefix}_{CLASSIFICATION_TARGET}_train.csv')
            c_test_path    = os.path.join(TRAIN_TEST_DIR, f'{prefix}_{CLASSIFICATION_TARGET}_test.csv')
            c_summary_path = os.path.join(TRAIN_TEST_DIR, f'{prefix}_{CLASSIFICATION_TARGET}_summary.csv')

            df_c_train.to_csv(c_train_path, index=False)
            df_c_test.to_csv(c_test_path, index=False)

            print(f"  [Classification] Saved Train: {os.path.basename(c_train_path)} ({len(df_c_train)} samples)")
            print(f"  [Classification] Saved Test : {os.path.basename(c_test_path)} ({len(df_c_test)} samples)")

            if 'dominant_species' in df_class.columns:
                print("  [Classification Summary]:")
                dist = df_class['dominant_species'].value_counts()
                summary_data = []
                for sp, count in dist.items():
                    sp_id = df_class[df_class['dominant_species'] == sp][CLASSIFICATION_TARGET].iloc[0]
                    summary_data.append({'species_name': sp, 'species_id': sp_id, 'sample_count': count})
                    print(f"    - {sp} (ID {sp_id}): {count} samples")

                pd.DataFrame(summary_data).to_csv(c_summary_path, index=False)
                print(f"  [+] Saved Summary to: {c_summary_path}")

        # --- REGRESSION (all targets) ---
        for reg_target in reg_targets:
            if reg_target not in df_base.columns:
                continue

            df_reg = pd.concat([df_base[[reg_target]], df_feat], axis=1).dropna(
                subset=[reg_target])
            df_r_train, df_r_test = train_test_split(
                df_reg, test_size=0.30, random_state=RANDOM_STATE)

            r_train_path   = os.path.join(TRAIN_TEST_DIR, f'{prefix}_{reg_target}_train.csv')
            r_test_path    = os.path.join(TRAIN_TEST_DIR, f'{prefix}_{reg_target}_test.csv')
            r_summary_path = os.path.join(TRAIN_TEST_DIR, f'{prefix}_{reg_target}_summary.csv')

            df_r_train.to_csv(r_train_path, index=False)
            df_r_test.to_csv(r_test_path, index=False)

            print(f"\n  [Regression] Saved Train: {os.path.basename(r_train_path)} ({len(df_r_train)} samples)")
            print(f"  [Regression] Saved Test : {os.path.basename(r_test_path)} ({len(df_r_test)} samples)")

            mean_tr = df_r_train[reg_target].mean()
            min_tr  = df_r_train[reg_target].min()
            max_tr  = df_r_train[reg_target].max()
            mean_ts = df_r_test[reg_target].mean()
            min_ts  = df_r_test[reg_target].min()
            max_ts  = df_r_test[reg_target].max()

            print(f"  [Regression Summary for '{reg_target}']: ")
            print(f"    Train -> Mean: {mean_tr:.2f}, Min: {min_tr:.2f}, Max: {max_tr:.2f}")
            print(f"    Test  -> Mean: {mean_ts:.2f}, Min: {min_ts:.2f}, Max: {max_ts:.2f}")

            pd.DataFrame({
                'Metric': ['Mean', 'Min', 'Max'],
                'Train Set': [mean_tr, min_tr, max_tr],
                'Test Set': [mean_ts, min_ts, max_ts]
            }).to_csv(r_summary_path, index=False)
            print(f"  [+] Saved Summary to: {r_summary_path}")

    print("\n====================================================")
    print("All tasks completed successfully!")
    print("====================================================")

if __name__ == '__main__':
    main()

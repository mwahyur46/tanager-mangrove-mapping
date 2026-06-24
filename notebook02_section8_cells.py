# =============================================================================
# NOTEBOOK 02 -- SECTION 8 (NEW): Evaluation vs GMW v3 (independent ground truth)
# =============================================================================
# Place src/evaluation.py in the src/ folder, then add these cells after
# section 7 (Visualization) in 02_classification.ipynb.
# =============================================================================


# ----------------------------------------------------------------------------
# MARKDOWN CELL
# ----------------------------------------------------------------------------
"""
## 8. Evaluation vs GMW v3 (Independent Ground Truth)

Section 5 evaluated against the pseudo-label test split, which is circular
(MVI/NDMI are both feature and label source). This section compares the
predicted extent against GMW v3, an independent reference never used in
training. These are the real accuracy figures.
"""


# ----------------------------------------------------------------------------
# CODE CELL: imports
# ----------------------------------------------------------------------------
"""
import importlib
import src.evaluation as _eval
importlib.reload(_eval)
from src.evaluation import (
    rasterize_gmw,
    evaluate_against_gmw,
    plot_confusion_matrix,
    plot_agreement_map,
)
"""


# ----------------------------------------------------------------------------
# CODE CELL: rasterize GMW v3 onto prediction grid
# ----------------------------------------------------------------------------
"""
# ============================================================
# Rasterize GMW v3 polygons to match the extent_map grid
# ============================================================
gmw_path = DATA_GMW / f'gmw_{SITE}_{SCENE_ID}.geojson'

gmw_raster = rasterize_gmw(
    str(gmw_path),
    reference_shape=extent_map.shape,
    transform=data['transform'],
    crs=data['crs'],
)
"""


# ----------------------------------------------------------------------------
# CODE CELL: evaluate (constrained to candidate zone for fair comparison)
# ----------------------------------------------------------------------------
"""
# ============================================================
# Evaluate RF extent vs GMW v3
# eval_mask = candidate_mask -> fair comparison (only where model
# was allowed to predict mangrove)
# ============================================================
gmw_eval = evaluate_against_gmw(
    extent_map,
    gmw_raster,
    eval_mask=candidate_mask,
    model_name='Random Forest',
)

# Save metrics
gmw_eval['metrics_table'].to_csv(
    OUT_RESULTS / f'gmw_eval_{SITE}_{SCENE_ID}.csv', index=False
)
"""


# ----------------------------------------------------------------------------
# CODE CELL: confusion matrix plot
# ----------------------------------------------------------------------------
"""
# ============================================================
# Confusion matrix vs GMW v3
# ============================================================
plot_confusion_matrix(
    gmw_eval['confusion_matrix'],
    model_name='RF vs GMW v3',
    normalize=False,
    save_path=str(OUT_FIGURES / f'confusion_gmw_{SITE}_{SCENE_ID}.png'),
)

# Row-normalized version (shows recall per class)
plot_confusion_matrix(
    gmw_eval['confusion_matrix'],
    model_name='RF vs GMW v3 (normalized)',
    normalize=True,
    save_path=str(OUT_FIGURES / f'confusion_gmw_norm_{SITE}_{SCENE_ID}.png'),
)
"""


# ----------------------------------------------------------------------------
# CODE CELL: spatial agreement map (TP / FP / FN)
# ----------------------------------------------------------------------------
"""
# ============================================================
# Spatial agreement map: where model agrees / over-predicts / misses
# ============================================================
plot_agreement_map(
    extent_map,
    gmw_raster,
    eval_mask=candidate_mask,
    site=SITE,
    save_path=str(OUT_FIGURES / f'agreement_{SITE}_{SCENE_ID}.png'),
)
"""

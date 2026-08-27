# =============================================================================
# evaluation.py
# Independent evaluation of predicted mangrove extent against GMW v3 ground truth
# =============================================================================
#
# Why a separate module:
#   evaluate_model() in classification.py evaluates against the PSEUDO-LABEL
#   test split, which is circular (MVI/NDMI are both feature and label source).
#   This module evaluates the predicted extent map against GMW v3, an
#   INDEPENDENT reference never used in training. This is the real accuracy.
#
# Required imports (already standard in the environment):
#   geopandas, rasterio, numpy, pandas, matplotlib, scikit-learn
# =============================================================================

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
)


# =============================================================================
# 1. Rasterize GMW v3 to match prediction grid
# =============================================================================

def rasterize_gmw(gmw_path: str,
                  reference_shape: tuple,
                  transform,
                  crs) -> np.ndarray:
    """
    Rasterize GMW v3 polygons onto the same grid as the prediction.

    Parameters
    ----------
    gmw_path        : path to GMW v3 geojson/shapefile
    reference_shape : (height, width) of the prediction raster
    transform       : affine transform of the prediction raster (data['transform'])
    crs             : CRS of the prediction raster (data['crs'])

    Returns
    -------
    2D np.ndarray uint8: 1=mangrove (GMW), 0=non-mangrove
    """
    gmw = gpd.read_file(gmw_path)

    # Reproject GMW to prediction CRS if needed
    if gmw.crs is not None and str(gmw.crs) != str(crs):
        gmw = gmw.to_crs(crs)

    h, w = reference_shape
    shapes = ((geom, 1) for geom in gmw.geometry if geom is not None)

    gmw_raster = rasterize(
        shapes=shapes,
        out_shape=(h, w),
        transform=transform,
        fill=0,
        dtype='uint8',
        all_touched=False,
    )

    print(f"  GMW v3 rasterized : {int(gmw_raster.sum()):,} mangrove px "
          f"({100 * gmw_raster.mean():.2f}%)")
    return gmw_raster


# =============================================================================
# 2. Evaluate predicted extent against GMW v3
# =============================================================================

def evaluate_against_gmw(extent_map: np.ndarray,
                         gmw_raster: np.ndarray,
                         eval_mask: np.ndarray = None,
                         model_name: str = "RF") -> dict:
    """
    Pixel-wise comparison of predicted extent against GMW v3 ground truth.

    Parameters
    ----------
    extent_map : 2D array from predict_extent() (1=mangrove, 0=non, -1=invalid)
    gmw_raster : 2D array from rasterize_gmw() (1=mangrove, 0=non)
    eval_mask  : optional 2D bool array limiting evaluation domain.
                 RECOMMENDED: pass candidate_mask so evaluation is fair
                 (comparing only where the model was allowed to predict).
                 If None, evaluates over all valid (non -1) pixels.
    model_name : label for printout

    Returns
    -------
    dict with keys: confusion_matrix, OA, kappa, precision, recall,
                    f1_mangrove, IoU, y_true, y_pred, metrics_table
    """
    # Domain: valid prediction pixels
    valid = (extent_map != -1)
    if eval_mask is not None:
        valid = valid & eval_mask

    y_pred = (extent_map[valid] == 1).astype(np.int8)
    y_true = (gmw_raster[valid] == 1).astype(np.int8)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    oa     = accuracy_score(y_true, y_pred)
    kappa  = cohen_kappa_score(y_true, y_pred)
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    f1_man = f1_score(y_true, y_pred, zero_division=0)

    # IoU (Jaccard) for mangrove class
    tn, fp, fn, tp = cm.ravel()
    iou = tp / max(tp + fp + fn, 1)

    print(f"\n  Evaluation vs GMW v3 -- {model_name}")
    print(f"  domain        : {'candidate zone' if eval_mask is not None else 'all valid'}")
    print(f"  pixels        : {valid.sum():,}")
    print(f"  -----------------------------------")
    print(f"  Overall Acc   : {oa:.4f}")
    print(f"  Cohen kappa   : {kappa:.4f}")
    print(f"  Precision     : {prec:.4f}  (of predicted mangrove, how many correct)")
    print(f"  Recall        : {rec:.4f}  (of true mangrove, how many found)")
    print(f"  F1 (mangrove) : {f1_man:.4f}")
    print(f"  IoU (mangrove): {iou:.4f}")
    print(f"  -----------------------------------")
    print(f"  Confusion matrix [rows=true, cols=pred]:")
    print(f"                 pred_non   pred_mang")
    print(f"  true_non     {tn:>10,} {fp:>10,}")
    print(f"  true_mang    {fn:>10,} {tp:>10,}")

    metrics_table = pd.DataFrame([{
        'Model'         : model_name,
        'OA'            : round(oa, 4),
        'Kappa'         : round(kappa, 4),
        'Precision'     : round(prec, 4),
        'Recall'        : round(rec, 4),
        'F1_mangrove'   : round(f1_man, 4),
        'IoU'           : round(iou, 4),
    }])

    return {
        'confusion_matrix' : cm,
        'OA'               : oa,
        'kappa'            : kappa,
        'precision'        : prec,
        'recall'           : rec,
        'f1_mangrove'      : f1_man,
        'IoU'              : iou,
        'y_true'           : y_true,
        'y_pred'           : y_pred,
        'metrics_table'    : metrics_table,
    }


# =============================================================================
# 3. Plot confusion matrix
# =============================================================================

def plot_confusion_matrix(cm: np.ndarray,
                          model_name: str = "RF",
                          normalize: bool = False,
                          save_path: str = None):
    """
    Plot confusion matrix with sklearn-style heatmap.

    Parameters
    ----------
    cm        : 2x2 confusion matrix from evaluate_against_gmw()
    normalize : if True, show row-normalized proportions
    save_path : optional path to save the figure
    """
    labels = ['Non-mangrove', 'Mangrove']

    cm_show = cm.astype(float)
    if normalize:
        row_sums = cm_show.sum(axis=1, keepdims=True)
        cm_show = np.divide(cm_show, row_sums,
                            out=np.zeros_like(cm_show), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm_show, cmap='Blues')

    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True (GMW v3)')
    ax.set_title(f'Confusion Matrix vs GMW v3 -- {model_name}')

    # Annotate cells
    thresh = cm_show.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = cm_show[i, j]
            txt = f'{val:.2f}' if normalize else f'{int(cm[i, j]):,}'
            ax.text(j, i, txt, ha='center', va='center',
                    color='white' if val > thresh else 'black',
                    fontsize=11)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved : {save_path}")
    plt.show()


# =============================================================================
# 4. Spatial agreement map (TP / FP / FN visualization)
# =============================================================================

def build_agree_array(extent_map: np.ndarray,
                      gmw_raster: np.ndarray,
                      eval_mask: np.ndarray = None) -> np.ndarray:
    """
    Build the commission-omission array from extent and GMW rasters.

    Returns
    -------
    2D np.ndarray uint8:
      0 = background (outside eval domain)
      1 = TP  (correct mangrove)
      2 = FP  (commission error -- predicted mangrove, GMW says no)
      3 = FN  (omission error  -- GMW says mangrove, model missed)
    """
    pred = (extent_map == 1)
    true = (gmw_raster == 1)
    if eval_mask is not None:
        pred = pred & eval_mask
        true = true & eval_mask
    agree = np.zeros(extent_map.shape, dtype=np.uint8)
    agree[true & pred]   = 1
    agree[~true & pred]  = 2
    agree[true & ~pred]  = 3
    return agree


def plot_agreement_map(extent_map: np.ndarray,
                       gmw_raster: np.ndarray,
                       eval_mask: np.ndarray = None,
                       site: str = "",
                       save_path: str = None):
    """
    Commission-omission error map: TP (correct), FP (commission), FN (omission).

    Plots in native pixel space. For a geographically projected version with
    lat/lon axes, use build_agree_array() then reproject_array_to_4326() from
    src.spatial_viz and render inline in the notebook.

    Color scheme:
      green  = TP  (both predict mangrove)
      red    = FP  (commission error -- model predicts mangrove, GMW says no)
      orange = FN  (omission error  -- GMW says mangrove, model misses)
    """
    agree = build_agree_array(extent_map, gmw_raster, eval_mask)

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    cmap = ListedColormap(['#f0f0f0', '#2ca02c', '#d62728', '#ff7f0e'])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(agree, cmap=cmap, vmin=0, vmax=3)
    ax.set_title(f'Commission-Omission Error Map -- {site}')
    ax.axis('off')

    legend = [
        Patch(color='#2ca02c', label='TP (correct mangrove)'),
        Patch(color='#d62728', label='FP (commission error)'),
        Patch(color='#ff7f0e', label='FN (omission error)'),
    ]
    ax.legend(handles=legend, loc='lower right', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved : {save_path}")
    plt.show()

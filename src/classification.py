# =============================================================================
# classification.py
# Pseudo-label generation, RF + XGBoost training, evaluation, save/load
# =============================================================================

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier


# =============================================================================
# 1. Pseudo-label Generation
# =============================================================================

def generate_pseudo_labels(indices: dict, thresholds: dict) -> np.ndarray:
    """
    Generate binary pseudo-labels using AND logic on MVI + NDMI.
    A pixel = mangrove (1) only if BOTH MVI > threshold AND NDMI > threshold.
    All other valid pixels = non-mangrove (0). Invalid pixels = -1.

    Parameters
    ----------
    indices    : dict of 2D index arrays from compute_all_indices()
    thresholds : dict of threshold values from apply_adaptive_threshold()

    Returns
    -------
    2D np.ndarray: 1=mangrove, 0=non-mangrove, -1=invalid
    """
    mvi_thresh  = thresholds.get('MVI')
    ndmi_thresh = thresholds.get('NDMI')

    if mvi_thresh is None or ndmi_thresh is None:
        raise ValueError("MVI and NDMI thresholds required — check adaptive threshold output")

    mvi  = indices['MVI']
    ndmi = indices['NDMI']

    labels = np.zeros(mvi.shape, dtype=np.int8)

    # Mangrove: both indices exceed threshold
    mangrove_mask = (mvi > mvi_thresh) & (ndmi > ndmi_thresh)
    labels[mangrove_mask] = 1

    # Invalid: NaN or inf in either index
    invalid_mask = ~(np.isfinite(mvi) & np.isfinite(ndmi))
    labels[invalid_mask] = -1

    n_mangrove = int(np.sum(labels == 1))
    n_nonmang  = int(np.sum(labels == 0))
    print(f"  Pseudo-labels generated:")
    print(f"  mangrove     : {n_mangrove:,} px")
    print(f"  non-mangrove : {n_nonmang:,} px")
    print(f"  invalid      : {int(np.sum(labels == -1)):,} px")

    return labels


# =============================================================================
# 2. Feature Matrix
# =============================================================================

def build_feature_matrix(indices: dict,
                          labels: np.ndarray) -> tuple:
    """
    Stack all 5 index arrays into (n_pixels, 5) feature matrix.
    Excludes invalid pixels (label == -1).

    Parameters
    ----------
    indices : dict of 2D index arrays
    labels  : 2D pseudo-label array from generate_pseudo_labels()

    Returns
    -------
    X            : np.ndarray (n_valid_pixels, 5)
    y            : np.ndarray (n_valid_pixels,)
    feature_names: list of str
    """
    feature_names = list(indices.keys())   # ['NDMI','MNDWI','MVI','SAVI','EMI']
    valid_mask    = labels != -1

    X = np.stack([indices[f][valid_mask] for f in feature_names], axis=1)
    y = labels[valid_mask].astype(np.int8)

    print(f"  Feature matrix : {X.shape[0]:,} samples x {X.shape[1]} features")
    print(f"  Features       : {feature_names}")

    return X, y, feature_names


# =============================================================================
# 3. Train / Test Split
# =============================================================================

def split_data(X: np.ndarray, y: np.ndarray,
               test_size: float = 0.2,
               random_state: int = 42) -> tuple:
    """
    Stratified train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    print(f"  Train samples  : {len(X_train):,}")
    print(f"  Test samples   : {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# =============================================================================
# 4. Model Training
# =============================================================================

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        n_estimators: int = 200,
                        random_state: int = 42) -> RandomForestClassifier:
    """
    Train Random Forest classifier.
    Primary model — used for final extent map.
    Uses class_weight='balanced' to handle mangrove pixel minority (typically < 10%).
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        oob_score=True,
        class_weight='balanced',
    )
    rf.fit(X_train, y_train)
    print(f"  RF OOB score   : {rf.oob_score_:.4f}")
    return rf


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  random_state: int = 42) -> XGBClassifier:
    """
    Train XGBoost classifier.
    Comparison model — results reported alongside RF.
    scale_pos_weight compensates for mangrove class minority.
    """
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    spw   = n_neg / max(n_pos, 1)
    print(f"  XGB scale_pos_weight: {spw:.2f}  (neg={n_neg:,} / pos={n_pos:,})")

    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=random_state,
        eval_metric='logloss',
        verbosity=0,
        scale_pos_weight=spw,
    )
    xgb.fit(X_train, y_train)
    return xgb


# =============================================================================
# 5. Evaluation
# =============================================================================

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str = "Model") -> dict:
    """
    Print classification report and return metrics dict.

    Returns
    -------
    dict: {'confusion_matrix': np.ndarray, 'y_pred': np.ndarray,
           'report': str}
    """
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=['Non-mangrove', 'Mangrove']
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  {model_name} — Classification Report:")
    print(report)

    return {'confusion_matrix': cm, 'y_pred': y_pred, 'report': report}


def compare_models(rf_metrics: dict, xgb_metrics: dict,
                   y_test: np.ndarray) -> pd.DataFrame:
    """
    Side-by-side accuracy comparison table for RF vs XGBoost.

    Parameters
    ----------
    rf_metrics  : dict from evaluate_model() for Random Forest
    xgb_metrics : dict from evaluate_model() for XGBoost
    y_test      : true labels (needed to compute per-class F1 from stored y_pred)

    Returns
    -------
    pd.DataFrame with columns: Model, OA, F1_mangrove, F1_nonmangrove, F1_macro
    """
    from sklearn.metrics import accuracy_score, f1_score

    rows = []
    for name, m in [('Random Forest', rf_metrics), ('XGBoost', xgb_metrics)]:
        y_pred = m['y_pred']
        f1_per_class = f1_score(y_test, y_pred, average=None, labels=[0, 1])
        rows.append({
            'Model'          : name,
            'OA'             : round(accuracy_score(y_test, y_pred), 4),
            'F1_nonmangrove' : round(f1_per_class[0], 4),
            'F1_mangrove'    : round(f1_per_class[1], 4),
            'F1_macro'       : round(f1_score(y_test, y_pred, average='macro'), 4),
        })

    df = pd.DataFrame(rows)
    print("\n  Model comparison:")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 6. Predict: Wall-to-Wall Extent Map
# =============================================================================

def predict_extent(model, indices: dict,
                   original_shape: tuple) -> np.ndarray:
    """
    Apply trained model to full scene — produces wall-to-wall extent map.

    Parameters
    ----------
    model          : fitted RF or XGBoost model
    indices        : dict of 2D index arrays (full scene)
    original_shape : (height, width) of original scene

    Returns
    -------
    2D np.ndarray: 1=mangrove, 0=non-mangrove, -1=invalid
    """
    feature_names = list(indices.keys())
    h, w = original_shape

    # Flatten all bands to (n_pixels, n_features)
    X_full = np.stack(
        [indices[f].ravel() for f in feature_names], axis=1
    ).astype(np.float32)

    # Mask invalid pixels
    valid_mask = np.all(np.isfinite(X_full), axis=1)

    predictions = np.full(h * w, -1, dtype=np.int8)
    predictions[valid_mask] = model.predict(X_full[valid_mask])

    extent_map = predictions.reshape(h, w)

    n_mangrove = int(np.sum(extent_map == 1))
    print(f"  Mangrove extent: {n_mangrove:,} px")
    return extent_map


# =============================================================================
# 7. Save / Load
# =============================================================================

def save_model(model, filepath: str):
    """Save model to .joblib — path: outputs/models/"""
    joblib.dump(model, filepath)
    print(f"  Model saved    : {filepath}")


def load_model(filepath: str):
    """Load model from .joblib"""
    return joblib.load(filepath)

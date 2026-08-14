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

def generate_pseudo_labels(indices: dict,
                            thresholds: dict,
                            candidate_mask: np.ndarray = None) -> np.ndarray:
    """
    Generate binary pseudo-labels using AND logic on MVI + NDMI,
    optionally constrained to a coastal candidate zone.

    Rule:
        mangrove (1) iff (MVI > t_mvi) AND (NDMI > t_ndmi)
                          AND (candidate_mask is True, if provided)
    Otherwise: non-mangrove (0) for valid pixels, -1 for invalid (NaN/inf).

    Spatial constraint: when candidate_mask is provided, pixels outside the
    mask are forced to label=0. This prevents inland vegetation (rainforest,
    revegetated mining, plantations) from being mislabeled as mangrove.

    Parameters
    ----------
    indices        : dict of 2D index arrays from compute_all_indices()
    thresholds     : dict from apply_adaptive_threshold()
    candidate_mask : optional 2D bool array (e.g. from
                     compute_coastal_candidate_mask())

    Returns
    -------
    2D np.ndarray: 1=mangrove, 0=non-mangrove, -1=invalid
    """
    mvi_thresh  = thresholds.get('MVI')
    ndmi_thresh = thresholds.get('NDMI')

    if mvi_thresh is None or ndmi_thresh is None or \
       not np.isfinite(mvi_thresh) or not np.isfinite(ndmi_thresh):
        raise ValueError("MVI and NDMI thresholds required (finite values)")

    mvi  = indices['MVI']
    ndmi = indices['NDMI']

    labels = np.zeros(mvi.shape, dtype=np.int8)

    # Threshold rule
    mangrove_mask = (mvi > mvi_thresh) & (ndmi > ndmi_thresh)

    # Apply spatial constraint
    if candidate_mask is not None:
        mangrove_mask = mangrove_mask & candidate_mask

    labels[mangrove_mask] = 1

    # Invalid: NaN or inf in either index
    invalid_mask = ~(np.isfinite(mvi) & np.isfinite(ndmi))
    labels[invalid_mask] = -1

    n_mangrove = int(np.sum(labels == 1))
    n_nonmang  = int(np.sum(labels == 0))
    n_invalid  = int(np.sum(labels == -1))
    print(f"  Pseudo-labels generated:")
    print(f"  mangrove     : {n_mangrove:,} px")
    print(f"  non-mangrove : {n_nonmang:,} px")
    print(f"  invalid      : {n_invalid:,} px")
    if candidate_mask is not None:
        print(f"  (constrained to coastal candidate zone)")

    return labels


# =============================================================================
# 2. Feature Matrix
# =============================================================================

def build_feature_matrix(indices: dict,
                          labels: np.ndarray,
                          extra_features: dict = None) -> tuple:
    """
    Stack index arrays (and optional diagnostic features) into a feature matrix.
    Excludes invalid pixels (label == -1).

    Parameters
    ----------
    indices        : dict of 2D index arrays (the 5 spectral indices)
    labels         : 2D pseudo-label array from generate_pseudo_labels()
    extra_features : optional dict of {name: 2D array} of additional per-pixel
                     diagnostic features (e.g. {'AbsDepth1640': depth_array}
                     from continuum.absorption_depth_1640()). Each array must
                     match the index array shape. Pixels where an extra feature
                     is NaN are dropped from the matrix to keep rows complete.

    Returns
    -------
    X            : np.ndarray (n_valid_pixels, n_features)
    y            : np.ndarray (n_valid_pixels,)
    feature_names: list of str

    Notes
    -----
    Extra features add spectral-shape information from Tanager's full spectrum
    (not reproducible from broadband multispectral sensors). They feed the
    classifier only; the adaptive threshold and pseudo-labels remain based on
    the spectral indices alone.
    """
    feature_names = list(indices.keys())
    feature_arrays = [indices[f] for f in feature_names]

    # Append optional diagnostic features
    if extra_features:
        for name, arr in extra_features.items():
            if arr.shape != feature_arrays[0].shape:
                raise ValueError(
                    f"extra_feature '{name}' shape {arr.shape} != "
                    f"index shape {feature_arrays[0].shape}"
                )
            feature_names.append(name)
            feature_arrays.append(arr)

    # Valid = labelled AND all feature values finite (handles extra-feature NaN)
    valid_mask = labels != -1
    for arr in feature_arrays:
        valid_mask = valid_mask & np.isfinite(arr)

    X = np.stack([arr[valid_mask] for arr in feature_arrays], axis=1)
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
    Train Random Forest classifier (primary model).
    Uses class_weight='balanced' to handle mangrove class minority.
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
    Train XGBoost classifier (comparison model).
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
    NOTE: this evaluates against pseudo-label test split, not against GMW v3.
    For ground truth evaluation, use a separate function against GMW v3.

    Returns
    -------
    dict: {'confusion_matrix': np.ndarray, 'y_pred': np.ndarray, 'report': str}
    """
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=['Non-mangrove', 'Mangrove']
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  {model_name} (pseudo-label test split):")
    print(report)

    return {'confusion_matrix': cm, 'y_pred': y_pred, 'report': report}


def compare_models(rf_metrics: dict, xgb_metrics: dict,
                   y_test: np.ndarray) -> pd.DataFrame:
    """
    Side-by-side accuracy comparison table for RF vs XGBoost.
    Evaluated against pseudo-label test split.
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
                   original_shape: tuple,
                   candidate_mask: np.ndarray = None,
                   extra_features: dict = None) -> np.ndarray:
    """
    Apply trained model to full scene, producing wall-to-wall extent map.

    Optional spatial constraint: when candidate_mask is provided, pixels
    outside the candidate zone are forced to label=0. This mirrors the
    training-time constraint and prevents inland false positives.

    Parameters
    ----------
    model          : fitted RF or XGBoost model
    indices        : dict of 2D index arrays (full scene)
    original_shape : (height, width) of original scene
    candidate_mask : optional 2D bool array
    extra_features : optional dict of {name: 2D array} of diagnostic features.
                     MUST match the extra_features passed to build_feature_matrix()
                     at training time, in the same key order, so feature columns
                     align between training and prediction.

    Returns
    -------
    2D np.ndarray: 1=mangrove, 0=non-mangrove, -1=invalid
    """
    feature_arrays = [indices[f] for f in indices.keys()]
    if extra_features:
        for name, arr in extra_features.items():
            feature_arrays.append(arr)

    h, w = original_shape

    # Flatten all features to (n_pixels, n_features)
    X_full = np.stack(
        [arr.ravel() for arr in feature_arrays], axis=1
    ).astype(np.float32)

    valid_mask = np.all(np.isfinite(X_full), axis=1)

    predictions = np.full(h * w, -1, dtype=np.int8)
    predictions[valid_mask] = model.predict(X_full[valid_mask])

    extent_map = predictions.reshape(h, w)

    # Apply spatial constraint if provided
    if candidate_mask is not None:
        outside_zone = (~candidate_mask) & (extent_map != -1)
        extent_map[outside_zone] = 0
        print(f"  Spatial constraint applied (outside candidate zone -> 0)")

    n_mangrove = int(np.sum(extent_map == 1))
    print(f"  Mangrove extent: {n_mangrove:,} px")
    return extent_map


# =============================================================================
# 7. Hyperparameter Tuning
# =============================================================================

# Gold-standard search spaces (classification variant)
# Adapted from model_utils.py (regression project reference).
# Scoring: F1 mangrove class (pos_label=1) - preferred over accuracy
# for minority-class remote sensing problems.

from scipy.stats import randint as _randint, uniform as _uniform

_RF_PARAM_DIST = {
    'n_estimators'     : _randint(100, 600),
    'max_depth'        : _randint(3, 30),
    'min_samples_split': _randint(2, 20),
    'min_samples_leaf' : _randint(1, 10),
    'max_features'     : ['sqrt', 'log2', 0.3, 0.5],
    'class_weight'     : ['balanced', 'balanced_subsample'],
}

_XGB_PARAM_DIST = {
    'n_estimators'     : _randint(100, 600),
    'max_depth'        : _randint(3, 10),
    'learning_rate'    : _uniform(0.01, 0.29),   # [0.01, 0.30]
    'subsample'        : _uniform(0.6, 0.4),      # [0.6, 1.0]
    'colsample_bytree' : _uniform(0.6, 0.4),      # [0.6, 1.0]
    'min_child_weight' : _randint(1, 10),
    'gamma'            : _uniform(0, 0.5),
    'reg_alpha'        : _uniform(0, 1.0),        # L1
    'reg_lambda'       : _uniform(0.5, 1.5),      # L2
}


def tune_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                       n_iter: int = 50,
                       cv: int = 5,
                       random_state: int = 42) -> tuple:
    """
    Tune Random Forest classifier via RandomizedSearchCV.

    Search space: gold-standard distributions covering tree depth,
    leaf size, feature sampling, and class weighting. Continuous
    parameters sampled via scipy.stats distributions for better
    coverage than fixed lists.

    Scoring: F1 for the mangrove class (pos_label=1). Preferred over
    overall accuracy for minority-class remote sensing problems.

    Parameters
    ----------
    X_train      : training feature matrix from build_feature_matrix()
    y_train      : training labels from split_data()
    n_iter       : random parameter combinations to try (default 50)
    cv           : cross-validation folds (default 5)
    random_state : random seed for reproducibility

    Returns
    -------
    best_model  : fitted RandomForestClassifier with best parameters
    best_params : dict of best hyperparameters
    best_score  : best CV F1 score (mangrove class, pseudo-label set)
    """
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    from sklearn.metrics import make_scorer, f1_score

    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scorer   = make_scorer(f1_score, pos_label=1, zero_division=0)

    search = RandomizedSearchCV(
        estimator           = RandomForestClassifier(
                                  random_state=random_state,
                                  n_jobs=-1,
                                  oob_score=False,
                              ),
        param_distributions = _RF_PARAM_DIST,
        n_iter              = n_iter,
        scoring             = scorer,
        cv                  = cv_split,
        verbose             = 1,
        random_state        = random_state,
        n_jobs              = -1,
        refit               = True,
    )

    print(f"  [RF] RandomizedSearchCV : {n_iter} iter x {cv}-fold CV")
    print(f"  Scoring                 : F1 (mangrove class)")
    print(f"  Training set            : {X_train.shape[0]:,} samples x "
          f"{X_train.shape[1]} features")

    search.fit(X_train, y_train)

    best_params = search.best_params_
    best_score  = search.best_score_
    best_model  = search.best_estimator_

    print(f"\n  Best CV F1 : {best_score:.4f}")
    print(f"  Best params:")
    for k, v in sorted(best_params.items()):
        print(f"    {k:<22}: {v}")

    return best_model, best_params, best_score


def tune_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                 n_iter: int = 50,
                 cv: int = 5,
                 random_state: int = 42) -> tuple:
    """
    Tune XGBoost classifier via RandomizedSearchCV.

    Search space: gold-standard distributions covering tree depth,
    learning rate, subsampling, column sampling, regularization (L1/L2),
    and minimum child weight. Continuous parameters sampled via
    scipy.stats.uniform for dense coverage.

    scale_pos_weight is set automatically from class ratio to handle
    mangrove minority class - not included in search space to avoid
    interaction with other imbalance-handling params.

    Scoring: F1 for the mangrove class (pos_label=1).

    Parameters
    ----------
    X_train      : training feature matrix from build_feature_matrix()
    y_train      : training labels from split_data()
    n_iter       : random parameter combinations to try (default 50)
    cv           : cross-validation folds (default 5)
    random_state : random seed for reproducibility

    Returns
    -------
    best_model  : fitted XGBClassifier with best parameters
    best_params : dict of best hyperparameters
    best_score  : best CV F1 score (mangrove class, pseudo-label set)
    """
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    from sklearn.metrics import make_scorer, f1_score
    from xgboost import XGBClassifier

    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    spw   = n_neg / max(n_pos, 1)

    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scorer   = make_scorer(f1_score, pos_label=1, zero_division=0)

    search = RandomizedSearchCV(
        estimator           = XGBClassifier(
                                  objective         = 'binary:logistic',
                                  scale_pos_weight  = spw,
                                  random_state      = random_state,
                                  n_jobs            = 1,    # parallelism via n_jobs below
                                  verbosity         = 0,
                                  eval_metric       = 'logloss',
                              ),
        param_distributions = _XGB_PARAM_DIST,
        n_iter              = n_iter,
        scoring             = scorer,
        cv                  = cv_split,
        verbose             = 1,
        random_state        = random_state,
        n_jobs              = -1,
        refit               = True,
    )

    print(f"  [XGB] RandomizedSearchCV : {n_iter} iter x {cv}-fold CV")
    print(f"  Scoring                  : F1 (mangrove class)")
    print(f"  scale_pos_weight         : {spw:.2f}  (neg={n_neg:,} / pos={n_pos:,})")
    print(f"  Training set             : {X_train.shape[0]:,} samples x "
          f"{X_train.shape[1]} features")

    search.fit(X_train, y_train)

    best_params = search.best_params_
    best_score  = search.best_score_
    best_model  = search.best_estimator_

    print(f"\n  Best CV F1 : {best_score:.4f}")
    print(f"  Best params:")
    for k, v in sorted(best_params.items()):
        print(f"    {k:<22}: {v}")

    return best_model, best_params, best_score, search


# =============================================================================
# 8. Save / Load
# =============================================================================

def save_model(model, filepath: str):
    """Save model to .joblib (path: outputs/models/)."""
    joblib.dump(model, filepath)
    print(f"  Model saved    : {filepath}")


def load_model(filepath: str):
    """Load model from .joblib."""
    return joblib.load(filepath)

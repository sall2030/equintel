"""
Train XGBoost models for win and place prediction.

Two models are trained:
  - win_model:   predicts P(horse finishes 1st)
  - place_model: predicts P(horse finishes in the places)

Both are calibrated with isotonic regression so the raw probability output
reflects true frequency, not just rank order.

Training uses a time-based split (last 18 months = validation) to avoid
data leakage from future races.
"""

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

from .features import MODEL_FEATURES


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

XGB_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "gamma": 1.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
}

VAL_MONTHS = 18  # Hold out last N months for validation


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train(
    features: pd.DataFrame,
    models_dir: str = "models",
    verbose: bool = True,
) -> dict:
    """
    Train win and place models.

    Args:
        features: Output of build_features()
        models_dir: Directory to save model artifacts
        verbose: Print training progress

    Returns:
        dict with 'win' and 'place' model objects and metrics
    """
    os.makedirs(models_dir, exist_ok=True)

    # Time-based train/val split
    cutoff = features["date"].max() - pd.DateOffset(months=VAL_MONTHS)
    train_df = features[features["date"] < cutoff].copy()
    val_df = features[features["date"] >= cutoff].copy()

    if verbose:
        print(f"Train: {len(train_df):,} rows ({train_df['date'].min().date()} -> {train_df['date'].max().date()})")
        print(f"Val:   {len(val_df):,} rows ({val_df['date'].min().date()} -> {val_df['date'].max().date()})")

    results = {}
    for target in ("won", "placed"):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {target.upper()} model...")
            print(f"{'='*60}")

        X_train, y_train = _prepare(train_df, target)
        X_val, y_val = _prepare(val_df, target)

        pos_rate = y_train.mean()
        if verbose:
            print(f"Positive rate in train: {pos_rate:.3f}")

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        # XGBoost with early stopping
        model = xgb.XGBClassifier(
            **{k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"},
            early_stopping_rounds=XGB_PARAMS["early_stopping_rounds"],
            scale_pos_weight=1.0,  # don't over-weight positives; calibration handles it
        )
        model.fit(
            X_train_s,
            y_train,
            eval_set=[(X_val_s, y_val)],
            verbose=50 if verbose else False,
        )

        # Wrap in isotonic calibration
        # Use the best iteration's predictions
        cal_model = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        cal_model.fit(X_val_s, y_val)

        # Metrics
        val_probs = cal_model.predict_proba(X_val_s)[:, 1]
        auc = roc_auc_score(y_val, val_probs)
        brier = brier_score_loss(y_val, val_probs)

        if verbose:
            print(f"\nValidation AUC:    {auc:.4f}")
            print(f"Validation Brier:  {brier:.4f}  (lower is better)")
            _print_top_features(model, n=15)

        # Save artifacts
        key = {"won": "win", "placed": "place"}[target]
        joblib.dump(cal_model, os.path.join(models_dir, f"{key}_model.pkl"))
        joblib.dump(scaler, os.path.join(models_dir, f"{key}_scaler.pkl"))

        results[key] = {
            "model": cal_model,
            "scaler": scaler,
            "auc": auc,
            "brier": brier,
            "feature_names": MODEL_FEATURES,
        }

    print(f"\nModels saved to {models_dir}/")
    return results


def _prepare(df: pd.DataFrame, target: str):
    """Return (X, y) arrays from feature DataFrame."""
    # Use only columns that exist
    cols = [c for c in MODEL_FEATURES if c in df.columns]
    X = df[cols].values.astype(float)
    y = df[target].values.astype(int)

    # XGBoost handles NaN natively, but StandardScaler does not
    # Impute NaN with column medians for scaling purposes
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    for j in range(X.shape[1]):
        X[nan_mask[:, j], j] = col_medians[j]

    return X, y


def _print_top_features(model, n=15):
    scores = model.feature_importances_
    pairs = sorted(zip(MODEL_FEATURES[:len(scores)], scores), key=lambda x: -x[1])
    print(f"\nTop {n} features by importance:")
    for name, score in pairs[:n]:
        bar = "█" * int(score * 200)
        print(f"  {name:<35} {score:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Load saved models
# ---------------------------------------------------------------------------

def load_models(models_dir: str = "models") -> dict:
    """Load previously trained model artifacts from disk."""
    result = {}
    for target in ("win", "place"):
        model_path = os.path.join(models_dir, f"{target}_model.pkl")
        scaler_path = os.path.join(models_dir, f"{target}_scaler.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No {target} model found at {model_path}. "
                "Run scripts/train_model.py first."
            )
        result[target] = {
            "model": joblib.load(model_path),
            "scaler": joblib.load(scaler_path),
            "feature_names": MODEL_FEATURES,
        }
    return result

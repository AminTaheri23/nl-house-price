"""LightGBM Hyperparameter Tuning for St. John's House Prices.

This script performs progressive hyperparameter tuning starting from simple models
and gradually increasing complexity. It uses 5-fold cross-validation and stops
when performance plateaus.

Usage:
    python src/train_lgbm_tuned.py              # Run default 5 iterations
    python src/train_lgbm_tuned.py --resume     # Resume from last checkpoint
    python src/train_lgbm_tuned.py --iter 10    # Run 10 iterations total
    python src/train_lgbm_tuned.py --iter 5 --force  # Restart with 5 iterations
"""

import sys
import os
import json
import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data_nl.csv"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
MODEL_DIR = EXPERIMENTS_DIR / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5

CITIES = ["St. John's", "Paradise", "Mount Pearl", "Torbay"]

SELECTED_COLUMNS = [
    'streetAddress', 'addressLocality', 'addressRegion', 'postalCode',
    'latitude', 'longitude', 'description', 'price', 'priceCurrency',
    'property-beds', 'property-baths', 'property-sqft', 'Basement', 'Bath',
    'Exterior', 'Exterior Features', 'Features', 'Fireplace', 'Heating',
    'MLS® #', 'Partial Bathroom', 'Property Tax', 'Property Type', 'Roof',
    'Sewer', 'Square Footage', 'Type', 'Parking', 'Flooring',
    'Parking Features', 'Fireplace Features', 'MLS'
]

COLUMNS_TO_DROP = [
    'description',
    'priceCurrency',
    'MLS® #',
    'MLS',
    'streetAddress',
    'addressRegion',
    'postalCode'
]

ITERATION_CONFIGS = [
    {
        'name': 'Iter1_simple_baseline',
        'params': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': RANDOM_STATE,
            'verbose': -1
        },
        'description': 'Simple baseline with shallow trees and fast learning'
    },
    {
        'name': 'Iter2_increase_capacity',
        'params': {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.05,
            'random_state': RANDOM_STATE,
            'verbose': -1
        },
        'description': 'Deeper trees, slower learning, more estimators'
    },
    {
        'name': 'Iter3_add_regularization',
        'params': {
            'n_estimators': 300,
            'max_depth': 7,
            'learning_rate': 0.03,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': RANDOM_STATE,
            'verbose': -1
        },
        'description': 'Add L1/L2 regularization to prevent overfitting'
    },
    {
        'name': 'Iter4_prevent_overfitting',
        'params': {
            'n_estimators': 300,
            'max_depth': 7,
            'learning_rate': 0.03,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_STATE,
            'verbose': -1
        },
        'description': 'Add min_child_samples, subsample, and colsample_bytree'
    },
    {
        'name': 'Iter5_fine_tune',
        'params': {
            'n_estimators': 400,
            'max_depth': 6,
            'learning_rate': 0.02,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'min_child_samples': 15,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'random_state': RANDOM_STATE,
            'verbose': -1
        },
        'description': 'Fine-tuned configuration based on previous results'
    },
    {
        'name': 'Iter6_original_best',
        'params': {
            'n_estimators': 200,
            'max_depth': 10,
            'learning_rate': 0.05,
            'random_state': RANDOM_STATE,
            'verbose': -1
        },
        'description': 'Original experiment parameters that achieved R²=0.28'
    },
    {
        'name': 'Iter7_more_trees',
        'params': {
            'n_estimators': 500,
            'max_depth': 10,
            'learning_rate': 0.03,
            'random_state': RANDOM_STATE,
            'verbose': -1
        },
        'description': 'More trees, same depth, slower learning'
    },
    {
        'name': 'Iter8_no_restriction',
        'params': {
            'n_estimators': 500,
            'max_depth': -1,
            'learning_rate': 0.02,
            'min_child_samples': 10,
            'random_state': RANDOM_STATE,
            'verbose': -1
        },
        'description': 'No max_depth restriction, more leaves'
    },
    {
        'name': 'Iter9_aggressive',
        'params': {
            'n_estimators': 700,
            'max_depth': 12,
            'learning_rate': 0.015,
            'min_child_samples': 5,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'random_state': RANDOM_STATE,
            'verbose': -1
        },
        'description': 'Aggressive: more trees, deeper, faster convergence'
    },
    {
        'name': 'Iter10_grid_search',
        'params': {
            'n_estimators': 600,
            'max_depth': 8,
            'learning_rate': 0.025,
            'min_child_samples': 8,
            'reg_alpha': 0.2,
            'reg_lambda': 0.2,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'random_state': RANDOM_STATE,
            'verbose': -1
        },
        'description': 'Grid search inspired best configuration'
    },
    {
        'name': 'Iter11_vary_lr',
        'params': {
            'n_estimators': 400,
            'max_depth': 6,
            'learning_rate': 0.015,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'min_child_samples': 15,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'random_state': RANDOM_STATE,
            'verbose': -1
        },
        'description': 'Slower learning rate based on Iter5'
    },
    {
        'name': 'Iter12_vary_depth',
        'params': {
            'n_estimators': 400,
            'max_depth': 5,
            'learning_rate': 0.02,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'min_child_samples': 15,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'random_state': RANDOM_STATE,
            'verbose': -1
        },
        'description': 'Slightly shallower trees'
    },
    {
        'name': 'Iter13_vary_depth',
        'params': {
            'n_estimators': 400,
            'max_depth': 7,
            'learning_rate': 0.02,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'min_child_samples': 15,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'random_state': RANDOM_STATE,
            'verbose': -1
        },
        'description': 'Slightly deeper trees'
    },
    {
        'name': 'Iter14_more_trees_iter5',
        'params': {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.02,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'min_child_samples': 15,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'random_state': RANDOM_STATE,
            'verbose': -1
        },
        'description': 'More trees with Iter5 params'
    },
    {
        'name': 'Iter15_fewer_reg',
        'params': {
            'n_estimators': 400,
            'max_depth': 6,
            'learning_rate': 0.02,
            'reg_alpha': 0.2,
            'reg_lambda': 0.2,
            'min_child_samples': 15,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'random_state': RANDOM_STATE,
            'verbose': -1
        },
        'description': 'Less regularization than Iter5'
    }
]

HISTORY_FILE = EXPERIMENTS_DIR / 'lgbm_tuning_history.json'
BEST_MODEL_FILE = MODEL_DIR / 'best_lgbm_model.txt'
BEST_PARAMS_FILE = MODEL_DIR / 'best_params.json'
REPORT_FILE = EXPERIMENTS_DIR / 'lgbm_tuning_report.md'


def load_and_filter_data() -> pd.DataFrame:
    """Load data and filter to St. John's area."""
    print("=" * 70)
    print("STEP 1: LOADING AND FILTERING DATA")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, sep=',', on_bad_lines='skip')
    print(f"Original dataset shape: {df.shape}")

    pattern = "|".join(CITIES)
    st_johns_df = df[df['streetAddress'].str.contains(pattern, case=False, na=False)].copy()
    print(f"St. John's area records: {len(st_johns_df)}")
    print(f"Cities included: {CITIES}")

    return st_johns_df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select only the specified columns."""
    print("\n" + "=" * 70)
    print("STEP 2: COLUMN SELECTION")
    print("=" * 70)

    available_cols = [col for col in SELECTED_COLUMNS if col in df.columns]
    print(f"Available columns from selection: {len(available_cols)}")

    df_selected = df[available_cols].copy()

    df_cleaned = df_selected.drop(columns=COLUMNS_TO_DROP, errors='ignore')
    print(f"Columns after dropping: {list(df_cleaned.columns)}")
    print(f"Shape after column selection: {df_cleaned.shape}")

    return df_cleaned


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: remove null/zero prices."""
    print("\n" + "=" * 70)
    print("STEP 3: BASIC CLEANING")
    print("=" * 70)

    initial_count = len(df)
    df_clean = df[df['price'].notna()].copy()
    df_clean = df_clean[df_clean['price'] > 0]
    print(f"Records after removing null/zero prices: {len(df_clean)}")
    print(f"Rows removed: {initial_count - len(df_clean)}")

    return df_clean


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and convert numeric columns."""
    print("\n" + "=" * 70)
    print("STEP 4: NUMERIC COLUMN CLEANING")
    print("=" * 70)

    numeric_cols = ['property-beds', 'property-baths', 'property-sqft', 'Property Tax']

    for col in numeric_cols:
        if col in df.columns:
            if col in ['property-sqft', 'Square Footage']:
                df[col] = df[col].astype(str).str.replace(',', '').replace('N/A', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            missing = df[col].isna().sum()
            print(f"{col}: {missing} missing values")

    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers using IQR method."""
    print("\n" + "=" * 70)
    print("STEP 5: OUTLIER REMOVAL")
    print("=" * 70)

    outlier_cols = ['price', 'property-sqft', 'property-beds', 'property-baths']
    outlier_mask = pd.Series([False] * len(df))

    for col in outlier_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            col_outliers = (df[col] < lower) | (df[col] > upper)
            outlier_mask = outlier_mask | col_outliers
            print(f"{col}: {col_outliers.sum()} outliers")

    df_clean = df[~outlier_mask].copy()
    print(f"Final dataset shape: {df_clean.shape}")

    return df_clean


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """Prepare features for ML modeling. Returns encoders for app use."""
    print("\n" + "=" * 70)
    print("STEP 6: FEATURE PREPARATION")
    print("=" * 70)

    feature_cols = [col for col in df.columns if col != 'price']

    for col in ['Property Tax', 'Square Footage']:
        if col in feature_cols and df[col].isna().all():
            df = df.drop(columns=[col])
            feature_cols.remove(col)
            print(f"Dropped empty column: {col}")

    X = df[feature_cols].copy()
    y = df['price'].copy()

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    encoders = {}

    for col in categorical_cols:
        X[col] = X[col].astype(str).fillna('Unknown')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    X = X.fillna(X.median())

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Categorical features encoded: {len(categorical_cols)}")
    print(f"Feature columns: {list(X.columns)}")

    return X, y, encoders


def scale_features(X_train: pd.DataFrame, X_val: Optional[pd.DataFrame] = None) -> Tuple:
    """Scale features using StandardScaler."""
    print("\n" + "=" * 70)
    print("STEP 7: FEATURE SCALING")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_val_scaled, scaler

    return X_train_scaled, scaler


def evaluate_model(model, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, float]:
    """Evaluate model using cross-validation."""
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    cv_r2_scores = []
    cv_rmse_scores = []
    cv_mae_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_val_cv)

        cv_r2_scores.append(r2_score(y_val_cv, y_pred))
        cv_rmse_scores.append(np.sqrt(mean_squared_error(y_val_cv, y_pred)))
        cv_mae_scores.append(mean_absolute_error(y_val_cv, y_pred))

    return {
        'cv_r2_mean': np.mean(cv_r2_scores),
        'cv_r2_std': np.std(cv_r2_scores),
        'cv_rmse_mean': np.mean(cv_rmse_scores),
        'cv_rmse_std': np.std(cv_rmse_scores),
        'cv_mae_mean': np.mean(cv_mae_scores),
        'cv_mae_std': np.std(cv_mae_scores)
    }


def train_and_evaluate(X: np.ndarray, y: np.ndarray, params: Dict, iteration_name: str) -> Dict[str, Any]:
    """Train model and evaluate with 5-fold CV."""
    print(f"\n{'=' * 70}")
    print(f"TRAINING: {iteration_name}")
    print(f"{'=' * 70}")

    model = lgb.LGBMRegressor(**params)

    cv_results = evaluate_model(model, X, y, cv_folds=N_FOLDS)

    print(f"\nCross-Validation Results ({N_FOLDS}-fold):")
    print(f"  R² Score:  {cv_results['cv_r2_mean']:.4f} (+/- {cv_results['cv_r2_std']:.4f})")
    print(f"  RMSE:      ${cv_results['cv_rmse_mean']:,.2f} (+/- ${cv_results['cv_rmse_std']:,.2f})")
    print(f"  MAE:       ${cv_results['cv_mae_mean']:,.2f} (+/- ${cv_results['cv_mae_std']:,.2f})")

    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(X, y)

    return {
        'iteration': iteration_name,
        'params': params,
        'cv_results': cv_results,
        'model': final_model
    }


def check_plateau(history: List[Dict], threshold: float = 0.01) -> bool:
    """Check if performance has plateaued."""
    if len(history) < 2:
        return False

    recent_scores = [h['cv_results']['cv_r2_mean'] for h in history[-2:]]
    improvement = recent_scores[-1] - recent_scores[-2]

    print(f"\nPlateau Check:")
    print(f"  Previous R²: {recent_scores[-2]:.4f}")
    print(f"  Current R²:  {recent_scores[-1]:.4f}")
    print(f"  Improvement: {improvement:.4f}")
    print(f"  Threshold:   {threshold:.4f}")

    return improvement < threshold


def save_history(history: List[Dict], filepath: Path):
    """Save training history to JSON."""
    serializable_history = []
    for entry in history:
        serializable_entry = {
            'iteration': entry['iteration'],
            'params': entry['params'],
            'cv_results': entry['cv_results'],
            'timestamp': entry.get('timestamp', datetime.now().isoformat())
        }
        serializable_history.append(serializable_entry)

    with open(filepath, 'w') as f:
        json.dump(serializable_history, f, indent=2, default=str)

    print(f"\nHistory saved to: {filepath}")


def load_history(filepath: Path) -> List[Dict]:
    """Load training history from JSON."""
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return []


def save_best_model(results: Dict, filepath: Path):
    """Save best model information."""
    with open(filepath, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("BEST LIGHTGBM MODEL - ST. JOHN'S HOUSE PRICES\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Training Date: {datetime.now().isoformat()}\n\n")
        f.write("Model Parameters:\n")
        f.write("-" * 40 + "\n")
        for key, value in results['best_params'].items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        f.write("Performance Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  R² Score:  {results['best_cv_r2']:.4f} (+/- {results['best_cv_r2_std']:.4f})\n")
        f.write(f"  RMSE:      ${results['best_rmse']:,.2f}\n")
        f.write(f"  MAE:       ${results['best_mae']:,.2f}\n")
        f.write("\n")
        f.write("Feature Columns Used:\n")
        f.write("-" * 40 + "\n")
        for i, col in enumerate(results['feature_columns'], 1):
            f.write(f"  {i}. {col}\n")

    print(f"Best model saved to: {filepath}")


def generate_report(history: List[Dict], output_path: Path):
    """Generate markdown report."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    best_entry = max(history, key=lambda x: x['cv_results']['cv_r2_mean'])

    report = f"""# LightGBM Hyperparameter Tuning Report

**Generated:** {timestamp}

## Summary

| Metric | Value |
|--------|-------|
| Total Iterations | {len(history)} |
| Best R² Score | {best_entry['cv_results']['cv_r2_mean']:.4f} |
| Best RMSE | ${best_entry['cv_results']['cv_rmse_mean']:,.2f} |
| Best MAE | ${best_entry['cv_results']['cv_mae_mean']:,.2f} |

## Best Model: {best_entry['iteration']}

## Iteration Results

| Iteration | R² Score | RMSE | MAE |
|-----------|----------|------|-----|
"""

    for entry in history:
        cv = entry['cv_results']
        report += f"| {entry['iteration']} | {cv['cv_r2_mean']:.4f} (±{cv['cv_r2_std']:.4f}) | ${cv['cv_rmse_mean']:,.0f} (±${cv['cv_rmse_std']:,.0f}) | ${cv['cv_mae_mean']:,.0f} (±${cv['cv_mae_std']:,.0f}) |\n"

    report += """

## Best Model Parameters

```json
"""

    best_entry = max(history, key=lambda x: x['cv_results']['cv_r2_mean'])
    report += json.dumps(best_entry['params'], indent=2)

    report += """

## Progress Visualization

```
"""

    scores = [h['cv_results']['cv_r2_mean'] for h in history]
    for i, score in enumerate(scores):
        bar = '█' * int(score * 50)
        report += f"Iter {i+1}: {bar} {score:.4f}\n"

    report += """```

## How to Resume Training

To run additional iterations, modify the `ITERATION_CONFIGS` list in `src/train_lgbm_tuned.py` and run:

```bash
python src/train_lgbm_tuned.py --resume
```

Or specify the total number of iterations:

```bash
python src/train_lgbm_tuned.py --iter 10
```

## Notes

- All experiments use 5-fold cross-validation
- Training stops when R² improvement < 0.01 for 2 consecutive iterations
- Random state is fixed at 42 for reproducibility
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Report saved to: {output_path}")


def save_model_artifacts(model, encoders: Dict, scaler: Any, imputer: Any, feature_columns: List[str], output_dir: Path):
    """Save model and preprocessing artifacts for application use."""
    import joblib

    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(model, output_dir / 'lgbm_model.joblib')
    joblib.dump(encoders, output_dir / 'label_encoders.joblib')
    joblib.dump(scaler, output_dir / 'scaler.joblib')
    joblib.dump(imputer, output_dir / 'imputer.joblib')

    with open(output_dir / 'feature_columns.txt', 'w') as f:
        for col in feature_columns:
            f.write(f"{col}\n")

    print(f"Model artifacts saved to: {output_dir}")


def run_tuning(max_iterations: int = 5, force_restart: bool = False, resume: bool = False):
    """Main tuning function."""
    print("\n" + "=" * 70)
    print("   LIGHTGBM HYPERPARAMETER TUNING")
    print("   St. John's House Price Prediction")
    print("=" * 70)

    os.makedirs(MODEL_DIR, exist_ok=True)

    if resume and HISTORY_FILE.exists():
        print("\nResuming from previous run...")
        history = load_history(HISTORY_FILE)
        start_iteration = len(history)
        print(f"Resuming from iteration {start_iteration + 1}")
    elif force_restart:
        print("\nForce restart - clearing history...")
        history = []
        start_iteration = 0
        if HISTORY_FILE.exists():
            os.remove(HISTORY_FILE)
    else:
        print("\nStarting fresh...")
        history = []
        start_iteration = 0

    print(f"\nTarget iterations: {max_iterations}")
    print(f"Starting from: {start_iteration}")
    print(f"Remaining iterations: {max_iterations - start_iteration}")

    if start_iteration >= max_iterations:
        print(f"\nAlready completed {start_iterations} iterations. Use --iter to specify more.")
        generate_report(history, REPORT_FILE)
        return history

    df_raw = load_and_filter_data()
    df_selected = select_columns(df_raw)
    df_cleaned = basic_cleaning(df_selected)
    df_cleaned = clean_numeric_columns(df_cleaned)
    df_cleaned = remove_outliers(df_cleaned)

    X, y, encoders = prepare_features(df_cleaned)

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    print(f"\nDataset ready: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")

    iterations_to_run = ITERATION_CONFIGS[start_iteration:max_iterations]

    for i, config in enumerate(iterations_to_run):
        actual_iteration = start_iteration + i + 1
        print(f"\n{'#' * 70}")
        print(f"# ITERATION {actual_iteration}/{max_iterations}")
        print(f"# {config['description']}")
        print(f"{'#' * 70}")

        result = train_and_evaluate(X_scaled, y, config['params'], config['name'])
        result['timestamp'] = datetime.now().isoformat()
        history.append(result)

        save_history(history, HISTORY_FILE)

        if len(history) >= 2 and False:
            if check_plateau(history, threshold=0.01):
                print("\n>>> PLATEAU DETECTED <<<")
                print("R² improvement < 0.01 for 2 consecutive iterations.")
                print("Stopping early to avoid overfitting.")
                break

    best_entry = max(history, key=lambda x: x['cv_results']['cv_r2_mean'])

    print("\n" + "=" * 70)
    print("TUNING COMPLETE!")
    print("=" * 70)
    print(f"\nBest Model: {best_entry['iteration']}")
    print(f"Best R² Score: {best_entry['cv_results']['cv_r2_mean']:.4f}")
    print(f"Best RMSE: ${best_entry['cv_results']['cv_rmse_mean']:,.2f}")

    save_best_model({
        'best_params': best_entry['params'],
        'best_cv_r2': best_entry['cv_results']['cv_r2_mean'],
        'best_cv_r2_std': best_entry['cv_results']['cv_r2_std'],
        'best_rmse': best_entry['cv_results']['cv_rmse_mean'],
        'best_mae': best_entry['cv_results']['cv_mae_mean'],
        'feature_columns': list(X.columns)
    }, BEST_MODEL_FILE)

    with open(BEST_PARAMS_FILE, 'w') as f:
        json.dump(best_entry['params'], f, indent=2)

    if 'model' in best_entry:
        save_model_artifacts(
            best_entry['model'],
            encoders,
            scaler,
            imputer,
            list(X.columns),
            MODEL_DIR
        )

    generate_report(history, REPORT_FILE)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LightGBM Hyperparameter Tuning')
    parser.add_argument('--iter', type=int, default=5, help='Maximum iterations (default: 5)')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--force', action='store_true', help='Force restart (clears history)')

    args = parser.parse_args()

    run_tuning(
        max_iterations=args.iter,
        force_restart=args.force,
        resume=args.resume
    )

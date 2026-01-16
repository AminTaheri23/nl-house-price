"""Run baseline experiment and save results."""

import sys
import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PATH, OUTPUT_DIR, RANDOM_STATE, TEST_SIZE, MODEL_PARAMS

warnings.filterwarnings('ignore')


def run_baseline():
    """Run baseline pipeline and capture metrics."""
    print("\n" + "=" * 70)
    print("   BASELINE EXPERIMENT - FULL DATASET")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {
        'experiment': 'baseline',
        'timestamp': datetime.now().isoformat(),
        'description': 'Full dataset ML pipeline'
    }

    print("\nLoading data...")
    df = pd.read_csv(DATA_PATH, sep=',', on_bad_lines='skip')
    results['original_shape'] = list(df.shape)
    print(f"Dataset shape: {df.shape}")

    print("Basic cleaning...")
    df = df[df['price'].notna()].copy()
    df = df[df['price'] > 0]
    print(f"After cleaning: {df.shape}")

    print("Cleaning numeric columns...")
    numeric_cols = ['property-beds', 'property-baths', 'property-sqft', 'Year Built',
                    'latitude', 'longitude', 'Garage Spaces', 'Parking Spaces']
    for col in numeric_cols:
        if col in df.columns:
            if col == 'property-sqft':
                df[col] = df[col].astype(str).str.replace(',', '').replace('N/A', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print("Removing outliers...")
    df = df[(df['price'] >= df['price'].quantile(0.01)) & (df['price'] <= df['price'].quantile(0.99))]

    outlier_cols = ['price', 'property-sqft', 'property-beds', 'property-baths']
    outlier_mask = pd.Series([False] * len(df))
    for col in outlier_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = outlier_mask | ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR))
    df = df[~outlier_mask]

    results['cleaned_shape'] = list(df.shape)

    print("Feature engineering...")
    df['house_age'] = 2026 - df['Year Built'].fillna(df['Year Built'].median())
    df['total_rooms'] = df['property-beds'].fillna(0) + df['property-baths'].fillna(0)
    df['log_sqft'] = np.log1p(df['property-sqft'].clip(lower=1))
    df['price_log'] = np.log1p(df['price'])

    feature_cols = ['latitude', 'longitude', 'property-beds', 'property-baths',
                    'property-sqft', 'house_age', 'log_sqft', 'price_log']
    available_cols = [col for col in feature_cols if col in df.columns]

    X = df[available_cols].drop('price_log', axis=1, errors='ignore')
    y = df['price']

    if 'price_log' in available_cols:
        X = df[available_cols].drop('price_log', axis=1)
    else:
        X = df[available_cols]

    print(f"Feature matrix: {X.shape}")

    X = X.fillna(X.median())

    print("Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train_scaled)
    X_test_imp = imputer.transform(X_test_scaled)

    feature_names = list(X.columns)
    results['features'] = feature_names

    print("Training models...")
    models = {
        'Random Forest': RandomForestRegressor(**MODEL_PARAMS['random_forest']),
        'LightGBM': lgb.LGBMRegressor(**MODEL_PARAMS['lightgbm']),
        'XGBoost': xgb.XGBRegressor(**MODEL_PARAMS['xgboost'])
    }

    model_results = {}
    feature_importance = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_imp, y_train)

        y_pred = model.predict(X_test_imp)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        cv_scores = cross_val_score(model, X_train_imp, y_train, cv=5, scoring='r2')

        model_results[name] = {
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'CV_R2_Mean': float(cv_scores.mean()),
            'CV_R2_Std': float(cv_scores.std())
        }

        print(f"  R²: {r2:.4f}, RMSE: ${rmse:,.2f}, MAE: ${mae:,.2f}")

        if hasattr(model, 'feature_importances_'):
            feature_importance[name] = dict(zip(feature_names, model.feature_importances_))

    results['model_results'] = model_results
    results['feature_importance'] = feature_importance

    print("\n" + "=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)
    for name, metrics in model_results.items():
        print(f"{name}: R²={metrics['R2']:.4f}, RMSE=${metrics['RMSE']:,.2f}")

    experiments_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments')
    os.makedirs(experiments_dir, exist_ok=True)

    with open(os.path.join(experiments_dir, 'baseline_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nBaseline results saved to: {experiments_dir}/baseline_results.json")

    return results


if __name__ == "__main__":
    run_baseline()

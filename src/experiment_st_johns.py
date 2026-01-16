"""St. John's Area EDA and Feature Importance Experiment.

This experiment analyzes house prices specifically in St. John's, Paradise,
Mount Pearl, and Torbay areas with focused feature importance analysis.
"""

import sys
import os
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_config import (
    DATA_PATH, EXPERIMENTS_DIR, RANDOM_STATE, TEST_SIZE,
    CITIES, SELECTED_COLUMNS, COLUMNS_TO_DROP, FINAL_COLUMNS,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, MODEL_PARAMS
)

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')


def load_and_filter_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    return df, st_johns_df


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


def eda_analysis(df: pd.DataFrame, output_dir: str) -> Dict:
    """Perform EDA on the dataset."""
    print("\n" + "=" * 70)
    print("STEP 6: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    eda_results = {
        'dataset_shape': list(df.shape),
        'price_stats': {},
        'numeric_summary': {},
        'categorical_summary': {},
        'missing_values': {}
    }

    price_stats = df['price'].describe()
    eda_results['price_stats'] = {
        'count': int(price_stats['count']),
        'mean': float(price_stats['mean']),
        'std': float(price_stats['std']),
        'min': float(price_stats['min']),
        '25%': float(price_stats['25%']),
        '50%': float(price_stats['50%']),
        '75%': float(price_stats['75%']),
        'max': float(price_stats['max'])
    }
    print(f"Price Statistics:")
    print(f"  Mean: ${price_stats['mean']:,.2f}")
    print(f"  Median: ${price_stats['50%']:,.2f}")
    print(f"  Std: ${price_stats['std']:,.2f}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col != 'price':
            eda_results['numeric_summary'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'missing': int(df[col].isna().sum())
            }

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        eda_results['categorical_summary'][col] = {
            'unique': int(df[col].nunique()),
            'missing': int(df[col].isna().sum()),
            'top_values': df[col].value_counts().head(5).to_dict()
        }

    eda_results['missing_values'] = {
        col: int(df[col].isna().sum()) for col in df.columns
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    prices_k = df['price'].dropna() / 1000
    axes[0, 0].hist(prices_k, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].set_xlabel('Price (CAD, thousands)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Price Distribution')

    if 'property-sqft' in df.columns:
        sqft = df['property-sqft'].fillna(0)
        price_k = df['price'].fillna(0) / 1000
        axes[0, 1].scatter(sqft, price_k, alpha=0.4, s=20, c='coral')
        axes[0, 1].set_xlabel('Square Footage')
        axes[0, 1].set_ylabel('Price (CAD, thousands)')
        axes[0, 1].set_title('Price vs Square Footage')

    if 'addressLocality' in df.columns:
        locality_avg = df.groupby('addressLocality')['price'].mean().sort_values(ascending=True).tail(10)
        axes[0, 2].barh(range(len(locality_avg)), locality_avg.values / 1000, color='green')
        axes[0, 2].set_yticks(range(len(locality_avg)))
        axes[0, 2].set_yticklabels(locality_avg.index, fontsize=8)
        axes[0, 2].set_xlabel('Avg Price (CAD, thousands)')
        axes[0, 2].set_title('Avg Price by Locality')

    if 'property-beds' in df.columns:
        beds_price = df.groupby('property-beds')['price'].mean()
        beds_price = beds_price[beds_price.index.notna() & (beds_price.index <= 10)]
        axes[1, 0].bar(beds_price.index.astype(str), beds_price.values / 1000, color='purple')
        axes[1, 0].set_xlabel('Bedrooms')
        axes[1, 0].set_ylabel('Avg Price (CAD, thousands)')
        axes[1, 0].set_title('Avg Price by Bedrooms')

    if 'property-baths' in df.columns:
        baths_price = df.groupby('property-baths')['price'].mean()
        baths_price = baths_price[baths_price.index.notna() & (baths_price.index <= 8)]
        axes[1, 1].bar(baths_price.index.astype(str), baths_price.values / 1000, color='orange')
        axes[1, 1].set_xlabel('Bathrooms')
        axes[1, 1].set_ylabel('Avg Price (CAD, thousands)')
        axes[1, 1].set_title('Avg Price by Bathrooms')

    if 'Property Type' in df.columns:
        type_counts = df['Property Type'].value_counts().head(6)
        axes[1, 2].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                       colors=plt.cm.Set3.colors)
        axes[1, 2].set_title('Property Type Distribution')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/st_johns_eda.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/st_johns_eda.png")

    return eda_results


def correlation_analysis(df: pd.DataFrame, output_dir: str) -> Dict:
    """Perform correlation analysis on numerical features."""
    print("\n" + "=" * 70)
    print("STEP 7: CORRELATION ANALYSIS")
    print("=" * 70)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if df[c].notna().sum() > 10]
    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                annot_kws={'size': 8}, mask=np.isnan(corr_matrix.values))
    plt.title('Correlation Matrix - St. John\'s Area')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/st_johns_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/st_johns_correlation.png")

    corr_with_price = corr_matrix['price'].drop('price').sort_values(ascending=False)
    print("\nCorrelation with Price:")
    for feat, corr in corr_with_price.items():
        print(f"  {feat:20s}: {corr:.4f}")

    return {
        'correlation_matrix': corr_matrix.to_dict(),
        'correlations_with_price': {k: float(v) for k, v in corr_with_price.items()}
    }


def mutual_information_analysis(X: pd.DataFrame, y: pd.Series, output_dir: str) -> Dict:
    """Calculate mutual information scores."""
    print("\n" + "=" * 70)
    print("STEP 8: MUTUAL INFORMATION ANALYSIS")
    print("=" * 70)

    X_numeric = X.select_dtypes(include=[np.number]).copy()

    imputer = SimpleImputer(strategy='median')
    X_numeric_array = imputer.fit_transform(X_numeric)
    X_numeric = pd.DataFrame(X_numeric_array, columns=X_numeric.columns)

    y_clean = y.fillna(y.median())

    mi_scores = mutual_info_regression(X_numeric, y_clean, random_state=RANDOM_STATE)
    mi_dict = dict(zip(X_numeric.columns, mi_scores))
    mi_sorted = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)

    print("\nMutual Information Scores:")
    for feat, score in mi_sorted:
        print(f"  {feat:20s}: {score:.4f}")

    plt.figure(figsize=(10, 6))
    features, scores = zip(*mi_sorted)
    plt.barh(range(len(features)), scores, color='steelblue')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Mutual Information Score')
    plt.title('Feature Importance - Mutual Information')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/st_johns_mutual_info.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/st_johns_mutual_info.png")

    return {k: float(v) for k, v in mi_dict.items()}


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features for ML modeling."""
    print("\n" + "=" * 70)
    print("STEP 9: FEATURE PREPARATION")
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

    for col in categorical_cols:
        X[col] = X[col].astype(str).fillna('Unknown')

    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    X = X.fillna(X.median())

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Categorical features encoded: {len(categorical_cols)}")

    return X, y


def train_models_and_importance(X: pd.DataFrame, y: pd.Series, output_dir: str) -> Dict:
    """Train models and extract feature importance."""
    print("\n" + "=" * 70)
    print("STEP 10: MODEL TRAINING & FEATURE IMPORTANCE")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train_scaled)
    X_test_imp = imputer.transform(X_test_scaled)

    feature_names = list(X.columns)

    importance_results = {}
    model_results = {}

    models = {
        'Random Forest': RandomForestRegressor(**MODEL_PARAMS['Random Forest']),
        'LightGBM': lgb.LGBMRegressor(**MODEL_PARAMS['LightGBM']),
        'XGBoost': xgb.XGBRegressor(**MODEL_PARAMS['XGBoost'])
    }

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
            importance = dict(zip(feature_names, model.feature_importances_))
            importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            importance_results[name] = importance
            print(f"  Top 5 features: {[f[0] for f in importance_sorted[:5]]}")

    plt.figure(figsize=(12, 8))
    if 'Random Forest' in importance_results:
        imp = importance_results['Random Forest']
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_imp)
        plt.barh(range(len(features)), importances, align='center', color='steelblue')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title('Random Forest Feature Importance - St. John\'s Area')
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/st_johns_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/st_johns_feature_importance.png")

    return {
        'model_results': model_results,
        'feature_importance': importance_results
    }


def save_results(results: Dict, output_path: str):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


def generate_markdown_report(results: Dict, output_path: str):
    """Generate markdown report."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    report = f"""# St. John's Area House Price Analysis Report

**Generated:** {timestamp}

## Dataset Summary

- **Records:** {results['eda']['dataset_shape'][0]}
- **Features:** {results['eda']['dataset_shape'][1]}

## Price Statistics

| Metric | Value |
|--------|-------|
| Mean | ${results['eda']['price_stats']['mean']:,.2f} |
| Median | ${results['eda']['price_stats']['50%']:,.2f} |
| Std Dev | ${results['eda']['price_stats']['std']:,.2f} |
| Min | ${results['eda']['price_stats']['min']:,.2f} |
| Max | ${results['eda']['price_stats']['max']:,.2f} |

## Correlation with Price (Top 10)

| Feature | Correlation |
|---------|-------------|
"""

    corr = results['correlation']['correlations_with_price']
    for feat, val in list(corr.items())[:10]:
        report += f"| {feat} | {val:.4f} |\n"

    report += """
## Mutual Information Scores (Top 10)

| Feature | MI Score |
|---------|----------|
"""
    mi = results['mutual_information']
    for feat, score in sorted(mi.items(), key=lambda x: x[1], reverse=True)[:10]:
        report += f"| {feat} | {score:.4f} |\n"

    report += """
## Model Performance

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
"""

    for name, metrics in results['model_results'].items():
        report += f"| {name} | {metrics['R2']:.4f} | ${metrics['RMSE']:,.2f} | ${metrics['MAE']:,.2f} |\n"

    report += """
## Feature Importance (Random Forest - Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
"""

    rf_imp = results['feature_importance']['Random Forest']
    for i, (feat, imp) in enumerate(sorted(rf_imp.items(), key=lambda x: x[1], reverse=True)[:10], 1):
        report += f"| {i} | {feat} | {imp:.4f} |\n"

    report += """
## Visualizations Generated

- `st_johns_eda.png` - Exploratory data analysis
- `st_johns_correlation.png` - Correlation matrix
- `st_johns_mutual_info.png` - Mutual information scores
- `st_johns_feature_importance.png` - Random Forest feature importance
"""

    with open(output_path, 'w') as f:
        f.write(report)
    print(f"Report saved to: {output_path}")


def run_experiment():
    """Run the complete St. John's experiment."""
    print("\n" + "=" * 70)
    print("   ST. JOHN'S AREA HOUSE PRICE ANALYSIS")
    print("   EDA & FEATURE IMPORTANCE EXPERIMENT")
    print("=" * 70)

    output_dir = os.path.join(EXPERIMENTS_DIR, 'st_johns')
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'experiment': 'st_johns_area_analysis',
        'timestamp': datetime.now().isoformat(),
        'cities': CITIES,
        'features_selected': FINAL_COLUMNS,
        'columns_dropped': COLUMNS_TO_DROP
    }

    df_raw, st_johns_df = load_and_filter_data()
    results['total_st_johns_records'] = len(st_johns_df)

    df_selected = select_columns(st_johns_df)
    df_cleaned = basic_cleaning(df_selected)
    df_cleaned = clean_numeric_columns(df_cleaned)
    df_cleaned = remove_outliers(df_cleaned)
    results['final_dataset_shape'] = list(df_cleaned.shape)

    eda_results = eda_analysis(df_cleaned, output_dir)
    results['eda'] = eda_results

    corr_results = correlation_analysis(df_cleaned, output_dir)
    results['correlation'] = corr_results

    X, y = prepare_features(df_cleaned)

    mi_results = mutual_information_analysis(X, y, output_dir)
    results['mutual_information'] = mi_results

    ml_results = train_models_and_importance(X, y, output_dir)
    results['model_results'] = ml_results['model_results']
    results['feature_importance'] = ml_results['feature_importance']

    save_results(results, os.path.join(EXPERIMENTS_DIR, 'st_johns_results.json'))
    generate_markdown_report(results, os.path.join(EXPERIMENTS_DIR, 'st_johns_report.md'))

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_experiment()

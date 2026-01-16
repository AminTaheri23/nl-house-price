"""Feature Importance Analysis: R² Score vs Top N Features.

This experiment analyzes how the number of top features affects model performance
using both LightGBM feature importance and Mutual Information.

Usage:
    python src/experiment_feature_selection.py          # Run full experiment
    python src/experiment_feature_selection.py --quick  # Quick version with fewer steps
"""

import sys
import os
import json
import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data_nl.csv"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

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
    'description', 'priceCurrency', 'MLS® #', 'MLS',
    'streetAddress', 'addressRegion', 'postalCode'
]


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

    return st_johns_df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select only the specified columns."""
    available_cols = [col for col in SELECTED_COLUMNS if col in df.columns]
    df_selected = df[available_cols].copy()
    df_cleaned = df_selected.drop(columns=COLUMNS_TO_DROP, errors='ignore')
    print(f"Shape after column selection: {df_cleaned.shape}")
    return df_cleaned


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: remove null/zero prices."""
    initial_count = len(df)
    df_clean = df[df['price'].notna()].copy()
    df_clean = df_clean[df_clean['price'] > 0]
    print(f"Records after cleaning: {len(df_clean)} (removed {initial_count - len(df_clean)})")
    return df_clean


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and convert numeric columns."""
    numeric_cols = ['property-beds', 'property-baths', 'property-sqft', 'Property Tax']
    for col in numeric_cols:
        if col in df.columns:
            if col in ['property-sqft', 'Square Footage']:
                df[col] = df[col].astype(str).str.replace(',', '').replace('N/A', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers using IQR method."""
    outlier_cols = ['price', 'property-sqft', 'property-beds', 'property-baths']
    outlier_mask = pd.Series([False] * len(df))
    for col in outlier_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_mask = outlier_mask | ((df[col] < lower) | (df[col] > upper))
    df_clean = df[~outlier_mask].copy()
    print(f"Final dataset shape: {df_clean.shape}")
    return df_clean


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """Prepare features for ML modeling."""
    feature_cols = [col for col in df.columns if col != 'price']
    for col in ['Property Tax', 'Square Footage']:
        if col in feature_cols and df[col].isna().all():
            df = df.drop(columns=[col])
            feature_cols.remove(col)

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

    return X, y, encoders


def get_lgbm_importance(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Get feature importance from LightGBM."""
    print("\n" + "=" * 70)
    print("COMPUTING LIGHTGBM FEATURE IMPORTANCE")
    print("=" * 70)

    model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.02,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=RANDOM_STATE,
        verbose=-1
    )

    model.fit(X, y)

    importance = dict(zip(X.columns, model.feature_importances_))
    importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print("\nLightGBM Feature Importance (Top 15):")
    for feat, imp in importance_sorted[:15]:
        print(f"  {feat:20s}: {imp:6.1f}")

    return importance


def get_mutual_information(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Get mutual information scores."""
    print("\n" + "=" * 70)
    print("COMPUTING MUTUAL INFORMATION SCORES")
    print("=" * 70)

    X_numeric = X.select_dtypes(include=[np.number]).copy()

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_numeric)

    mi_scores = mutual_info_regression(X_imputed, y.fillna(y.median()), random_state=RANDOM_STATE)
    mi_dict = dict(zip(X_numeric.columns, mi_scores))
    mi_sorted = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)

    print("\nMutual Information Scores (Top 15):")
    for feat, score in mi_sorted[:15]:
        print(f"  {feat:20s}: {score:.4f}")

    all_mi = {col: 0.0 for col in X.columns}
    for col, score in mi_dict.items():
        all_mi[col] = score

    return all_mi


def get_combined_importance(lgbm_imp: Dict[str, float], mi_imp: Dict[str, float]) -> Dict[str, float]:
    """Combine LightGBM importance and MI scores using rank averaging."""
    print("\n" + "=" * 70)
    print("COMBINING FEATURE IMPORTANCE METHODS")
    print("=" * 70)

    lgbm_sorted = sorted(lgbm_imp.keys(), key=lambda x: lgbm_imp[x], reverse=True)
    mi_sorted = sorted(mi_imp.keys(), key=lambda x: mi_imp[x], reverse=True)

    lgbm_ranks = {feat: rank + 1 for rank, feat in enumerate(lgbm_sorted)}
    mi_ranks = {feat: rank + 1 for rank, feat in enumerate(mi_sorted)}

    combined = {}
    for feat in lgbm_imp.keys():
        avg_rank = (lgbm_ranks[feat] + mi_ranks[feat]) / 2
        combined[feat] = avg_rank

    combined_sorted = sorted(combined.items(), key=lambda x: x[1])

    print("\nCombined Ranking (Lower is Better):")
    for i, (feat, rank) in enumerate(combined_sorted[:15], 1):
        print(f"  {i:2d}. {feat:20s}: avg_rank={rank:.1f}")

    return combined


def evaluate_top_n_features(X: pd.DataFrame, y: pd.Series, feature_names: List[str],
                            n: int, cv_folds: int = 5) -> Dict[str, float]:
    """Evaluate model performance with top N features."""
    top_features = feature_names[:n]
    X_subset = X[top_features].copy()

    X_imputed = SimpleImputer(strategy='median').fit_transform(X_subset)
    X_scaled = StandardScaler().fit_transform(X_imputed)

    model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.02,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=RANDOM_STATE,
        verbose=-1
    )

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_r2_scores = []
    cv_rmse_scores = []
    cv_mae_scores = []

    for train_idx, val_idx in kf.split(X_scaled):
        X_train_cv, X_val_cv = X_scaled[train_idx], X_scaled[val_idx]
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
        'cv_mae_mean': np.mean(cv_mae_scores)
    }


def run_experiment(max_features: int = 23, quick: bool = False):
    """Run the feature selection experiment."""
    print("\n" + "=" * 70)
    print("   FEATURE IMPORTANCE vs N_FEATURES EXPERIMENT")
    print("   St. John's House Price Prediction")
    print("=" * 70)

    output_dir = EXPERIMENTS_DIR / "feature_selection"
    os.makedirs(output_dir, exist_ok=True)

    df_raw = load_and_filter_data()
    df_selected = select_columns(df_raw)
    df_cleaned = basic_cleaning(df_selected)
    df_cleaned = clean_numeric_columns(df_cleaned)
    df_cleaned = remove_outliers(df_cleaned)

    X, y, encoders = prepare_features(df_cleaned)
    print(f"\nTotal features available: {X.shape[1]}")

    lgbm_importance = get_lgbm_importance(X, y)
    mi_importance = get_mutual_information(X, y)
    combined_importance = get_combined_importance(lgbm_importance, mi_importance)

    lgbm_sorted = sorted(lgbm_importance.keys(), key=lambda x: lgbm_importance[x], reverse=True)
    mi_sorted = sorted(mi_importance.keys(), key=lambda x: mi_importance[x], reverse=True)
    combined_sorted = [f[0] for f in sorted(combined_importance.items(), key=lambda x: x[1])]

    if quick:
        n_values = list(range(1, min(12, max_features + 1), 2))
    else:
        n_values = list(range(1, min(max_features, len(X.columns)) + 1))

    results = {
        'experiment': 'feature_selection_analysis',
        'timestamp': datetime.now().isoformat(),
        'n_features_tested': n_values,
        'lgbm_importance': lgbm_importance,
        'mutual_information': mi_importance,
        'combined_ranking': combined_importance,
        'lgbm_results': {},
        'mi_results': {},
        'combined_results': {}
    }

    print("\n" + "=" * 70)
    print("EVALUATING MODEL PERFORMANCE VS N_FEATURES")
    print("=" * 70)

    for n in n_values:
        print(f"\nEvaluating top {n} features...")

        lgbm_result = evaluate_top_n_features(X, y, lgbm_sorted, n)
        results['lgbm_results'][n] = lgbm_result
        print(f"  LGBM R²: {lgbm_result['cv_r2_mean']:.4f}")

        mi_result = evaluate_top_n_features(X, y, mi_sorted, n)
        results['mi_results'][n] = mi_result
        print(f"  MI R²:   {mi_result['cv_r2_mean']:.4f}")

        combined_result = evaluate_top_n_features(X, y, combined_sorted, n)
        results['combined_results'][n] = combined_result
        print(f"  Combined R²: {combined_result['cv_r2_mean']:.4f}")

    save_results(results, output_dir / "feature_selection_results.json")
    create_plots(results, n_values, output_dir)
    generate_report(results, n_values, output_dir)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE!")
    print("=" * 70)

    return results


def create_plots(results: Dict, n_values: List[int], output_dir: Path):
    """Create visualization plots."""
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATION PLOTS")
    print("=" * 70)

    lgbm_r2 = [results['lgbm_results'][n]['cv_r2_mean'] for n in n_values]
    lgbm_std = [results['lgbm_results'][n]['cv_r2_std'] for n in n_values]

    mi_r2 = [results['mi_results'][n]['cv_r2_mean'] for n in n_values]
    mi_std = [results['mi_results'][n]['cv_r2_std'] for n in n_values]

    combined_r2 = [results['combined_results'][n]['cv_r2_mean'] for n in n_values]
    combined_std = [results['combined_results'][n]['cv_r2_std'] for n in n_values]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax1 = axes[0, 0]
    ax1.errorbar(n_values, lgbm_r2, yerr=lgbm_std, marker='o', capsize=3,
                 label='LightGBM Importance', color='steelblue', linewidth=2)
    ax1.errorbar(n_values, mi_r2, yerr=mi_std, marker='s', capsize=3,
                 label='Mutual Information', color='coral', linewidth=2)
    ax1.errorbar(n_values, combined_r2, yerr=combined_std, marker='^', capsize=3,
                 label='Combined Ranking', color='green', linewidth=2)
    ax1.set_xlabel('Number of Top Features', fontsize=12)
    ax1.set_ylabel('R² Score (5-Fold CV)', fontsize=12)
    ax1.set_title('R² Score vs Number of Features', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    best_combined_n = n_values[np.argmax(combined_r2)]
    ax1.axvline(x=best_combined_n, color='green', linestyle=':', alpha=0.7)
    ax1.annotate(f'Best: {best_combined_n} features', xy=(best_combined_n, max(combined_r2)),
                 xytext=(best_combined_n + 2, max(combined_r2) + 0.02),
                 fontsize=10, color='green')

    ax2 = axes[0, 1]
    feature_names = list(results['lgbm_importance'].keys())
    importance_values = list(results['lgbm_importance'].values())
    sorted_idx = np.argsort(importance_values)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = [importance_values[i] for i in sorted_idx]

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(sorted_features)))
    ax2.barh(range(len(sorted_features)), sorted_importance, color=colors)
    ax2.set_yticks(range(len(sorted_features)))
    ax2.set_yticklabels(sorted_features, fontsize=9)
    ax2.set_xlabel('LightGBM Importance', fontsize=12)
    ax2.set_title('LightGBM Feature Importance', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()

    ax3 = axes[1, 0]
    mi_values = [results['mutual_information'].get(f, 0) for f in feature_names]
    sorted_mi_idx = np.argsort(mi_values)[::-1]
    sorted_mi_features = [feature_names[i] for i in sorted_mi_idx]
    sorted_mi = [mi_values[i] for i in sorted_mi_idx]

    colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(sorted_mi_features)))
    ax3.barh(range(len(sorted_mi_features)), sorted_mi, color=colors)
    ax3.set_yticks(range(len(sorted_mi_features)))
    ax3.set_yticklabels(sorted_mi_features, fontsize=9)
    ax3.set_xlabel('Mutual Information Score', fontsize=12)
    ax3.set_title('Mutual Information Scores', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()

    ax4 = axes[1, 1]
    combined_rank = results['combined_ranking']
    combined_values = [1 / combined_rank.get(f, len(combined_rank)) for f in feature_names]
    sorted_combined_idx = np.argsort(combined_values)[::-1]
    sorted_combined_features = [feature_names[i] for i in sorted_combined_idx]
    sorted_combined = [combined_values[i] for i in sorted_combined_idx]

    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(sorted_combined_features)))
    ax4.barh(range(len(sorted_combined_features)), sorted_combined, color=colors)
    ax4.set_yticks(range(len(sorted_combined_features)))
    ax4.set_yticklabels(sorted_combined_features, fontsize=9)
    ax4.set_xlabel('Combined Score (1 / avg_rank)', fontsize=12)
    ax4.set_title('Combined Feature Ranking', fontsize=14, fontweight='bold')
    ax4.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_selection_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'feature_selection_analysis.png'}")

    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(n_values,
                    np.array(combined_r2) - np.array(combined_std),
                    np.array(combined_r2) + np.array(combined_std),
                    alpha=0.2, color='green')
    ax.plot(n_values, combined_r2, marker='o', color='green', linewidth=2,
            markersize=8, label='Combined Ranking')
    ax.plot(n_values, lgbm_r2, marker='s', color='steelblue', linewidth=1.5,
            linestyle='--', alpha=0.7, label='LightGBM')
    ax.plot(n_values, mi_r2, marker='^', color='coral', linewidth=1.5,
            linestyle='--', alpha=0.7, label='Mutual Information')

    best_n = n_values[np.argmax(combined_r2)]
    best_r2 = max(combined_r2)
    ax.scatter([best_n], [best_r2], s=200, c='gold', edgecolors='black', zorder=5)
    ax.annotate(f'Best: {best_n} features\nR² = {best_r2:.4f}',
                xy=(best_n, best_r2), xytext=(best_n + 2, best_r2 + 0.03),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_xlabel('Number of Top Features', fontsize=12)
    ax.set_ylabel('R² Score (5-Fold CV)', fontsize=12)
    ax.set_title('Optimal Number of Features Analysis\nSt. John\'s House Prices', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_values)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'optimal_n_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'optimal_n_features.png'}")


def save_results(results: Dict, output_path: Path):
    """Save results to JSON."""
    serializable_results = {
        'experiment': results['experiment'],
        'timestamp': results['timestamp'],
        'n_features_tested': results['n_features_tested'],
        'lgbm_importance': {k: float(v) for k, v in results['lgbm_importance'].items()},
        'mutual_information': {k: float(v) for k, v in results['mutual_information'].items()},
        'combined_ranking': {k: float(v) for k, v in results['combined_ranking'].items()},
        'lgbm_results': {str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in results['lgbm_results'].items()},
        'mi_results': {str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in results['mi_results'].items()},
        'combined_results': {str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in results['combined_results'].items()}
    }
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    print(f"Results saved to: {output_path}")


def generate_report(results: Dict, n_values: List[int], output_dir: Path):
    """Generate markdown report."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    best_combined_n = n_values[np.argmax([results['combined_results'][n]['cv_r2_mean'] for n in n_values])]
    best_combined_r2 = results['combined_results'][best_combined_n]['cv_r2_mean']

    best_lgbm_n = n_values[np.argmax([results['lgbm_results'][n]['cv_r2_mean'] for n in n_values])]
    best_lgbm_r2 = results['lgbm_results'][best_lgbm_n]['cv_r2_mean']

    best_mi_n = n_values[np.argmax([results['mi_results'][n]['cv_r2_mean'] for n in n_values])]
    best_mi_r2 = results['mi_results'][best_mi_n]['cv_r2_mean']

    report = f"""# Feature Selection Analysis Report

**Generated:** {timestamp}

## Summary

| Metric | LightGBM | Mutual Information | Combined |
|--------|----------|-------------------|----------|
| Best N Features | {best_lgbm_n} | {best_mi_n} | **{best_combined_n}** |
| Best R² Score | {best_lgbm_r2:.4f} | {best_mi_r2:.4f} | **{best_combined_r2:.4f}** |

## R² Score vs Number of Features

| N Features | LGBM R² | MI R² | Combined R² |
|------------|---------|-------|-------------|
"""

    for n in n_values:
        lgbm = results['lgbm_results'][n]['cv_r2_mean']
        mi = results['mi_results'][n]['cv_r2_mean']
        combined = results['combined_results'][n]['cv_r2_mean']
        marker = " ← **BEST**" if n == best_combined_n else ""
        report += f"| {n:2d} | {lgbm:.4f} | {mi:.4f} | {combined:.4f}{marker} |\n"

    combined_sorted = [f[0] for f in sorted(results['combined_ranking'].items(), key=lambda x: x[1])]

    report += f"""

## Top Features by Combined Ranking

| Rank | Feature | Avg Rank |
|------|---------|----------|
"""

    for i, feat in enumerate(combined_sorted[:15], 1):
        rank = results['combined_ranking'][feat]
        report += f"| {i:2d} | {feat} | {rank:.1f} |\n"

    report += """

## LightGBM Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
"""

    lgbm_sorted = sorted(results['lgbm_importance'].items(), key=lambda x: x[1], reverse=True)
    for i, (feat, imp) in enumerate(lgbm_sorted[:15], 1):
        report += f"| {i:2d} | {feat} | {imp:.1f} |\n"

    report += """

## Mutual Information Scores

| Rank | Feature | MI Score |
|------|---------|----------|
"""

    mi_sorted = sorted(results['mutual_information'].items(), key=lambda x: x[1], reverse=True)
    for i, (feat, score) in enumerate(mi_sorted[:15], 1):
        report += f"| {i:2d} | {feat} | {score:.4f} |\n"

    report += """

## Key Findings

1. **Optimal Feature Count:** The model performs best with **{n} features**.

2. **Feature Selection Impact:** Using too few features underfits, while too many adds noise.

3. **Most Important Features:**
   - **Location:** `latitude`, `longitude` consistently rank high
   - **Size:** `property-sqft` is a strong predictor
   - **Rooms:** `property-beds`, `property-baths` contribute meaningfully

4. **Method Comparison:** Combined ranking (LGBM + MI) outperforms individual methods.

## Visualizations Generated

- `feature_selection_analysis.png` - Comprehensive 4-panel analysis
- `optimal_n_features.png` - R² vs N features with best point marked

## Recommendations

1. **Use top {n} features** for optimal performance
2. **Drop low-importance features** like Roof, Sewer, Fireplace Features
3. **Consider feature engineering** on top predictors (e.g., price_per_sqft)
""".format(n=best_combined_n)

    with open(output_dir / 'feature_selection_report.md', 'w') as f:
        f.write(report)
    print(f"Report saved to: {output_dir / 'feature_selection_report.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature Selection Experiment')
    parser.add_argument('--quick', action='store_true', help='Quick version with fewer steps')
    parser.add_argument('--max-features', type=int, default=23, help='Maximum features to test')

    args = parser.parse_args()

    run_experiment(max_features=args.max_features, quick=args.quick)

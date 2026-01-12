"""Main pipeline for NL House Price Prediction.

A comprehensive ML pipeline with outlier removal, feature engineering,
multiple models (including LightGBM), and visualizations.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader
from data_cleaner import DataCleaner
from feature_engineering import FeatureEngineer
from trainer import Trainer
from evaluator import Evaluator
from visualizer import Visualizer
from models import get_models, get_model_params
from config import DATA_PATH, OUTPUT_DIR, RANDOM_STATE

import warnings
warnings.filterwarnings('ignore')


def run_full_pipeline():
    """Execute the complete ML pipeline."""
    
    print("\n" + "=" * 70)
    print("   NL HOUSE PRICE PREDICTION - MODULAR ML PIPELINE   ")
    print("   WITH OUTLIER REMOVAL & LightGBM   ")
    print("=" * 70)
    
    # Step 1: Load Data
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
    loader.get_basic_info()
    loader.get_target_stats()
    
    # Step 2: Basic Cleaning
    print("\n" + "=" * 70)
    print("STEP 2: BASIC DATA CLEANING")
    print("=" * 70)
    
    cleaner = DataCleaner(df)
    df = cleaner.clean_basic()
    
    # Step 3: Clean Numeric Columns
    print("\n" + "=" * 70)
    print("STEP 3: CLEANING NUMERIC COLUMNS")
    print("=" * 70)
    
    numeric_cols = ['property-beds', 'property-baths', 'property-sqft', 'Year Built', 
                   'latitude', 'longitude', 'Garage Spaces', 'Parking Spaces']
    df = cleaner.clean_numeric_columns(numeric_cols)
    
    # Step 4: Remove Outliers
    print("\n" + "=" * 70)
    print("STEP 4: OUTLIER REMOVAL")
    print("=" * 70)
    
    df = cleaner.remove_price_outliers(lower_percentile=1, upper_percentile=99)
    
    outlier_cols = ['price', 'property-sqft', 'property-beds', 'property-baths']
    df = cleaner.remove_outliers_iqr(outlier_cols, multiplier=1.5)
    
    df = cleaner.remove_outliers_isolation_forest(
        ['price', 'property-sqft', 'property-beds', 'property-baths'],
        contamination=0.05
    )
    
    cleaning_report = cleaner.get_cleaning_report()
    
    # Step 5: Feature Engineering
    print("\n" + "=" * 70)
    print("STEP 5: FEATURE ENGINEERING")
    print("=" * 70)
    
    engineer = FeatureEngineer(df)
    df = engineer.create_price_features()
    df = engineer.create_property_features()
    df = engineer.create_age_features()
    df = engineer.create_location_features()
    df = engineer.create_property_type_features()
    feature_summary = engineer.get_feature_summary()
    
    # Step 6: Prepare Features for ML
    print("\n" + "=" * 70)
    print("STEP 6: FEATURE PREPARATION FOR ML")
    print("=" * 70)
    
    feature_df = engineer.select_features_for_ml()
    
    X = feature_df.drop('price', axis=1)
    y = feature_df['price']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Features: {list(X.columns)}")
    
    # Step 7: Train-Test Split
    print("\n" + "=" * 70)
    print("STEP 7: TRAIN-TEST SPLIT")
    print("=" * 70)
    
    trainer = Trainer(X, y, test_size=0.2, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = trainer.split_data()
    
    # Step 8: Scale Features
    print("\n" + "=" * 70)
    print("STEP 8: FEATURE SCALING")
    print("=" * 70)
    
    X_train_scaled, X_test_scaled = trainer.scale_features()
    X_train_imp, X_test_imp = trainer.impute_missing(X_train_scaled, X_test_scaled)
    
    # Step 9: Train Models
    print("\n" + "=" * 70)
    print("STEP 9: MODEL TRAINING")
    print("=" * 70)
    
    models = get_models()
    print(f"Training {len(models)} models...")
    
    results = trainer.train_all_models(X_train_imp, y_train, models)
    trainer.get_best_model()
    
    # Step 10: Hyperparameter Tuning for LightGBM
    print("\n" + "=" * 70)
    print("STEP 10: HYPERPARAMETER TUNING (LightGBM)")
    print("=" * 70)
    
    lgb_params = {
        'n_estimators': [100, 150],
        'max_depth': [5, 10],
        'learning_rate': [0.05, 0.1],
    }
    
    best_lgb, best_params = trainer.hyperparameter_tuning(
        X_train_imp, y_train, 'LightGBM', lgb_params
    )
    
    # Step 11: Evaluate Models
    print("\n" + "=" * 70)
    print("STEP 11: MODEL EVALUATION")
    print("=" * 70)
    
    evaluator = Evaluator(y_test, X_test_imp)
    
    for name, model in trainer.models.items():
        if hasattr(model, 'predict'):
            evaluator.add_model(name, model)
    
    evaluation_results = evaluator.evaluate_all()
    ranking = evaluator.get_ranking()
    
    # Step 12: Feature Importance
    print("\n" + "=" * 70)
    print("STEP 12: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    feature_names = list(X.columns)
    importance_results = evaluator.get_feature_importance(feature_names)
    
    # Step 13: Visualizations
    print("\n" + "=" * 70)
    print("STEP 13: VISUALIZATIONS")
    print("=" * 70)
    
    viz = Visualizer(OUTPUT_DIR)
    viz.create_eda_summary(df)
    viz.plot_price_distribution(df)
    viz.plot_price_vs_sqft(df)
    viz.plot_locality_prices(df)
    viz.plot_bedroom_prices(df)
    viz.plot_property_type_distribution(df)
    viz.plot_model_comparison(evaluation_results)
    
    if evaluator.best_model is not None:
        y_pred = evaluator.best_model.predict(X_test_imp)
        viz.plot_actual_vs_predicted(y_test, y_pred, evaluator.best_model_name)
        viz.plot_feature_importance(importance_results, feature_names)
    
    # Step 14: Save Results
    print("\n" + "=" * 70)
    print("STEP 14: SAVING RESULTS")
    print("=" * 70)
    
    results_path = f'{OUTPUT_DIR}/model_results.json'
    evaluator.feature_importance = importance_results
    evaluator.save_results(results_path)
    
    # Final Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\nDataset Statistics:")
    print(f"  - Original rows: {cleaning_report['original_shape'][0]}")
    print(f"  - Cleaned rows: {cleaning_report['final_shape'][0]}")
    print(f"  - Total outliers removed: {cleaning_report['rows_removed']}")
    
    print(f"\nData Split:")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Testing samples: {len(X_test)}")
    
    print(f"\nBest Model: {evaluator.best_model_name}")
    if evaluator.best_model_name in evaluation_results:
        best_metrics = evaluation_results[evaluator.best_model_name]
        print(f"  - R² Score: {best_metrics['R2']:.4f}")
        print(f"  - RMSE: ${best_metrics['RMSE']:,.2f}")
        print(f"  - MAE: ${best_metrics['MAE']:,.2f}")
    
    print(f"\nModels Trained: {len([k for k in results.keys() if 'error' not in results.get(k, {})])}")
    print(f"  Including: LightGBM, XGBoost, Random Forest, Gradient Boosting")
    
    print(f"\nOutput Files in {OUTPUT_DIR}:")
    import os
    for f in os.listdir(OUTPUT_DIR):
        print(f"  - {f}")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    return {
        'df': df,
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'models': trainer.models,
        'results': evaluation_results,
        'best_model': evaluator.best_model,
        'feature_importance': importance_results
    }


if __name__ == "__main__":
    results = run_full_pipeline()

"""Evaluator module for model performance evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)
from typing import Dict, Any, Tuple
import json
from datetime import datetime


class Evaluator:
    """Evaluate trained models on test data."""
    
    def __init__(self, y_test: pd.Series, X_test: np.ndarray = None):
        self.y_test = y_test
        self.X_test = X_test
        self.models = {}
        self.evaluation_results = {}
        self.best_model = None
        self.best_model_name = None
        
    def add_model(self, name: str, model: Any):
        """Add a trained model for evaluation."""
        self.models[name] = model
        
    def evaluate_all(self) -> Dict:
        """Evaluate all models on test data."""
        print("\n" + "=" * 60)
        print("STEP 7: MODEL EVALUATION")
        print("=" * 60)
        
        for name, model in self.models.items():
            try:
                y_pred = model.predict(self.X_test)
                
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                mape = mean_absolute_percentage_error(self.y_test, y_pred) * 100
                
                self.evaluation_results[name] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'MAPE': mape
                }
                
                print(f"\n{name}:")
                print(f"  RMSE: ${rmse:,.2f}")
                print(f"  MAE: ${mae:,.2f}")
                print(f"  R² Score: {r2:.4f}")
                print(f"  MAPE: {mape:.2f}%")
                
            except Exception as e:
                print(f"\n{name}: Error - {str(e)[:50]}")
                self.evaluation_results[name] = {'error': str(e)}
        
        self._find_best_model()
        return self.evaluation_results
    
    def _find_best_model(self):
        """Find the best performing model based on R² score."""
        valid_results = {k: v for k, v in self.evaluation_results.items() if 'error' not in v}
        
        if valid_results:
            best = max(valid_results.items(), key=lambda x: x[1]['R2'])
            self.best_model_name = best[0]
            self.best_model = self.models[best[0]]
            
            print("\n" + "=" * 60)
            print("BEST MODEL")
            print("=" * 60)
            print(f"Model: {self.best_model_name}")
            print(f"R² Score: {best[1]['R2']:.4f}")
            print(f"RMSE: ${best[1]['RMSE']:,.2f}")
            print(f"MAE: ${best[1]['MAE']:,.2f}")
    
    def get_ranking(self) -> pd.DataFrame:
        """Get model ranking by R² score."""
        valid_results = {k: v for k, v in self.evaluation_results.items() if 'error' not in v}
        
        ranking_df = pd.DataFrame(valid_results).T
        ranking_df = ranking_df.sort_values('R2', ascending=False)
        
        print("\n" + "=" * 60)
        print("MODEL RANKING BY R² SCORE")
        print("=" * 60)
        print(ranking_df)
        
        return ranking_df
    
    def get_feature_importance(self, feature_names: list) -> Dict:
        """Get feature importance from tree-based models."""
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        
        importance_results = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                importance_dict = dict(zip(feature_names, importance))
                importance_results[name] = importance_dict
                
                print(f"\n{name} Feature Importances:")
                sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                for feat, imp in sorted_imp[:10]:
                    print(f"  {feat:20s}: {imp:.4f}")
        
        self.feature_importance = importance_results
        return importance_results
    
    def save_results(self, output_path: str):
        """Save evaluation results to JSON file."""
        results = {
            'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'best_model': self.best_model_name,
            'best_r2': self.evaluation_results.get(self.best_model_name, {}).get('R2', 0),
            'results': self.evaluation_results,
            'feature_importance': self.feature_importance
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")
        return results


if __name__ == "__main__":
    from data_loader import DataLoader
    from data_cleaner import DataCleaner
    from feature_engineering import FeatureEngineer
    from trainer import Trainer
    from models import get_models
    from config import DATA_PATH
    
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
    
    cleaner = DataCleaner(df)
    df = cleaner.clean_basic()
    df = cleaner.clean_numeric_columns(['property-beds', 'property-baths', 'property-sqft', 'Year Built'])
    df = cleaner.remove_price_outliers()
    
    engineer = FeatureEngineer(df)
    df = engineer.run_pipeline()
    feature_df = engineer.select_features_for_ml()
    
    X = feature_df.drop('price', axis=1)
    y = feature_df['price']
    
    trainer = Trainer(X, y)
    X_train, X_test, y_train, y_test = trainer.split_data()
    X_train_scaled, X_test_scaled = trainer.scale_features()
    X_train_imp, X_test_imp = trainer.impute_missing(X_train_scaled, X_test_scaled)
    
    models = get_models()
    trainer.train_all_models(X_train_imp, y_train, models)
    best_name, best_model = trainer.get_best_model()
    
    evaluator = Evaluator(y_test, X_test_imp)
    evaluator.add_model(best_name, best_model)
    evaluator.evaluate_all()
    evaluator.get_ranking()

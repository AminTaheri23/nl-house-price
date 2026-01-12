"""Trainer module for model training with cross-validation."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class Trainer:
    """Train and evaluate ML models."""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        
        self.X_train = None
        self.X_test = self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.model_results = {}
        
    def split_data(self) -> Tuple:
        """Split data into train and test sets."""
        print("\n" + "=" * 60)
        print("STEP 4: TRAIN-TEST SPLIT")
        print("=" * 60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Training set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")
        print(f"Ratio: {1-self.test_size:.0%}/{self.test_size:.0%}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self) -> Tuple:
        """Scale numerical features."""
        print("\n" + "=" * 60)
        print("STEP 5: FEATURE SCALING")
        print("=" * 60)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        self.scaler = scaler
        
        print(f"Scaled training features: {X_train_scaled.shape}")
        print(f"Scaled testing features: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled
    
    def impute_missing(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple:
        """Impute missing values."""
        print("\n" + "=" * 60)
        print("MISSING VALUE IMPUTATION")
        print("=" * 60)
        
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        print(f"Imputed training features: {X_train_imputed.shape}")
        print(f"Missing values after imputation: {np.isnan(X_train_imputed).sum()}")
        
        self.imputer = imputer
        
        return X_train_imputed, X_test_imputed
    
    def train_all_models(self, X_train: np.ndarray, y_train: pd.Series, models: Dict[str, Any]) -> Dict:
        """Train all models and evaluate with cross-validation."""
        print("\n" + "=" * 60)
        print("STEP 6: MODEL TRAINING")
        print("=" * 60)
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                model.fit(X_train, y_train)
                self.models[name] = model
                
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                results[name] = {
                    'cv_r2_mean': cv_scores.mean(),
                    'cv_r2_std': cv_scores.std(),
                    'model': model
                }
                
                print(f"  CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)[:50]}")
                results[name] = {
                    'cv_r2_mean': 0,
                    'cv_r2_std': 0,
                    'error': str(e)
                }
        
        self.model_results = results
        
        print("\n" + "-" * 60)
        print("MODEL COMPARISON (CV R² Scores)")
        print("-" * 60)
        
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['cv_r2_mean'], reverse=True)
        
        for name, result in sorted_results:
            print(f"{name:25s}: {result['cv_r2_mean']:.4f} (+/- {result['cv_r2_std']:.4f})")
        
        return results
    
    def get_best_model(self) -> Tuple[str, Any]:
        """Get the best performing model."""
        valid_results = {k: v for k, v in self.model_results.items() if 'error' not in v}
        
        if not valid_results:
            return None, None
        
        best_name = max(valid_results.items(), key=lambda x: x[1]['cv_r2_mean'])[0]
        best_model = self.models[best_name]
        
        print(f"\nBest model: {best_name}")
        print(f"CV R² Score: {valid_results[best_name]['cv_r2_mean']:.4f}")
        
        return best_name, best_model
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: pd.Series, 
                              model_name: str, param_grid: Dict) -> Tuple:
        """Perform hyperparameter tuning using GridSearchCV."""
        print("\n" + "=" * 60)
        print(f"HYPERPARAMETER TUNING: {model_name}")
        print("=" * 60)
        
        from sklearn.ensemble import RandomForestRegressor
        import lightgbm as lgb
        import xgboost as xgb
        
        model_map = {
            'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
            'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }
        
        if model_name not in model_map:
            print(f"Model {model_name} not supported for tuning")
            return None, None
        
        model = model_map[model_name]
        
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=5, 
            scoring='r2', 
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV R² score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_


if __name__ == "__main__":
    from data_loader import DataLoader
    from data_cleaner import DataCleaner
    from feature_engineering import FeatureEngineer
    from models import get_models
    from config import DATA_PATH
    
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
    
    cleaner = DataCleaner(df)
    df = cleaner.clean_basic()
    df = cleaner.clean_numeric_columns(['property-beds', 'property-baths', 'property-sqft', 'Year Built'])
    df = cleaner.remove_price_outliers()
    
    engineer = FeatureEngineer(df)
    df_features = engineer.run_pipeline()
    feature_df = engineer.select_features_for_ml()
    
    X = feature_df.drop('price', axis=1)
    y = feature_df['price']
    
    trainer = Trainer(X, y)
    X_train, X_test, y_train, y_test = trainer.split_data()
    X_train_scaled, X_test_scaled = trainer.scale_features()
    X_train_imp, X_test_imp = trainer.impute_missing(X_train_scaled, X_test_scaled)
    
    models = get_models()
    results = trainer.train_all_models(X_train_imp, y_train, models)
    trainer.get_best_model()

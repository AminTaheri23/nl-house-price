"""Models module with multiple ML algorithms including LightGBM."""

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb
from typing import Dict, Any


def get_models() -> Dict[str, Any]:
    """Get dictionary of all models."""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, 
            max_depth=15, 
            random_state=42, 
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, 
            max_depth=5, 
            random_state=42
        ),
        'K-Neighbors': KNeighborsRegressor(n_neighbors=5),
        'SVR': SVR(kernel='rbf'),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            random_state=42,
            verbosity=0
        ),
        'AdaBoost': AdaBoostRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
    }
    return models


def get_model_params() -> Dict[str, Dict]:
    """Get default hyperparameters for tuning."""
    return {
        'Linear Regression': {},
        'Ridge Regression': {'alpha': [0.1, 1.0, 10.0]},
        'Lasso Regression': {'alpha': [0.1, 1.0, 10.0]},
        'ElasticNet': {'alpha': [0.1, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]},
        'Decision Tree': {'max_depth': [5, 10, 15, 20]},
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1]
        },
        'K-Neighbors': {'n_neighbors': [3, 5, 7, 10]},
        'LightGBM': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 100]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [5, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1]
        },
        'AdaBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.5]
        }
    }


def get_tuned_model(model_name: str, params: Dict = None) -> Any:
    """Get a model with specific hyperparameters."""
    if params is None:
        params = {}
    
    base_models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=params.get('n_estimators', 200),
            max_depth=params.get('max_depth', 20),
            min_samples_split=params.get('min_samples_split', 5),
            min_samples_leaf=params.get('min_samples_leaf', 2),
            random_state=42,
            n_jobs=-1
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=params.get('n_estimators', 200),
            max_depth=params.get('max_depth', 10),
            learning_rate=params.get('learning_rate', 0.05),
            num_leaves=params.get('num_leaves', 31),
            random_state=42,
            verbose=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=params.get('n_estimators', 200),
            max_depth=params.get('max_depth', 8),
            learning_rate=params.get('learning_rate', 0.05),
            random_state=42,
            verbosity=0
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 5),
            learning_rate=params.get('learning_rate', 0.1),
            random_state=42
        )
    }
    
    return base_models.get(model_name)


def print_model_summary():
    """Print summary of available models."""
    models = get_models()
    print("\n" + "=" * 60)
    print("AVAILABLE MODELS")
    print("=" * 60)
    
    for i, (name, model) in enumerate(models.items(), 1):
        print(f"{i:2d}. {name}")
    
    print(f"\nTotal models: {len(models)}")
    print("Tree-based models (handle NaN): Random Forest, Gradient Boosting")
    print("Gradient boosting libraries: LightGBM, XGBoost")

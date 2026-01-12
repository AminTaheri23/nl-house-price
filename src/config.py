"""Configuration settings for NL House Price Prediction project."""

import os
from datetime import datetime

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data_nl.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ML Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Model hyperparameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'lightgbm': {
        'n_estimators': 200,
        'max_depth': 10,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': RANDOM_STATE,
        'verbose': -1
    },
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.05,
        'random_state': RANDOM_STATE,
        'verbosity': 0
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'random_state': RANDOM_STATE
    }
}

# Feature columns
KEY_FEATURES = [
    'latitude',
    'longitude',
    'property-beds',
    'property-baths',
    'property-sqft',
    'Year Built'
]

DERIVED_FEATURES = [
    'price_per_sqft',
    'house_age',
    'total_rooms',
    'log_sqft'
]

# Outlier detection
OUTLIER_METHODS = ['iqr', 'zscore', 'isolation_forest']

# Visualization settings
FIG_SIZE = (14, 10)
DPI = 150

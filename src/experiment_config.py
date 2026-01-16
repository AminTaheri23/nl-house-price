"""Experiment configuration for St. John's area analysis."""

import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "data_nl.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

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
    'description',
    'priceCurrency',
    'MLS® #',
    'MLS',
    'streetAddress',
    'addressRegion',
    'postalCode'
]

FINAL_COLUMNS = [
    'addressLocality', 'latitude', 'longitude', 'price',
    'property-beds', 'property-baths', 'property-sqft', 'Basement', 'Bath',
    'Exterior', 'Exterior Features', 'Features', 'Fireplace', 'Heating',
    'Partial Bathroom', 'Property Tax', 'Property Type', 'Roof',
    'Sewer', 'Square Footage', 'Type', 'Parking', 'Flooring',
    'Parking Features', 'Fireplace Features'
]

NUMERIC_FEATURES = [
    'latitude', 'longitude', 'property-beds', 'property-baths',
    'property-sqft', 'Property Tax', 'Square Footage'
]

CATEGORICAL_FEATURES = [
    'addressLocality', 'Basement', 'Bath', 'Exterior', 'Exterior Features',
    'Features', 'Fireplace', 'Heating', 'Partial Bathroom', 'Property Type',
    'Roof', 'Sewer', 'Type', 'Parking', 'Flooring', 'Parking Features',
    'Fireplace Features'
]

MODEL_PARAMS = {
    'Random Forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'LightGBM': {
        'n_estimators': 200,
        'max_depth': 10,
        'learning_rate': 0.05,
        'random_state': RANDOM_STATE,
        'verbose': -1
    },
    'XGBoost': {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.05,
        'random_state': RANDOM_STATE,
        'verbosity': 0
    }
}

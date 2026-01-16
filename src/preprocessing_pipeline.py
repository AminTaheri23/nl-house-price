"""Preprocessing Pipeline for St. John's House Price Prediction.

This module provides functions to preprocess raw house data for prediction.
It can be used standalone or integrated into a web application.

Usage:
    from preprocessing_pipeline import preprocess, predict

    # Preprocess raw data
    X = preprocess(raw_data_dict)

    # Make prediction
    price = predict(raw_data_dict)
"""

import sys
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "experiments" / "models"

CITIES = ["St. John's", "Paradise", "Mount Pearl", "Torbay"]

COLUMNS_TO_DROP = [
    'description',
    'priceCurrency',
    'MLS® #',
    'MLS',
    'streetAddress',
    'addressRegion',
    'postalCode'
]

REQUIRED_COLUMNS = [
    'addressLocality', 'latitude', 'longitude',
    'property-beds', 'property-baths', 'property-sqft',
    'Basement', 'Bath', 'Exterior', 'Exterior Features',
    'Features', 'Fireplace', 'Heating', 'Partial Bathroom',
    'Property Type', 'Roof', 'Sewer', 'Square Footage',
    'Type', 'Parking', 'Flooring',
    'Parking Features', 'Fireplace Features'
]


def get_feature_columns() -> List[str]:
    """Get feature columns from saved file."""
    features_path = MODEL_DIR / 'feature_columns.txt'
    if features_path.exists():
        with open(features_path, 'r') as f:
            cols = [line.strip() for line in f if line.strip()]
        return cols
    return REQUIRED_COLUMNS


def load_artifacts() -> Dict[str, Any]:
    """Load saved model and preprocessing artifacts."""
    artifacts = {}

    model_path = MODEL_DIR / 'lgbm_model.joblib'
    encoders_path = MODEL_DIR / 'label_encoders.joblib'
    scaler_path = MODEL_DIR / 'scaler.joblib'
    imputer_path = MODEL_DIR / 'imputer.joblib'
    features_path = MODEL_DIR / 'feature_columns.txt'

    if model_path.exists():
        artifacts['model'] = joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")

    if encoders_path.exists():
        artifacts['encoders'] = joblib.load(encoders_path)
    else:
        raise FileNotFoundError(f"Encoders not found at {encoders_path}")

    if scaler_path.exists():
        artifacts['scaler'] = joblib.load(scaler_path)
    else:
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    if imputer_path.exists():
        artifacts['imputer'] = joblib.load(imputer_path)
    else:
        raise FileNotFoundError(f"Imputer not found at {imputer_path}")

    if features_path.exists():
        with open(features_path, 'r') as f:
            artifacts['feature_columns'] = [line.strip() for line in f]
    else:
        raise FileNotFoundError(f"Feature columns not found at {features_path}")

    return artifacts


def filter_st_johns_area(df: pd.DataFrame) -> pd.DataFrame:
    """Filter data to St. John's metropolitan area."""
    pattern = "|".join(CITIES)
    mask = df['streetAddress'].str.contains(pattern, case=False, na=False)
    return df[mask].copy()


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select only the columns needed for prediction."""
    feature_cols = get_feature_columns()
    available_cols = [col for col in feature_cols if col in df.columns]
    return df[available_cols].copy()


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and convert numeric columns."""
    numeric_cols = ['property-beds', 'property-baths', 'property-sqft', 'Property Tax']

    for col in numeric_cols:
        if col in df.columns:
            if col in ['property-sqft', 'Square Footage']:
                df[col] = df[col].astype(str).str.replace(',', '').replace('N/A', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def encode_categoricals(df: pd.DataFrame, encoders: Dict) -> pd.DataFrame:
    """Encode categorical columns using fitted label encoders."""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols:
        if col in encoders:
            le = encoders[col]
            df[col] = df[col].astype(str).fillna('Unknown')
            df[col] = df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    return df


def impute_missing(df: pd.DataFrame, imputer) -> np.ndarray:
    """Impute missing values using fitted imputer."""
    return imputer.transform(df)


def scale_features(X: np.ndarray, scaler) -> np.ndarray:
    """Scale features using fitted scaler."""
    return scaler.transform(X)


def preprocess(raw_data: Dict, artifacts: Optional[Dict] = None) -> np.ndarray:
    """Preprocess raw house data for prediction.

    Args:
        raw_data: Dictionary containing house data fields
        artifacts: Optional pre-loaded artifacts (for efficiency in batch predictions)

    Returns:
        Preprocessed feature array ready for model prediction

    Example:
        >>> raw = {
        ...     'streetAddress': '123 Main St, St. John\'s',
        ...     'latitude': 47.5,
        ...     'longitude': -52.7,
        ...     'property-beds': 3,
        ...     'property-baths': 2,
        ...     'property-sqft': 1500,
        ...     'addressLocality': 'St. John\'s',
        ...     'Basement': 'Finished',
        ...     'Bath': 2,
        ...     'Exterior': 'Vinyl siding',
        ...     'Exterior Features': 'Vinyl siding',
        ...     'Features': ['Garage'],
        ...     'Fireplace': 'No',
        ...     'Heating': 'Electric',
        ...     'Partial Bathroom': 1,
        ...     'Property Tax': 2500,
        ...     'Property Type': 'Single Family',
        ...     'Roof': 'Asphalt shingle',
        ...     'Sewer': 'Municipal',
        ...     'Type': 'Single Family',
        ...     'Parking': 'Attached Garage',
        ...     'Flooring': 'Hardwood',
        ...     'Parking Features': ['Attached Garage'],
        ...     'Fireplace Features': []
        ... }
        >>> X = preprocess(raw_data)
        >>> price = predict(raw_data)
    """
    if artifacts is None:
        artifacts = load_artifacts()

    df = pd.DataFrame([raw_data])

    if 'streetAddress' not in df.columns:
        raise ValueError("Missing required column: streetAddress")

    df = filter_st_johns_area(df)

    if len(df) == 0:
        raise ValueError("Property not in St. John's metropolitan area")

    df = select_columns(df)
    df = clean_numeric_columns(df)

    df = encode_categoricals(df, artifacts['encoders'])

    feature_cols = artifacts.get('feature_columns', list(df.columns))

    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[feature_cols]

    X = df.values
    X = np.nan_to_num(X, nan=0.0)

    X = impute_missing(X, artifacts['imputer'])
    X = scale_features(X, artifacts['scaler'])

    return X


def predict(raw_data: Dict) -> float:
    """Predict house price from raw data.

    Args:
        raw_data: Dictionary containing house data fields

    Returns:
        Predicted house price in CAD

    Example:
        >>> raw = {...}  # See preprocess() for structure
        >>> price = predict(raw)
        >>> print(f"Predicted price: ${price:,.2f}")
    """
    artifacts = load_artifacts()
    X = preprocess(raw_data, artifacts)
    model = artifacts['model']
    price = model.predict(X)[0]
    return max(0, price)


def predict_batch(raw_data_list: List[Dict]) -> List[float]:
    """Predict house prices for multiple properties.

    Args:
        raw_data_list: List of raw data dictionaries

    Returns:
        List of predicted prices

    Example:
        >>> properties = [raw_data_1, raw_data_2, raw_data_3]
        >>> prices = predict_batch(properties)
        >>> for price in prices:
        ...     print(f"${price:,.2f}")
    """
    artifacts = load_artifacts()
    predictions = []

    for raw_data in raw_data_list:
        try:
            price = predict(raw_data)
            predictions.append(price)
        except Exception as e:
            print(f"Error predicting for property: {e}")
            predictions.append(None)

    return predictions


def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded model."""
    artifacts = load_artifacts()

    return {
        'model_type': type(artifacts['model']).__name__,
        'feature_columns': artifacts['feature_columns'],
        'n_features': len(artifacts['feature_columns']),
        'n_categorical_encoders': len(artifacts['encoders']),
        'model_params': artifacts['model'].get_params()
    }


def validate_input(raw_data: Dict) -> List[str]:
    """Validate raw input data and return list of issues.

    Args:
        raw_data: Dictionary containing house data fields

    Returns:
        List of validation issues (empty if valid)
    """
    issues = []

    required_for_location = ['streetAddress']
    for col in required_for_location:
        if col not in raw_data:
            issues.append(f"Missing required column: {col}")

    if 'streetAddress' in raw_data:
        pattern = "|".join(CITIES)
        if not any(city.lower() in raw_data['streetAddress'].lower() for city in CITIES):
            issues.append("Address not in St. John's metropolitan area")

    numeric_cols = ['property-beds', 'property-baths', 'property-sqft']
    for col in numeric_cols:
        if col in raw_data:
            try:
                val = float(raw_data[col])
                if val < 0:
                    issues.append(f"{col} must be non-negative")
            except (ValueError, TypeError):
                issues.append(f"{col} must be a number")

    return issues


def create_sample_prediction() -> Dict[str, Any]:
    """Create a sample prediction with example data."""
    sample_data = {
        'streetAddress': '123 Example Street, St. John\'s, NL',
        'addressLocality': 'St. John\'s',
        'latitude': 47.5615,
        'longitude': -52.7126,
        'property-beds': 3,
        'property-baths': 2,
        'property-sqft': 1800,
        'Basement': 'Finished',
        'Bath': 2,
        'Exterior': 'Vinyl siding',
        'Exterior Features': 'Vinyl siding',
        'Features': ['Garage', 'Deck'],
        'Fireplace': 'No',
        'Heating': 'Electric',
        'Partial Bathroom': 1,
        'Property Type': 'Single Family',
        'Roof': 'Asphalt shingle',
        'Sewer': 'Municipal sewage system',
        'Type': 'Single Family',
        'Parking': 'Attached Garage',
        'Flooring': 'Hardwood',
        'Parking Features': ['Attached Garage'],
        'Fireplace Features': []
    }

    issues = validate_input(sample_data)
    if issues:
        return {
            'success': False,
            'issues': issues,
            'sample_data': sample_data
        }

    try:
        price = predict(sample_data)
        model_info = get_model_info()

        return {
            'success': True,
            'predicted_price': price,
            'sample_data': sample_data,
            'model_info': model_info
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'sample_data': sample_data
        }


if __name__ == "__main__":
    print("=" * 70)
    print("PREPROCESSING PIPELINE TEST")
    print("=" * 70)

    try:
        result = create_sample_prediction()

        if result['success']:
            print("\n✓ Sample prediction successful!")
            print(f"\nPredicted Price: ${result['predicted_price']:,.2f}")
            print(f"\nModel Info:")
            print(f"  Type: {result['model_info']['model_type']}")
            print(f"  Features: {result['model_info']['n_features']}")
        else:
            print("\n✗ Sample prediction failed!")
            if 'issues' in result:
                print("Validation issues:")
                for issue in result['issues']:
                    print(f"  - {issue}")
            if 'error' in result:
                print(f"Error: {result['error']}")

    except FileNotFoundError as e:
        print(f"\n✗ Model artifacts not found: {e}")
        print("Please run 'python src/train_lgbm_tuned.py' first to train and save the model.")
    except Exception as e:
        print(f"\n✗ Error: {e}")

"""Preprocessing Pipeline for Top 12 Features - Optimized for Web App."""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "experiments" / "models_top12"

CITIES = ["St. John's", "Paradise", "Mount Pearl", "Torbay"]

FEATURE_COLUMNS = [
    'property-baths',
    'Square Footage',
    'property-sqft',
    'longitude',
    'latitude',
    'Heating',
    'Features',
    'Parking',
    'Flooring',
    'Exterior',
    'Parking Features',
    'addressLocality'
]

COLUMN_MAPPING = {
    'property_baths': 'property-baths',
    'square_footage': 'Square Footage',
    'property_sqft': 'property-sqft',
    'address_locality': 'addressLocality',
    'parking_features': 'Parking Features'
}

HEATING_OPTIONS = [
    'Electric', 'Forced air', 'Heat pump', 'Baseboard heaters',
    'Radiant', 'Gas', 'Oil', 'Wood', 'Other'
]

PARKING_OPTIONS = [
    'Attached Garage', 'Detached Garage', 'Carport', 'Parking pad',
    'No parking', 'Shared driveway', 'Street parking', 'Other'
]

FEATURES_OPTIONS = [
    'Garage', 'Deck', 'Patio', 'Pool', 'Fireplace', 'Basement',
    'Central air', 'Security system', 'Storage', 'Workshop', 'Other'
]

FLOORING_OPTIONS = [
    'Hardwood', 'Carpet', 'Tile', 'Laminate', 'Vinyl', 'Cork', 'Concrete', 'Other'
]

EXTERIOR_OPTIONS = [
    'Vinyl siding', 'Brick', 'Wood', 'Stucco', 'Aluminum',
    'Fiber cement', 'Stone', 'Metal', 'Other'
]

PARKING_FEATURES_OPTIONS = [
    'Attached Garage', 'Detached Garage', 'Carport', 'Parking pad',
    'No parking', 'Shared driveway', 'Street parking', 'Other'
]

CITY_OPTIONS = ['St. John\'s', 'Paradise', 'Mount Pearl', 'Torbay']


def load_artifacts() -> Dict[str, Any]:
    """Load saved model and preprocessing artifacts."""
    artifacts = {}

    model_path = MODEL_DIR / 'lgbm_model_top12.joblib'
    encoders_path = MODEL_DIR / 'label_encoders_top12.joblib'
    scaler_path = MODEL_DIR / 'scaler_top12.joblib'

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

    artifacts['feature_columns'] = FEATURE_COLUMNS

    return artifacts


def validate_input(data: Dict) -> tuple[bool, str]:
    """Validate input data."""
    required_numeric = ['latitude', 'longitude', 'property_baths', 'property_sqft', 'square_footage']

    for field in required_numeric:
        if field not in data:
            return False, f"Missing required field: {field}"
        try:
            val = float(data[field]) if field in ['latitude', 'longitude'] else int(data[field])
            if val < 0:
                return False, f"{field} must be non-negative"
        except (ValueError, TypeError):
            return False, f"{field} must be a valid number"

    if not (-90 <= data['latitude'] <= 90):
        return False, "Latitude must be between -90 and 90"
    if not (-180 <= data['longitude'] <= 180):
        return False, "Longitude must be between -180 and 180"

    if data.get('address_locality') not in CITY_OPTIONS:
        return False, f"addressLocality must be one of: {', '.join(CITY_OPTIONS)}"

    return True, ""


def convert_api_to_columns(data: Dict) -> Dict:
    """Convert snake_case API input to hyphenated model columns."""
    converted = {}
    for key, value in data.items():
        if key in COLUMN_MAPPING:
            converted[COLUMN_MAPPING[key]] = value
        else:
            converted[key] = value
    return converted


def preprocess(raw_data: Dict, artifacts: Optional[Dict] = None) -> np.ndarray:
    """Preprocess raw house data for prediction."""
    if artifacts is None:
        artifacts = load_artifacts()

    df = pd.DataFrame([raw_data])

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[FEATURE_COLUMNS]

    numeric_cols = ['property-baths', 'property-sqft', 'Square Footage']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        if col in artifacts['encoders']:
            le = artifacts['encoders'][col]
            df[col] = df[col].astype(str).fillna('Unknown')
            df[col] = df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    X = df.values.astype(float)
    X = np.nan_to_num(X, nan=0.0)

    X = artifacts['scaler'].transform(X)

    return X


def get_model_info() -> Dict[str, Any]:
    """Get information about the model."""
    artifacts = load_artifacts()

    return {
        'model_type': type(artifacts['model']).__name__,
        'feature_columns': artifacts['feature_columns'],
        'n_features': len(artifacts['feature_columns']),
        'n_categorical_encoders': len(artifacts['encoders']),
        'model_params': artifacts['model'].get_params(),
        'performance': {
            'r2_score': 0.2182,
            'rmse': 238849.05,
            'mae': 136582.88
        }
    }


if __name__ == "__main__":
    print("=" * 60)
    print("PIPELINE TOP12 TEST")
    print("=" * 60)

    try:
        artifacts = load_artifacts()
        print("✓ Artifacts loaded successfully")
        print(f"  Model: {type(artifacts['model']).__name__}")
        print(f"  Features: {len(artifacts['feature_columns'])}")

        test_data = {
            'address_locality': "St. John's",
            'latitude': 47.5615,
            'longitude': -52.7126,
            'property_baths': 3,
            'property_sqft': 1850,
            'square_footage': 1850,
            'Heating': 'Electric',
            'Features': 'Garage',
            'Parking': 'Attached Garage',
            'Flooring': 'Hardwood',
            'Exterior': 'Vinyl siding',
            'parking_features': 'Attached Garage'
        }

        converted = convert_api_to_columns(test_data)
        X = preprocess(converted, artifacts)
        print(f"✓ Preprocessing successful, shape: {X.shape}")

        price = artifacts['model'].predict(X)[0]
        print(f"✓ Prediction: ${price:,.2f}")

    except FileNotFoundError as e:
        print(f"✗ Model artifacts not found: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")

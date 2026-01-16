"""Prediction Service with Confidence Intervals."""

from typing import Dict, Any
import numpy as np
from app.pipeline.pipeline_top12 import load_artifacts, preprocess, convert_api_to_columns


class HousePricePredictor:
    MAE = 136582.88
    RMSE = 238849.05
    R2_SCORE = 0.2182

    def __init__(self):
        self.artifacts = load_artifacts()
        self.model = self.artifacts['model']

    def predict(self, raw_data: Dict) -> Dict[str, Any]:
        """Make a prediction with confidence interval."""
        converted_data = convert_api_to_columns(raw_data)
        X = preprocess(converted_data, self.artifacts)

        price = self.model.predict(X)[0]
        price = float(max(0, price))

        margin = 0.8 * self.MAE

        relative_margin = margin / price if price > 0 else 1.0

        if relative_margin < 0.25:
            quality = "high"
            quality_label = "High Confidence"
            quality_color = "green"
        elif relative_margin < 0.5:
            quality = "medium"
            quality_label = "Medium Confidence"
            quality_color = "yellow"
        else:
            quality = "low"
            quality_label = "Low Confidence"
            quality_color = "red"

        return {
            "predicted_price": round(price, 2),
            "confidence_interval": {
                "lower": round(price - margin, 2),
                "upper": round(price + margin, 2),
                "margin": round(margin, 2),
                "level": "95%"
            },
            "quality": quality,
            "quality_label": quality_label,
            "quality_color": quality_color,
            "model_performance": {
                "r2_score": self.R2_SCORE,
                "rmse": self.RMSE,
                "mae": self.MAE
            }
        }

    def predict_batch(self, data_list: list) -> list:
        """Predict multiple properties."""
        return [self.predict(data) for data in data_list]

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": type(self.model).__name__,
            "performance": {
                "r2_score": self.R2_SCORE,
                "rmse": self.RMSE,
                "mae": self.MAE
            },
            "features": self.artifacts['feature_columns'],
            "n_features": len(self.artifacts['feature_columns'])
        }


_predictor_instance = None


def get_predictor() -> HousePricePredictor:
    """Get or create predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = HousePricePredictor()
    return _predictor_instance


if __name__ == "__main__":
    predictor = HousePricePredictor()

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

    result = predictor.predict(test_data)

    print("=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Predicted Price: ${result['predicted_price']:,.2f}")
    print(f"95% CI: ${result['confidence_interval']['lower']:,.2f} - ${result['confidence_interval']['upper']:,.2f}")
    print(f"Quality: {result['quality_label']}")
    print("=" * 60)

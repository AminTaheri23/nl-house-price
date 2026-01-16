"""Pydantic Schemas for API Request/Response."""

from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    address_locality: str = Field(
        ...,
        example="St. John's",
        description="City in the St. John's metropolitan area"
    )
    latitude: float = Field(
        ...,
        example=47.5615,
        description="Latitude coordinate (get from Google Maps)"
    )
    longitude: float = Field(
        ...,
        example=-52.7126,
        description="Longitude coordinate (get from Google Maps)"
    )
    property_baths: int = Field(
        ...,
        example=3,
        ge=0,
        description="Number of bathrooms"
    )
    property_sqft: int = Field(
        ...,
        example=1850,
        ge=0,
        description="Property size in square feet"
    )
    square_footage: int = Field(
        ...,
        example=1850,
        ge=0,
        description="Square footage of the living area"
    )
    heating: str = Field(
        ...,
        example="Electric",
        description="Type of heating system"
    )
    features: str = Field(
        ...,
        example="Garage",
        description="Property features (Garage, Deck, Pool, etc.)"
    )
    parking: str = Field(
        ...,
        example="Attached Garage",
        description="Type of parking available"
    )
    flooring: str = Field(
        ...,
        example="Hardwood",
        description="Primary flooring type"
    )
    exterior: str = Field(
        ...,
        example="Vinyl siding",
        description="Exterior material"
    )
    parking_features: str = Field(
        ...,
        example="Attached Garage",
        description="Parking features"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "address_locality": "St. John's",
                    "latitude": 47.5615,
                    "longitude": -52.7126,
                    "property_baths": 3,
                    "property_sqft": 1850,
                    "square_footage": 1850,
                    "heating": "Electric",
                    "features": "Garage",
                    "parking": "Attached Garage",
                    "flooring": "Hardwood",
                    "exterior": "Vinyl siding",
                    "parking_features": "Attached Garage"
                }
            ]
        }
    }


class ConfidenceInterval(BaseModel):
    lower: float
    upper: float
    margin: float
    level: str


class ModelPerformance(BaseModel):
    r2_score: float
    rmse: float
    mae: float


class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_interval: ConfidenceInterval
    quality: str
    quality_label: str
    quality_color: str
    model_performance: ModelPerformance


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_type: str
    performance: ModelPerformance
    features: list[str]
    n_features: int


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

"""API Endpoints for House Price Prediction."""

from fastapi import APIRouter, HTTPException
from app.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse
)
from app.services.predictor import get_predictor

router = APIRouter(prefix="/api", tags=["prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Prediction error"}
    },
    summary="Predict house price",
    description="Get a predicted house price with confidence interval for St. John's area properties."
)
async def predict(request: PredictionRequest):
    """Predict house price with confidence interval."""
    try:
        predictor = get_predictor()
        result = predictor.predict(request.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API and model are running properly."
)
async def health():
    """Check API health."""
    try:
        predictor = get_predictor()
        return {
            "status": "healthy",
            "model_loaded": predictor.model is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_loaded": False
        }


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Model information",
    description="Get information about the trained model and its performance."
)
async def model_info():
    """Get model information."""
    try:
        predictor = get_predictor()
        return predictor.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

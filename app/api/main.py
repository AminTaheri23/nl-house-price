"""FastAPI Application for NL House Price Prediction."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router
from app.services.predictor import get_predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor = get_predictor()
    yield


app = FastAPI(
    title="NL House Price Predictor API",
    description="""
    ## St. John's Metropolitan Area House Price Prediction

    This API predicts house prices in the St. John's metropolitan area (St. John's, Paradise, Mount Pearl, Torbay) 
    using a LightGBM machine learning model trained on local real estate data.

    ### Features
    - **Prediction with Confidence Intervals**: Get price estimates with 95% confidence ranges
    - **Quality Indicators**: High/Medium/Low confidence based on prediction uncertainty
    - **Model Performance Metrics**: R², RMSE, and MAE scores

    ### Model Information
    - **Algorithm**: LightGBM
    - **Features Used**: 12 (location, size, and property features)
    - **R² Score**: 0.2182
    - **RMSE**: $238,849
    - **MAE**: $136,583

    ### Usage
    Send a POST request to `/api/predict` with the required property features.
    """,
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://nl-house-price-frontend.onrender.com",
        "https://nl-house-price-1.onrender.com",
        "https://*.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/", tags=["root"])
async def root():
    return {
        "message": "NL House Price Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "predict": "/api/predict"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

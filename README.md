# NL House Price Predictor

A full-stack web application for predicting house prices in the St. John's metropolitan area (St. John's, Paradise, Mount Pearl, Torbay) using a LightGBM machine learning model.

## Features

- **House Price Prediction**: Get estimated prices for properties in NL
- **Confidence Intervals**: 95% confidence range with quality indicators
- **12 Key Features**: Location, size, and property characteristics
- **Responsive Design**: Works on desktop and mobile

## Tech Stack

- **Backend**: FastAPI + Uvicorn + LightGBM
- **Frontend**: React + Vite + TailwindCSS
- **Deployment**: Render

## Quick Start

### Local Development

1. **Backend**:
```bash
cd app
pip install -r requirements.txt
uvicorn api.main:app --reload
```

2. **Frontend**:
```bash
cd frontend
npm install
npm run dev
```

3. Open http://localhost:5173

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Get prediction with CI |
| GET | `/api/health` | Health check |
| GET | `/api/info` | Model information |

### Example Request

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Example Response

```json
{
  "predicted_price": 385000.00,
  "confidence_interval": {
    "lower": 275734.12,
    "upper": 494265.88,
    "margin": 109265.88,
    "level": "95%"
  },
  "quality": "medium",
  "quality_label": "Medium Confidence",
  "quality_color": "yellow",
  "model_performance": {
    "r2_score": 0.2182,
    "rmse": 238849.05,
    "mae": 136582.88
  }
}
```

## Model Performance

| Metric | Value |
|--------|-------|
| R² Score | 0.2182 |
| RMSE | $238,849 |
| MAE | $136,583 |

## Deployment to Render

### Option 1: Single Repo (API + Frontend)

1. Push to GitHub
2. Create Render web service for API:
   - Build command: `pip install -r app/requirements.txt`
   - Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
3. Create Render static site for frontend:
   - Build command: `cd frontend && npm install && npm run build`
   - Publish path: `frontend/dist`

### Option 2: Separate Services

Deploy API and frontend as separate services for better scaling.

## Project Structure

```
nl-house-price/
├── app/
│   ├── api/
│   │   ├── main.py           # FastAPI app
│   │   ├── endpoints.py      # /predict, /health, /info
│   │   └── schemas.py        # Pydantic models
│   ├── pipeline/
│   │   └── pipeline_top12.py # Preprocessing
│   ├── services/
│   │   └── predictor.py      # Prediction logic
│   ├── requirements.txt
│   └── runtime.txt
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── services/         # API calls
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
├── experiments/
│   └── models_top12/         # Trained model artifacts
└── render.yaml               # Render deployment config
```

## Getting Coordinates from Google Maps

1. Go to maps.google.com
2. Search for the property address
3. Right-click on the location
4. The first item in the menu shows the coordinates
5. Click to copy (format: `47.5615, -52.7126`)

## Confidence Interval

The confidence interval is based on the model's MAE (Mean Absolute Error):
- **High Confidence**: ±25% of predicted price
- **Medium Confidence**: ±25-50% of predicted price
- **Low Confidence**: >50% of predicted price

## License

MIT

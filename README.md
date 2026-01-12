# NL House Price Prediction

A comprehensive machine learning pipeline for predicting house prices in Newfoundland and Labrador, Canada.

## Project Overview

This project analyzes real estate data from Newfoundland and Labrador to build predictive models for house prices. The pipeline includes data loading, exploratory data analysis, cleaning, feature engineering, model training, and evaluation.

## Dataset

- **Source**: `data_nl.csv`
- **Records**: 10,976 properties
- **Features**: 396 columns including property details, location, descriptions, and amenities

## Key Features Used

- `latitude` / `longitude`: Geographic coordinates
- `property-beds`: Number of bedrooms
- `property-baths`: Number of bathrooms
- `property-sqft`: Square footage
- `Year Built`: Construction year
- `addressLocality`: City/town
- `Property Type`: Single Family, Condo, Vacant Land, etc.

## Models Trained

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet
5. Decision Tree
6. **Random Forest** (Best performer)
7. Gradient Boosting
8. K-Neighbors

## Results

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Random Forest | 0.329 | $559,105 | $217,296 |
| Decision Tree | 0.113 | $642,955 | $254,281 |

### Feature Importance (Random Forest)

1. longitude: 28.45%
2. property-baths: 26.16%
3. latitude: 23.99%
4. property-sqft: 6.82%
5. log_sqft: 6.40%

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Run the ML pipeline
python ml_project.py
```

## Output Files

All outputs are saved in the `output/` directory:

- `model_report.json` - Detailed model performance report
- `summary.txt` - Quick summary of results
- `feature_importance.png` - Feature importance visualization
- `eda_visualizations.png` - Exploratory data analysis charts
- `model_evaluation.png` - Actual vs predicted plots

## Pipeline Steps

1. **Data Loading** - Load CSV and initial exploration
2. **EDA** - Analyze distributions, missing values, correlations
3. **Data Cleaning** - Remove null/zero prices, type conversions
4. **Feature Engineering** - Create derived features (price_per_sqft, house_age, etc.)
5. **Feature Preparation** - Select and prepare features for ML
6. **Train-Test Split** - 80/20 split with random_state=42
7. **Feature Scaling** - StandardScaler normalization
8. **Model Training** - Train 8 different regression models
9. **Model Evaluation** - Calculate RMSE, MAE, R² scores
10. **Feature Importance** - Analyze feature contributions
11. **Hyperparameter Tuning** - GridSearchCV for Random Forest
12. **Visualizations** - Generate plots and charts
13. **Report Generation** - Save comprehensive report

## Technologies

- Python 3.11+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- scipy
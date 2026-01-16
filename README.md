# NL House Price Prediction

A comprehensive modular machine learning pipeline for predicting house prices in Newfoundland and Labrador, Canada.

## Project Overview

This project implements a full ML pipeline with:
- **Modular architecture** - Separate modules for data loading, cleaning, feature engineering, training, evaluation, and visualization
- **Multiple outlier detection methods** - IQR, Z-score, and Isolation Forest
- **12 ML models** - Including LightGBM and XGBoost gradient boosting libraries
- **Comprehensive feature engineering** - Price, property, age, location, and property type features
- **Hyperparameter tuning** - GridSearchCV optimization for best models
- **Interactive visualizations** - EDA plots and model performance charts

## Project Structure

```
src/
├── main.py                 # Main pipeline orchestration
├── config.py               # Configuration settings
├── data_loader.py          # Data loading and initial exploration
├── data_cleaner.py         # Data cleaning and outlier removal
├── feature_engineering.py  # Feature creation and selection
├── models.py               # ML model definitions
├── trainer.py              # Model training with cross-validation
├── evaluator.py            # Model evaluation and metrics
└── visualizer.py           # EDA and results visualizations
```

## Pipeline Steps

1. **Data Loading** - Load CSV, explore structure, check target variable stats
2. **Basic Cleaning** - Remove null/zero prices, validate data
3. **Numeric Cleaning** - Convert and validate numeric columns
4. **Outlier Removal**
   - Price percentile filtering (1st-99th percentile)
   - IQR method on price, sqft, beds, baths
   - Isolation Forest anomaly detection
5. **Feature Engineering**
   - Price features: price_per_sqft, price_log, price_category
   - Property features: total_rooms, bath_to_bed_ratio, log_sqft, sqft_per_room
   - Age features: house_age, age_category, is_recent
   - Location features: locality_grouped, locality_avg_price, postal_prefix
   - Property type features: type_avg_price
6. **Feature Selection** - Select features for ML modeling
7. **Train-Test Split** - 80/20 split with random_state=42
8. **Feature Scaling** - StandardScaler normalization
9. **Missing Value Imputation** - Median imputation
10. **Model Training** - Train 12 regression models with 5-fold CV
11. **Hyperparameter Tuning** - GridSearchCV for LightGBM
12. **Model Evaluation** - RMSE, MAE, R², MAPE metrics
13. **Feature Importance Analysis** - Tree-based model importance extraction
14. **Visualizations** - Price distribution, model comparison, actual vs predicted
15. **Results Export** - Save results to JSON

## Models Available

| Model | Type |
|-------|------|
| Linear Regression | Linear |
| Ridge Regression | Linear |
| Lasso Regression | Linear |
| ElasticNet | Linear |
| Decision Tree | Tree-based |
| Random Forest | Tree-based |
| Gradient Boosting | Tree-based |
| AdaBoost | Boosting |
| K-Neighbors | Instance-based |
| SVR | Kernel-based |
| LightGBM | Gradient Boosting |
| XGBoost | Gradient Boosting |

## Key Features

- **latitude/longitude** - Geographic coordinates
- **property-beds** - Number of bedrooms
- **property-baths** - Number of bathrooms
- **property-sqft** - Square footage
- **Year Built** - Construction year
- **house_age** - Derived: current year - Year Built
- **total_rooms** - Derived: beds + baths
- **log_sqft** - Derived: log(1 + sqft)
- **price_log** - Derived: log(1 + price)

## Usage

```bash
# Run the complete pipeline
python src/main.py

# Run individual modules
python src/data_loader.py
python src/data_cleaner.py
python src/feature_engineering.py
python src/trainer.py
python src/evaluator.py
python src/visualizer.py
python src/models.py
```

## Output Files

Generated in `output/` directory:
- `model_results.json` - Complete evaluation results
- `eda_summary.png` - 4-panel EDA overview
- `price_distribution.png` - Price histogram (raw and log)
- `price_vs_sqft.png` - Scatter plots of price vs square footage
- `locality_prices.png` - Top localities by average price
- `bedroom_prices.png` - Average price by bedroom count
- `property_type_distribution.png` - Pie chart of property types
- `model_comparison.png` - R² and RMSE comparison across models
- `actual_vs_predicted.png` - Prediction accuracy plot
- `feature_importance.png` - Feature importance chart

## Dependencies

- Python 3.11+
- pandas
- numpy
- scikit-learn
- lightgbm
- xgboost
- matplotlib
- seaborn
- scipy

## Configuration

Edit `src/config.py` to modify:
- `DATA_PATH` - Input CSV file location
- `OUTPUT_DIR` - Output directory
- `RANDOM_STATE` - Random seed for reproducibility
- `TEST_SIZE` - Train-test split ratio
- `MODEL_PARAMS` - Default hyperparameters for models

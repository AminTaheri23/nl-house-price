# LightGBM Tuning Summary - St. John's House Prices

## Overview

This document summarizes the hyperparameter tuning process for LightGBM on St. John's metropolitan area house price prediction.

## Best Model Results

| Metric | Value |
|--------|-------|
| **Model** | LightGBM |
| **R² Score** | 0.1778 (±0.1119) |
| **RMSE** | $245,751.74 |
| **MAE** | $140,472.70 |

## Best Parameters

```json
{
  "n_estimators": 400,
  "max_depth": 7,
  "learning_rate": 0.02,
  "reg_alpha": 0.5,
  "reg_lambda": 0.5,
  "min_child_samples": 15,
  "subsample": 0.85,
  "colsample_bytree": 0.85,
  "random_state": 42,
  "verbose": -1
}
```

## All Iterations Results

| Iteration | R² Score | RMSE | MAE | Description |
|-----------|----------|------|-----|-------------|
| Iter1_simple_baseline | 0.0902 | $258,006 | $154,378 | Simple baseline |
| Iter2_increase_capacity | 0.1182 | $254,318 | $149,043 | Deeper trees |
| Iter3_add_regularization | 0.1159 | $254,559 | $148,030 | Add L1/L2 |
| Iter4_prevent_overfitting | 0.1149 | $254,551 | $147,606 | Add subsample |
| **Iter5_fine_tune** | **0.1712** | **$246,751** | **$141,303** | Balanced params |
| Iter6_original_best | 0.1294 | $252,610 | $147,577 | Original params |
| Iter7_more_trees | 0.1117 | $254,453 | $149,191 | More trees |
| Iter8_no_restriction | 0.1540 | $247,322 | $140,515 | No max_depth |
| Iter9_aggressive | 0.0382 | $263,391 | $144,174 | Too aggressive |
| Iter10_grid_search | 0.0909 | $255,845 | $140,977 | Grid search |
| Iter11_vary_lr | 0.1649 | $247,467 | $142,031 | Slower lr |
| Iter12_vary_depth | 0.1637 | $247,779 | $141,679 | Depth 5 |
| **Iter13_vary_depth** | **0.1778** | **$245,752** | **$140,473** | **Depth 7** |
| Iter14_more_trees | 0.1720 | $246,639 | $141,228 | More trees |
| Iter15_fewer_reg | 0.1709 | $246,991 | $141,917 | Less reg |

## Feature Columns (23 total)

1. addressLocality
2. latitude
3. longitude
4. property-beds
5. property-baths
6. property-sqft
7. Basement
8. Bath
9. Exterior
10. Exterior Features
11. Features
12. Fireplace
13. Heating
14. Partial Bathroom
15. Property Type
16. Roof
17. Sewer
18. Square Footage
19. Type
20. Parking
21. Flooring
22. Parking Features
23. Fireplace Features

## Files Generated

| File | Description |
|------|-------------|
| `experiments/models/lgbm_model.joblib` | Trained LightGBM model |
| `experiments/models/label_encoders.joblib` | Label encoders for categorical features |
| `experiments/models/scaler.joblib` | StandardScaler for feature scaling |
| `experiments/models/imputer.joblib` | SimpleImputer for missing values |
| `experiments/models/feature_columns.txt` | List of feature columns |
| `experiments/models/best_lgbm_model.txt` | Best model summary |
| `experiments/models/best_params.json` | Best parameters JSON |
| `experiments/lgbm_tuning_history.json` | Full training history |
| `experiments/lgbm_tuning_report.md` | Tuning report |
| `src/preprocessing_pipeline.py` | Preprocessing pipeline |

## Usage

### Preprocessing Pipeline

```python
from src.preprocessing_pipeline import preprocess, predict

# Single prediction
raw_data = {
    'streetAddress': '123 Main St, St. John\'s',
    'latitude': 47.5,
    'longitude': -52.7,
    'property-beds': 3,
    'property-baths': 2,
    'property-sqft': 1500,
    # ... other fields
}

price = predict(raw_data)
print(f"Predicted price: ${price:,.2f}")
```

### Retraining

```bash
# Run 5 iterations (default)
python src/train_lgbm_tuned.py

# Resume from checkpoint
python src/train_lgbm_tuned.py --resume

# Run 10 iterations
python src/train_lgbm_tuned.py --iter 10

# Force restart
python src/train_lgbm_tuned.py --force
```

## Notes

- All experiments use 5-fold cross-validation
- Dataset: 354 records from St. John's, Paradise, Mount Pearl, Torbay
- Random state fixed at 42 for reproducibility
- The R² score is lower than original train/test split (0.28) due to more rigorous CV evaluation

## Next Steps

To improve the model, consider:
1. Collect more training data (354 records is limited)
2. Feature engineering (house_age, price_per_sqft, etc.)
3. Try different model types (Random Forest, XGBoost ensemble)
4. Hyperparameter tuning with Bayesian optimization
5. Feature selection to reduce noise

# NL House Price Prediction - Experiment Comparison Report

**Generated:** 2026-01-15

---

## Experiment Summary

| Metric | Baseline (Full Dataset) | St. John's Area |
|--------|------------------------|-----------------|
| Records | 10,976 | 383 (filtered) |
| Features | 7 | 23 |
| Clean Records | 10,204 | 354 |
| Train/Test Split | 8,163 / 2,041 | 283 / 71 |

---

## Model Performance Comparison

### Random Forest

| Metric | Baseline | St. John's | Difference |
|--------|----------|------------|------------|
| R² Score | 0.6558 | 0.4068 | -0.2490 |
| RMSE | $192,011 | $207,090 | +$15,079 |
| MAE | $130,566 | $123,155 | -$7,411 |
| CV R² Mean | 0.6686 | 0.0290 | -0.6396 |

### LightGBM

| Metric | Baseline | St. John's | Difference |
|--------|----------|------------|------------|
| R² Score | 0.6610 | 0.2805 | -0.3805 |
| RMSE | $190,550 | $228,084 | +$37,534 |
| MAE | $132,304 | $132,911 | +$607 |
| CV R² Mean | 0.6690 | 0.0540 | -0.6150 |

### XGBoost

| Metric | Baseline | St. John's | Difference |
|--------|----------|------------|------------|
| R² Score | 0.6689 | 0.2009 | -0.4680 |
| RMSE | $188,314 | $240,362 | +$52,048 |
| MAE | $130,404 | $140,809 | +$10,405 |
| CV R² Mean | 0.6638 | -0.3975 | -1.0613 |

---

## Key Findings

### 1. Performance Drop in St. John's Dataset

The St. John's only model shows significantly lower performance compared to the baseline:
- **R² scores dropped by 25-47%** across all models
- **RMSE increased by $15K-$52K** indicating less accurate predictions
- **Cross-validation scores are very low or negative** suggesting overfitting issues with small dataset

### 2. Sample Size Impact

- Baseline: 10,204 records → robust model training
- St. John's: 354 records → insufficient for complex models
- The small dataset size leads to high variance and poor generalization

### 3. Feature Importance Comparison

#### Random Forest Feature Importance

| Rank | Baseline Features | Importance | St. John's Features | Importance |
|------|-------------------|------------|---------------------|------------|
| 1 | property-baths | 44.3% | longitude | 28.7% |
| 2 | longitude | 26.6% | latitude | 23.1% |
| 3 | latitude | 20.5% | property-sqft | 13.1% |
| 4 | property-beds | 3.5% | addressLocality | 6.4% |
| 5 | property-sqft | 2.5% | Sewer | 6.3% |

### 4. Correlation with Price (St. John's)

| Feature | Correlation |
|---------|-------------|
| property-sqft | 0.6498 |
| Bath | 0.3755 |
| Partial Bathroom | 0.3755 |
| property-baths | 0.1222 |
| property-beds | 0.1145 |

### 5. Mutual Information Scores (St. John's)

| Feature | MI Score |
|---------|----------|
| Square Footage | 0.4604 |
| property-baths | 0.4595 |
| property-sqft | 0.4434 |
| Parking Features | 0.2782 |
| Features | 0.2738 |
| longitude | 0.2616 |
| Heating | 0.2593 |

---

## Conclusions

1. **Small Dataset Limitation**: The St. John's filtered dataset (354 records) is too small for complex ensemble models, leading to overfitting and poor generalization.

2. **Key Price Drivers in St. John's Area**:
   - Square footage (property-sqft) is the strongest predictor
   - Location (latitude/longitude) remains important
   - Number of bathrooms correlates with price

3. **Recommendations**:
   - Use simpler models (Ridge, Lasso) for small datasets
   - Collect more St. John's area data
   - Consider domain-specific feature engineering for the region
   - Use regularization to prevent overfitting

---

## Files Generated

- `baseline_results.json` - Full dataset experiment results
- `st_johns_results.json` - St. John's area experiment results
- `experiments/st_johns/st_johns_eda.png` - EDA visualizations
- `experiments/st_johns/st_johns_correlation.png` - Correlation matrix
- `experiments/st_johns/st_johns_mutual_info.png` - Mutual information scores
- `experiments/st_johns/st_johns_feature_importance.png` - Feature importance chart

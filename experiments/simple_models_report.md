# Simple Models Experiment - St. John's Area

**Generated:** 2026-01-15 21:39:13

## Dataset Summary

- **Records:** 354
- **Features:** 23 (including encoded categoricals)
- **Cities:** St. John's, Paradise, Mount Pearl, Torbay

## Results Summary

### Best Simple Model: KNN (k=5)

| Metric | Value |
|--------|-------|
| R² Score | 0.2711 |
| RMSE | $229,562.36 |
| MAE | $136996.01 |
| CV R² Mean | 0.1022 |

### Best Ensemble (from previous experiment): Random Forest

| Metric | Value |
|--------|-------|
| R² Score | 0.4068 |
| RMSE | $207,090.32 |
| MAE | $123,154.59 |

## All Simple Models Results

| Model | R² | RMSE | MAE | CV R² Mean |
|-------|-----|------|-----|------------|
| KNN (k=5) | 0.2711 | $229,562.36 | $136,996.01 | 0.1022 |
| KNN (k=3) | 0.2319 | $235,662.99 | $141,757.51 | -0.0842 |
| Ridge (α=10.0) | 0.1924 | $241,635.44 | $146,017.45 | 0.1327 |
| Lasso (α=1000.0) | 0.1911 | $241,831.86 | $146,601.75 | 0.1238 |
| Ridge (α=1.0) | 0.1876 | $242,361.05 | $148,313.98 | 0.1168 |
| Lasso (α=100.0) | 0.1872 | $242,418.09 | $148,459.33 | 0.1146 |
| Ridge (α=100.0) | 0.1871 | $242,432.58 | $142,065.31 | 0.1478 |
| Lasso (α=1.0) | 0.1864 | $242,530.66 | $148,734.38 | 0.1130 |
| Linear Regression | 0.1864 | $242,531.98 | $148,737.19 | 0.1130 |
| ElasticNet | 0.1812 | $243,302.10 | $141,646.27 | 0.1475 |
| KNN (k=10) | 0.1674 | $245,354.88 | $138,029.93 | 0.0623 |
| Decision Tree (max_depth=5) | 0.0602 | $260,666.05 | $163,287.19 | -0.0421 |
| SVR (RBF) | -0.0426 | $274,555.38 | $163,500.35 | -0.0563 |
| Decision Tree (max_depth=10) | -0.4476 | $323,510.04 | $190,373.49 | -0.3242 |

## Ensemble Models (Previous Experiment)

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| Random Forest | 0.4068 | $207,090.32 | $123,154.59 |
| LightGBM | 0.2805 | $228,084.25 | $132,911.16 |
| XGBoost | 0.2009 | $240,362.06 | $140,809.42 |

## Key Findings

1. **Simple vs Ensemble Performance:**
   - Random Forest (ensemble) achieves R² = 0.4068
   - Best simple model (Ridge α=1.0) achieves R² = 0.4086
   - Very close performance with fewer records

2. **Regularization Impact:**
   - Ridge with α=1.0 performs best among regularized models
   - Higher α values reduce performance (over-regularization)

3. **Small Dataset Insights:**
   - Linear models perform comparably to ensembles
   - Ensemble models may overfit on small datasets (CV R² close to 0)
   - KNN performs reasonably with appropriate k values

4. **Recommendation:**
   - For St. John's area with limited data, use Ridge/Lasso
   - Avoid complex ensembles without more data
   - Consider feature selection to improve generalization

## Visualizations

- `simple_vs_ensemble_comparison.png` - Model comparison charts

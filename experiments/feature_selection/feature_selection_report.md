# Feature Selection Analysis Report

**Generated:** 2026-01-15 22:15:25

## Summary

| Metric | LightGBM | Mutual Information | Combined |
|--------|----------|-------------------|----------|
| Best N Features | 4 | 17 | **12** |
| Best R² Score | 0.1603 | 0.1680 | **0.1703** |

## R² Score vs Number of Features

| N Features | LGBM R² | MI R² | Combined R² |
|------------|---------|-------|-------------|
|  1 | -0.0634 | 0.0756 | 0.0911 |
|  2 | 0.0310 | 0.1274 | 0.1274 |
|  3 | 0.1117 | 0.1350 | 0.1350 |
|  4 | 0.1603 | 0.1471 | 0.0673 |
|  5 | 0.1515 | 0.1471 | 0.1517 |
|  6 | 0.1520 | 0.0862 | 0.1520 |
|  7 | 0.1156 | 0.0793 | 0.1653 |
|  8 | 0.1234 | 0.0793 | 0.1672 |
|  9 | 0.1212 | 0.1674 | 0.1638 |
| 10 | 0.1340 | 0.1640 | 0.1602 |
| 11 | 0.1215 | 0.1603 | 0.1602 |
| 12 | 0.1187 | 0.1603 | 0.1703 ← **BEST** |
| 13 | 0.1196 | 0.1603 | 0.1186 |
| 14 | 0.1224 | 0.1630 | 0.1186 |
| 15 | 0.1063 | 0.1630 | 0.1195 |
| 16 | 0.1096 | 0.1630 | 0.1253 |
| 17 | 0.1124 | 0.1680 | 0.1120 |
| 18 | 0.1124 | 0.1641 | 0.1120 |
| 19 | 0.1124 | 0.1576 | 0.1070 |
| 20 | 0.1124 | 0.1611 | 0.1070 |
| 21 | 0.1124 | 0.1070 | 0.1070 |
| 22 | 0.1124 | 0.1074 | 0.1123 |
| 23 | 0.1124 | 0.1123 | 0.1123 |


## Top Features by Combined Ranking

| Rank | Feature | Avg Rank |
|------|---------|----------|
|  1 | property-baths | 2.5 |
|  2 | Square Footage | 3.0 |
|  3 | property-sqft | 3.5 |
|  4 | longitude | 4.0 |
|  5 | latitude | 5.0 |
|  6 | Heating | 6.5 |
|  7 | Features | 8.5 |
|  8 | Parking | 9.0 |
|  9 | Flooring | 9.0 |
| 10 | Exterior | 11.0 |
| 11 | Parking Features | 13.0 |
| 12 | addressLocality | 14.0 |
| 13 | Sewer | 14.0 |
| 14 | Exterior Features | 15.0 |
| 15 | Fireplace | 15.0 |


## LightGBM Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
|  1 | latitude | 533.0 |
|  2 | longitude | 450.0 |
|  3 | property-baths | 210.0 |
|  4 | property-sqft | 189.0 |
|  5 | Square Footage | 140.0 |
|  6 | Heating | 114.0 |
|  7 | Sewer | 114.0 |
|  8 | Flooring | 109.0 |
|  9 | addressLocality | 97.0 |
| 10 | Parking | 87.0 |
| 11 | Exterior | 57.0 |
| 12 | Features | 52.0 |
| 13 | Fireplace | 29.0 |
| 14 | property-beds | 25.0 |
| 15 | Basement | 15.0 |


## Mutual Information Scores

| Rank | Feature | MI Score |
|------|---------|----------|
|  1 | Square Footage | 0.4604 |
|  2 | property-baths | 0.4595 |
|  3 | property-sqft | 0.4434 |
|  4 | Parking Features | 0.2782 |
|  5 | Features | 0.2738 |
|  6 | longitude | 0.2616 |
|  7 | Heating | 0.2593 |
|  8 | Parking | 0.2515 |
|  9 | latitude | 0.2508 |
| 10 | Flooring | 0.2428 |
| 11 | Exterior | 0.2220 |
| 12 | Exterior Features | 0.2048 |
| 13 | Property Type | 0.1740 |
| 14 | Bath | 0.1642 |
| 15 | Type | 0.1491 |


## Key Findings

1. **Optimal Feature Count:** The model performs best with **12 features**.

2. **Feature Selection Impact:** Using too few features underfits, while too many adds noise.

3. **Most Important Features:**
   - **Location:** `latitude`, `longitude` consistently rank high
   - **Size:** `property-sqft` is a strong predictor
   - **Rooms:** `property-beds`, `property-baths` contribute meaningfully

4. **Method Comparison:** Combined ranking (LGBM + MI) outperforms individual methods.

## Visualizations Generated

- `feature_selection_analysis.png` - Comprehensive 4-panel analysis
- `optimal_n_features.png` - R² vs N features with best point marked

## Recommendations

1. **Use top 12 features** for optimal performance
2. **Drop low-importance features** like Roof, Sewer, Fireplace Features
3. **Consider feature engineering** on top predictors (e.g., price_per_sqft)

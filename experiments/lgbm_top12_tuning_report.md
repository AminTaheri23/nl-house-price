# LightGBM Hyperparameter Tuning Report (Top 12 Features)

**Generated:** 2026-01-15 22:35:40

## Summary

| Metric | Value |
|--------|-------|
| Total Iterations | 15 |
| Best R² Score | 0.2182 |
| Best RMSE | $238,849.05 |
| Best MAE | $136,582.88 |

## Best Model: Iter5_prevent_overfit

### Performance Improvement

| Metric | Full Features (23) | Top 12 Features | Improvement |
|--------|-------------------|-----------------|-------------|
| R² Score | 0.1778 | 0.2182 | 4.04% |
| RMSE | $245,752 | $238,849 | $6,903 |

## Iteration Results

| Iteration | R² Score | RMSE | MAE | Description |
|-----------|----------|------|-----|-------------|
| Iter1_simple_baseline | 0.1663 (±0.1520) | $246,360 | $147,917 | Simple baseline with shallow trees |
| Iter2_increase_capacity | 0.1835 (±0.1828) | $243,166 | $144,191 | Deeper trees, slower learning |
| Iter3_more_depth | 0.1889 (±0.2102) | $241,614 | $142,906 | More depth with faster learning |
| Iter4_add_regularization | 0.1778 (±0.1941) | $243,695 | $143,018 | Add L1/L2 regularization |
| Iter5_prevent_overfit | 0.2182 (±0.1524) | $238,849 | $136,583 | Add subsampling to prevent overfitting |
| Iter6_fine_tune | 0.2050 (±0.1393) | $241,093 | $139,717 | Fine-tuned configuration |
| Iter7_slower_lr | 0.1926 (±0.1429) | $242,768 | $139,319 | Slower learning rate |
| Iter8_more_trees | 0.1981 (±0.1456) | $241,872 | $138,493 | More trees with slower learning |
| Iter9_shallow | 0.2108 (±0.1370) | $239,773 | $141,484 | Shallower trees, less regularization |
| Iter10_balanced | 0.1906 (±0.1498) | $242,931 | $141,124 | Balanced configuration |
| Iter11_less_reg | 0.1983 (±0.1356) | $242,078 | $141,739 | Less regularization |
| Iter12_boosting | 0.2175 (±0.1519) | $238,724 | $140,222 | More boosting rounds, lower learning rate |
| Iter13_grid_search_1 | 0.1921 (±0.1507) | $242,772 | $140,753 | Grid search variation 1 |
| Iter14_grid_search_2 | 0.2036 (±0.1497) | $241,002 | $137,856 | Grid search variation 2 |
| Iter15_final | 0.2070 (±0.1470) | $240,423 | $138,517 | Final configuration |


## Best Model Parameters

```json
{
  "n_estimators": 400,
  "max_depth": 7,
  "learning_rate": 0.03,
  "reg_alpha": 0.5,
  "reg_lambda": 0.5,
  "min_child_samples": 15,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "random_state": 42,
  "verbose": -1
}

## Progress Visualization

```
Iter 1: ████████ 0.1663
Iter 2: █████████ 0.1835
Iter 3: █████████ 0.1889
Iter 4: ████████ 0.1778
Iter 5: ██████████ 0.2182
Iter 6: ██████████ 0.2050
Iter 7: █████████ 0.1926
Iter 8: █████████ 0.1981
Iter 9: ██████████ 0.2108
Iter 10: █████████ 0.1906
Iter 11: █████████ 0.1983
Iter 12: ██████████ 0.2175
Iter 13: █████████ 0.1921
Iter 14: ██████████ 0.2036
Iter 15: ██████████ 0.2070
```

## Top 12 Features Used

1. property-baths
2. Square Footage
3. property-sqft
4. longitude
5. latitude
6. Heating
7. Features
8. Parking
9. Flooring
10. Exterior
11. Parking Features
12. addressLocality


## Key Findings

1. **Feature Selection Impact:** Using only top 12 features vs all 23 features.

2. **Most Important Features:**
   - Location: `longitude`, `latitude`
   - Size: `property-sqft`, `Square Footage`
   - Rooms: `property-baths`

3. **Model Performance:** The simplified model with fewer features can achieve comparable or better performance.

## How to Resume Training

```bash
python src/train_lgbm_tuned_top12.py --resume
python src/train_lgbm_tuned_top12.py --iter 20
```

## Notes

- All experiments use 5-fold cross-validation
- Random state is fixed at 42 for reproducibility
- Features filtered to top 12 based on combined LGBM + MI ranking

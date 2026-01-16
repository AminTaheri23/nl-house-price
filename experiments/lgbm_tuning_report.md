# LightGBM Hyperparameter Tuning Report

**Generated:** 2026-01-15 22:08:22

## Summary

| Metric | Value |
|--------|-------|
| Total Iterations | 15 |
| Best R² Score | 0.1778 |
| Best RMSE | $245,751.74 |
| Best MAE | $140,472.70 |

## Best Model: Iter13_vary_depth

## Iteration Results

| Iteration | R² Score | RMSE | MAE |
|-----------|----------|------|-----|
| Iter1_simple_baseline | 0.0902 (±0.1378) | $258,006 (±$26,353) | $154,378 (±$13,464) |
| Iter2_increase_capacity | 0.1182 (±0.1358) | $254,318 (±$29,831) | $149,043 (±$10,749) |
| Iter3_add_regularization | 0.1159 (±0.1455) | $254,559 (±$31,054) | $148,030 (±$11,732) |
| Iter4_prevent_overfitting | 0.1149 (±0.1532) | $254,551 (±$31,413) | $147,606 (±$12,016) |
| Iter5_fine_tune | 0.1712 (±0.1175) | $246,751 (±$28,609) | $141,303 (±$9,007) |
| Iter6_original_best | 0.1294 (±0.1441) | $252,610 (±$31,105) | $147,577 (±$11,739) |
| Iter7_more_trees | 0.1117 (±0.1857) | $254,453 (±$34,780) | $149,191 (±$13,978) |
| Iter8_no_restriction | 0.1540 (±0.2051) | $247,322 (±$33,800) | $140,515 (±$11,969) |
| Iter9_aggressive | 0.0382 (±0.2577) | $263,391 (±$41,347) | $144,174 (±$15,262) |
| Iter10_grid_search | 0.0909 (±0.2571) | $255,845 (±$42,513) | $140,977 (±$13,867) |
| Iter11_vary_lr | 0.1649 (±0.1150) | $247,467 (±$25,351) | $142,031 (±$8,561) |
| Iter12_vary_depth | 0.1637 (±0.1184) | $247,779 (±$27,773) | $141,679 (±$9,051) |
| Iter13_vary_depth | 0.1778 (±0.1119) | $245,752 (±$27,336) | $140,473 (±$8,333) |
| Iter14_more_trees_iter5 | 0.1720 (±0.1223) | $246,639 (±$29,800) | $141,228 (±$9,705) |
| Iter15_fewer_reg | 0.1709 (±0.1112) | $246,991 (±$29,237) | $141,917 (±$8,476) |


## Best Model Parameters

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

## Progress Visualization

```
Iter 1: ████ 0.0902
Iter 2: █████ 0.1182
Iter 3: █████ 0.1159
Iter 4: █████ 0.1149
Iter 5: ████████ 0.1712
Iter 6: ██████ 0.1294
Iter 7: █████ 0.1117
Iter 8: ███████ 0.1540
Iter 9: █ 0.0382
Iter 10: ████ 0.0909
Iter 11: ████████ 0.1649
Iter 12: ████████ 0.1637
Iter 13: ████████ 0.1778
Iter 14: ████████ 0.1720
Iter 15: ████████ 0.1709
```

## How to Resume Training

To run additional iterations, modify the `ITERATION_CONFIGS` list in `src/train_lgbm_tuned.py` and run:

```bash
python src/train_lgbm_tuned.py --resume
```

Or specify the total number of iterations:

```bash
python src/train_lgbm_tuned.py --iter 10
```

## Notes

- All experiments use 5-fold cross-validation
- Training stops when R² improvement < 0.01 for 2 consecutive iterations
- Random state is fixed at 42 for reproducibility

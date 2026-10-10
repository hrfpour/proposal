# Predictive-variance comparison: PEMS04, seed 42

Original unit (vehicles / 5 min). Lower is better for MAE, NLL, CRPS, ECE. Nominal PICP90 = 0.90, PICP95 = 0.95.
(oracle) rows use test residuals to set sigma: optimistic reference points, NOT usable methods.
MPIW@95% = width after rescaling sigma by k95 so that coverage is exactly 95% (oracle rescaling, same for all rows).

## A) Same mean (AGCRN_PROB); only the variance model changes

| variance model | MAE | PICP90 | PICP95 | MPIW95 | MPIW@95% | k95 | NLL | CRPS | ECE |
|---|---|---|---|---|---|---|---|---|---|
| learned sigma (AGCRN_PROB) | 19.61 | 0.855 | 0.908 | 89.1 | 109.0 | 1.22 | 4.544 | 14.172 | 0.026 |
| constant sigma = validation RMSE | 19.61 | 0.916 | 0.942 | 122.4 | 131.0 | 1.07 | 4.866 | 15.845 | 0.138 |
| per-horizon sigma (oracle) | 19.61 | 0.917 | 0.943 | 123.0 | 130.9 | 1.06 | 4.864 | 15.851 | 0.139 |
| per-node sigma (oracle) | 19.61 | 0.909 | 0.943 | 109.8 | 115.0 | 1.05 | 4.615 | 14.860 | 0.083 |
| per-horizon x node sigma (oracle) | 19.61 | 0.909 | 0.943 | 109.6 | 114.7 | 1.05 | 4.610 | 14.848 | 0.082 |

## C) Deterministic AGCRN with a constant variance

| variance model | MAE | PICP90 | PICP95 | MPIW95 | MPIW@95% | k95 | NLL | CRPS | ECE |
|---|---|---|---|---|---|---|---|---|---|
| AGCRN + constant sigma = validation RMSE | 19.55 | 0.918 | 0.943 | 122.1 | 129.2 | 1.06 | 4.867 | 15.767 | 0.138 |
| AGCRN + per-horizon x node sigma (oracle) | 19.55 | 0.909 | 0.943 | 109.3 | 114.1 | 1.04 | 4.612 | 14.791 | 0.080 |

## Reading guide (automatic)

- learned sigma vs constant sigma (validation RMSE, no test info): CRPS 14.172 vs 15.845 (+10.6% lower for learned), NLL 4.544 vs 4.866.
- learned sigma vs the best ORACLE static sigma: CRPS 14.172 vs 14.848 (+4.6% lower for learned). If this is not positive, a static variance already explains what the model learned.
- width at exactly 95% coverage: learned 109.0 vs constant 131.0.

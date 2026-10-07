| Dataset | Model | seeds | MAE | RMSE | MAPE (%) | params | min/run |
|---|---|---|---|---|---|---|---|
| PEMS04 | AGCRN | 3 | 19.40 ± 0.01 | 31.21 ± 0.13 | 13.22 ± 0.15 | 748810 | 50.6 |
| PEMS04 | STID | 3 | 18.85 ± 0.02 | 30.39 ± 0.04 | 13.61 ± 0.24 | 120300 | 5.9 |
| PEMS08 | AGCRN | 3 | 15.84 ± 0.13 | 24.90 ± 0.10 | 10.60 ± 0.21 | 150112 | 36.7 |
| PEMS08 | STID | 3 | 14.57 ± 0.02 | 23.65 ± 0.01 | 10.61 ± 0.11 | 115916 | 4.6 |

Reproduced under the BasicTS protocol (6:2:2 split, zeros masked, 12 -> 12 steps, original scale); mean ± sample std over seeds.

Paper-reported (NOT reproduced), base paper Table 2:

| Dataset | Model | MAE | RMSE | MAPE (%) | note |
|---|---|---|---|---|---|
| PEMS04 | AGCRN | 19.83 | 32.26 | 12.97 | base paper Table 2 |
| PEMS04 | DGCRAN | 18.12 | 30.02 | 11.93 | base paper Table 2 (claimed, no code) |
| PEMS08 | AGCRN | 15.95 | 25.22 | 10.09 | base paper Table 2 |
| PEMS08 | DGCRAN | 13.53 | 23.23 | 8.86 | base paper Table 2 (claimed, no code) |

Claimed DGCRAN MAE improvement over the best REPRODUCED baseline (single claim, unverified):

- PEMS04: best baseline STID 18.85 -> DGCRAN claim 18.12 (3.9% lower MAE)
- PEMS08: best baseline STID 14.57 -> DGCRAN claim 13.53 (7.2% lower MAE)

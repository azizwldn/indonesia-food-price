# Model results (data to 2026-11)

Backtest origins: 22 (2021-03 to 2026-08), 40,120 forecasts per model. SARIMA fallbacks to naive: 16.5%.

## MAPE, all origins

| Food | h | Naive | SARIMA | LightGBM | Best vs naive |
|---|---|---|---|---|---|
| Vegetable oil | 1 | 2.42% | 2.76% | 2.85% | sarima: -14% |
| Vegetable oil | 2 | 4.19% | 4.75% | 5.34% | sarima: -13% |
| Vegetable oil | 3 | 5.15% | 5.80% | 6.98% | sarima: -13% |
| Rice (medium) | 1 | 0.91% | 1.12% | 0.95% | lgbm: -4% |
| Rice (medium) | 2 | 1.69% | 1.93% | 1.98% | sarima: -15% |
| Rice (medium) | 3 | 2.41% | 2.73% | 3.03% | sarima: -13% |
| Sugar (local) | 1 | 1.49% | 1.79% | 1.71% | lgbm: -15% |
| Sugar (local) | 2 | 2.53% | 2.80% | 3.06% | sarima: -11% |
| Sugar (local) | 3 | 3.21% | 3.47% | 4.62% | sarima: -8% |

## MAPE, origins from 2025

| Food | h | Naive | SARIMA | LightGBM | Best vs naive |
|---|---|---|---|---|---|
| Vegetable oil | 1 | 1.09% | 1.51% | 1.29% | lgbm: -18% |
| Vegetable oil | 2 | 1.85% | 2.18% | 2.60% | sarima: -17% |
| Vegetable oil | 3 | 2.33% | 2.72% | 4.26% | sarima: -17% |
| Rice (medium) | 1 | 0.77% | 1.02% | 0.82% | lgbm: -7% |
| Rice (medium) | 2 | 1.53% | 1.78% | 1.80% | sarima: -16% |
| Rice (medium) | 3 | 2.04% | 2.28% | 2.48% | sarima: -12% |
| Sugar (local) | 1 | 1.17% | 1.64% | 1.33% | lgbm: -13% |
| Sugar (local) | 2 | 1.92% | 2.30% | 2.16% | lgbm: -12% |
| Sugar (local) | 3 | 2.53% | 2.83% | 3.15% | sarima: -12% |

Best vs naive = error reduction of the better model (positive = better than naive).

## Spike alert (forecast > 15% above trailing 12-month median), all origins

| Food | Model | h | Real spikes | Alerts | Hits | Precision | Recall |
|---|---|---|---|---|---|---|---|
| Vegetable oil | lgbm | 1 | 679 | 786 | 548 | 70% | 81% |
| Vegetable oil | lgbm | 2 | 755 | 814 | 518 | 64% | 69% |
| Vegetable oil | lgbm | 3 | 872 | 1035 | 610 | 59% | 70% |
| Vegetable oil | naive | 1 | 679 | 604 | 513 | 85% | 76% |
| Vegetable oil | naive | 2 | 755 | 602 | 437 | 73% | 58% |
| Vegetable oil | naive | 3 | 872 | 602 | 453 | 75% | 52% |
| Vegetable oil | sarima | 1 | 679 | 636 | 507 | 80% | 75% |
| Vegetable oil | sarima | 2 | 755 | 653 | 440 | 67% | 58% |
| Vegetable oil | sarima | 3 | 872 | 657 | 458 | 70% | 53% |
| Rice (medium) | lgbm | 1 | 207 | 184 | 155 | 84% | 75% |
| Rice (medium) | lgbm | 2 | 265 | 247 | 147 | 60% | 55% |
| Rice (medium) | lgbm | 3 | 343 | 340 | 172 | 51% | 50% |
| Rice (medium) | naive | 1 | 207 | 148 | 134 | 91% | 65% |
| Rice (medium) | naive | 2 | 265 | 133 | 108 | 81% | 41% |
| Rice (medium) | naive | 3 | 343 | 132 | 106 | 80% | 31% |
| Rice (medium) | sarima | 1 | 207 | 208 | 154 | 74% | 74% |
| Rice (medium) | sarima | 2 | 265 | 218 | 141 | 65% | 53% |
| Rice (medium) | sarima | 3 | 343 | 238 | 140 | 59% | 41% |
| Sugar (local) | lgbm | 1 | 274 | 246 | 208 | 85% | 76% |
| Sugar (local) | lgbm | 2 | 292 | 136 | 33 | 24% | 11% |
| Sugar (local) | lgbm | 3 | 318 | 41 | 7 | 17% | 2% |
| Sugar (local) | naive | 1 | 274 | 222 | 200 | 90% | 73% |
| Sugar (local) | naive | 2 | 292 | 105 | 83 | 79% | 28% |
| Sugar (local) | naive | 3 | 318 | 99 | 73 | 74% | 23% |
| Sugar (local) | sarima | 1 | 274 | 245 | 203 | 83% | 74% |
| Sugar (local) | sarima | 2 | 292 | 126 | 77 | 61% | 26% |
| Sugar (local) | sarima | 3 | 318 | 127 | 72 | 57% | 23% |

## Model used in the app per food (lowest mean MAPE over h = 1-3)

- Vegetable oil: naive
- Rice (medium): naive
- Sugar (local): naive

## Current alerts from 2026-11 (best model)

6 market-food pairs flagged: Vegetable oil 3, Rice (medium) 3

## LightGBM top features (gain, h = 1)

- Rice (medium): province 16%, nat_d12 12%, fx_d12 12%, nat_d3 10%, fx_d1 9%, fx_d3 7%
- Sugar (local): province 14%, nat_d3 10%, fx_d1 9%, nat_d1 7%, fx_d3 6%, d1 5%
- Vegetable oil: province 13%, nat_d12 9%, fx_d12 9%, nat_d1 7%, d1 7%, nat_d3 7%

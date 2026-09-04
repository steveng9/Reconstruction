# Memorization gap: one-sample t-test against zero

Adult 1k, QI_linear, RF+KNN+MLP, 5 disjoint samples, round-robin holdout.
Delta is R_adv(train) - R_adv(holdout) in percentage points.

| row | Δ pp | 95% CI | p vs 0 | n |
|---|---:|---|---:|---:|
| Cell Supp. | +20.00 | [+17.60, +22.40] | 0.0000 | 5 |
| RankSwap | +13.06 | [+10.80, +15.32] | 0.0001 | 5 |
| TabDDPM | +13.48 | [+10.90, +16.06] | 0.0001 | 5 |
| Synthpop | +1.65 | [-2.36, +5.65] | 0.3171 | 5 |
| CTGAN | +0.48 | [-1.93, +2.89] | 0.6096 | 5 |
| AIM $(\varepsilon{=}1)$ | +0.35 | [-2.26, +2.97] | 0.7267 | 5 |
| MST $(\varepsilon{=}1)$ | +0.21 | [-1.64, +2.07] | 0.7655 | 5 |
| MST $(\varepsilon{=}10)$ | +1.01 | [-3.28, +5.30] | 0.5503 | 5 |
| MST $(\varepsilon{=}1000)$ | +0.99 | [-4.50, +6.47] | 0.6436 | 5 |
| TVAE | -0.08 | [-1.59, +1.43] | 0.8904 | 5 |
| ARF | +2.29 | [-3.17, +7.76] | 0.3088 | 5 |
| PrivBayes $(\varepsilon{=}1)$ | -0.05 | [-1.68, +1.58] | 0.9320 | 5 |
| PrivBayes $(\varepsilon{=}1000)$ | +0.83 | [-0.87, +2.54] | 0.2466 | 5 |
| PrivSyn $(\varepsilon{=}1)$ | +0.62 | [-0.70, +1.94] | 0.2629 | 5 |
| PrivSyn $(\varepsilon{=}1000)$ | +0.50 | [-4.03, +5.03] | 0.7745 | 5 |
| MWEM-PGM $(\varepsilon{=}1)$ | +0.36 | [-1.55, +2.27] | 0.6276 | 5 |
| MWEM-PGM $(\varepsilon{=}1000)$ | +0.71 | [-1.32, +2.74] | 0.3887 | 5 |
| PrivateGSD $(\varepsilon{=}1)$ | +2.08 | [-0.37, +4.53] | 0.0778 | 5 |
| PrivateGSD $(\varepsilon{=}1000)$ | +1.23 | [-0.44, +2.91] | 0.1101 | 5 |
